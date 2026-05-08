//! Transport seam for streaming an LLM chat completion.
//!
//! All chat traffic to the agent loop currently goes through Warp's backend
//! via [`ServerApi::generate_multi_agent_output`]. Adding a client-routed
//! Ollama provider requires plugging in a different transport without forking
//! the agent loop. This module introduces a small trait that the existing
//! server-routed path and the client-routed Ollama path both implement; the
//! agent loop only sees the trait.
//!
//! Status:
//! * [`ServerApiChatTransport`] — production path, behaviour-equivalent to the
//!   pre-refactor direct call.
//! * [`LocalOllamaChatTransport`] — chat-only translator over
//!   [`ai::ollama::OllamaTransport`]. See [`super::ollama_translate`] for the
//!   request/response translation matrix and supported scope.

use std::sync::Arc;

use anyhow::anyhow;
use async_trait::async_trait;
use warp_multi_agent_api as api;

use ::ai::ollama::OllamaTransport;

use crate::ai::llms::LLMModelHost;
use crate::server::server_api::{AIApiError, AIOutputStream, ServerApi};

use super::ollama_translate::{
    build_chat_request, chunks_to_response_events, pick_target_task_id, ResponseIds,
};

/// Streams an [`api::ResponseEvent`] sequence in response to a built
/// [`api::Request`].
///
/// Implementations must be cancellation-safe: dropping the returned stream
/// must abort any in-flight network work.
#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
pub trait LlmChatTransport: Send + Sync {
    async fn stream(
        &self,
        request: api::Request,
    ) -> Result<AIOutputStream<api::ResponseEvent>, Arc<AIApiError>>;
}

/// Server-routed transport. Forwards requests to
/// [`ServerApi::generate_multi_agent_output`].
pub struct ServerApiChatTransport {
    server_api: Arc<ServerApi>,
}

impl ServerApiChatTransport {
    pub fn new(server_api: Arc<ServerApi>) -> Self {
        Self { server_api }
    }
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl LlmChatTransport for ServerApiChatTransport {
    async fn stream(
        &self,
        request: api::Request,
    ) -> Result<AIOutputStream<api::ResponseEvent>, Arc<AIApiError>> {
        self.server_api.generate_multi_agent_output(&request).await
    }
}

/// Client-routed transport for a local (or self-hosted) Ollama daemon.
///
/// Holds an [`OllamaTransport`] (production: `HttpOllamaTransport`; tests
/// inject a fake) and a model tag stripped of the `ollama:` provider prefix
/// — see [`crate::ai::ollama_discovery`] for the prefix convention.
pub struct LocalOllamaChatTransport {
    ollama: Arc<dyn OllamaTransport>,
    /// The bare Ollama tag (e.g. `llama3.1:8b`). The wrapping `LLMId` (e.g.
    /// `ollama:llama3.1:8b`) is stripped at construction time.
    model_tag: String,
}

impl LocalOllamaChatTransport {
    pub fn new(ollama: Arc<dyn OllamaTransport>, model_tag: String) -> Self {
        Self { ollama, model_tag }
    }
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl LlmChatTransport for LocalOllamaChatTransport {
    async fn stream(
        &self,
        request: api::Request,
    ) -> Result<AIOutputStream<api::ResponseEvent>, Arc<AIApiError>> {
        let chat_request = build_chat_request(self.model_tag.clone(), &request);
        let target_task_id = pick_target_task_id(&request);
        let chunk_stream = self.ollama.chat_stream(chat_request).await.map_err(|e| {
            Arc::new(AIApiError::Other(anyhow!(
                "Ollama chat_stream failed: {e}"
            )))
        })?;
        Ok(chunks_to_response_events(
            chunk_stream,
            target_task_id,
            ResponseIds::new(),
        ))
    }
}

/// Picks the right transport for a model whose host is `host`.
///
/// Models discovered from a local Ollama daemon carry
/// `LLMModelHost::LocalOllama` in their `host_configs`; everything else
/// falls through to the server transport.
pub fn chat_transport_for_host(
    host: Option<&LLMModelHost>,
    ollama: Option<(Arc<dyn OllamaTransport>, String)>,
    server_api: Arc<ServerApi>,
) -> Arc<dyn LlmChatTransport> {
    match (host, ollama) {
        (Some(LLMModelHost::LocalOllama), Some((transport, model_tag))) => {
            Arc::new(LocalOllamaChatTransport::new(transport, model_tag))
        }
        _ => Arc::new(ServerApiChatTransport::new(server_api)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::ai::ollama::dto::{
        ChatMessage, ChatRequest, ChatRole, ChatStreamChunk, OllamaModelTag, ShowModelResponse,
    };
    use ::ai::ollama::OllamaError;
    use async_trait::async_trait;
    use futures::stream;
    use futures::stream::BoxStream;
    use futures::StreamExt;

    fn empty_request() -> api::Request {
        api::Request::default()
    }

    /// Test-only `LlmChatTransport` that yields a fixed sequence of events.
    /// Intended for callers that want to drive the agent loop without a real
    /// server — kept inside the test module so production code never depends
    /// on it.
    struct FakeChatTransport {
        events: std::sync::Mutex<Option<Vec<Result<api::ResponseEvent, Arc<AIApiError>>>>>,
    }

    impl FakeChatTransport {
        fn new(events: Vec<Result<api::ResponseEvent, Arc<AIApiError>>>) -> Self {
            Self {
                events: std::sync::Mutex::new(Some(events)),
            }
        }
    }

    #[async_trait]
    impl LlmChatTransport for FakeChatTransport {
        async fn stream(
            &self,
            _request: api::Request,
        ) -> Result<AIOutputStream<api::ResponseEvent>, Arc<AIApiError>> {
            let events = self
                .events
                .lock()
                .unwrap()
                .take()
                .expect("FakeChatTransport::stream called more than once");
            Ok(Box::pin(stream::iter(events)))
        }
    }

    /// Test double for `OllamaTransport` returning a canned chunk stream.
    struct StubOllama {
        chunks: std::sync::Mutex<Option<Vec<Result<ChatStreamChunk, OllamaError>>>>,
    }

    impl StubOllama {
        fn new(chunks: Vec<Result<ChatStreamChunk, OllamaError>>) -> Self {
            Self {
                chunks: std::sync::Mutex::new(Some(chunks)),
            }
        }
    }

    #[async_trait]
    impl OllamaTransport for StubOllama {
        async fn list_models(&self) -> Result<Vec<OllamaModelTag>, OllamaError> {
            Ok(vec![])
        }
        async fn show_model(&self, _name: &str) -> Result<ShowModelResponse, OllamaError> {
            Err(OllamaError::Protocol("not implemented in stub".into()))
        }
        async fn chat_stream(
            &self,
            _request: ChatRequest,
        ) -> Result<BoxStream<'static, Result<ChatStreamChunk, OllamaError>>, OllamaError>
        {
            let chunks = self
                .chunks
                .lock()
                .unwrap()
                .take()
                .expect("chat_stream called more than once");
            Ok(Box::pin(stream::iter(chunks)))
        }
    }

    fn task_with(messages: Vec<api::Message>) -> api::Task {
        api::Task {
            id: "task-1".into(),
            description: String::new(),
            dependencies: None,
            messages,
            summary: String::new(),
            server_data: String::new(),
        }
    }

    fn user_query_input(q: &str) -> api::request::Input {
        #[allow(deprecated)]
        api::request::Input {
            context: None,
            r#type: Some(api::request::input::Type::UserQuery(
                api::request::input::UserQuery {
                    query: q.to_string(),
                    referenced_attachments: Default::default(),
                    mode: None,
                    intended_agent: 0,
                },
            )),
        }
    }

    fn done_chunk(text: &str) -> ChatStreamChunk {
        ChatStreamChunk {
            model: "m".into(),
            created_at: None,
            message: ChatMessage {
                role: ChatRole::Assistant,
                content: text.into(),
                images: None,
                tool_calls: None,
                tool_call_id: None,
            },
            done: true,
            done_reason: Some("stop".into()),
            prompt_eval_count: None,
            eval_count: None,
            total_duration: None,
        }
    }

    #[tokio::test]
    async fn local_ollama_translates_a_full_chat() {
        let stub = Arc::new(StubOllama::new(vec![Ok(done_chunk("Hi!"))]));
        let transport = LocalOllamaChatTransport::new(stub, "llama3.1:8b".into());

        let request = api::Request {
            task_context: Some(api::request::TaskContext {
                tasks: vec![task_with(vec![])],
            }),
            input: Some(user_query_input("hello")),
            ..Default::default()
        };
        let mut stream = transport.stream(request).await.unwrap();
        let mut count = 0;
        while let Some(event) = stream.next().await {
            event.expect("each event ok");
            count += 1;
        }
        // Init + Begin/Add + Append + Commit + Finished
        assert_eq!(count, 5);
    }

    #[tokio::test]
    async fn fake_transport_round_trips_a_response_event() {
        // Sanity-check that the trait shape supports test doubles.
        let transport = FakeChatTransport::new(vec![Ok(api::ResponseEvent::default())]);

        let mut stream = transport.stream(empty_request()).await.unwrap();
        let first = stream.next().await.expect("one event").expect("ok");
        let _ = first;
        assert!(stream.next().await.is_none());
    }
}


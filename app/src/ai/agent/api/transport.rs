//! Transport seam for streaming an LLM chat completion.
//!
//! All chat traffic to the agent loop currently goes through Warp's backend
//! via [`ServerApi::generate_multi_agent_output`]. Adding a client-routed
//! Ollama provider requires plugging in a different transport without forking
//! the agent loop. This module introduces a small trait that the existing
//! server-routed path and the future Ollama path both implement; the agent
//! loop only sees the trait.
//!
//! Status:
//! * [`ServerApiChatTransport`] — production path, behaviour-equivalent to the
//!   pre-refactor direct call.
//! * [`LocalOllamaChatTransport`] — stub that compiles, has tests, and
//!   surfaces a clear error when invoked. The full
//!   `warp_multi_agent_api::Request` ↔ Ollama `/api/chat` translation lands
//!   together with the Ollama settings work in Milestone 3 of
//!   `specs/ollama-integration/PLAN.md`.

use std::sync::Arc;

use anyhow::anyhow;
use async_trait::async_trait;
use futures::stream;
use warp_multi_agent_api as api;

use crate::ai::llms::LLMModelHost;
use crate::server::server_api::{AIApiError, AIOutputStream, ServerApi};

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
/// **Stub**. The full `Request` ↔ Ollama-chat translation matrix is not yet
/// implemented; calling [`Self::stream`] returns a single-error stream that
/// the existing agent-loop error handling surfaces to the user. The HTTP
/// transport itself lives in [`crate::ai::ollama`] (re-exported from the
/// `ai` crate) and will be wired in once Ollama settings exist (M3).
#[allow(dead_code)] // Wired into routing in M3.
#[derive(Default)]
pub struct LocalOllamaChatTransport {
    // Intentionally empty until translation lands.
}

#[allow(dead_code)] // Stream impl is a stub until M4 lands; constructor is used by routing.
impl LocalOllamaChatTransport {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl LlmChatTransport for LocalOllamaChatTransport {
    async fn stream(
        &self,
        _request: api::Request,
    ) -> Result<AIOutputStream<api::ResponseEvent>, Arc<AIApiError>> {
        // Surface a single error event so callers see a real failure rather
        // than a silently-empty stream.
        let err = AIApiError::Other(anyhow!(
            "Ollama transport is not yet implemented. \
             Pick a server-routed model or disable the OllamaProvider feature flag."
        ));
        let stream = stream::once(async move { Err(Arc::new(err)) });
        Ok(Box::pin(stream))
    }
}

/// Picks the right transport for a model whose host is `host`.
///
/// Models discovered from a local Ollama daemon carry
/// `LLMModelHost::LocalOllama` in their `host_configs`; everything else
/// falls through to the server transport.
pub fn chat_transport_for_host(
    host: Option<&LLMModelHost>,
    server_api: Arc<ServerApi>,
) -> Arc<dyn LlmChatTransport> {
    match host {
        Some(LLMModelHost::LocalOllama) => Arc::new(LocalOllamaChatTransport::new()),
        _ => Arc::new(ServerApiChatTransport::new(server_api)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[tokio::test]
    async fn local_ollama_stub_returns_single_error_event() {
        let transport = LocalOllamaChatTransport::new();
        let mut stream = transport
            .stream(empty_request())
            .await
            .expect("stub must return a stream, not an immediate Err");

        let first = stream.next().await.expect("stub yields one event");
        assert!(first.is_err(), "stub must surface an error to the caller");

        assert!(
            stream.next().await.is_none(),
            "stub must terminate after the error event"
        );
    }

    #[tokio::test]
    async fn fake_transport_round_trips_a_response_event() {
        // Sanity-check that the trait shape supports test doubles. This also
        // covers the future M3 wiring point where a real fake will drive the
        // agent loop without any HTTP.
        let transport = FakeChatTransport::new(vec![Ok(api::ResponseEvent::default())]);

        let mut stream = transport.stream(empty_request()).await.unwrap();
        let first = stream.next().await.expect("one event").expect("ok");
        // ResponseEvent has no PartialEq; we only verify the stream yielded
        // exactly one Ok and then terminated.
        let _ = first;
        assert!(stream.next().await.is_none());
    }
}

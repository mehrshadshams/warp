//! HTTP transport that talks to a local Ollama daemon.
//!
//! The trait is small on purpose — only the operations the agent actually
//! needs — so it is easy to mock in tests via [`FakeOllamaTransport`] (see
//! the test module). The default implementation uses Warp's
//! [`http_client::Client`] for parity with the rest of the codebase.

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};

use super::config::OllamaConfig;
use super::dto::{
    ChatRequest, ChatStreamChunk, ListTagsResponse, OllamaModelTag, ShowModelRequest,
    ShowModelResponse,
};
use super::error::OllamaError;
use super::ndjson::NdjsonParser;

/// Operations the agent layer expects from any Ollama endpoint.
#[async_trait]
pub trait OllamaTransport: Send + Sync + 'static {
    /// `GET /api/tags` — list installed models.
    async fn list_models(&self) -> Result<Vec<OllamaModelTag>, OllamaError>;

    /// `POST /api/show` — fetch capabilities / metadata for one model.
    async fn show_model(&self, name: &str) -> Result<ShowModelResponse, OllamaError>;

    /// `POST /api/chat?stream=true` — start a streaming chat completion.
    /// The stream surfaces NDJSON chunks one at a time and ends after the
    /// chunk with `done: true` (or with an error).
    async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, Result<ChatStreamChunk, OllamaError>>, OllamaError>;
}

// -----------------------------------------------------------------------------
// HTTP implementation
// -----------------------------------------------------------------------------

pub struct HttpOllamaTransport {
    client: http_client::Client,
    config: OllamaConfig,
}

impl std::fmt::Debug for HttpOllamaTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpOllamaTransport")
            .field("config", &self.config)
            .finish()
    }
}

impl HttpOllamaTransport {
    pub fn new(config: OllamaConfig) -> Result<Self, OllamaError> {
        config.validate()?;
        Ok(Self {
            client: http_client::Client::new(),
            config,
        })
    }

    /// Inject a custom `http_client::Client` (used by tests with
    /// `Client::new_for_test`).
    pub fn with_client(
        config: OllamaConfig,
        client: http_client::Client,
    ) -> Result<Self, OllamaError> {
        config.validate()?;
        Ok(Self { client, config })
    }

    pub fn config(&self) -> &OllamaConfig {
        &self.config
    }
}

#[async_trait]
impl OllamaTransport for HttpOllamaTransport {
    async fn list_models(&self) -> Result<Vec<OllamaModelTag>, OllamaError> {
        let url = self.config.endpoint("api/tags")?;
        let endpoint = url.to_string();
        let resp = self
            .client
            .get(url)
            .timeout(self.config.request_timeout)
            .send()
            .await
            .map_err(|e| OllamaError::Unreachable {
                endpoint: endpoint.clone(),
                source: anyhow::Error::new(e),
            })?;
        let resp = resp
            .error_for_status()
            .map_err(|e| OllamaError::HttpStatus {
                status: e.source.status().map(|s| s.as_u16()).unwrap_or(0),
                body: e.to_string(),
            })?;
        let body = resp.bytes_stream();
        let bytes = collect_body(body).await?;
        let parsed: ListTagsResponse = serde_json::from_slice(&bytes)
            .map_err(|e| OllamaError::Protocol(format!("invalid /api/tags JSON: {e}")))?;
        Ok(parsed.models)
    }

    async fn show_model(&self, name: &str) -> Result<ShowModelResponse, OllamaError> {
        let url = self.config.endpoint("api/show")?;
        let endpoint = url.to_string();
        let resp = self
            .client
            .post(url)
            .json(&ShowModelRequest { name })
            .timeout(self.config.request_timeout)
            .send()
            .await
            .map_err(|e| OllamaError::Unreachable {
                endpoint: endpoint.clone(),
                source: anyhow::Error::new(e),
            })?;
        if resp.status().as_u16() == 404 {
            return Err(OllamaError::ModelNotFound {
                model: name.to_string(),
            });
        }
        let resp = resp
            .error_for_status()
            .map_err(|e| OllamaError::HttpStatus {
                status: e.source.status().map(|s| s.as_u16()).unwrap_or(0),
                body: e.to_string(),
            })?;
        let bytes = collect_body(resp.bytes_stream()).await?;
        serde_json::from_slice(&bytes)
            .map_err(|e| OllamaError::Protocol(format!("invalid /api/show JSON: {e}")))
    }

    async fn chat_stream(
        &self,
        mut request: ChatRequest,
    ) -> Result<BoxStream<'static, Result<ChatStreamChunk, OllamaError>>, OllamaError> {
        request.stream = true;
        if request.keep_alive.is_none() {
            if let Some(d) = self.config.keep_alive {
                request.keep_alive = Some(format_keep_alive(d));
            }
        }

        let url = self.config.endpoint("api/chat")?;
        let endpoint = url.to_string();
        let resp = self
            .client
            .post(url)
            .json(&request)
            .timeout(self.config.request_timeout)
            .send()
            .await
            .map_err(|e| OllamaError::Unreachable {
                endpoint: endpoint.clone(),
                source: anyhow::Error::new(e),
            })?;
        let resp = resp
            .error_for_status()
            .map_err(|e| OllamaError::HttpStatus {
                status: e.source.status().map(|s| s.as_u16()).unwrap_or(0),
                body: e.to_string(),
            })?;

        let bytes = resp.bytes_stream();
        let stream = ndjson_chunk_stream(bytes);
        Ok(Box::pin(stream))
    }
}

fn format_keep_alive(d: std::time::Duration) -> String {
    // Ollama parses Go-style durations; seconds is universally accepted.
    format!("{}s", d.as_secs())
}

async fn collect_body(
    mut stream: impl futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Unpin,
) -> Result<bytes::BytesMut, OllamaError> {
    let mut buf = bytes::BytesMut::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| OllamaError::Stream(e.to_string()))?;
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}

fn ndjson_chunk_stream<S>(
    bytes: S,
) -> impl futures::Stream<Item = Result<ChatStreamChunk, OllamaError>> + Send
where
    S: futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send + 'static,
{
    use futures::stream;
    let parser = NdjsonParser::new();
    stream::unfold(
        (
            Box::pin(bytes),
            parser,
            Vec::<ChatStreamChunk>::new(),
            false,
        ),
        |state| async move {
            let (mut bytes, mut parser, mut pending, mut done) = state;

            loop {
                if let Some(next) = pending.pop() {
                    return Some((Ok(next), (bytes, parser, pending, done)));
                }
                if done {
                    // Flush any trailing record without a final newline.
                    return match parser.finish::<ChatStreamChunk>() {
                        Ok(Some(chunk)) => Some((Ok(chunk), (bytes, parser, pending, true))),
                        Ok(None) => None,
                        Err(e) => Some((Err(e), (bytes, parser, pending, true))),
                    };
                }

                match bytes.as_mut().next().await {
                    Some(Ok(buf)) => match parser.push::<ChatStreamChunk>(&buf) {
                        Ok(mut new_chunks) => {
                            // Emit in arrival order; Vec::pop is LIFO so reverse.
                            new_chunks.reverse();
                            pending = new_chunks;
                        }
                        Err(e) => return Some((Err(e), (bytes, parser, pending, true))),
                    },
                    Some(Err(e)) => {
                        return Some((
                            Err(OllamaError::Stream(e.to_string())),
                            (bytes, parser, pending, true),
                        ));
                    }
                    None => {
                        done = true;
                    }
                }
            }
        },
    )
}

// -----------------------------------------------------------------------------
// Test helpers — public under cfg(test) so the M2 agent layer can reuse them.
// -----------------------------------------------------------------------------

#[cfg(test)]
pub mod testing {
    use super::*;
    use std::sync::Mutex;

    /// In-memory transport that returns pre-recorded responses. Useful for
    /// unit tests of the agent integration layer.
    pub struct FakeOllamaTransport {
        pub list_models_response: Mutex<Option<Result<Vec<OllamaModelTag>, OllamaError>>>,
        pub show_model_response: Mutex<Option<Result<ShowModelResponse, OllamaError>>>,
        pub chat_chunks: Mutex<Option<Vec<Result<ChatStreamChunk, OllamaError>>>>,
        pub last_chat_request: Mutex<Option<ChatRequest>>,
    }

    impl FakeOllamaTransport {
        pub fn new() -> Self {
            Self {
                list_models_response: Mutex::new(None),
                show_model_response: Mutex::new(None),
                chat_chunks: Mutex::new(None),
                last_chat_request: Mutex::new(None),
            }
        }
    }

    #[async_trait]
    impl OllamaTransport for FakeOllamaTransport {
        async fn list_models(&self) -> Result<Vec<OllamaModelTag>, OllamaError> {
            self.list_models_response
                .lock()
                .unwrap()
                .take()
                .unwrap_or_else(|| Ok(vec![]))
        }
        async fn show_model(&self, _name: &str) -> Result<ShowModelResponse, OllamaError> {
            self.show_model_response
                .lock()
                .unwrap()
                .take()
                .unwrap_or_else(|| {
                    Ok(ShowModelResponse {
                        modelfile: None,
                        parameters: None,
                        template: None,
                        details: None,
                        capabilities: vec![],
                        model_info: None,
                    })
                })
        }
        async fn chat_stream(
            &self,
            request: ChatRequest,
        ) -> Result<BoxStream<'static, Result<ChatStreamChunk, OllamaError>>, OllamaError> {
            *self.last_chat_request.lock().unwrap() = Some(request);
            let chunks = self.chat_chunks.lock().unwrap().take().unwrap_or_default();
            Ok(Box::pin(futures::stream::iter(chunks)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::FakeOllamaTransport;
    use super::*;
    use crate::ollama::dto::{ChatMessage, ChatRole};
    use serde_json::json;

    #[test]
    fn rejects_invalid_config_at_construction() {
        let mut cfg = OllamaConfig::default();
        cfg.base_url = url::Url::parse("ftp://localhost").unwrap();
        let err = HttpOllamaTransport::new(cfg).unwrap_err();
        assert!(matches!(err, OllamaError::InvalidConfig { .. }));
    }

    #[tokio::test]
    async fn fake_transport_round_trips_chat_request() {
        let fake = FakeOllamaTransport::new();
        *fake.chat_chunks.lock().unwrap() = Some(vec![
            Ok(ChatStreamChunk {
                model: "m".into(),
                created_at: None,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: "hi".into(),
                    images: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                done: false,
                done_reason: None,
                prompt_eval_count: None,
                eval_count: None,
                total_duration: None,
            }),
            Ok(ChatStreamChunk {
                model: "m".into(),
                created_at: None,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: "".into(),
                    images: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                done: true,
                done_reason: Some("stop".into()),
                prompt_eval_count: Some(2),
                eval_count: Some(3),
                total_duration: None,
            }),
        ]);

        let req = ChatRequest {
            model: "m".into(),
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "ping".into(),
                images: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false, // transport flips this to true
            tools: None,
            options: None,
            keep_alive: None,
        };
        let mut stream = fake.chat_stream(req).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(c) = stream.next().await {
            chunks.push(c.unwrap());
        }
        assert_eq!(chunks.len(), 2);
        assert!(chunks[1].done);
        let _ = json!({}); // silence unused import in some configs
    }
}

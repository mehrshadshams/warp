//! Client-side transport for talking to a local (or self-hosted) Ollama
//! daemon.
//!
//! Unlike the OpenAI / Anthropic / Gemini providers, Ollama is **not** routed
//! through Warp's backend: requests are made directly from the client to a
//! user-controlled endpoint (default `http://localhost:11434`). Anything
//! that touches the network here therefore needs to be conservative about
//! security (loopback by default, no credentials, scheme allow-list) and
//! about test-determinism (no live HTTP in unit tests).
//!
//! The module is intentionally narrow: it exposes only the small subset of
//! Ollama's HTTP surface that the agent needs (list models, probe a model,
//! chat-completion with streaming + tool calling).

pub mod config;
pub mod dto;
pub mod error;
pub mod ndjson;
pub mod tool;
pub mod transport;

pub use config::OllamaConfig;
pub use error::OllamaError;
pub use transport::{HttpOllamaTransport, OllamaTransport};

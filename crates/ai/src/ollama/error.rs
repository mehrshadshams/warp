//! Error type for the Ollama transport.
//!
//! All variants are designed to surface actionable user-facing messages
//! (e.g. "is `ollama serve` running?"). Lower-level details are preserved
//! via `#[source]` for logging.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum OllamaError {
    /// User-supplied configuration is malformed or violates the loopback
    /// safety rail.
    #[error("invalid Ollama configuration: {reason}")]
    InvalidConfig { reason: String },

    /// The HTTP request failed before any response was received. Most often
    /// this means the daemon is not running.
    #[error("could not reach Ollama daemon at {endpoint}: {source}")]
    Unreachable {
        endpoint: String,
        #[source]
        source: anyhow::Error,
    },

    /// Ollama returned a non-2xx response. `body` is truncated to keep logs
    /// bounded.
    #[error("Ollama returned HTTP {status}: {body}")]
    HttpStatus { status: u16, body: String },

    /// The user requested a model that the daemon does not have installed.
    #[error("model `{model}` is not installed; run `ollama pull {model}`")]
    ModelNotFound { model: String },

    /// Failed to (de)serialize a request or NDJSON chunk.
    #[error("Ollama protocol error: {0}")]
    Protocol(String),

    /// Underlying I/O / streaming failure mid-response.
    #[error("Ollama stream error: {0}")]
    Stream(String),
}

impl OllamaError {
    /// True when the failure is a transient network condition that callers
    /// might want to retry automatically.
    pub fn is_retryable(&self) -> bool {
        matches!(self, OllamaError::Unreachable { .. } | OllamaError::Stream(_))
    }
}

//! Wire-format DTOs for the subset of Ollama's HTTP API used by Warp.
//!
//! Reference: <https://github.com/ollama/ollama/blob/main/docs/api.md>
//!
//! These types intentionally mirror Ollama's JSON shape exactly. Translation
//! to and from Warp-internal agent types lives in higher layers (see
//! [`super::tool`] for tool-schema translation, and the `M2` agent integration
//! work for message translation).

use serde::{Deserialize, Serialize};
use serde_json::Value;

// -----------------------------------------------------------------------------
// /api/tags
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ListTagsResponse {
    pub models: Vec<OllamaModelTag>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModelTag {
    /// Fully-qualified name including tag, e.g. `llama3.1:8b`.
    pub name: String,
    /// Size in bytes.
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub details: Option<OllamaModelDetails>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModelDetails {
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub families: Option<Vec<String>>,
    #[serde(default)]
    pub parameter_size: Option<String>,
    #[serde(default)]
    pub quantization_level: Option<String>,
}

// -----------------------------------------------------------------------------
// /api/show
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ShowModelRequest<'a> {
    pub name: &'a str,
}

/// Response from `/api/show`. Most fields are optional because Ollama omits
/// them for older models.
#[derive(Debug, Clone, Deserialize)]
pub struct ShowModelResponse {
    #[serde(default)]
    pub modelfile: Option<String>,
    #[serde(default)]
    pub parameters: Option<String>,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub details: Option<OllamaModelDetails>,
    /// Capability hints surfaced by newer Ollama versions (e.g.
    /// `["completion", "tools", "vision"]`).
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub model_info: Option<Value>,
}

impl ShowModelResponse {
    pub fn supports_tools(&self) -> bool {
        self.capabilities.iter().any(|c| c == "tools")
    }
    pub fn supports_vision(&self) -> bool {
        self.capabilities.iter().any(|c| c == "vision")
    }
}

// -----------------------------------------------------------------------------
// /api/chat
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    /// We always set this to `true`; non-streaming is unused.
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OllamaTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<ChatOptions>,
    /// e.g. `"5m"`. See `OllamaConfig::keep_alive`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ChatOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: ChatRole,
    /// Always present; may be empty when `tool_calls` carries the payload.
    pub content: String,
    /// Base64-encoded image bytes for vision-capable models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Used on `role: tool` follow-ups to identify which call this answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Some Ollama versions omit `id`. Generate one client-side when missing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallFunction {
    pub name: String,
    /// Ollama returns arguments as a JSON object (not a string), unlike
    /// OpenAI. We preserve that shape.
    pub arguments: Value,
}

/// OpenAI-compatible tool description that Ollama accepts on `/api/chat`.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct OllamaTool {
    #[serde(rename = "type")]
    pub kind: &'static str, // always "function"
    pub function: OllamaToolFunction,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct OllamaToolFunction {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the arguments. Caller is responsible for
    /// validity; the daemon will surface schema errors per request.
    pub parameters: Value,
}

/// One NDJSON line from a streaming `/api/chat` response.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatStreamChunk {
    pub model: String,
    #[serde(default)]
    pub created_at: Option<String>,
    pub message: ChatMessage,
    /// `true` on the final chunk, which also carries token counts and the
    /// `done_reason` when present.
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub prompt_eval_count: Option<u32>,
    #[serde(default)]
    pub eval_count: Option<u32>,
    #[serde(default)]
    pub total_duration: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tags_response() {
        let raw = r#"{
            "models": [
                {
                    "name": "llama3.1:8b",
                    "size": 4661211808,
                    "digest": "abc",
                    "details": { "family": "llama", "parameter_size": "8B" }
                }
            ]
        }"#;
        let parsed: ListTagsResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.models.len(), 1);
        assert_eq!(parsed.models[0].name, "llama3.1:8b");
    }

    #[test]
    fn parses_show_with_capabilities() {
        let raw = r#"{
            "modelfile": "FROM llama3.1",
            "capabilities": ["completion", "tools"]
        }"#;
        let show: ShowModelResponse = serde_json::from_str(raw).unwrap();
        assert!(show.supports_tools());
        assert!(!show.supports_vision());
    }

    #[test]
    fn parses_chunk_with_tool_call() {
        let raw = r#"{
            "model": "llama3.1:8b",
            "created_at": "2026-05-06T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": { "name": "search", "arguments": {"q": "rust"} }
                }]
            },
            "done": false
        }"#;
        let chunk: ChatStreamChunk = serde_json::from_str(raw).unwrap();
        assert!(!chunk.done);
        let calls = chunk.message.tool_calls.as_ref().unwrap();
        assert_eq!(calls[0].function.name, "search");
    }

    #[test]
    fn parses_final_chunk_with_token_counts() {
        let raw = r#"{
            "model": "llama3.1:8b",
            "message": { "role": "assistant", "content": "" },
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 12,
            "eval_count": 34
        }"#;
        let chunk: ChatStreamChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.done);
        assert_eq!(chunk.eval_count, Some(34));
    }

    #[test]
    fn skips_nones_when_serializing_chat_request() {
        let req = ChatRequest {
            model: "llama3.1:8b".into(),
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "hi".into(),
                images: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: true,
            tools: None,
            options: None,
            keep_alive: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("tools"));
        assert!(!json.contains("options"));
        assert!(!json.contains("keep_alive"));
        assert!(json.contains("\"stream\":true"));
    }
}

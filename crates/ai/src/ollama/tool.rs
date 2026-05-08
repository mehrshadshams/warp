//! Translation between Warp's internal tool description and Ollama's
//! OpenAI-compatible tool schema.
//!
//! Kept deliberately small: the only thing we need on the request side is
//! `(name, description, json_schema) -> OllamaTool`. The reverse direction
//! (parsing tool calls *from* Ollama) is just deserializing
//! [`super::dto::ToolCall`] — no helper needed.

use serde_json::Value;
use uuid::Uuid;

use super::dto::{OllamaTool, OllamaToolFunction, ToolCall};

/// Minimal description of a tool, owned by the agent layer. We avoid taking
/// a dependency on `warp_multi_agent_api` types here so this module stays
/// transport-only.
#[derive(Debug, Clone)]
pub struct ToolDescriptor {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's parameters. Must be a JSON object.
    pub parameters: Value,
}

impl ToolDescriptor {
    pub fn to_ollama_tool(&self) -> OllamaTool {
        OllamaTool {
            kind: "function",
            function: OllamaToolFunction {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: self.parameters.clone(),
            },
        }
    }
}

/// Some Ollama versions omit `id` on `tool_calls`. The agent loop needs
/// a stable identifier to match the eventual `role: tool` reply, so we
/// synthesise a v4 UUID when missing.
pub fn ensure_tool_call_id(call: &mut ToolCall) -> &str {
    if call.id.is_none() {
        call.id = Some(Uuid::new_v4().to_string());
    }
    call.id.as_deref().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn descriptor_round_trips_into_ollama_shape() {
        let d = ToolDescriptor {
            name: "search".into(),
            description: "Search the codebase".into(),
            parameters: json!({
                "type": "object",
                "properties": { "q": { "type": "string" } },
                "required": ["q"],
            }),
        };
        let tool = d.to_ollama_tool();
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "search");
        assert_eq!(json["function"]["parameters"]["required"][0], "q");
    }

    #[test]
    fn synthesises_missing_tool_call_id() {
        let mut call = ToolCall {
            id: None,
            function: super::super::dto::ToolCallFunction {
                name: "x".into(),
                arguments: json!({}),
            },
        };
        let id = ensure_tool_call_id(&mut call).to_string();
        assert!(!id.is_empty());
        // Idempotent.
        assert_eq!(ensure_tool_call_id(&mut call), id);
    }
}

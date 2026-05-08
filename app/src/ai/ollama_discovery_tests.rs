use std::sync::Mutex;

use ai::ollama::{
    dto::{ChatRequest, ChatStreamChunk, OllamaModelDetails, OllamaModelTag, ShowModelResponse},
    error::OllamaError,
    transport::OllamaTransport,
};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use futures::StreamExt;

use super::*;
use crate::ai::llms::{LLMModelHost, LLMProvider};

fn tag(name: &str, parameter_size: Option<&str>) -> OllamaModelTag {
    OllamaModelTag {
        name: name.to_string(),
        size: 0,
        digest: String::new(),
        details: parameter_size.map(|p| OllamaModelDetails {
            family: None,
            families: None,
            parameter_size: Some(p.to_string()),
            quantization_level: None,
        }),
    }
}

#[test]
fn translates_basic_tag_with_parameter_size() {
    let info = llm_info_from_ollama_tag(&tag("llama3.1:8b", Some("8B")));
    assert_eq!(info.display_name, "llama3.1:8b");
    assert_eq!(info.base_model_name, "llama3.1");
    assert_eq!(info.id.as_str(), "llama3.1:8b");
    assert_eq!(info.provider, LLMProvider::Ollama);
    assert_eq!(info.description.as_deref(), Some("local · 8B"));
    assert!(!info.vision_supported);
    assert_eq!(info.usage_metadata.request_multiplier, 0);

    let host_cfg = info
        .host_configs
        .get(&LLMModelHost::LocalOllama)
        .expect("LocalOllama host config must be present");
    assert!(host_cfg.enabled);
    assert_eq!(host_cfg.model_routing_host, LLMModelHost::LocalOllama);
}

#[test]
fn tag_without_colon_uses_full_name_as_base() {
    let info = llm_info_from_ollama_tag(&tag("mistral", None));
    assert_eq!(info.base_model_name, "mistral");
    assert_eq!(info.display_name, "mistral");
    // Falls back to bare "local" when no parameter_size.
    assert_eq!(info.description.as_deref(), Some("local"));
}

// Local fake — `FakeOllamaTransport` in the `ai` crate is `#[cfg(test)]`
// only and not visible from `app`.
struct StubTransport {
    tags: Mutex<Option<Result<Vec<OllamaModelTag>, OllamaError>>>,
}

#[async_trait]
impl OllamaTransport for StubTransport {
    async fn list_models(&self) -> Result<Vec<OllamaModelTag>, OllamaError> {
        self.tags
            .lock()
            .unwrap()
            .take()
            .expect("list_models called more than once or never primed")
    }

    async fn show_model(&self, _name: &str) -> Result<ShowModelResponse, OllamaError> {
        unimplemented!("not used in discovery tests")
    }

    async fn chat_stream(
        &self,
        _request: ChatRequest,
    ) -> Result<BoxStream<'static, Result<ChatStreamChunk, OllamaError>>, OllamaError> {
        Ok(stream::empty().boxed())
    }
}

#[tokio::test]
async fn discover_preserves_order_and_translates_each_tag() {
    let transport = StubTransport {
        tags: Mutex::new(Some(Ok(vec![
            tag("llama3.1:8b", Some("8B")),
            tag("qwen2.5-coder:14b", Some("14B")),
            tag("mistral", None),
        ]))),
    };

    let infos = discover_ollama_models(&transport)
        .await
        .expect("discovery should succeed");

    assert_eq!(infos.len(), 3);
    assert_eq!(infos[0].id.as_str(), "llama3.1:8b");
    assert_eq!(infos[1].id.as_str(), "qwen2.5-coder:14b");
    assert_eq!(infos[2].id.as_str(), "mistral");
    for info in &infos {
        assert_eq!(info.provider, LLMProvider::Ollama);
        assert!(info.host_configs.contains_key(&LLMModelHost::LocalOllama));
    }
}

#[tokio::test]
async fn discover_propagates_transport_errors() {
    let transport = StubTransport {
        tags: Mutex::new(Some(Err(OllamaError::HttpStatus {
            status: 503,
            body: "service unavailable".to_string(),
        }))),
    };

    let err = discover_ollama_models(&transport)
        .await
        .expect_err("transport error should propagate");
    assert!(matches!(err, OllamaError::HttpStatus { status: 503, .. }));
}

#[tokio::test]
async fn discover_empty_list_is_ok() {
    let transport = StubTransport {
        tags: Mutex::new(Some(Ok(vec![]))),
    };
    let infos = discover_ollama_models(&transport).await.unwrap();
    assert!(infos.is_empty());
}

// ---- merge_choices_with_ollama -----------------------------------------

fn server_info(id: &str) -> LLMInfo {
    let mut info = llm_info_from_ollama_tag(&tag(id, None));
    // Pretend it came from the server: not LocalOllama-routed.
    info.host_configs.clear();
    info.provider = LLMProvider::Unknown;
    info
}

#[test]
fn merge_appends_ollama_after_existing_and_preserves_order() {
    let existing = vec![server_info("auto"), server_info("claude-sonnet-4")];
    let ollama = vec![
        llm_info_from_ollama_tag(&tag("llama3.1:8b", Some("8B"))),
        llm_info_from_ollama_tag(&tag("mistral", None)),
    ];

    let merged = merge_choices_with_ollama(&existing, ollama);

    assert_eq!(merged.len(), 4);
    assert_eq!(merged[0].id.as_str(), "auto");
    assert_eq!(merged[1].id.as_str(), "claude-sonnet-4");
    assert_eq!(merged[2].id.as_str(), "llama3.1:8b");
    assert_eq!(merged[3].id.as_str(), "mistral");
}

#[test]
fn merge_existing_wins_on_id_collision() {
    let existing = vec![server_info("llama3.1:8b")];
    let ollama = vec![llm_info_from_ollama_tag(&tag("llama3.1:8b", Some("8B")))];

    let merged = merge_choices_with_ollama(&existing, ollama);

    assert_eq!(merged.len(), 1);
    // Existing entry preserved (no LocalOllama host_config).
    assert!(merged[0].host_configs.is_empty());
    assert_eq!(merged[0].provider, LLMProvider::Unknown);
}

#[test]
fn merge_dedupes_within_ollama_list() {
    let ollama = vec![
        llm_info_from_ollama_tag(&tag("llama3.1:8b", None)),
        llm_info_from_ollama_tag(&tag("llama3.1:8b", Some("8B"))),
        llm_info_from_ollama_tag(&tag("mistral", None)),
    ];

    let merged = merge_choices_with_ollama(&[], ollama);

    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].id.as_str(), "llama3.1:8b");
    assert_eq!(merged[1].id.as_str(), "mistral");
}

#[test]
fn merge_with_empty_ollama_returns_existing_clone() {
    let existing = vec![server_info("auto")];
    let merged = merge_choices_with_ollama(&existing, vec![]);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].id.as_str(), "auto");
}

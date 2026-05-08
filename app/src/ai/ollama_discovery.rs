//! Ollama model discovery.
//!
//! Translates the Ollama `/api/tags` response into Warp's [`LLMInfo`] shape so
//! local models can appear in the model picker alongside server-routed models.
//!
//! This module is intentionally a pure translation layer plus a thin async
//! façade over [`OllamaTransport`]. Wiring the discovered models into the live
//! [`super::llms::LLMPreferences`] singleton (and triggering a refresh on
//! settings change) is a follow-up — see PLAN.md M3.3.

use std::collections::HashMap;

use ai::ollama::{
    dto::OllamaModelTag,
    error::OllamaError,
    transport::OllamaTransport,
};

use crate::ai::llms::{
    LLMContextWindow, LLMInfo, LLMModelHost, LLMProvider, LLMUsageMetadata, RoutingHostConfig,
};

/// Build an [`LLMInfo`] entry for an Ollama model tag.
///
/// The returned info advertises the model as `provider = Ollama` with a single
/// `LocalOllama` routing host, vision/usage left at conservative defaults.
/// Capability hints (tools, vision) are not present in `/api/tags`; they come
/// from `/api/show`. A future refinement can call `show_model` per tag and
/// upgrade `vision_supported` / `disable_reason` accordingly.
pub fn llm_info_from_ollama_tag(tag: &OllamaModelTag) -> LLMInfo {
    let id = tag.name.clone();
    let display_name = tag.name.clone();
    let base_model_name = tag
        .name
        .split(':')
        .next()
        .unwrap_or(&tag.name)
        .to_string();

    let description = tag
        .details
        .as_ref()
        .and_then(|d| d.parameter_size.clone())
        .map(|p| format!("local · {p}"))
        .or(Some("local".to_string()));

    let mut host_configs = HashMap::new();
    host_configs.insert(
        LLMModelHost::LocalOllama,
        RoutingHostConfig {
            enabled: true,
            model_routing_host: LLMModelHost::LocalOllama,
        },
    );

    LLMInfo {
        display_name,
        base_model_name,
        id: id.into(),
        reasoning_level: None,
        // Local models don't consume server-side request quota.
        usage_metadata: LLMUsageMetadata {
            request_multiplier: 0,
            credit_multiplier: None,
        },
        description,
        disable_reason: None,
        // Conservative default; refined by /api/show in a follow-up.
        vision_supported: false,
        spec: None,
        provider: LLMProvider::Ollama,
        host_configs,
        discount_percentage: None,
        context_window: LLMContextWindow::default(),
    }
}

/// Fetch the locally installed Ollama models and translate them to
/// [`LLMInfo`]. Order from the daemon is preserved.
pub async fn discover_ollama_models(
    transport: &dyn OllamaTransport,
) -> Result<Vec<LLMInfo>, OllamaError> {
    let tags = transport.list_models().await?;
    Ok(tags.iter().map(llm_info_from_ollama_tag).collect())
}

#[cfg(test)]
#[path = "ollama_discovery_tests.rs"]
mod tests;

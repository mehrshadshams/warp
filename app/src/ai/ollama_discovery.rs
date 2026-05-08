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
use std::time::Duration;

use ai::ollama::{
    config::OllamaConfig, dto::OllamaModelTag, error::OllamaError, transport::OllamaTransport,
};
use url::Url;

use crate::ai::llms::{
    LLMContextWindow, LLMInfo, LLMModelHost, LLMProvider, LLMUsageMetadata, RoutingHostConfig,
};
use crate::settings::ai::AISettings;

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
    let base_model_name = tag.name.split(':').next().unwrap_or(&tag.name).to_string();

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

/// Merge server-provided [`LLMInfo`] choices with Ollama-discovered ones.
///
/// Ollama entries are appended after the server entries so that the existing
/// default (which references a server-provided id) keeps working, and the
/// model picker shows local models in a contiguous block at the bottom.
///
/// If an Ollama-provided id collides with an existing entry, the existing
/// entry wins (the daemon shouldn't be authoritative for server-managed
/// models). The Ollama list is otherwise deduped by id within itself,
/// preserving first occurrence.
pub fn merge_choices_with_ollama(
    existing: &[LLMInfo],
    ollama: Vec<LLMInfo>,
) -> Vec<LLMInfo> {
    use std::collections::HashSet;

    let mut seen: HashSet<crate::ai::llms::LLMId> =
        existing.iter().map(|i| i.id.clone()).collect();
    let mut merged: Vec<LLMInfo> = existing.to_vec();
    for info in ollama {
        if seen.insert(info.id.clone()) {
            merged.push(info);
        }
    }
    merged
}

/// Strip any [`LLMInfo`] entries whose `provider` is [`LLMProvider::Ollama`].
///
/// Used before re-applying an updated set of Ollama-discovered choices so that
/// stale local entries (from a previous discovery run, or before the user
/// disabled the integration) don't linger in the picker.
pub fn strip_ollama_entries(choices: &[LLMInfo]) -> Vec<LLMInfo> {
    choices
        .iter()
        .filter(|i| !matches!(i.provider, LLMProvider::Ollama))
        .cloned()
        .collect()
}

/// Filter discovered models down to the user-selected subset, if any.
/// An empty `selected` means "no filter — show every discovered model".
pub fn filter_by_user_selection(infos: Vec<LLMInfo>, selected: &[String]) -> Vec<LLMInfo> {
    if selected.is_empty() {
        return infos;
    }
    let allow: std::collections::HashSet<&str> = selected.iter().map(String::as_str).collect();
    infos
        .into_iter()
        .filter(|i| allow.contains(i.id.as_str()))
        .collect()
}

/// Build an [`OllamaConfig`] from the user's [`AISettings`].
///
/// Returns:
/// - `None` if Ollama is disabled or the resolved base URL is blank.
/// - `Some(Ok(config))` if the settings produce a valid, validated config.
/// - `Some(Err(_))` if the settings are present but malformed (bad URL,
///   non-loopback host without the opt-in, etc.) — callers can surface this
///   as a banner.
pub fn ollama_config_from_settings(
    ai_settings: &AISettings,
) -> Option<Result<OllamaConfig, OllamaError>> {
    if !*ai_settings.ollama_enabled {
        return None;
    }
    let raw = ai_settings.resolved_ollama_base_url()?;
    let url = match Url::parse(&raw) {
        Ok(u) => u,
        Err(e) => {
            return Some(Err(OllamaError::InvalidConfig {
                reason: format!("base URL `{raw}` is not a valid URL: {e}"),
            }));
        }
    };
    let mut cfg = OllamaConfig {
        base_url: url,
        loopback_only: !*ai_settings.ollama_allow_remote_hosts,
        ..OllamaConfig::default()
    };
    let keep_alive_raw = ai_settings.ollama_keep_alive.trim();
    if !keep_alive_raw.is_empty() {
        // Best-effort parse: accept `Ns`, `Nm`, `Nh`. On parse error we leave
        // `keep_alive = None` so the daemon's own default applies.
        if let Some(d) = parse_keep_alive(keep_alive_raw) {
            cfg.keep_alive = Some(d);
        }
    }
    if let Err(e) = cfg.validate() {
        return Some(Err(e));
    }
    Some(Ok(cfg))
}

fn parse_keep_alive(raw: &str) -> Option<Duration> {
    let raw = raw.trim();
    let (num, unit) = raw.split_at(raw.find(|c: char| !c.is_ascii_digit())?);
    let n: u64 = num.parse().ok()?;
    match unit {
        "s" => Some(Duration::from_secs(n)),
        "m" => Some(Duration::from_secs(n * 60)),
        "h" => Some(Duration::from_secs(n * 60 * 60)),
        _ => None,
    }
}

#[cfg(test)]
#[path = "ollama_discovery_tests.rs"]
mod tests;

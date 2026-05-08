use super::*;

// -- DisableReason::should_clear_preference tests --

#[test]
fn should_clear_preference_admin_disabled() {
    // AdminDisabled always clears, regardless of BYOK status.
    assert!(DisableReason::AdminDisabled.should_clear_preference(false));
    assert!(DisableReason::AdminDisabled.should_clear_preference(true));
}

#[test]
fn should_clear_preference_unavailable() {
    assert!(DisableReason::Unavailable.should_clear_preference(false));
    assert!(DisableReason::Unavailable.should_clear_preference(true));
}

#[test]
fn should_not_clear_preference_out_of_requests() {
    // Transient — never clears.
    assert!(!DisableReason::OutOfRequests.should_clear_preference(false));
    assert!(!DisableReason::OutOfRequests.should_clear_preference(true));
}

#[test]
fn should_not_clear_preference_provider_outage() {
    // Transient — never clears.
    assert!(!DisableReason::ProviderOutage.should_clear_preference(false));
    assert!(!DisableReason::ProviderOutage.should_clear_preference(true));
}

#[test]
fn should_clear_preference_requires_upgrade_without_byok() {
    // No BYOK key → server will reject → clear.
    assert!(DisableReason::RequiresUpgrade.should_clear_preference(false));
}

#[test]
fn should_not_clear_preference_requires_upgrade_with_byok() {
    // BYOK key present → server allows → keep.
    assert!(!DisableReason::RequiresUpgrade.should_clear_preference(true));
}

#[test]
fn llm_info_deserializes_without_base_model_name() {
    let raw = r#"{
            "display_name": "gpt-4o",
            "id": "gpt-4o",
            "usage_metadata": {
                "request_multiplier": 1,
                "credit_multiplier": null
            },
            "description": null,
            "disable_reason": null,
            "vision_supported": false,
            "spec": null,
            "provider": "Unknown"
        }"#;

    let info: LLMInfo = serde_json::from_str(raw).expect("should deserialize");
    assert_eq!(info.display_name, "gpt-4o");
    assert_eq!(info.base_model_name, "gpt-4o");
}

#[test]
fn llm_info_deserializes_host_configs_as_vec() {
    // Wire format from server: host_configs is a Vec
    let raw = r#"{
            "display_name": "gpt-4o",
            "id": "gpt-4o",
            "usage_metadata": { "request_multiplier": 1, "credit_multiplier": null },
            "provider": "OpenAI",
            "host_configs": [
                { "enabled": true, "model_routing_host": "DirectApi" },
                { "enabled": false, "model_routing_host": "AwsBedrock" }
            ]
        }"#;

    let info: LLMInfo = serde_json::from_str(raw).expect("should deserialize vec format");
    assert_eq!(info.display_name, "gpt-4o");
    assert_eq!(info.host_configs.len(), 2);
    assert!(
        info.host_configs
            .get(&LLMModelHost::DirectApi)
            .unwrap()
            .enabled
    );
    assert!(
        !info
            .host_configs
            .get(&LLMModelHost::AwsBedrock)
            .unwrap()
            .enabled
    );
}

#[test]
fn llm_info_round_trip_serializes_and_deserializes() {
    // Start with wire format (Vec)
    let wire_json = r#"{
            "display_name": "claude-3",
            "base_model_name": "claude-3",
            "id": "claude-3",
            "usage_metadata": { "request_multiplier": 2, "credit_multiplier": 1.5 },
            "description": "A powerful model",
            "vision_supported": true,
            "provider": "Anthropic",
            "host_configs": [
                { "enabled": true, "model_routing_host": "DirectApi" }
            ]
        }"#;

    // Deserialize from wire format
    let info: LLMInfo = serde_json::from_str(wire_json).expect("should deserialize");

    // Serialize (produces HashMap format)
    let serialized = serde_json::to_string(&info).expect("should serialize");

    // Deserialize again (from HashMap format)
    let round_tripped: LLMInfo =
        serde_json::from_str(&serialized).expect("should deserialize after round trip");

    assert_eq!(info, round_tripped);
}

// -- Ollama provider / LocalOllama host variant snapshot tests --

#[test]
fn llm_provider_ollama_round_trip() {
    let serialized = serde_json::to_string(&LLMProvider::Ollama).expect("serialize");
    assert_eq!(serialized, "\"Ollama\"");
    let parsed: LLMProvider = serde_json::from_str(&serialized).expect("deserialize");
    assert_eq!(parsed, LLMProvider::Ollama);
}

#[test]
fn llm_model_host_local_ollama_round_trip() {
    let serialized = serde_json::to_string(&LLMModelHost::LocalOllama).expect("serialize");
    assert_eq!(serialized, "\"LocalOllama\"");
    let parsed: LLMModelHost = serde_json::from_str(&serialized).expect("deserialize");
    assert_eq!(parsed, LLMModelHost::LocalOllama);
}

#[test]
fn llm_model_host_unknown_variant_falls_back() {
    // Unknown wire values must deserialize to LLMModelHost::Unknown, not panic.
    let parsed: LLMModelHost =
        serde_json::from_str("\"SomeFutureHost\"").expect("should accept unknown variant");
    assert_eq!(parsed, LLMModelHost::Unknown);
}

#[test]
fn llm_info_deserializes_ollama_provider_with_local_host() {
    let raw = r#"{
            "display_name": "llama3.1:8b",
            "id": "llama3.1:8b",
            "usage_metadata": { "request_multiplier": 0, "credit_multiplier": 0 },
            "provider": "Ollama",
            "vision_supported": false,
            "host_configs": [
                { "enabled": true, "model_routing_host": "LocalOllama" }
            ]
        }"#;

    let info: LLMInfo = serde_json::from_str(raw).expect("should deserialize ollama model");
    assert_eq!(info.provider, LLMProvider::Ollama);
    assert!(
        info.host_configs
            .get(&LLMModelHost::LocalOllama)
            .expect("LocalOllama host present")
            .enabled
    );
}

#[test]
fn llm_provider_ollama_has_no_icon() {
    // Ollama models surface a "Local" badge instead of a brand icon.
    assert!(LLMProvider::Ollama.icon().is_none());
}

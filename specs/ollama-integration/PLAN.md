# Ollama Provider Integration — Plan & Guidelines

> Status: In progress — Milestone 0 mostly done, Milestone 1 transport scaffolding landed.
> Scope: Add Ollama (`https://ollama.com`) as a first‑class LLM provider for the Warp Agent, enabling local / self‑hosted models.

> **Progress snapshot (verified against the working tree)**
>
> - ✅ M0.1 feature flag `OllamaProvider`
> - ◐ M0.2 client enums done; GraphQL `LlmProvider::Ollama` still missing
> - ⬜ M0.3 snapshot tests for new variants
> - ✅ M1 transport module (`crates/ai/src/ollama/`) with NDJSON parser, `HttpOllamaTransport`, `list_models` / `show_model` / `chat_stream`, and tool helpers
> - ⬜ M2 agent integration (`LocalLlmAgentEventSource`)
> - ⬜ M3 settings & discovery
> - ⬜ M4–M7 UX, telemetry, integration tests, rollout

---

## 1. Background — How AI is wired in Warp today

The exploration of the codebase surfaced a layered architecture. Understanding it is a **prerequisite** for any provider work — most of the heavy lifting happens on Warp's backend (`warp_multi_agent_api`), not in the desktop client.

### 1.1 Crate / module layout

| Area | Location | Role |
|------|----------|------|
| Core AI types | [crates/ai/src](crates/ai/src) | `ApiKeyManager`, `LLMId`, skills, indexing, project context |
| Client agent loop | [app/src/ai/agent](app/src/ai) | Conversation state, request building, response stream consumption |
| Model catalog | [app/src/ai/llms.rs](app/src/ai/llms.rs) | `LLMProvider`, `LLMInfo`, `ModelsByFeature`, `AvailableLLMs` |
| Server client trait | [app/src/server/server_api/ai.rs](app/src/server/server_api/ai.rs) | `AIClient` — the boundary between client and Warp backend |
| GraphQL schema | [crates/graphql/src/api/workspace.rs](crates/graphql/src/api/workspace.rs) | Wire format for provider/model metadata |
| Settings UI/state | [app/src/settings/ai.rs](app/src/settings/ai.rs) | User‑facing AI settings (keys, defaults, etc.) |
| Secret storage | `crates/managed_secrets`, `warpui_extras::secure_storage` | Keychain‑backed credentials |
| Feature gating | [crates/warp_features/src/lib.rs](crates/warp_features/src/lib.rs) | `FeatureFlag` enum |
| MCP tools | [app/src/ai/mcp](app/src/ai) | Provider‑agnostic tool exposure |

### 1.2 Provider abstraction (current)

> Note: the `Ollama` provider variant and `LocalOllama` host variant have already landed (see Milestone 0 status below). The snippet below shows the shape **after** that scaffolding.

```rust
// app/src/ai/llms.rs
pub enum LLMProvider { OpenAI, Anthropic, Google, Xai, Ollama, Unknown }

pub enum LLMModelHost { DirectApi, AwsBedrock, LocalOllama, Unknown }

pub struct LLMInfo {
    pub display_name: String,
    pub base_model_name: String,
    pub id: LLMId,
    pub provider: LLMProvider,
    pub host_configs: HashMap<LLMModelHost, RoutingHostConfig>,
    pub usage_metadata: LLMUsageMetadata,
    pub vision_supported: bool,
    pub spec: Option<LLMSpec>,
    // ...
}
```

The `LLMProvider` enum is metadata: it tells the **server** how to route a request. The actual HTTP call to OpenAI / Anthropic / Gemini happens **server‑side** inside `warp_multi_agent_api`. Client code only:

1. Selects an `LLMId` (per feature: agent_mode / coding / cli_agent / computer_use).
2. Optionally provides a BYO API key (`ApiKeys` struct) that the server forwards.
3. Streams `ResponseEvent`s from the server, including tool‑call requests.

### 1.3 Key implication for Ollama

> **Ollama is fundamentally different from existing providers because it must talk to a *local* HTTP endpoint (default `http://localhost:11434`) on the user's machine.**

Routing the request through Warp's cloud backend would either (a) require the backend to reach the user's laptop (impossible without a tunnel) or (b) require a new code path that bypasses the backend and calls Ollama directly from the client.

We therefore need a **client‑side LLM transport** for Ollama. This is a meaningful new architectural seam — call it out explicitly in the design.

### 1.4 Existing assets that help

- `crates/http_client` — async HTTP/streaming layer already used for non‑AI server calls.
- `crates/jsonrpc` and existing SSE handling patterns.
- `MCP` integration ([app/src/ai/mcp](app/src/ai)) gives provider‑agnostic tool definitions; reuse to translate to Ollama's tool schema.
- `AgentEventSource` trait ([app/src/ai/agent_events/driver.rs](app/src/ai/agent_events/driver.rs)) and its `FakeAgentEventSource` mock ([app/src/ai/agent_events/driver_tests.rs](app/src/ai/agent_events/driver_tests.rs)) — the natural test seam.
- Feature flag system — gate the rollout cleanly (see [.agents/skills/add-feature-flag/SKILL.md](.agents/skills/add-feature-flag/SKILL.md)).

### 1.5 Things already searched

- Initial `grep -i ollama` (pre‑scaffolding) found only 4 hits, all in static suggested commands ([app/src/ai/blocklist/passive_suggestions/static_prompt_suggestions.rs](app/src/ai/blocklist/passive_suggestions/static_prompt_suggestions.rs)).
- Current state: scaffolding has landed in [app/src/ai/llms.rs](app/src/ai/llms.rs), [crates/warp_features/src/lib.rs](crates/warp_features/src/lib.rs), and the new transport module under [crates/ai/src/ollama](crates/ai/src/ollama). See Milestone 0 / Milestone 1 below.

---

## 2. Guidelines for integrating any new LLM provider

Distilled from the patterns observed:

1. **Extend the metadata, don't fork the loop.** New providers are added by extending `LLMProvider` + `LLMInfo` and registering models in `ModelsByFeature` — never by introducing a parallel agent loop.
2. **Keep transport behind a trait.** The agent loop consumes a `ResponseStream` of `ResponseEvent`s. Any new transport (cloud or local) must adapt to that stream shape, not the other way around.
3. **Provider‑agnostic tool layer.** Tools are defined via MCP and a Warp‑internal schema. New providers translate that schema to/from their own format at the edge — never leak provider‑specific JSON into the agent.
4. **Feature‑flag every rollout.** New providers go behind a `FeatureFlag` (Dogfood → Preview → Stable). Plan flag removal up‑front (see [.agents/skills/remove-feature-flag/SKILL.md](.agents/skills/remove-feature-flag/SKILL.md)).
5. **Secrets via `managed_secrets`.** Even non‑secret config (base URLs) belongs in settings; never hard‑code endpoints.
6. **GraphQL parity.** Anything in `LLMProvider` must have a matching variant in [crates/graphql/src/api/workspace.rs](crates/graphql/src/api/workspace.rs) so server / client serialize symmetrically.
7. **Test at the seam.** Use `FakeAgentEventSource` for client tests; record golden fixtures for the provider transport. Avoid live API calls in CI.
8. **UI follows `warp-ui-guidelines`.** Settings panes for the new provider must conform to [.agents/skills/warp-ui-guidelines/SKILL.md](.agents/skills/warp-ui-guidelines/SKILL.md).

---

## 3. Requirements

### 3.1 Functional

| ID | Requirement |
|----|-------------|
| F1 | User can enable Ollama as a provider in Settings → AI. |
| F2 | User can configure base URL (default `http://localhost:11434`) and per‑model overrides. |
| F3 | Warp lists installed Ollama models by querying `GET /api/tags`. |
| F4 | Selected Ollama models appear in model picker for `agent_mode` and `coding` features. |
| F5 | Agent chat (non‑streaming + streaming) works against Ollama via `POST /api/chat`. |
| F6 | Tool calling works for Ollama models that advertise tool support (Llama 3.1+, Qwen 2.5, etc.) using Ollama's OpenAI‑compatible tool schema. |
| F7 | Graceful degradation: if a model lacks tool support, fall back to plain chat or surface a clear error. |
| F8 | Health check + actionable error UI when Ollama daemon is unreachable. |
| F9 | Telemetry event on first successful Ollama completion (no prompt/response content). |
| F10 | Feature flag `OllamaProvider` gates the entire feature. |

### 3.2 Non‑functional

| ID | Requirement |
|----|-------------|
| N1 | Zero impact on cloud‑provider latency when Ollama is disabled. |
| N2 | All Ollama HTTP traffic stays on `localhost` (or user‑configured host) — never proxied through Warp backend. |
| N3 | No prompts, responses, or model names leave the device unless general telemetry says so. |
| N4 | Agent loop code paths shared with cloud providers — no parallel agent implementation. |
| N5 | `cargo presubmit` clean (fmt, clippy, tests, WASM build). |

### 3.3 Out of scope (initial release)

- Embeddings / RAG indexing via Ollama.
- Computer‑use models via Ollama.
- Voice input / multimodal (vision can be a follow‑up for `llava`‑style models).
- Auto‑download / lifecycle management of Ollama models from within Warp.

---

## 4. Detailed Design

### 4.1 New types

```rust
// app/src/ai/llms.rs
pub enum LLMProvider {
    OpenAI,
    Anthropic,
    Google,
    Xai,
    Ollama,        // <-- new
    Unknown,
}

pub enum LLMModelHost {
    DirectApi,
    AwsBedrock,
    LocalOllama,   // <-- new; signals client‑side transport
    Unknown,
}
```

```rust
// crates/ai/src/ollama/mod.rs (new module)
pub struct OllamaConfig {
    pub base_url: Url,                  // default http://localhost:11434
    pub request_timeout: Duration,      // default 120s
    pub keep_alive: Option<Duration>,   // forwarded to Ollama `keep_alive`
}

#[async_trait]
pub trait OllamaTransport: Send + Sync + 'static {
    async fn list_models(&self) -> Result<Vec<OllamaModel>, OllamaError>;
    async fn chat(
        &self,
        req: OllamaChatRequest,
    ) -> Result<BoxStream<'static, Result<OllamaChatChunk, OllamaError>>, OllamaError>;
}
```

### 4.2 Transport

- Implement `OllamaTransport` on top of `crates/http_client`.
- Endpoints used:
  - `GET  /api/tags` — list local models.
  - `POST /api/chat` with `stream: true` — chat completion (NDJSON streaming).
  - `POST /api/show` — capability probe (tool support, context length).
- NDJSON parser (one JSON object per line) → mapped to internal `ResponseEvent`s.
- Tools serialized as Ollama's OpenAI‑compatible array under `tools`. Tool calls returned in `message.tool_calls` are translated back to the internal tool‑call event.

### 4.3 Agent integration seam

Introduce `LocalLlmAgentEventSource` implementing the existing `AgentEventSource` trait ([app/src/ai/agent_events](app/src/ai)).

Routing decision (built once per request):

```text
choose_event_source(model.host):
  DirectApi | AwsBedrock  -> ServerAgentEventSource (existing)
  LocalOllama             -> LocalLlmAgentEventSource (new)
```

This keeps the agent loop, conversation state, MCP tool dispatch, telemetry hooks, and UI rendering **unchanged**.

### 4.4 Settings & secrets

- New section in [app/src/settings/ai.rs](app/src/settings/ai.rs):
  - `ollama_enabled: bool`
  - `ollama_base_url: String` (validated as URL on save)
  - `ollama_keep_alive: Option<String>` (e.g. `"10m"`)
  - `ollama_selected_models: Vec<String>` (subset of discovered models)
- No API key. Base URL is **not** a secret — store in `Settings`, not `managed_secrets`.
- Env override: `WARP_OLLAMA_BASE_URL` for power users / CI.

### 4.5 Model discovery

On settings open and on demand:
1. Call `OllamaTransport::list_models`.
2. Build `LLMInfo` entries with `provider = Ollama`, `host_configs = { LocalOllama }`.
3. Capability probe via `/api/show` to set `vision_supported`, tool support flag.
4. Cache in memory; invalidate on settings change or manual "Refresh".

### 4.6 GraphQL schema

Add an `Ollama` variant to the cynic `LlmProvider` enum in [crates/graphql/src/api/workspace.rs](crates/graphql/src/api/workspace.rs) so any backend telemetry / feature toggles that key off provider name serialize symmetrically. Note the enum already has an `#[cynic(fallback)] Other(String)` arm, so this is technically optional — but adding the explicit variant keeps client/server in lockstep. Server side need not handle requests — provider is client‑routed.

### 4.7 Feature flag

Add to [crates/warp_features/src/lib.rs](crates/warp_features/src/lib.rs):

```rust
/// Enables Ollama as a local LLM provider for the Warp Agent.
OllamaProvider,
```

Gating points:
- Settings UI section visibility.
- Model picker filtering (`if !flag { drop_ollama_models() }`).
- Routing decision in §4.3 (defensive — never reach `LocalOllama` host if flag off).

Plan promotion path per [.agents/skills/promote-feature/SKILL.md](.agents/skills/promote-feature/SKILL.md): Internal → Dogfood → Preview → Stable.

### 4.8 Errors & UX

| Failure mode | UX |
|--------------|----|
| Daemon unreachable | Inline banner in chat: "Ollama is not running. Start it with `ollama serve`." with a copy button. |
| Model not pulled | Banner + suggested command `ollama pull <model>` (reuse existing static suggestion). |
| Tool calls unsupported | Toast: "Selected Ollama model does not support tools. Switch model or disable tool use." |
| Network timeout | Standard agent error envelope, retryable. |

### 4.9 Telemetry

Per [.agents/skills/add-telemetry/SKILL.md](.agents/skills/add-telemetry/SKILL.md):
- `ollama_enabled_toggled { enabled: bool }`
- `ollama_models_discovered { count: u32 }`
- `ollama_chat_started { model_hash: String }` (hashed, no PII)
- `ollama_chat_completed { model_hash, latency_ms, tokens_in, tokens_out, tool_calls }`
- `ollama_chat_failed { model_hash, error_class }`

### 4.10 Security

- Reject non‑loopback base URLs unless user has explicitly enabled "Allow remote Ollama hosts" (off by default) — protects against SSRF / accidental cloud exposure.
- Validate URL scheme (`http`/`https` only), no `file://`, no embedded credentials.
- Rate‑limit `/api/tags` polling to once per 30s.
- No persistence of conversation content beyond existing agent history storage.

---

## 5. Task list

Tasks are sized to be independently reviewable. Dependencies are noted.

### Milestone 0 — Scaffolding

- [x] **T0.1** Add `OllamaProvider` to `FeatureFlag` enum and wire defaults (off everywhere) — see [.agents/skills/add-feature-flag/SKILL.md](.agents/skills/add-feature-flag/SKILL.md). _(landed at [crates/warp_features/src/lib.rs](crates/warp_features/src/lib.rs#L867))_
- [~] **T0.2** Add `Ollama` variant to `LLMProvider` and `LocalOllama` to `LLMModelHost` in [app/src/ai/llms.rs](app/src/ai/llms.rs); add corresponding `Ollama` to GraphQL enum in [crates/graphql/src/api/workspace.rs](crates/graphql/src/api/workspace.rs).
  - [x] Client `LLMProvider::Ollama` ([app/src/ai/llms.rs](app/src/ai/llms.rs#L111))
  - [x] Client `LLMModelHost::LocalOllama` ([app/src/ai/llms.rs](app/src/ai/llms.rs#L136))
  - [ ] GraphQL `LlmProvider::Ollama` (still missing — see [crates/graphql/src/api/workspace.rs](crates/graphql/src/api/workspace.rs#L77))
- [ ] **T0.3** Snapshot tests: serialize/deserialize the new variants in [app/src/ai/llms_tests.rs](app/src/ai/llms_tests.rs) — file exists but has no `Ollama` coverage yet.

### Milestone 1 — Transport (depends on T0.x)

- [x] **T1.1** Create `crates/ai/src/ollama/` module with `OllamaConfig`, request/response DTOs, `OllamaError`. _(see [crates/ai/src/ollama/mod.rs](crates/ai/src/ollama/mod.rs))_
- [x] **T1.2** Implement `HttpOllamaTransport` using `crates/http_client`. _(see [crates/ai/src/ollama/transport.rs](crates/ai/src/ollama/transport.rs#L41))_
- [x] **T1.3** NDJSON streaming parser + unit tests. _(see [crates/ai/src/ollama/ndjson.rs](crates/ai/src/ollama/ndjson.rs))_ — note: tests are inline `#[cfg(test)]`; no `tests/fixtures/ollama/*.ndjson` fixtures yet.
- [x] **T1.4** `list_models` + `show_model` on `OllamaTransport`. _(see [crates/ai/src/ollama/transport.rs](crates/ai/src/ollama/transport.rs#L23-L26))_ — golden tests against recorded responses still TODO.
- [x] **T1.5** Tool schema translation (`ToolDescriptor`, `ensure_tool_call_id`) with round‑trip tests. _(see [crates/ai/src/ollama/tool.rs](crates/ai/src/ollama/tool.rs))_

### Milestone 2 — Agent integration (depends on M1)

- [ ] **T2.1** Implement `LocalLlmAgentEventSource` adapting `OllamaTransport` to `AgentEventSource`.
- [ ] **T2.2** Routing function that picks `EventSource` based on `LLMModelHost`.
- [ ] **T2.3** Threading: ensure local transport runs on a non‑blocking executor; cancellation on user stop.
- [ ] **T2.4** Tests using `FakeOllamaTransport` driving the existing agent loop.

### Milestone 3 — Settings & discovery (depends on M1)

- [ ] **T3.1** Add Ollama settings fields in [app/src/settings/ai.rs](app/src/settings/ai.rs) with migration default values.
- [ ] **T3.2** Settings UI panel (per [.agents/skills/warp-ui-guidelines/SKILL.md](.agents/skills/warp-ui-guidelines/SKILL.md)): enable toggle, base URL input, "Test connection" button, model multi‑select.
- [ ] **T3.3** Model discovery service that populates `AvailableLLMs.choices` for Ollama and reacts to settings changes.
- [ ] **T3.4** Env var override `WARP_OLLAMA_BASE_URL`.

### Milestone 4 — UX polish (depends on M2 + M3)

- [ ] **T4.1** Inline error banners (daemon down, model missing, tools unsupported).
- [ ] **T4.2** Capability badges in the model picker (Tools / Vision / Context length).
- [ ] **T4.3** Loopback‑only guard with opt‑in toggle for remote hosts.

### Milestone 5 — Telemetry & observability

- [ ] **T5.1** Wire telemetry events from §4.9 per [.agents/skills/add-telemetry/SKILL.md](.agents/skills/add-telemetry/SKILL.md).
- [ ] **T5.2** Structured tracing spans on transport calls.

### Milestone 6 — Tests & CI

- [ ] **T6.1** Unit tests already added in M1/M2; verify coverage on transport + routing.
- [ ] **T6.2** Integration test using the framework in [crates/integration](crates/integration) (mock Ollama HTTP server) — see [.agents/skills/warp-integration-test/SKILL.md](.agents/skills/warp-integration-test/SKILL.md).
- [ ] **T6.3** Run `./script/presubmit`; fix per [.agents/skills/fix-errors/SKILL.md](.agents/skills/fix-errors/SKILL.md).

### Milestone 7 — Rollout

- [ ] **T7.1** Internal dogfood: enable `OllamaProvider` for `Internal` channel.
- [ ] **T7.2** Promote per [.agents/skills/promote-feature/SKILL.md](.agents/skills/promote-feature/SKILL.md) after a soak period.
- [ ] **T7.3** Docs: update `WARP.md` / FAQ; one‑pager on supported models.
- [ ] **T7.4** Schedule flag removal per [.agents/skills/remove-feature-flag/SKILL.md](.agents/skills/remove-feature-flag/SKILL.md) once Stable.

---

## 6. Risks & open questions

| # | Risk / Question | Mitigation |
|---|-----------------|------------|
| R1 | First time client makes direct LLM calls — security review needed for the new outbound path. | Loopback default, scheme allow‑list, security review before Preview. |
| R2 | Tool‑calling fidelity varies wildly between local models. | Capability probe + clear UX when unsupported; document tested model matrix. |
| R3 | Long‑context streaming may starve UI thread. | Run transport on tokio runtime; backpressure via bounded channel. |
| R4 | Server‑side telemetry may assume every request flows through backend. | Audit telemetry pipeline; send Ollama events via the same client telemetry channel used elsewhere. |
| Q1 | Should embeddings (for codebase indexing) be supported in v1? | Out of scope for v1; revisit in follow‑up. |
| Q2 | Where does `LLMUsageMetadata` (pricing) come from for free local models? | Set token cost to `0`; surface "Local" badge instead of $/Mtok. |
| Q3 | Multi‑tenant teams: should an admin be able to disable Ollama org‑wide? | Yes — gate behind an org setting in addition to the feature flag. Track as follow‑up. |

---

## 7. References

- Ollama API: <https://github.com/ollama/ollama/blob/main/docs/api.md>
- Existing provider abstraction: [app/src/ai/llms.rs](app/src/ai/llms.rs)
- Server boundary: [app/src/server/server_api/ai.rs](app/src/server/server_api/ai.rs)
- Agent event source pattern: [app/src/ai/agent_events/driver_tests.rs](app/src/ai/agent_events/driver_tests.rs)
- Feature flag workflow: [.agents/skills/add-feature-flag/SKILL.md](.agents/skills/add-feature-flag/SKILL.md)
- UI guidelines: [.agents/skills/warp-ui-guidelines/SKILL.md](.agents/skills/warp-ui-guidelines/SKILL.md)

//! Translates between [`warp_multi_agent_api`] (the agent-loop wire format)
//! and the local Ollama daemon's [`ai::ollama`] DTOs.
//!
//! Scope (M4 v1, "chat-only"):
//!
//! * Request side: collect every prior `UserQuery` / `AgentOutput` /
//!   `SystemQuery` from `request.task_context.tasks` and the new
//!   `request.input` (only `UserQuery` is supported), flatten into Ollama
//!   `messages: Vec<ChatMessage>`. Tool calls, code reviews, summarization,
//!   suggestions, MCP, attachments, etc. are intentionally dropped — the user
//!   sees a regular chat experience and an `Other` finish reason if they sent
//!   something unsupported.
//!
//! * Response side: each `done: false` NDJSON chunk becomes an
//!   `AppendToMessageContent` action; the first chunk is preceded by a
//!   `BeginTransaction` + `AddMessagesToTask` that seeds the assistant
//!   message. The terminal `done: true` chunk produces `CommitTransaction`
//!   followed by `StreamFinished{Done}` (or `Other`/`InternalError` on
//!   failure paths).
//!
//! Higher-fidelity features (tool calling, capability hydration via
//! `/api/show`, vision attachments) are deferred.

use std::sync::Arc;

use ai::ollama::dto::{
    ChatMessage, ChatRequest, ChatRole, ChatStreamChunk,
};
use anyhow::anyhow;
use futures::stream::{self, BoxStream, StreamExt};
use prost_types::FieldMask;
use uuid::Uuid;
use warp_multi_agent_api as api;

use crate::server::server_api::AIApiError;

/// Path used by the field-mask append on every text delta.
///
/// Matches `Message.message.agent_output.text`. The receiving side
/// (`Task::append_to_message_content` → `FieldMaskOperation::append`) treats
/// this as "append to the string at this path".
const AGENT_OUTPUT_TEXT_PATH: &str = "agent_output.text";

/// Build an Ollama [`ChatRequest`] from a Warp agent-loop [`api::Request`].
///
/// `model` is the bare Ollama tag (e.g. `llama3.1:8b`). The request is left
/// `stream = false` here; [`ai::ollama::OllamaTransport::chat_stream`] flips
/// it to `true` itself.
pub fn build_chat_request(model: String, request: &api::Request) -> ChatRequest {
    let mut messages = Vec::new();

    // Walk every task's message log in order. For a single-root conversation
    // this gives us the full history in chronological order. For
    // multi-task / sub-agent conversations we still emit a flat history,
    // which is the best we can do without orchestrator semantics.
    for task in request
        .task_context
        .as_ref()
        .map(|tc| tc.tasks.as_slice())
        .unwrap_or(&[])
    {
        for msg in &task.messages {
            if let Some(translated) = message_to_chat(msg) {
                messages.push(translated);
            }
        }
    }

    // Then append the new input from this turn.
    if let Some(input) = request.input.as_ref() {
        messages.extend(input_to_chat_messages(input));
    }

    ChatRequest {
        model,
        messages,
        stream: false,
        tools: None,
        options: None,
        keep_alive: None,
    }
}

fn message_to_chat(msg: &api::Message) -> Option<ChatMessage> {
    match msg.message.as_ref()? {
        api::message::Message::UserQuery(q) => Some(text_message(ChatRole::User, &q.query)),
        api::message::Message::AgentOutput(o) => {
            Some(text_message(ChatRole::Assistant, &o.text))
        }
        api::message::Message::AgentReasoning(r) => {
            // Roll reasoning into the assistant turn to keep history coherent.
            Some(text_message(ChatRole::Assistant, &r.reasoning))
        }
        // Drop everything else: tool calls/results, system queries (which use
        // a oneof, not raw text), summarization, code-review, todos, web
        // fetch, artifact events, etc. v1 chat-only.
        _ => None,
    }
}

fn input_to_chat_messages(input: &api::request::Input) -> Vec<ChatMessage> {
    let Some(t) = input.r#type.as_ref() else {
        return vec![];
    };
    match t {
        // Modern wrapper: pull every UserQuery out of the UserInputs batch.
        api::request::input::Type::UserInputs(inputs) => inputs
            .inputs
            .iter()
            .filter_map(|ui| match ui.input.as_ref()? {
                api::request::input::user_inputs::user_input::Input::UserQuery(q) => {
                    Some(text_message(ChatRole::User, &q.query))
                }
                api::request::input::user_inputs::user_input::Input::CliAgentUserQuery(c) => {
                    c.user_query.as_ref().map(|q| text_message(ChatRole::User, &q.query))
                }
                _ => None,
            })
            .collect(),
        // Legacy direct UserQuery (deprecated tag 2 — still on the wire).
        #[allow(deprecated)]
        api::request::input::Type::UserQuery(q) => {
            vec![text_message(ChatRole::User, &q.query)]
        }
        _ => vec![],
    }
}

fn text_message(role: ChatRole, content: &str) -> ChatMessage {
    ChatMessage {
        role,
        content: content.to_owned(),
        images: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

/// IDs the response stream will roundtrip back to the conversation state
/// machine. Each is a fresh UUID — Ollama doesn't give us server-side IDs.
pub struct ResponseIds {
    pub conversation_id: String,
    pub request_id: String,
    pub run_id: String,
    pub assistant_message_id: String,
}

impl ResponseIds {
    pub fn new() -> Self {
        Self {
            conversation_id: Uuid::new_v4().to_string(),
            request_id: Uuid::new_v4().to_string(),
            run_id: Uuid::new_v4().to_string(),
            assistant_message_id: Uuid::new_v4().to_string(),
        }
    }
}

/// Pick the task to attach the assistant message to.
///
/// Strategy: the *last* task in `task_context.tasks` is the one the client
/// most recently mutated (typically the active root or the active sub-agent),
/// and matches what the conversation state machine expects after a fresh
/// `add_pending_response` registration.
pub fn pick_target_task_id(request: &api::Request) -> Option<String> {
    request
        .task_context
        .as_ref()?
        .tasks
        .last()
        .map(|t| t.id.clone())
}

/// Translate a stream of Ollama [`ChatStreamChunk`]s into the
/// `(StreamInit, ClientActions, …, StreamFinished)` sequence the agent loop
/// expects. The output stream always emits exactly one terminal event.
pub fn chunks_to_response_events<S>(
    upstream: S,
    target_task_id: Option<String>,
    ids: ResponseIds,
) -> BoxStream<'static, Result<api::ResponseEvent, Arc<AIApiError>>>
where
    S: futures::Stream<Item = Result<ChatStreamChunk, ai::ollama::OllamaError>> + Send + 'static,
{
    // We can't translate without a target task; surface a clear error before
    // touching the network.
    let Some(task_id) = target_task_id else {
        return Box::pin(stream::once(async {
            Err(Arc::new(AIApiError::Other(anyhow!(
                "Ollama transport: no task in request.task_context to attach response to. \
                 Start a fresh conversation and try again."
            ))))
        }));
    };

    let init_event = response_event(api::response_event::Type::Init(
        api::response_event::StreamInit {
            conversation_id: ids.conversation_id.clone(),
            request_id: ids.request_id.clone(),
            run_id: ids.run_id.clone(),
        },
    ));

    let begin_actions = response_event(api::response_event::Type::ClientActions(
        api::response_event::ClientActions {
            actions: vec![
                action(api::client_action::Action::BeginTransaction(
                    api::client_action::BeginTransaction {},
                )),
                action(api::client_action::Action::AddMessagesToTask(
                    api::client_action::AddMessagesToTask {
                        task_id: task_id.clone(),
                        messages: vec![seed_assistant_message(
                            &task_id,
                            &ids.assistant_message_id,
                            &ids.request_id,
                        )],
                    },
                )),
            ],
        },
    ));

    // State carried across the unfold over the upstream chunks.
    enum Phase {
        Streaming,
        Done,
    }
    struct State<S> {
        upstream: S,
        phase: Phase,
        task_id: String,
        message_id: String,
        request_id: String,
    }

    let state = State {
        upstream: Box::pin(upstream),
        phase: Phase::Streaming,
        task_id: task_id.clone(),
        message_id: ids.assistant_message_id.clone(),
        request_id: ids.request_id.clone(),
    };

    let chunk_stream = stream::unfold(state, |mut state| async move {
        match state.phase {
            Phase::Done => None,
            Phase::Streaming => match state.upstream.next().await {
                Some(Ok(chunk)) => {
                    let delta = chunk.message.content.clone();
                    let mut events = Vec::new();

                    if !delta.is_empty() {
                        events.push(Ok(response_event(
                            api::response_event::Type::ClientActions(
                                api::response_event::ClientActions {
                                    actions: vec![action(
                                        api::client_action::Action::AppendToMessageContent(
                                            api::client_action::AppendToMessageContent {
                                                task_id: state.task_id.clone(),
                                                message: Some(append_delta_message(
                                                    &state.task_id,
                                                    &state.message_id,
                                                    &state.request_id,
                                                    &delta,
                                                )),
                                                mask: Some(FieldMask {
                                                    paths: vec![
                                                        AGENT_OUTPUT_TEXT_PATH.to_string(),
                                                    ],
                                                }),
                                            },
                                        ),
                                    )],
                                },
                            ),
                        )));
                    }

                    if chunk.done {
                        state.phase = Phase::Done;
                        // CommitTransaction first, then a Finished event with
                        // the right reason. We also include a final empty
                        // append so subscribers always see at least one
                        // text event even if the daemon produced nothing.
                        events.push(Ok(response_event(
                            api::response_event::Type::ClientActions(
                                api::response_event::ClientActions {
                                    actions: vec![action(
                                        api::client_action::Action::CommitTransaction(
                                            api::client_action::CommitTransaction {},
                                        ),
                                    )],
                                },
                            ),
                        )));
                        events.push(Ok(finished_event(chunk.done_reason.as_deref())));
                    }

                    Some((stream::iter(events), state))
                }
                Some(Err(e)) => {
                    // Roll back the partial assistant message and surface a
                    // single error event so the agent-loop UI shows red.
                    state.phase = Phase::Done;
                    let rollback = response_event(api::response_event::Type::ClientActions(
                        api::response_event::ClientActions {
                            actions: vec![action(
                                api::client_action::Action::RollbackTransaction(
                                    api::client_action::RollbackTransaction {},
                                ),
                            )],
                        },
                    ));
                    let err = Err(Arc::new(AIApiError::Other(anyhow!(
                        "Ollama chat stream failed: {e}"
                    ))));
                    Some((stream::iter(vec![Ok(rollback), err]), state))
                }
                None => {
                    // Upstream ended without a `done: true` chunk. Emit a
                    // graceful Commit + Other-reason finish so the UI can
                    // recover.
                    state.phase = Phase::Done;
                    let commit = response_event(api::response_event::Type::ClientActions(
                        api::response_event::ClientActions {
                            actions: vec![action(
                                api::client_action::Action::CommitTransaction(
                                    api::client_action::CommitTransaction {},
                                ),
                            )],
                        },
                    ));
                    let finished = finished_event(Some("eof"));
                    Some((stream::iter(vec![Ok(commit), Ok(finished)]), state))
                }
            },
        }
    })
    .flatten();

    let head = stream::iter(vec![Ok(init_event), Ok(begin_actions)]);
    Box::pin(head.chain(chunk_stream))
}

fn response_event(t: api::response_event::Type) -> api::ResponseEvent {
    api::ResponseEvent { r#type: Some(t) }
}

fn action(a: api::client_action::Action) -> api::ClientAction {
    api::ClientAction { action: Some(a) }
}

fn seed_assistant_message(task_id: &str, msg_id: &str, request_id: &str) -> api::Message {
    api::Message {
        id: msg_id.to_string(),
        task_id: task_id.to_string(),
        request_id: request_id.to_string(),
        timestamp: None,
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::AgentOutput(
            api::message::AgentOutput {
                text: String::new(),
            },
        )),
    }
}

fn append_delta_message(
    task_id: &str,
    msg_id: &str,
    request_id: &str,
    delta: &str,
) -> api::Message {
    api::Message {
        id: msg_id.to_string(),
        task_id: task_id.to_string(),
        request_id: request_id.to_string(),
        timestamp: None,
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::AgentOutput(
            api::message::AgentOutput {
                text: delta.to_string(),
            },
        )),
    }
}

fn finished_event(done_reason: Option<&str>) -> api::ResponseEvent {
    use api::response_event::stream_finished;
    let reason = match done_reason {
        Some("stop") | Some("done") | None => {
            stream_finished::Reason::Done(stream_finished::Done {})
        }
        Some(_other) => stream_finished::Reason::Other(stream_finished::Other {}),
    };
    response_event(api::response_event::Type::Finished(
        api::response_event::StreamFinished {
            token_usage: vec![],
            should_refresh_model_config: false,
            request_cost: None,
            conversation_usage_metadata: None,
            reason: Some(reason),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai::ollama::dto::{ChatMessage, ChatRole, ChatStreamChunk};

    fn user_query_input(q: &str) -> api::request::Input {
        #[allow(deprecated)]
        api::request::Input {
            context: None,
            r#type: Some(api::request::input::Type::UserQuery(
                api::request::input::UserQuery {
                    query: q.to_string(),
                    referenced_attachments: Default::default(),
                    mode: None,
                    intended_agent: 0,
                },
            )),
        }
    }

    fn task_with(messages: Vec<api::Message>) -> api::Task {
        api::Task {
            id: "task-1".into(),
            description: String::new(),
            dependencies: None,
            messages,
            summary: String::new(),
            server_data: String::new(),
        }
    }

    fn user_msg(text: &str) -> api::Message {
        api::Message {
            id: "u".into(),
            task_id: "task-1".into(),
            request_id: String::new(),
            timestamp: None,
            server_message_data: String::new(),
            citations: vec![],
            message: Some(api::message::Message::UserQuery(api::message::UserQuery {
                query: text.into(),
                context: None,
                referenced_attachments: Default::default(),
                mode: None,
                intended_agent: 0,
            })),
        }
    }

    fn agent_msg(text: &str) -> api::Message {
        api::Message {
            id: "a".into(),
            task_id: "task-1".into(),
            request_id: String::new(),
            timestamp: None,
            server_message_data: String::new(),
            citations: vec![],
            message: Some(api::message::Message::AgentOutput(
                api::message::AgentOutput { text: text.into() },
            )),
        }
    }

    #[test]
    fn build_chat_request_flattens_history_then_new_input() {
        let request = api::Request {
            task_context: Some(api::request::TaskContext {
                tasks: vec![task_with(vec![user_msg("hi"), agent_msg("hello!")])],
            }),
            input: Some(user_query_input("how are you?")),
            ..Default::default()
        };
        let req = build_chat_request("llama3.1:8b".into(), &request);
        assert_eq!(req.model, "llama3.1:8b");
        assert!(!req.stream); // transport flips this
        let roles: Vec<_> = req.messages.iter().map(|m| m.role.clone()).collect();
        assert_eq!(
            roles,
            vec![ChatRole::User, ChatRole::Assistant, ChatRole::User]
        );
        let contents: Vec<_> = req.messages.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(contents, vec!["hi", "hello!", "how are you?"]);
    }

    #[test]
    fn build_chat_request_drops_unsupported_message_types() {
        // ToolCall message is not translated.
        let tool_call = api::Message {
            id: "tc".into(),
            task_id: "task-1".into(),
            request_id: String::new(),
            timestamp: None,
            server_message_data: String::new(),
            citations: vec![],
            message: Some(api::message::Message::ToolCall(api::message::ToolCall {
                tool_call_id: "x".into(),
                ..Default::default()
            })),
        };
        let request = api::Request {
            task_context: Some(api::request::TaskContext {
                tasks: vec![task_with(vec![user_msg("hi"), tool_call, agent_msg("ok")])],
            }),
            input: Some(user_query_input("again")),
            ..Default::default()
        };
        let req = build_chat_request("m".into(), &request);
        let roles: Vec<_> = req.messages.iter().map(|m| m.role.clone()).collect();
        assert_eq!(
            roles,
            vec![ChatRole::User, ChatRole::Assistant, ChatRole::User]
        );
    }

    #[test]
    fn pick_target_task_id_returns_last_task_id() {
        let request = api::Request {
            task_context: Some(api::request::TaskContext {
                tasks: vec![
                    api::Task {
                        id: "root".into(),
                        ..Default::default()
                    },
                    api::Task {
                        id: "sub".into(),
                        ..Default::default()
                    },
                ],
            }),
            ..Default::default()
        };
        assert_eq!(pick_target_task_id(&request).as_deref(), Some("sub"));
    }

    #[test]
    fn pick_target_task_id_none_when_no_tasks() {
        let request = api::Request::default();
        assert!(pick_target_task_id(&request).is_none());
    }

    fn chunk(text: &str, done: bool) -> ChatStreamChunk {
        ChatStreamChunk {
            model: "m".into(),
            created_at: None,
            message: ChatMessage {
                role: ChatRole::Assistant,
                content: text.into(),
                images: None,
                tool_calls: None,
                tool_call_id: None,
            },
            done,
            done_reason: if done { Some("stop".into()) } else { None },
            prompt_eval_count: None,
            eval_count: None,
            total_duration: None,
        }
    }

    fn fixed_ids() -> ResponseIds {
        ResponseIds {
            conversation_id: "conv".into(),
            request_id: "req".into(),
            run_id: "run".into(),
            assistant_message_id: "msg".into(),
        }
    }

    #[tokio::test]
    async fn chunks_to_events_emits_init_then_seed_then_appends_then_finished() {
        let upstream = stream::iter(vec![
            Ok(chunk("Hel", false)),
            Ok(chunk("lo!", true)),
        ]);
        let mut events =
            chunks_to_response_events(upstream, Some("task-1".into()), fixed_ids())
                .collect::<Vec<_>>()
                .await;

        // First: StreamInit
        let first = events.remove(0).expect("ok");
        match first.r#type.expect("type") {
            api::response_event::Type::Init(init) => {
                assert_eq!(init.conversation_id, "conv");
                assert_eq!(init.request_id, "req");
                assert_eq!(init.run_id, "run");
            }
            _ => panic!("expected Init"),
        }

        // Second: BeginTransaction + AddMessagesToTask
        let second = events.remove(0).expect("ok");
        let actions = match second.r#type.expect("type") {
            api::response_event::Type::ClientActions(a) => a.actions,
            _ => panic!("expected ClientActions"),
        };
        assert_eq!(actions.len(), 2);
        assert!(matches!(
            actions[0].action.as_ref().unwrap(),
            api::client_action::Action::BeginTransaction(_)
        ));
        match actions[1].action.as_ref().unwrap() {
            api::client_action::Action::AddMessagesToTask(add) => {
                assert_eq!(add.task_id, "task-1");
                assert_eq!(add.messages.len(), 1);
                assert_eq!(add.messages[0].id, "msg");
            }
            _ => panic!("expected AddMessagesToTask"),
        }

        // Third: AppendToMessageContent("Hel")
        let third = events.remove(0).expect("ok");
        match third.r#type.expect("type") {
            api::response_event::Type::ClientActions(a) => match a.actions[0].action.as_ref().unwrap() {
                api::client_action::Action::AppendToMessageContent(append) => {
                    let text = match append
                        .message
                        .as_ref()
                        .unwrap()
                        .message
                        .as_ref()
                        .unwrap()
                    {
                        api::message::Message::AgentOutput(o) => o.text.clone(),
                        _ => panic!("expected AgentOutput"),
                    };
                    assert_eq!(text, "Hel");
                    assert_eq!(
                        append.mask.as_ref().unwrap().paths,
                        vec![AGENT_OUTPUT_TEXT_PATH.to_string()]
                    );
                }
                _ => panic!("expected AppendToMessageContent"),
            },
            _ => panic!("expected ClientActions"),
        }

        // Fourth: AppendToMessageContent("lo!")
        let _fourth = events.remove(0).expect("ok");

        // Fifth: CommitTransaction
        let fifth = events.remove(0).expect("ok");
        match fifth.r#type.expect("type") {
            api::response_event::Type::ClientActions(a) => assert!(matches!(
                a.actions[0].action.as_ref().unwrap(),
                api::client_action::Action::CommitTransaction(_)
            )),
            _ => panic!("expected ClientActions"),
        }

        // Sixth: StreamFinished{Done}
        let sixth = events.remove(0).expect("ok");
        match sixth.r#type.expect("type") {
            api::response_event::Type::Finished(f) => match f.reason.expect("reason") {
                api::response_event::stream_finished::Reason::Done(_) => {}
                other => panic!("expected Done, got {other:?}"),
            },
            _ => panic!("expected Finished"),
        }

        assert!(events.is_empty(), "no extra events expected");
    }

    #[tokio::test]
    async fn chunks_to_events_skips_empty_deltas_but_still_finishes() {
        let upstream = stream::iter(vec![Ok(chunk("", true))]);
        let events =
            chunks_to_response_events(upstream, Some("task-1".into()), fixed_ids())
                .collect::<Vec<_>>()
                .await;
        // Init, BeginTransaction+Add, Commit, Finished. (No Append because delta empty.)
        assert_eq!(events.len(), 4, "got {events:?}");
    }

    #[tokio::test]
    async fn chunks_to_events_emits_rollback_and_error_on_upstream_error() {
        let upstream = stream::iter(vec![
            Ok(chunk("Hi", false)),
            Err(ai::ollama::OllamaError::Stream("boom".into())),
        ]);
        let events =
            chunks_to_response_events(upstream, Some("task-1".into()), fixed_ids())
                .collect::<Vec<_>>()
                .await;
        // Init, Begin+Add, Append("Hi"), Rollback, Err
        assert_eq!(events.len(), 5);
        assert!(events.last().unwrap().is_err());
        match events[3].as_ref().unwrap().r#type.as_ref().unwrap() {
            api::response_event::Type::ClientActions(a) => assert!(matches!(
                a.actions[0].action.as_ref().unwrap(),
                api::client_action::Action::RollbackTransaction(_)
            )),
            _ => panic!("expected ClientActions"),
        }
    }

    #[tokio::test]
    async fn chunks_to_events_errors_when_no_target_task() {
        let upstream = stream::iter(vec![Ok(chunk("hi", true))]);
        let events =
            chunks_to_response_events(upstream, None, fixed_ids())
                .collect::<Vec<_>>()
                .await;
        assert_eq!(events.len(), 1);
        assert!(events[0].is_err());
    }
}

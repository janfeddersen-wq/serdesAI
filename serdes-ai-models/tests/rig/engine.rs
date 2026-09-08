//! Turn execution engine: bridges Responses API turns onto a serdesAI model.
//!
//! A turn is prepared by resolving conversation history (from
//! `previous_response_id` chaining or by replaying the full input), executed
//! against the backing [`Model`], and optionally persisted for later
//! retrieval and chaining.

use super::convert::{
    input_to_history, new_id, output_items_from_response, tool_choice, tool_definitions,
};
use super::error::{ResponsesError, codes};
use super::store::{InMemoryResponseStore, SessionResponseCache, StoredResponse};
use chrono::Utc;
use futures::StreamExt;
use serdes_ai_core::messages::{
    ModelRequestPart, ModelResponsePart, ModelResponseStreamEvent, StreamCompleteEvent, TextPart,
    ThinkingPart, ThinkingPartDelta, ToolCallArgs, ToolCallPart,
};
use serdes_ai_core::{FinishReason, ModelRequest, ModelResponse, ModelSettings, RequestUsage};
use serdes_ai_models::model::{Model, ModelRequestParameters};
use serdes_ai_models::openai::responses::events::StreamEvent;
use serdes_ai_models::openai::responses::wire::*;
use std::collections::HashMap;
use std::sync::Arc;
async fn emit<S, F>(
    event: StreamEvent,
    sequence: &mut u64,
    sink: &mut S,
) -> Result<(), ResponsesError>
where
    S: FnMut(StreamEvent) -> F + Send,
    F: std::future::Future<Output = Result<(), ResponsesError>> + Send,
{
    sink(event).await?;
    *sequence += 1;
    Ok(())
}

/// Executes Responses API turns against a backing serdesAI model.
pub struct ResponsesEngine {
    model: Arc<dyn Model>,
    store: Arc<InMemoryResponseStore>,
}

impl ResponsesEngine {
    /// Create an engine serving `model` with an in-memory response store.
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            store: Arc::new(InMemoryResponseStore::default()),
        }
    }

    /// The backing model.
    #[must_use]
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    /// The backing response store.
    pub fn response_store(&self) -> &Arc<InMemoryResponseStore> {
        &self.store
    }

    /// Fetch a stored response by ID.
    pub async fn get_response(&self, id: &str) -> Option<StoredResponse> {
        self.store.get(id).await
    }

    /// Validate a request and resolve its conversation history.
    ///
    /// When `previous_response_id` is set, history is resolved from the
    /// connection-local session cache first (websocket turns with
    /// `store: false`) and the shared store second. Instructions sent on a
    /// chained turn replace the stored instructions.
    pub async fn prepare(
        &self,
        request: &CreateResponseRequest,
        session: Option<&SessionResponseCache>,
    ) -> Result<PreparedTurn, ResponsesError> {
        if request.background == Some(true) {
            return Err(ResponsesError::InvalidRequest(
                "background mode is not supported by this server".to_string(),
            ));
        }

        let tools = tool_definitions(request.tools.as_deref())?;
        let mut params = ModelRequestParameters::new().with_tools(tools);
        params.stream_usage = true;
        if let Some(choice) = tool_choice(request.tool_choice.as_ref()) {
            params = params.with_tool_choice(choice);
        }

        let settings = ModelSettings {
            temperature: request.temperature,
            top_p: request.top_p,
            max_tokens: request.max_output_tokens,
            parallel_tool_calls: request.parallel_tool_calls,
            ..ModelSettings::default()
        };

        let history = if let Some(previous_id) = &request.previous_response_id {
            let stored = match session.and_then(|cache| cache.get(previous_id)) {
                Some(stored) => Some(stored),
                None => self.store.get(previous_id).await,
            };
            let Some(stored) = stored else {
                return Err(ResponsesError::PreviousResponseNotFound(
                    previous_id.clone(),
                ));
            };
            let mut history = apply_instructions(stored.history, request.instructions.as_deref());
            let new_items = input_to_history(&request.input, None, &history)?;
            history.extend(new_items);
            history
        } else {
            input_to_history(&request.input, request.instructions.as_deref(), &[])?
        };

        Ok(PreparedTurn {
            request: request.clone(),
            response_id: new_id("resp_"),
            created_at: Utc::now().timestamp(),
            history,
            params,
            settings,
        })
    }

    /// Execute a turn without streaming.
    pub async fn execute(&self, turn: PreparedTurn) -> Result<TurnOutput, ResponsesError> {
        let response = self
            .model
            .request(&turn.history, &turn.settings, &turn.params)
            .await
            .map_err(|err| ResponsesError::Model(err.to_string()))?;

        let output = output_items_from_response(&response);
        let mut object = self.skeleton(&turn);
        object.output = output;
        finish_object(&mut object, response.finish_reason, response.usage.as_ref());

        Ok(TurnOutput {
            response: object,
            history: turn.history,
            model_response: response,
        })
    }

    /// Execute a turn with streaming, passing each event to `sink` in order.
    ///
    /// Events carry consecutive `sequence_number`s starting at 0. On model
    /// failure or premature stream end, open items are closed, the response
    /// is finished as `failed`, `response.failed` is emitted, and the error
    /// is returned.
    pub async fn execute_streaming<S, F>(
        &self,
        turn: PreparedTurn,
        sink: &mut S,
    ) -> Result<TurnOutput, ResponsesError>
    where
        S: FnMut(StreamEvent) -> F + Send,
        F: std::future::Future<Output = Result<(), ResponsesError>> + Send,
    {
        let mut object = self.skeleton(&turn);
        let mut sequence: u64 = 0;

        let mut stream = self
            .model
            .request_stream(&turn.history, &turn.settings, &turn.params)
            .await
            .map_err(|err| ResponsesError::Model(err.to_string()))?;

        emit(
            StreamEvent::ResponseCreated {
                sequence_number: sequence,
                response: object.clone(),
            },
            &mut sequence,
            sink,
        )
        .await?;
        emit(
            StreamEvent::ResponseInProgress {
                sequence_number: sequence,
                response: object.clone(),
            },
            &mut sequence,
            sink,
        )
        .await?;

        let mut assembler = StreamAssembler::new();
        let mut failure: Option<String> = None;
        let mut usage: Option<RequestUsage> = None;
        let mut finish_reason: Option<FinishReason> = None;
        let mut terminal_seen = false;

        while let Some(item) = stream.next().await {
            let event = match item {
                Ok(event) => event,
                Err(err) => {
                    failure = Some(err.to_string());
                    break;
                }
            };
            match event {
                ModelResponseStreamEvent::PartStart(start) => {
                    for event in assembler.part_started(&start, &mut sequence) {
                        sink(event).await?;
                    }
                }
                ModelResponseStreamEvent::PartDelta(delta) => {
                    if let Some(event) = assembler.part_delta(&delta, &mut sequence) {
                        sink(event).await?;
                    }
                }
                ModelResponseStreamEvent::PartEnd(end) => {
                    for event in assembler.part_ended(&end.index, &mut sequence) {
                        sink(event).await?;
                    }
                }
                ModelResponseStreamEvent::StreamComplete(complete) => {
                    usage = usage_from_stream_complete(&complete);
                    finish_reason = Some(complete.finish_reason);
                    terminal_seen = true;
                    break;
                }
            }
        }

        for event in assembler.close_open_items(&mut sequence) {
            sink(event).await?;
        }
        object.output = assembler.completed_items();
        let parts = assembler.take_parts();

        if !terminal_seen && failure.is_none() {
            failure = Some("model stream ended without a terminal event".to_string());
        }

        if let Some(reason) = failure {
            object.status = ResponseStatus::Failed;
            object.error = Some(ErrorBodyRef {
                code: codes::MODEL_ERROR.to_string(),
                message: reason.clone(),
            });
            emit(
                StreamEvent::ResponseFailed {
                    sequence_number: sequence,
                    response: object.clone(),
                },
                &mut sequence,
                sink,
            )
            .await?;
            return Err(ResponsesError::Model(reason));
        }

        finish_object(&mut object, finish_reason, usage.as_ref());
        let final_event = if object.status == ResponseStatus::Incomplete {
            StreamEvent::ResponseIncomplete {
                sequence_number: sequence,
                response: object.clone(),
            }
        } else {
            StreamEvent::ResponseCompleted {
                sequence_number: sequence,
                response: object.clone(),
            }
        };
        emit(final_event, &mut sequence, sink).await?;

        Ok(TurnOutput {
            response: object,
            history: turn.history,
            model_response: ModelResponse {
                parts,
                model_name: Some(turn.request.model.clone()),
                timestamp: Utc::now(),
                finish_reason,
                usage,
                vendor_id: None,
                vendor_details: None,
                kind: "response".to_string(),
            },
        })
    }

    /// Persist a finished turn according to the request's `store` flag.
    ///
    /// `store: true` (the default) writes to the shared store. `store: false`
    /// on a websocket session writes to the connection-local cache so the
    /// next turn on the same socket can chain via `previous_response_id`.
    pub async fn persist(
        &self,
        request: &CreateResponseRequest,
        session: Option<&SessionResponseCache>,
        output: &TurnOutput,
    ) {
        // Stored history includes this turn's assistant response so chained
        // turns replay the full conversation.
        let mut history = output.history.clone();
        let mut assistant_request = ModelRequest::new();
        assistant_request.add_part(ModelRequestPart::ModelResponse(Box::new(
            output.model_response.clone(),
        )));
        history.push(assistant_request);

        let stored = StoredResponse {
            id: output.response.id.clone(),
            response: output.response.clone(),
            history,
            stored_at: Utc::now(),
        };
        if request.store.unwrap_or(true) {
            self.store.put(stored).await;
        } else if let Some(session) = session {
            session.put(stored);
        }
    }

    fn skeleton(&self, turn: &PreparedTurn) -> ResponseObject {
        ResponseObject::in_progress(
            turn.response_id.clone(),
            turn.created_at,
            turn.request.model.clone(),
            &turn.request,
        )
    }
}

/// A validated request with resolved history, ready to execute.
pub struct PreparedTurn {
    /// The original request.
    pub request: CreateResponseRequest,
    /// Server-assigned response ID.
    pub response_id: String,
    /// Creation time (Unix seconds).
    pub created_at: i64,
    /// Full conversation history including the current turn's input.
    pub history: Vec<ModelRequest>,
    /// Model request parameters (tools, tool choice).
    pub params: ModelRequestParameters,
    /// Model settings mapped from the request.
    pub settings: ModelSettings,
}

/// The result of an executed turn.
pub struct TurnOutput {
    /// The final response object.
    pub response: ResponseObject,
    /// The conversation history up to (not including) this turn's output.
    pub history: Vec<ModelRequest>,
    /// The model response produced this turn, appended to the stored history
    /// by [`ResponsesEngine::persist`] so chained turns see the assistant's
    /// own reply.
    pub model_response: ModelResponse,
}

/// Replace any instructions in stored history with the current request's.
///
/// If the chained request carries no instructions, the stored instructions
/// are kept.
fn apply_instructions(
    mut history: Vec<ModelRequest>,
    instructions: Option<&str>,
) -> Vec<ModelRequest> {
    if let Some(instructions) = instructions {
        if history
            .first()
            .is_some_and(|request| request.kind == super::convert::INSTRUCTIONS_KIND)
        {
            history.remove(0);
        }
        history.insert(0, super::convert::instructions_request(instructions));
    }
    history
}

/// Map a stream-complete event onto request usage, if it reported any tokens.
fn usage_from_stream_complete(event: &StreamCompleteEvent) -> Option<RequestUsage> {
    if event.input_tokens.is_none() && event.output_tokens.is_none() {
        return None;
    }
    Some(RequestUsage {
        request_tokens: event.input_tokens,
        response_tokens: event.output_tokens,
        cache_creation_tokens: event.cache_creation_tokens,
        cache_read_tokens: event.cache_read_tokens,
        ..RequestUsage::default()
    })
}

/// Set final status, incomplete details, and usage on a response object.
fn finish_object(
    object: &mut ResponseObject,
    finish_reason: Option<FinishReason>,
    usage: Option<&RequestUsage>,
) {
    if let Some(usage) = usage {
        let input = usage.request_tokens;
        let output = usage.response_tokens;
        object.usage = Some(ResponseUsage {
            input_tokens: input,
            output_tokens: output,
            total_tokens: usage.total_tokens.or_else(|| match (input, output) {
                (Some(input), Some(output)) => Some(input + output),
                (Some(input), None) => Some(input),
                (None, Some(output)) => Some(output),
                (None, None) => None,
            }),
        });
    }
    match finish_reason {
        Some(FinishReason::Length) => {
            object.status = ResponseStatus::Incomplete;
            object.incomplete_details = Some(IncompleteDetails {
                reason: "max_output_tokens".to_string(),
            });
        }
        Some(FinishReason::Error) | Some(FinishReason::ContentFilter) => {
            object.status = ResponseStatus::Failed;
            object.error = Some(ErrorBodyRef {
                code: codes::MODEL_ERROR.to_string(),
                message: format!("model finished with {finish_reason:?}"),
            });
        }
        _ => {
            object.status = ResponseStatus::Completed;
        }
    }
}

/// Accumulates open output items while translating model stream events.
///
/// Each part index maps to at most one open output item. Completed items are
/// kept in completion order.
struct StreamAssembler {
    next_output_index: u64,
    completed: Vec<OutputItem>,
    completed_parts: Vec<ModelResponsePart>,
    open: HashMap<usize, OpenItem>,
}

enum OpenItem {
    Text {
        output_index: u64,
        item_id: String,
        text: String,
    },
    Reasoning {
        output_index: u64,
        item_id: String,
        thinking: ThinkingPart,
    },
    FunctionCall {
        output_index: u64,
        item_id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
    /// A part kind with no Responses item mapping (file, builtin tool call).
    Unmapped,
}

impl StreamAssembler {
    fn new() -> Self {
        Self {
            next_output_index: 0,
            completed: Vec::new(),
            completed_parts: Vec::new(),
            open: HashMap::new(),
        }
    }

    fn part_started(
        &mut self,
        start: &serdes_ai_core::messages::PartStartEvent,
        sequence: &mut u64,
    ) -> Vec<StreamEvent> {
        let output_index = self.next_output_index;
        self.next_output_index += 1;
        match &start.part {
            ModelResponsePart::Text(text) => {
                let item_id = new_id("msg_");
                let mut events = vec![
                    StreamEvent::OutputItemAdded {
                        sequence_number: *sequence,
                        output_index,
                        item: OutputItem::Message {
                            id: item_id.clone(),
                            role: "assistant".to_string(),
                            status: OutputItemStatus::InProgress,
                            content: Vec::new(),
                        },
                    },
                    StreamEvent::ContentPartAdded {
                        sequence_number: *sequence + 1,
                        item_id: item_id.clone(),
                        output_index,
                        content_index: 0,
                        part: OutputContent::OutputText {
                            text: String::new(),
                            annotations: Vec::new(),
                        },
                    },
                ];
                *sequence += 2;
                self.open.insert(
                    start.index,
                    OpenItem::Text {
                        output_index,
                        item_id,
                        text: String::new(),
                    },
                );
                if !text.content.is_empty() {
                    if let Some(event) = self.text_delta(start.index, &text.content, sequence) {
                        events.push(event);
                    }
                }
                events
            }
            ModelResponsePart::Thinking(thinking) => {
                let item_id = new_id("rs_");
                let mut events = vec![
                    StreamEvent::OutputItemAdded {
                        sequence_number: *sequence,
                        output_index,
                        item: OutputItem::Reasoning {
                            id: item_id.clone(),
                            summary: Vec::new(),
                            encrypted_content: super::convert::encrypted_content(thinking),
                        },
                    },
                    StreamEvent::ReasoningSummaryPartAdded {
                        sequence_number: *sequence + 1,
                        item_id: item_id.clone(),
                        output_index,
                        summary_index: 0,
                        part: SummaryTextItem::new(String::new()),
                    },
                ];
                *sequence += 2;
                self.open.insert(
                    start.index,
                    OpenItem::Reasoning {
                        output_index,
                        item_id,
                        thinking: ThinkingPart {
                            content: String::new(),
                            ..thinking.clone()
                        },
                    },
                );
                if !thinking.content.is_empty() {
                    if let Some(event) = self.reasoning_delta(
                        start.index,
                        &ThinkingPartDelta::new(&thinking.content),
                        sequence,
                    ) {
                        events.push(event);
                    }
                }
                events
            }
            ModelResponsePart::ToolCall(call) => {
                let item_id = new_id("fc_");
                let call_id = call.tool_call_id.clone().unwrap_or_else(|| new_id("call_"));
                let initial_args = call.args.to_json_string().unwrap_or_default();
                let mut events = vec![StreamEvent::OutputItemAdded {
                    sequence_number: *sequence,
                    output_index,
                    item: OutputItem::FunctionCall {
                        id: item_id.clone(),
                        call_id: call_id.clone(),
                        name: call.tool_name.clone(),
                        arguments: String::new(),
                        status: OutputItemStatus::InProgress,
                    },
                }];
                *sequence += 1;
                self.open.insert(
                    start.index,
                    OpenItem::FunctionCall {
                        output_index,
                        item_id,
                        call_id,
                        name: call.tool_name.clone(),
                        arguments: initial_args.clone(),
                    },
                );
                if !initial_args.is_empty() && initial_args != "{}" && initial_args != "null" {
                    if let Some(OpenItem::FunctionCall { item_id, .. }) =
                        self.open.get(&start.index)
                    {
                        events.push(StreamEvent::FunctionCallArgumentsDelta {
                            sequence_number: *sequence,
                            item_id: item_id.clone(),
                            output_index,
                            delta: initial_args,
                        });
                        *sequence += 1;
                    }
                }
                events
            }
            ModelResponsePart::File(_) | ModelResponsePart::BuiltinToolCall(_) => {
                self.open.insert(start.index, OpenItem::Unmapped);
                Vec::new()
            }
        }
    }

    fn text_delta(&mut self, index: usize, delta: &str, sequence: &mut u64) -> Option<StreamEvent> {
        let open = self.open.get_mut(&index)?;
        match open {
            OpenItem::Text {
                output_index,
                item_id,
                text,
            } => {
                text.push_str(delta);
                let event = StreamEvent::OutputTextDelta {
                    sequence_number: *sequence,
                    item_id: item_id.clone(),
                    output_index: *output_index,
                    content_index: 0,
                    delta: delta.to_string(),
                };
                *sequence += 1;
                Some(event)
            }
            _ => None,
        }
    }

    fn reasoning_delta(
        &mut self,
        index: usize,
        delta: &ThinkingPartDelta,
        sequence: &mut u64,
    ) -> Option<StreamEvent> {
        let open = self.open.get_mut(&index)?;
        match open {
            OpenItem::Reasoning {
                output_index,
                item_id,
                thinking,
            } => {
                delta.apply(thinking);
                let event = StreamEvent::ReasoningSummaryTextDelta {
                    sequence_number: *sequence,
                    item_id: item_id.clone(),
                    output_index: *output_index,
                    summary_index: 0,
                    delta: delta.content_delta.clone(),
                };
                *sequence += 1;
                Some(event)
            }
            _ => None,
        }
    }

    fn part_delta(
        &mut self,
        delta: &serdes_ai_core::messages::PartDeltaEvent,
        sequence: &mut u64,
    ) -> Option<StreamEvent> {
        use serdes_ai_core::messages::ModelResponsePartDelta;
        match &delta.delta {
            ModelResponsePartDelta::Text(text) => {
                self.text_delta(delta.index, &text.content_delta, sequence)
            }
            ModelResponsePartDelta::Thinking(thinking) => {
                self.reasoning_delta(delta.index, thinking, sequence)
            }
            ModelResponsePartDelta::ToolCall(tool) => {
                let open = self.open.get_mut(&delta.index)?;
                match open {
                    OpenItem::FunctionCall {
                        output_index,
                        item_id,
                        arguments,
                        ..
                    } => {
                        arguments.push_str(&tool.args_delta);
                        let event = StreamEvent::FunctionCallArgumentsDelta {
                            sequence_number: *sequence,
                            item_id: item_id.clone(),
                            output_index: *output_index,
                            delta: tool.args_delta.clone(),
                        };
                        *sequence += 1;
                        Some(event)
                    }
                    _ => None,
                }
            }
            ModelResponsePartDelta::BuiltinToolCall(_) => None,
        }
    }

    fn part_ended(&mut self, index: &usize, sequence: &mut u64) -> Vec<StreamEvent> {
        let Some(open) = self.open.remove(index) else {
            return Vec::new();
        };
        let mut events = Vec::new();
        match open {
            OpenItem::Text {
                output_index,
                item_id,
                text,
            } => {
                let item = OutputItem::Message {
                    id: item_id.clone(),
                    role: "assistant".to_string(),
                    status: OutputItemStatus::Completed,
                    content: vec![OutputContent::OutputText {
                        text: text.clone(),
                        annotations: Vec::new(),
                    }],
                };
                events.push(StreamEvent::ContentPartDone {
                    sequence_number: *sequence,
                    item_id,
                    output_index,
                    content_index: 0,
                    part: OutputContent::OutputText {
                        text: text.clone(),
                        annotations: Vec::new(),
                    },
                });
                *sequence += 1;
                events.push(StreamEvent::OutputItemDone {
                    sequence_number: *sequence,
                    output_index,
                    item: item.clone(),
                });
                *sequence += 1;
                self.completed.push(item);
                self.completed_parts
                    .push(ModelResponsePart::Text(TextPart::new(text)));
            }
            OpenItem::Reasoning {
                output_index,
                item_id,
                thinking,
            } => {
                let text = &thinking.content;
                events.push(StreamEvent::ReasoningSummaryTextDone {
                    sequence_number: *sequence,
                    item_id: item_id.clone(),
                    output_index,
                    summary_index: 0,
                    text: text.clone(),
                });
                *sequence += 1;
                events.push(StreamEvent::ReasoningSummaryPartDone {
                    sequence_number: *sequence,
                    item_id: item_id.clone(),
                    output_index,
                    summary_index: 0,
                    part: SummaryTextItem::new(text.clone()),
                });
                *sequence += 1;
                let item = OutputItem::Reasoning {
                    id: item_id,
                    summary: vec![SummaryTextItem::new(text.clone())],
                    encrypted_content: super::convert::encrypted_content(&thinking),
                };
                self.completed_parts
                    .push(ModelResponsePart::Thinking(thinking));
                events.push(StreamEvent::OutputItemDone {
                    sequence_number: *sequence,
                    output_index,
                    item: item.clone(),
                });
                *sequence += 1;
                self.completed.push(item);
            }
            OpenItem::FunctionCall {
                output_index,
                item_id,
                call_id,
                name,
                arguments,
            } => {
                self.completed_parts
                    .push(ModelResponsePart::ToolCall(ToolCallPart {
                        tool_name: name.clone(),
                        args: ToolCallArgs::from(arguments.clone()),
                        tool_call_id: Some(call_id.clone()),
                        id: None,
                        provider_details: None,
                    }));
                events.push(StreamEvent::FunctionCallArgumentsDone {
                    sequence_number: *sequence,
                    item_id: item_id.clone(),
                    output_index,
                    arguments: arguments.clone(),
                });
                *sequence += 1;
                let item = OutputItem::FunctionCall {
                    id: item_id,
                    call_id,
                    name,
                    arguments,
                    status: OutputItemStatus::Completed,
                };
                events.push(StreamEvent::OutputItemDone {
                    sequence_number: *sequence,
                    output_index,
                    item: item.clone(),
                });
                *sequence += 1;
                self.completed.push(item);
            }
            OpenItem::Unmapped => {}
        }
        events
    }

    fn close_open_items(&mut self, sequence: &mut u64) -> Vec<StreamEvent> {
        let mut indexes: Vec<usize> = self.open.keys().copied().collect();
        indexes.sort_unstable();
        let mut events = Vec::new();
        for index in indexes {
            events.extend(self.part_ended(&index, sequence));
        }
        events
    }

    fn completed_items(&mut self) -> Vec<OutputItem> {
        std::mem::take(&mut self.completed)
    }

    fn take_parts(&mut self) -> Vec<ModelResponsePart> {
        std::mem::take(&mut self.completed_parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serdes_ai_models::mock::FunctionModel;

    fn request(model: &str, input: &str) -> CreateResponseRequest {
        CreateResponseRequest {
            model: model.to_string(),
            input: ResponseInput::Text(input.to_string()),
            instructions: Some("be brief".to_string()),
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            stream: None,
            background: None,
            store: None,
            previous_response_id: None,
            reasoning: None,
            parallel_tool_calls: None,
            metadata: None,
            user: None,
            truncation: None,
            include: None,
            text: None,
            service_tier: None,
        }
    }

    #[tokio::test]
    async fn rejects_background_mode() {
        let engine = ResponsesEngine::new(Arc::new(FunctionModel::constant_text("hi")));
        let mut req = request("m", "hello");
        req.background = Some(true);
        let err = engine
            .prepare(&req, None)
            .await
            .err()
            .expect("expected error");
        assert!(matches!(err, ResponsesError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn chained_turn_resolves_history_and_replaces_instructions() {
        let engine = ResponsesEngine::new(Arc::new(FunctionModel::constant_text("first")));
        let first = engine.prepare(&request("m", "hello"), None).await.unwrap();
        let output = engine.execute(first).await.unwrap();
        let first_id = output.response.id.clone();
        engine.persist(&request("m", "hello"), None, &output).await;

        let mut chained = request("m", "and then?");
        chained.previous_response_id = Some(first_id);
        chained.instructions = Some("new instructions".to_string());
        let turn = engine.prepare(&chained, None).await.unwrap();

        let system: Vec<String> = turn
            .history
            .iter()
            .flat_map(|r| r.parts.iter())
            .filter_map(|part| match part {
                ModelRequestPart::SystemPrompt(s) => Some(s.content.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(system, vec!["new instructions".to_string()]);

        let user_texts: Vec<String> = turn
            .history
            .iter()
            .flat_map(|r| r.parts.iter())
            .filter_map(|part| match part {
                ModelRequestPart::UserPrompt(u) => match &u.content {
                    serdes_ai_core::messages::UserContent::Text(t) => Some(t.clone()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(user_texts.len(), 2);
        assert_eq!(user_texts[1], "and then?");
    }

    #[tokio::test]
    async fn unknown_previous_response_id_is_404() {
        let engine = ResponsesEngine::new(Arc::new(FunctionModel::constant_text("hi")));
        let mut req = request("m", "hello");
        req.previous_response_id = Some("resp_missing".to_string());
        let err = engine
            .prepare(&req, None)
            .await
            .err()
            .expect("expected error");
        assert!(
            matches!(err, ResponsesError::PreviousResponseNotFound(id) if id == "resp_missing")
        );
    }

    #[tokio::test]
    async fn session_cache_serves_store_false_continuations() {
        let engine = ResponsesEngine::new(Arc::new(FunctionModel::constant_text("ok")));
        let session = SessionResponseCache::default();
        let mut first = request("m", "hello");
        first.store = Some(false);
        let output = engine
            .execute(engine.prepare(&first, Some(&session)).await.unwrap())
            .await
            .unwrap();
        engine.persist(&first, Some(&session), &output).await;
        assert!(engine.get_response(&output.response.id).await.is_none());

        let mut second = request("m", "next");
        second.store = Some(false);
        second.previous_response_id = Some(output.response.id.clone());
        let turn = engine.prepare(&second, Some(&session)).await.unwrap();
        assert!(turn.history.len() >= 2);
    }

    #[tokio::test]
    async fn streaming_emits_ordered_events_and_usage() {
        use futures::stream;
        use serdes_ai_core::messages::{
            ModelResponsePartDelta, PartDeltaEvent, PartEndEvent, PartStartEvent,
            StreamCompleteEvent, TextPartDelta,
        };

        let model = FunctionModel::with_stream(move |_, _| {
            Box::pin(stream::iter(vec![
                Ok(ModelResponseStreamEvent::PartStart(PartStartEvent::new(
                    0,
                    ModelResponsePart::Text(serdes_ai_core::messages::TextPart::new("hello")),
                ))),
                Ok(ModelResponseStreamEvent::PartDelta(PartDeltaEvent::new(
                    0,
                    ModelResponsePartDelta::Text(TextPartDelta {
                        content_delta: " world".to_string(),
                        provider_details: None,
                    }),
                ))),
                Ok(ModelResponseStreamEvent::PartEnd(PartEndEvent::new(0))),
                Ok(ModelResponseStreamEvent::StreamComplete(
                    StreamCompleteEvent {
                        finish_reason: FinishReason::Stop,
                        input_tokens: Some(3),
                        output_tokens: Some(2),
                        cache_creation_tokens: None,
                        cache_read_tokens: None,
                    },
                )),
            ]))
        });
        let engine = ResponsesEngine::new(Arc::new(model));
        let turn = engine.prepare(&request("m", "hi"), None).await.unwrap();
        let mut events = Vec::new();
        let output = engine
            .execute_streaming(turn, &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            })
            .await
            .unwrap();

        let kinds: Vec<&'static str> = events.iter().map(|e| e.kind()).collect();
        assert_eq!(
            kinds,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        let sequences: Vec<u64> = events.iter().map(|e| e.sequence_number()).collect();
        let expected: Vec<u64> = (0..events.len() as u64).collect();
        assert_eq!(sequences, expected);
        assert_eq!(output.response.status, ResponseStatus::Completed);
        match &output.response.output[0] {
            OutputItem::Message { content, .. } => match &content[0] {
                OutputContent::OutputText { text, .. } => assert_eq!(text, "hello world"),
                other => panic!("unexpected content: {other:?}"),
            },
            other => panic!("unexpected item: {other:?}"),
        }
        let usage = output.response.usage.as_ref().unwrap();
        assert_eq!(usage.input_tokens, Some(3));
        assert_eq!(usage.output_tokens, Some(2));
        assert_eq!(usage.total_tokens, Some(5));
    }
}

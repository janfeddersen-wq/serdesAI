//! Websocket transport for the Open Responses model.
//!
//! One flat `response.create` frame starts a turn; the server streams wire
//! events back over the same socket. Turns keep per-conversation session
//! state (see [`super::session`]) so chained turns send only new input,
//! and recover from stale continuation ids, dead sockets, and the server's
//! connection lifetime limit — but only before any caller-visible event
//! escapes.

use super::events::{StreamEvent, failure, failure_kind, translate};
use super::session::{ChannelSink, CollectSink, EventSink, MAX_ATTEMPTS, close_socket};
use super::wire::{ResponseObject, WsErrorEnvelope, codes};
use super::{RequestOverlay, ResponsesApiRequest};
use crate::error::ModelError;
use crate::model::{ModelRequestParameters, StreamedResponse};
use crate::openai::responses::OpenAIResponsesModel;
use serde::Serialize;
use serdes_ai_core::messages::{ModelRequest, ModelResponseStreamEvent};
use serdes_ai_core::{FinishReason, ModelResponse, ModelSettings, RequestUsage};
use serdes_ai_streaming::websocket::{WebSocketConfig, WebSocketStream, WsStreamMessage};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// The wire frame that initiates a websocket turn.
///
/// The codex wire form is flat: `type` plus the response parameters at the
/// top level (`{"type":"response.create","model":…,"input":…}`), with no
/// `response` wrapper. The live backend reads `model` from the frame root
/// and reports `None` when it is nested. HTTP and the websocket share one
/// request type; the frame flattens it, and the unified request leaves
/// `stream` unset on this transport.
#[derive(Serialize)]
struct ResponseCreateFrame<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    #[serde(flatten)]
    response: &'a ResponsesApiRequest,
}

/// Outcome of one websocket attempt.
enum AttemptOutcome {
    /// The turn reached a terminal event; carries the final response object.
    Finished(Box<ResponseObject>, FinishReason),
    /// Recoverable before any event escaped; retry with adjusted session.
    Retry(RetryKind),
    /// Terminal failure; carries the error to surface.
    Failed(ModelError),
}

/// Recoverable failure modes.
enum RetryKind {
    /// Stale `previous_response_id`: clear continuation, replay everything.
    StaleContinuation,
    /// Socket is dead or rejected: reconnect and replay.
    Reconnect,
}

/// Run one turn over the websocket transport.
///
/// `sink` receives every model event; events are only emitted once the turn
/// is committed (never across internal retries), so a caller-visible event
/// implies no further replay. Returns the final response object.
async fn run_ws_turn(
    model: &OpenAIResponsesModel,
    messages: &[ModelRequest],
    settings: &ModelSettings,
    params: &ModelRequestParameters,
    sink: &mut dyn EventSink,
) -> Result<ResponseObject, ModelError> {
    let fingerprints: Vec<u64> = messages.iter().map(super::session::fingerprint).collect();
    let conv = model.conversation(messages).await;
    let mut state = conv.lock().await;
    let mut streamed_any = false;
    let mut last_cause: Option<String> = None;

    for _attempt in 0..MAX_ATTEMPTS {
        // Reconnect if needed. A fresh socket means a fresh server-side
        // session, so continuation state from the old socket is void.
        if state.socket.is_none() {
            let mut config = WebSocketConfig::new(model.base_url.clone());
            config.headers = model.request_headers();
            state.socket = Some(
                WebSocketStream::connect(config)
                    .await
                    .map_err(|e| ModelError::Connection(e.to_string()))?,
            );
            state.previous_response_id = None;
            state.sent_fingerprints.clear();
        }

        let (skip, previous) = state.plan(&fingerprints, messages);
        let request = build_request(model, messages, settings, params, skip, previous)?;
        let frame = ResponseCreateFrame {
            kind: "response.create",
            response: &request,
        };
        tracing::debug!(frame = %serde_json::to_string(&frame).unwrap_or_default(), "sending response.create");

        let socket = state.socket.as_mut().expect("socket ensured above");
        let mut outcome = match socket.send_json(&frame).await {
            Ok(()) => None,
            Err(e) => Some(if streamed_any {
                AttemptOutcome::Failed(ModelError::Connection(e.to_string()))
            } else {
                last_cause = Some(e.to_string());
                tracing::warn!(error = %e, "send failed before any event; reconnecting");
                AttemptOutcome::Retry(RetryKind::Reconnect)
            }),
        };
        if outcome.is_none() {
            outcome = Some(read_ws_events(socket, sink, &mut streamed_any).await);
        }

        match outcome.expect("outcome set") {
            AttemptOutcome::Finished(response, _reason) => {
                // With chaining on, the next turn continues this response;
                // with it off, every turn replays its full input.
                if model.chaining {
                    state.previous_response_id = Some(response.id.clone());
                    state.sent_fingerprints = fingerprints.clone();
                }
                state.last_used = std::time::Instant::now();
                return Ok(*response);
            }
            AttemptOutcome::Retry(RetryKind::StaleContinuation) => {
                state.previous_response_id = None;
                state.sent_fingerprints.clear();
                continue;
            }
            AttemptOutcome::Retry(RetryKind::Reconnect) => {
                // Close the retired socket with a handshake so the peer
                // sees a Close frame rather than a reset; a socket that
                // already failed may reject the handshake, which is fine.
                if let Some(mut socket) = state.socket.take() {
                    close_socket(&mut socket).await;
                }
                continue;
            }
            AttemptOutcome::Failed(error) => {
                // A failed read or cancelled sink may leave response frames
                // unread. They must not become the next turn's response.
                if let Some(mut socket) = state.take_socket() {
                    close_socket(&mut socket).await;
                }
                return Err(error);
            }
        }
    }

    Err(ModelError::Connection(match last_cause {
        Some(cause) => {
            format!("websocket turn exhausted retries; last cause: {cause}")
        }
        None => "websocket turn exhausted retries".to_string(),
    }))
}

/// Read frames for one attempt, translating events into the sink until the
/// turn reaches a terminal event, a recoverable error, or a failure. The
/// caller owns the session and applies the retry adjustment.
async fn read_ws_events(
    socket: &mut WebSocketStream,
    sink: &mut dyn EventSink,
    streamed_any: &mut bool,
) -> AttemptOutcome {
    loop {
        let message = match socket.next_message().await {
            Some(Ok(message)) => message,
            Some(Err(e)) => {
                if !*streamed_any {
                    tracing::warn!(error = %e, "socket error before any event; reconnecting");
                }
                return if *streamed_any {
                    AttemptOutcome::Failed(ModelError::Connection(e.to_string()))
                } else {
                    AttemptOutcome::Retry(RetryKind::Reconnect)
                };
            }
            None => {
                if !*streamed_any {
                    tracing::warn!("socket closed by peer before any event; reconnecting");
                }
                return if *streamed_any {
                    AttemptOutcome::Failed(ModelError::Connection(
                        "connection closed mid-turn".to_string(),
                    ))
                } else {
                    AttemptOutcome::Retry(RetryKind::Reconnect)
                };
            }
        };
        let text = match message {
            WsStreamMessage::Text(text) => text,
            WsStreamMessage::Close => {
                if !*streamed_any {
                    tracing::warn!("close frame before any event; reconnecting");
                }
                return if *streamed_any {
                    AttemptOutcome::Failed(ModelError::Connection(
                        "connection closed mid-turn".to_string(),
                    ))
                } else {
                    AttemptOutcome::Retry(RetryKind::Reconnect)
                };
            }
            WsStreamMessage::Ping | WsStreamMessage::Pong | WsStreamMessage::Binary(_) => continue,
        };

        let event = match super::events::decode(&text) {
            Ok(super::events::DataEvent::Event(event)) => *event,
            Ok(super::events::DataEvent::Error(envelope)) => {
                let code = envelope.error.code.as_str();
                if code == codes::PREVIOUS_RESPONSE_NOT_FOUND && !*streamed_any {
                    return AttemptOutcome::Retry(RetryKind::StaleContinuation);
                }
                if code == codes::WEBSOCKET_CONNECTION_LIMIT_REACHED && !*streamed_any {
                    return AttemptOutcome::Retry(RetryKind::Reconnect);
                }
                return AttemptOutcome::Failed(envelope_error(&envelope));
            }
            Err(error) => return AttemptOutcome::Failed(error),
        };

        // Capture the terminal response object before translation consumes
        // the event; the terminal model event must be the last one sent.
        let mut terminal: Option<(ResponseObject, FinishReason)> = None;
        match &event {
            StreamEvent::ResponseCompleted { response, .. } => {
                terminal = Some((response.clone(), FinishReason::Stop));
            }
            StreamEvent::ResponseIncomplete { response, .. } => {
                terminal = Some((response.clone(), FinishReason::Length));
            }
            StreamEvent::ResponseFailed { response, .. } => {
                return AttemptOutcome::Failed(failure(response));
            }
            _ => {}
        }

        for translated in translate(event) {
            match translated {
                Ok(event) => {
                    if sink.send(event).await.is_err() {
                        return AttemptOutcome::Failed(ModelError::Cancelled);
                    }
                    *streamed_any = true;
                }
                Err(error) => return AttemptOutcome::Failed(error),
            }
        }

        if let Some((response, reason)) = terminal {
            return AttemptOutcome::Finished(Box::new(response), reason);
        }
    }
}

/// Build the request body for a websocket turn.
///
/// The mapping is shared with the HTTP path (see
/// `OpenAIResponsesModel::compose_request`); the websocket pins its
/// transport fields: delta-only input past the session's skip point,
/// `store: false`, `stream` omitted entirely, no routing
/// fields the transport has never sent.
pub(super) fn build_request(
    model: &OpenAIResponsesModel,
    messages: &[ModelRequest],
    settings: &ModelSettings,
    params: &ModelRequestParameters,
    skip: usize,
    previous_response_id: Option<String>,
) -> Result<ResponsesApiRequest, ModelError> {
    model.compose_request(
        messages,
        settings,
        params,
        RequestOverlay {
            skip,
            stream: None,
            store: Some(false),
            previous_response_id,
            service_tier: None,
            truncation: None,
        },
    )
}

/// Map an error envelope onto a model error.
fn envelope_error(envelope: &WsErrorEnvelope) -> ModelError {
    ModelError::provider(
        "openai",
        envelope.error.code.clone(),
        envelope.error.message.clone(),
        failure_kind(&envelope.error.code),
        None,
    )
}

/// Run one turn over the websocket transport and fold the collected events
/// into a complete response.
pub(crate) async fn request(
    model: &OpenAIResponsesModel,
    messages: &[ModelRequest],
    settings: &ModelSettings,
    params: &ModelRequestParameters,
) -> Result<ModelResponse, ModelError> {
    let mut sink = CollectSink(Vec::new());
    let response = run_ws_turn(model, messages, settings, params, &mut sink).await?;
    Ok(response_from_events(
        sink.0,
        &model.model_name,
        &response.id,
    ))
}

/// Stream one turn over the websocket transport through a channel.
pub(crate) fn stream(
    model: &OpenAIResponsesModel,
    messages: &[ModelRequest],
    settings: &ModelSettings,
    params: &ModelRequestParameters,
) -> Result<StreamedResponse, ModelError> {
    let (tx, rx) = mpsc::channel::<Result<ModelResponseStreamEvent, ModelError>>(64);
    let model = model.clone();
    let messages = messages.to_vec();
    let settings = settings.clone();
    let params = params.clone();

    tokio::spawn(async move {
        let result = {
            let mut sink = ChannelSink(&tx);
            run_ws_turn(&model, &messages, &settings, &params, &mut sink)
                .await
                .map(|_| ())
        };
        if let Err(error) = result {
            // A failure after events escaped still reaches the caller as
            // an error item; a failure before that is the only item.
            let _ = tx.send(Err(error)).await;
        }
    });

    Ok(Box::pin(ReceiverStream::new(rx)))
}

/// Fold collected stream events into a complete response.
fn response_from_events(
    events: Vec<ModelResponseStreamEvent>,
    model_name: &str,
    response_id: &str,
) -> ModelResponse {
    use serdes_ai_core::messages::ModelResponsePartDelta;

    let mut parts: Vec<serdes_ai_core::messages::ModelResponsePart> = Vec::new();
    let mut finish_reason = None;
    let mut usage = None;

    for event in events {
        match event {
            ModelResponseStreamEvent::PartStart(start) => {
                if start.index < parts.len() {
                    parts[start.index] = start.part;
                } else {
                    parts.push(start.part);
                }
            }
            ModelResponseStreamEvent::PartDelta(delta) => {
                if let Some(part) = parts.get_mut(delta.index) {
                    match delta.delta {
                        ModelResponsePartDelta::Text(_)
                        | ModelResponsePartDelta::ToolCall(_)
                        | ModelResponsePartDelta::Thinking(_)
                        | ModelResponsePartDelta::BuiltinToolCall(_) => {
                            let _ = delta.delta.apply(part);
                        }
                    }
                }
            }
            ModelResponseStreamEvent::PartEnd(_) => {}
            ModelResponseStreamEvent::StreamComplete(complete) => {
                finish_reason = Some(complete.finish_reason);
                usage = Some(RequestUsage {
                    request_tokens: complete.input_tokens,
                    response_tokens: complete.output_tokens,
                    total_tokens: match (complete.input_tokens, complete.output_tokens) {
                        (Some(input), Some(output)) => Some(input + output),
                        (input, output) => input.or(output),
                    },
                    cache_creation_tokens: complete.cache_creation_tokens,
                    cache_read_tokens: complete.cache_read_tokens,
                    details: None,
                });
            }
        }
    }

    ModelResponse {
        parts,
        model_name: Some(model_name.to_string()),
        timestamp: chrono::Utc::now(),
        finish_reason,
        usage,
        vendor_id: Some(response_id.to_string()),
        vendor_details: None,
        kind: "response".to_string(),
    }
}

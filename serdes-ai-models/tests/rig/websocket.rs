//! Open Responses websocket transport.
//!
//! `GET /v1/responses` upgrades to a websocket. The client sends
//! `{"type":"response.create", …flat response parameters…}` text frames; the server
//! answers with the same event objects the SSE transport emits, one JSON
//! object per text frame, with `sequence_number` restarting at 0 each turn.
//!
//! Semantics (Open Responses websocket profile):
//!
//! - Turns are sequential: the server processes one `response.create` at a
//!   time.
//! - `stream`, `stream_options`, and `background` must be omitted; events
//!   always stream over the socket and background mode is unsupported.
//! - `store: false` turns keep their continuation state in a connection-local
//!   cache, so nothing is persisted globally while still allowing
//!   `previous_response_id` chaining on the same connection (the codex CLI
//!   default profile).
//! - When a continuation references an evicted or unknown response ID, the
//!   server replies with a `previous_response_not_found` envelope; the
//!   referenced ID is evicted from the session cache so the client is pushed
//!   to replay the full input, matching how the codex CLI recovers.
//! - The connection has a lifetime limit (default 60 minutes, enforced
//!   between turns only); exceeding it closes the socket with a
//!   `websocket_connection_limit_reached` envelope, which clients treat as a
//!   reconnect signal.
//! - Errors are wrapped as `{"type":"error","status_code":N,"error":{...}}`.
//! - The end of a turn is signaled by its terminal event
//!   (`response.completed`, `response.incomplete`, or `response.failed`),
//!   not by a channel close; turns are sequential, so the client always knows
//!   which turn an event belongs to.

use super::engine::ResponsesEngine;
use super::error::{FromResponsesError, ResponsesError, WsErrorEnvelope, codes};
use super::store::SessionResponseCache;
use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use serdes_ai_models::openai::responses::events::StreamEvent;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Configuration for websocket sessions.
#[derive(Debug, Clone)]
pub struct WebSocketSessionConfig {
    /// Maximum connection lifetime before the server refuses new turns.
    ///
    /// Enforced only between turns, never mid-turn.
    pub connection_ttl: Duration,
}

impl Default for WebSocketSessionConfig {
    fn default() -> Self {
        Self {
            connection_ttl: Duration::from_secs(60 * 60),
        }
    }
}

/// Keys carried by codex websocket turns that only apply to HTTP requests.
/// They are stripped before the payload is parsed.
const FORBIDDEN_KEYS: [&str; 3] = ["stream", "stream_options", "background"];

/// Outgoing frame sender shared by turn execution and control paths.
type FrameSender = mpsc::Sender<Message>;

/// Serve one websocket connection until the client closes or the lifetime
/// limit is reached.
pub async fn handle_socket(
    socket: WebSocket,
    engine: Arc<ResponsesEngine>,
    config: WebSocketSessionConfig,
) {
    // Bounded queues let socket backpressure reach the model stream. Reading
    // stays active during a turn so a Close cancels even a stalled model.
    let (mut ws_tx, mut ws_rx) = socket.split();
    let (input_tx, mut input_rx) = mpsc::channel(16);
    let (disconnected_tx, mut disconnected_rx) = tokio::sync::oneshot::channel::<()>();
    let reader = tokio::spawn(async move {
        while let Some(frame) = ws_rx.next().await {
            match frame {
                Ok(Message::Close(_)) | Err(_) => break,
                Ok(frame) => {
                    // Reject overflow rather than parking the only reader that
                    // can cancel a stalled turn on disconnect.
                    if input_tx.try_send(frame).is_err() {
                        break;
                    }
                }
            }
        }
        let _ = disconnected_tx.send(());
    });
    let (msg_tx, mut msg_rx) = mpsc::channel::<Message>(64);
    let forwarder = tokio::spawn(async move {
        while let Some(message) = msg_rx.recv().await {
            if ws_tx.send(message).await.is_err() {
                break;
            }
        }
        // Send a close frame for a clean shutdown when the client has not
        // already closed.
        let _ = ws_tx.close().await;
    });

    let session = SessionResponseCache::default();
    let connected_at = Instant::now();
    // The lifetime limit is enforced between turns only: a connection always
    // gets to serve its first turn, matching the Open Responses reconnect
    // contract (a fresh connection is a fresh allowance).
    let mut turns_served = false;

    loop {
        let receive = async {
            // A zero TTL is the rig's one-turn-per-connection mode.
            if config.connection_ttl.is_zero() && !turns_served {
                Ok(input_rx.recv().await)
            } else {
                tokio::time::timeout(
                    config.connection_ttl.saturating_sub(connected_at.elapsed()),
                    input_rx.recv(),
                )
                .await
            }
        };
        let frame = match receive.await {
            Ok(Some(frame)) => frame,
            Ok(None) => break,
            Err(_) => {
                let envelope = WsErrorEnvelope::from_error(&ResponsesError::ConnectionLimitReached);
                let _ = msg_tx.send(Message::Text(envelope.to_json().into())).await;
                let _ = msg_tx.send(Message::Close(None)).await;
                break;
            }
        };
        match frame {
            Message::Text(text) => {
                let mut sender = msg_tx.clone();
                let rejection = tokio::select! {
                    biased;
                    _ = &mut disconnected_rx => break,
                    _ = msg_tx.closed() => break,
                    result = run_turn(
                        engine.as_ref(), &session, connected_at, turns_served,
                        &config, text.as_str(), &mut sender,
                    ) => result,
                };
                if let Some(error) = rejection {
                    // A rejected turn emitted no events, so the error
                    // envelope is the only signal the client gets. Evict the
                    // referenced continuation ID so a retried chain with the
                    // same ID fails fast and the client replays full input.
                    if let Some(id) = error.previous_response_id() {
                        session.evict(&id);
                    }
                    let envelope = WsErrorEnvelope::from_error(&error);
                    let _ = sender
                        .send(Message::Text(envelope.to_json().to_string().into()))
                        .await;
                    if envelope.error.code == codes::WEBSOCKET_CONNECTION_LIMIT_REACHED {
                        let _ = sender.send(Message::Close(None)).await;
                        break;
                    }
                } else {
                    turns_served = true;
                }
            }
            Message::Ping(payload) => {
                let _ = msg_tx.send(Message::Pong(payload)).await;
            }
            Message::Pong(_) => {}
            Message::Close(_) => break,
            Message::Binary(_) => {
                let envelope = WsErrorEnvelope::from_error(&ResponsesError::InvalidRequest(
                    "binary frames are not supported; send response.create as a text frame"
                        .to_string(),
                ));
                let _ = msg_tx
                    .send(Message::Text(envelope.to_json().to_string().into()))
                    .await;
            }
        }
    }
    reader.abort();
    let _ = reader.await;
    drop(msg_tx);
    let _ = forwarder.await;
}

/// Validate and run a single `response.create` frame.
///
/// Returns `Some(error)` when the turn was rejected before any event was
/// emitted (validation, unknown continuation, expired connection). Errors
/// during execution are streamed as a `response.failed` event, and the turn
/// still resolves as `None` because the connection remains usable.
async fn run_turn(
    engine: &ResponsesEngine,
    session: &SessionResponseCache,
    connected_at: Instant,
    turns_served: bool,
    config: &WebSocketSessionConfig,
    text: &str,
    sender: &mut FrameSender,
) -> Option<ResponsesError> {
    let frame: Value = match serde_json::from_str(text) {
        Ok(frame) => frame,
        Err(err) => {
            return Some(ResponsesError::InvalidRequest(format!(
                "frame is not valid JSON: {err}"
            )));
        }
    };
    let kind = match frame.get("type").and_then(Value::as_str) {
        Some(kind) => kind,
        None => {
            return Some(ResponsesError::InvalidRequest(
                "frame must carry a \"type\" field".to_string(),
            ));
        }
    };
    if kind != "response.create" {
        return Some(ResponsesError::InvalidRequest(format!(
            "unsupported frame type '{kind}'; only response.create is accepted"
        )));
    }
    // Codex sends `response.create` frames with the parameters flat on the
    // frame root (no `response` wrapper); HTTP-only keys arrive alongside
    // them and are ignored rather than rejected.
    let mut response_object = match frame {
        Value::Object(object) => object,
        _ => {
            return Some(ResponsesError::InvalidRequest(
                "frame must be a JSON object".to_string(),
            ));
        }
    };
    response_object.remove("type");
    for key in FORBIDDEN_KEYS {
        response_object.remove(key);
    }

    // The lifetime limit is enforced between turns only, so an in-flight turn
    // always completes.
    if turns_served && connected_at.elapsed() >= config.connection_ttl {
        return Some(ResponsesError::ConnectionLimitReached);
    }

    let request: serdes_ai_models::openai::responses::wire::CreateResponseRequest =
        match serde_json::from_value(Value::Object(response_object)) {
            Ok(request) => request,
            Err(err) => {
                return Some(ResponsesError::InvalidRequest(format!(
                    "invalid response payload: {err}"
                )));
            }
        };

    let turn = match engine.prepare(&request, Some(session)).await {
        Ok(turn) => turn,
        Err(error) => return Some(error),
    };

    let mut streamed_anything = false;
    let mut sink = |event: StreamEvent| {
        streamed_anything = true;
        let sender = &*sender;
        async move {
            let payload = serde_json::to_string(&event).expect("wire event serializes");
            sender
                .send(Message::Text(payload.into()))
                .await
                .map_err(|_| ResponsesError::Model("socket receiver closed".into()))
        }
    };

    match engine.execute_streaming(turn, &mut sink).await {
        Ok(output) => {
            engine.persist(&request, Some(session), &output).await;
            None
        }
        Err(error) => {
            // A mid-stream failure already produced response.failed; only
            // acquisition failures (nothing streamed) need an envelope.
            if streamed_anything {
                tracing::warn!("websocket turn failed mid-stream: {error}");
                None
            } else {
                Some(error)
            }
        }
    }
}

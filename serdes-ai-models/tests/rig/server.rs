//! Wire-accurate axum server used as a **test rig** for the client in this
//! crate's integration tests. Not a product surface.
//!
//! Routes:
//!
//! - `POST /v1/responses` — create a response (JSON, or SSE when
//!   `stream: true`)
//! - `GET /v1/responses` — websocket upgrade; the Open Responses websocket
//!   transport with connection-local session state
//! - `GET /health` — liveness probe

use super::engine::{PreparedTurn, ResponsesEngine};
use super::error::{HttpErrorEnvelope, ResponsesError};
use super::websocket::{self, WebSocketSessionConfig};
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{State, ws::WebSocketUpgrade},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures::StreamExt;
use serdes_ai_models::openai::responses::events::StreamEvent;
use serdes_ai_models::openai::responses::wire::{
    CreateResponseRequest, OutputItem, OutputItemStatus,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

/// Server error types.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    /// Failed to bind the listen address.
    #[error("failed to bind to address: {0}")]
    Bind(String),
    /// The server stopped with an error.
    #[error("server error: {0}")]
    Serve(String),
}

/// Shared state for the HTTP handlers.
pub struct ResponsesServerState {
    /// Turn engine backing every route.
    pub engine: Arc<ResponsesEngine>,
    /// Websocket session configuration.
    pub websocket_config: WebSocketSessionConfig,
}

/// The Responses API server.
pub struct ResponsesServer {
    state: Arc<ResponsesServerState>,
}

impl ResponsesServer {
    /// Create a server for the given engine.
    #[must_use]
    pub fn new(engine: ResponsesEngine) -> Self {
        Self::from_engine(Arc::new(engine))
    }

    /// Create a server for a shared engine.
    #[must_use]
    pub fn from_engine(engine: Arc<ResponsesEngine>) -> Self {
        Self {
            state: Arc::new(ResponsesServerState {
                engine,
                websocket_config: WebSocketSessionConfig::default(),
            }),
        }
    }

    /// Override the websocket session configuration.
    #[must_use]
    pub fn with_websocket_config(self, config: WebSocketSessionConfig) -> Self {
        Self {
            state: Arc::new(ResponsesServerState {
                engine: self.state.engine.clone(),
                websocket_config: config,
            }),
        }
    }

    /// Build the axum router.
    pub fn router(&self) -> Router {
        Router::new()
            .route(
                "/v1/responses",
                get(upgrade_websocket).post(create_response),
            )
            .route("/health", get(health))
            .with_state(self.state.clone())
    }

    /// Start serving on the given address; blocks until shutdown.
    ///
    /// # Errors
    ///
    /// Returns an error if the address cannot be bound or the server fails.
    pub async fn serve(self, addr: impl Into<SocketAddr>) -> Result<(), ServerError> {
        let addr = addr.into();
        let router = self.router();
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| ServerError::Bind(e.to_string()))?;
        axum::serve(listener, router)
            .await
            .map_err(|e| ServerError::Serve(e.to_string()))?;
        Ok(())
    }
}

/// Map engine errors onto OpenAI-style HTTP error envelopes.
fn error_response(error: ResponsesError) -> Response {
    let status = StatusCode::from_u16(error.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let body = HttpErrorEnvelope {
        error: error.body(),
    };
    (status, Json(body)).into_response()
}

/// POST /v1/responses.
async fn create_response(State(state): State<Arc<ResponsesServerState>>, body: Bytes) -> Response {
    let request: CreateResponseRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(err) => {
            return error_response(ResponsesError::InvalidRequest(format!(
                "failed to parse request body: {err}"
            )));
        }
    };

    let turn = match state.engine.prepare(&request, None).await {
        Ok(turn) => turn,
        Err(err) => return error_response(err),
    };

    if request.stream == Some(true) {
        stream_response(state, turn, request).await
    } else {
        match state.engine.execute(turn).await {
            Ok(output) => {
                state.engine.persist(&request, None, &output).await;
                Json(output.response).into_response()
            }
            Err(err) => error_response(err),
        }
    }
}

/// SSE variant of `POST /v1/responses`.
///
/// Events are written as `data: {json}\n\n` frames, terminated by the
/// `data: [DONE]\n\n` sentinel. The HTTP status cannot change once the body
/// starts, so failures after the first event surface as `response.failed`
/// events followed by `[DONE]`.
async fn stream_response(
    state: Arc<ResponsesServerState>,
    turn: PreparedTurn,
    request: CreateResponseRequest,
) -> Response {
    let (event_tx, event_rx) = mpsc::channel::<Result<String, std::io::Error>>(64);
    let engine = state.engine.clone();

    tokio::spawn(async move {
        let mut sink = |event: StreamEvent| {
            let sender = &event_tx;
            async move {
                let payload = serde_json::to_string(&event).expect("wire event serializes");
                sender
                    .send(Ok(payload))
                    .await
                    .map_err(|_| ResponsesError::Model("stream receiver closed".into()))
            }
        };
        let result = tokio::select! {
            biased;
            _ = event_tx.closed() => return,
            result = engine.execute_streaming(turn, &mut sink) => result,
        };
        match result {
            Ok(output) => engine.persist(&request, None, &output).await,
            Err(err) => tracing::warn!("streaming turn failed: {err}"),
        }
        let _ = event_tx.send(Ok("[DONE]".to_string())).await;
    });

    let body_stream = tokio_stream::wrappers::ReceiverStream::new(event_rx).map(|item| {
        item.map(|payload| {
            let mut frame = String::with_capacity(payload.len() + 8);
            frame.push_str("data: ");
            frame.push_str(&payload);
            frame.push_str("\n\n");
            frame
        })
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body_stream))
        .unwrap_or_else(|err| {
            error_response(ResponsesError::Model(format!(
                "failed to build stream response: {err}"
            )))
        })
}

/// GET /v1/responses — websocket upgrade for the Open Responses transport.
async fn upgrade_websocket(
    State(state): State<Arc<ResponsesServerState>>,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| {
        websocket::handle_socket(socket, state.engine.clone(), state.websocket_config.clone())
    })
    .into_response()
}

/// GET /health.
async fn health() -> &'static str {
    "ok"
}

/// A test-only SSE turn that streams a delta and the `data: [DONE]`
/// sentinel but never a terminal `response.completed`, `response.failed`,
/// or `response.incomplete` event. Contract tests point the client here to
/// pin that the sentinel without a terminal event surfaces as an error
/// instead of ending the stream silently.
pub fn malformed_sse_router() -> Router {
    Router::new().route("/v1/responses", post(malformed_sse_turn))
}

async fn malformed_sse_turn() -> Response {
    let events = [
        StreamEvent::OutputItemAdded {
            sequence_number: 0,
            output_index: 0,
            item: OutputItem::Message {
                id: "msg_malformed".to_string(),
                role: "assistant".to_string(),
                status: OutputItemStatus::InProgress,
                content: Vec::new(),
            },
        },
        StreamEvent::OutputTextDelta {
            sequence_number: 1,
            item_id: "msg_malformed".to_string(),
            output_index: 0,
            content_index: 0,
            delta: "ok".to_string(),
        },
    ];
    let mut body = String::new();
    for event in events {
        body.push_str("data: ");
        body.push_str(&serde_json::to_string(&event).expect("event serializes"));
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from(body))
        .expect("static response body")
}

//! Wire-accurate Open Responses test rig, adopted as serdes-ai-models test
//! support.
//!
//! The rig is a fake but protocol-faithful Open Responses server: an axum
//! endpoint (`POST /v1/responses` for JSON and SSE, `GET /v1/responses` for
//! the Open Responses websocket transport, plus `/health`) backed by an
//! in-memory engine that bridges wire turns onto any serdesAI
//! [`Model`](serdes_ai_models::Model). Integration tests point reqwest or
//! tokio-tungstenite at a spawned rig so the Responses client is validated
//! against real protocol shapes instead of hand-built fakes.
//!
//! The rig is test support, not a product surface, and lives here so the
//! responses client's integration tests can run against real protocol
//! shapes. The wire and event model it serves is the crate's own
//! `openai::responses::{wire, events}`.
//!
//! Declare `mod rig;` from a test binary in `tests/` to use it:
//!
//! ```ignore
//! mod rig;
//!
//! use rig::{recording_model, spawn_server};
//!
//! #[tokio::test]
//! async fn basic_turn() {
//!     let (model, calls) = recording_model();
//!     let addr = spawn_server(model).await;
//!     // POST http://{addr}/v1/responses ...
//! }
//! ```

// Helpers are shared across test binaries; not every binary uses every one.
#![allow(dead_code)]

pub mod convert;
pub mod engine;
pub mod error;
pub mod server;
pub mod store;
pub mod websocket;

use crate::rig::engine::ResponsesEngine;
use crate::rig::server::{ResponsesServer, malformed_sse_router};
use crate::rig::websocket::WebSocketSessionConfig;
use serdes_ai_models::mock::FunctionModel;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

/// A function model that records how many model requests each turn saw.
///
/// The non-streaming and streaming paths share the record, so tests can
/// assert on history lengths regardless of transport.
pub fn recording_model() -> (FunctionModel, Arc<std::sync::Mutex<Vec<usize>>>) {
    let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
    let calls_stream = calls.clone();
    let calls_out = calls.clone();
    let model = FunctionModel::with_both(
        move |requests, _| {
            calls.lock().unwrap().push(requests.len());
            serdes_ai_core::ModelResponse::text("ok")
        },
        move |requests, _| {
            calls_stream.lock().unwrap().push(requests.len());
            let part = serdes_ai_core::messages::TextPart::new("ok");
            Box::pin(futures::stream::iter(vec![
                Ok(serdes_ai_core::ModelResponseStreamEvent::part_start(
                    0,
                    serdes_ai_core::ModelResponsePart::Text(part),
                )),
                Ok(serdes_ai_core::ModelResponseStreamEvent::StreamComplete(
                    serdes_ai_core::messages::StreamCompleteEvent {
                        finish_reason: serdes_ai_core::FinishReason::Stop,
                        input_tokens: Some(1),
                        output_tokens: Some(1),
                        cache_creation_tokens: None,
                        cache_read_tokens: None,
                    },
                )),
            ]))
        },
    );
    (model, calls_out)
}

/// Start the responses server on an ephemeral port and return its address.
pub async fn spawn_server(model: FunctionModel) -> SocketAddr {
    spawn_server_with_ws_config(model, WebSocketSessionConfig::default()).await
}

/// Start the responses server with a custom websocket session config.
pub async fn spawn_server_with_ws_config(
    model: FunctionModel,
    websocket_config: WebSocketSessionConfig,
) -> SocketAddr {
    let engine = ResponsesEngine::new(Arc::new(model));
    let server = ResponsesServer::new(engine).with_websocket_config(websocket_config);
    let router = server.router();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    addr
}

/// Start the malformed-SSE contract server on an ephemeral port: a turn
/// whose SSE body streams a delta and the `data: [DONE]` sentinel but
/// never a terminal event.
pub async fn spawn_malformed_sse_server() -> SocketAddr {
    let router = malformed_sse_router();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    addr
}

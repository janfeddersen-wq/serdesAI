//! Raw-protocol tests against the wire-accurate Open Responses rig adopted
//! under `tests/rig`: JSON and SSE turns over HTTP plus websocket turns
//! (sequential turns, session-local `store: false` state, continuation
//! errors, forbidden keys, ping/pong, connection lifetime).

mod rig;

use futures::{SinkExt, StreamExt};
use rig::{recording_model, spawn_server, spawn_server_with_ws_config};
use serde_json::{Value, json};
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;

fn client() -> reqwest::Client {
    reqwest::Client::new()
}

#[tokio::test]
async fn json_turn_returns_completed_response() {
    let (model, calls) = recording_model();
    let addr = spawn_server(model).await;

    let response = client()
        .post(format!("http://{addr}/v1/responses"))
        .json(&json!({
            "model": "my-model",
            "instructions": "be brief",
            "input": "hello",
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.unwrap();
    assert!(body["id"].as_str().unwrap().starts_with("resp_"));
    assert_eq!(body["status"], "completed");
    assert_eq!(body["model"], "my-model");
    assert_eq!(body["output"][0]["content"][0]["text"], "ok");
    // instructions + user prompt form the history
    assert_eq!(*calls.lock().unwrap(), vec![2]);
}

// Websocket integration tests for the Open Responses transport: sequential
// turns, connection-local state for `store: false`, continuation recovery,
// forbidden keys, ping/pong, and the connection lifetime limit.

type Ws =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

async fn connect(addr: &std::net::SocketAddr) -> Ws {
    let url = format!("ws://{addr}/v1/responses");
    tokio_tungstenite::connect_async(url)
        .await
        .expect("websocket upgrade on GET /v1/responses")
        .0
}

/// Send a response.create frame and collect events until the turn's terminal
/// event (or an error envelope) arrives.
async fn run_turn(ws: &mut Ws, response: Value) -> Vec<Value> {
    // Codex sends response.create frames flat: parameters on the frame root.
    let mut frame = response;
    frame["type"] = json!("response.create");
    ws.send(Message::Text(frame.to_string().into()))
        .await
        .unwrap();

    let mut events = Vec::new();
    loop {
        let message = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("timed out waiting for event")
            .expect("socket closed mid-turn")
            .expect("socket error mid-turn");
        let Message::Text(text) = message else {
            continue;
        };
        let event: Value = serde_json::from_str(&text).unwrap();
        let done = matches!(
            event["type"].as_str(),
            Some("response.completed" | "response.incomplete" | "response.failed" | "error")
        );
        events.push(event);
        if done {
            return events;
        }
    }
}

fn find<'a>(events: &'a [Value], kind: &str) -> &'a Value {
    events
        .iter()
        .find(|event| event["type"] == kind)
        .unwrap_or_else(|| panic!("no {kind} event in {events:?}"))
}

#[tokio::test]
async fn turns_stream_events_with_terminal_completion() {
    let (model, _calls) = recording_model();
    let addr = spawn_server_with_ws_config(model, Default::default()).await;
    let mut ws = connect(&addr).await;

    let events = run_turn(
        &mut ws,
        json!({"model": "m", "instructions": "sys", "input": "hi"}),
    )
    .await;

    assert_eq!(find(&events, "response.created")["sequence_number"], 0);
    let completed = find(&events, "response.completed");
    assert_eq!(completed["response"]["status"], "completed");
    assert!(
        completed["response"]["id"]
            .as_str()
            .unwrap()
            .starts_with("resp_")
    );
    // wire names use the unprefixed forms for item-level events
    assert!(
        events
            .iter()
            .any(|event| event["type"] == "response.output_item.added")
    );
    assert!(
        events
            .iter()
            .any(|event| event["type"] == "response.output_text.delta")
    );

    // A second turn on the same socket works (sequential turns).
    let events = run_turn(&mut ws, json!({"model": "m", "input": "again"})).await;
    assert!(
        events
            .iter()
            .any(|event| event["type"] == "response.completed")
    );
}

#[tokio::test]
async fn store_false_state_lives_in_the_session() {
    let (model, calls) = recording_model();
    let addr = spawn_server_with_ws_config(model, Default::default()).await;
    let mut ws = connect(&addr).await;

    // codex profile: store:false chaining on a single connection
    let events = run_turn(
        &mut ws,
        json!({"model": "m", "instructions": "sys", "input": "first", "store": false}),
    )
    .await;
    let first_id = find(&events, "response.completed")["response"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    let events = run_turn(
        &mut ws,
        json!({
            "model": "m",
            "previous_response_id": first_id,
            "input": "second",
            "store": false,
        }),
    )
    .await;
    assert_eq!(
        find(&events, "response.completed")["response"]["status"],
        "completed"
    );

    // Turn 2 saw the full chained history: instructions + first prompt +
    // first model response + second prompt.
    let calls = calls.lock().unwrap();
    assert_eq!(*calls, vec![2, 4]);
}

#[tokio::test]
async fn unknown_continuation_reports_error_and_connection_survives() {
    let (model, _calls) = recording_model();
    let addr = spawn_server_with_ws_config(model, Default::default()).await;
    let mut ws = connect(&addr).await;

    let events = run_turn(
        &mut ws,
        json!({"model": "m", "previous_response_id": "resp_missing", "input": "x"}),
    )
    .await;
    let error = find(&events, "error");
    assert_eq!(error["error"]["code"], "previous_response_not_found");
    assert_eq!(error["status_code"], 404);

    // Full-input replay still works on the same connection.
    let events = run_turn(&mut ws, json!({"model": "m", "input": "replayed"})).await;
    assert!(
        events
            .iter()
            .any(|event| event["type"] == "response.completed")
    );
}

#[tokio::test]
async fn stream_keys_are_ignored() {
    let (model, _calls) = recording_model();
    let addr = spawn_server_with_ws_config(model, Default::default()).await;
    let mut ws = connect(&addr).await;

    // The live backend accepts HTTP-only keys on websocket turns; the rig
    // strips them instead of rejecting the frame.
    let events = run_turn(&mut ws, json!({"model": "m", "input": "x", "stream": true})).await;
    assert!(
        find(&events, "response.completed").is_object(),
        "turn should complete with stream key present"
    );

    // Non response.create frames are rejected too.
    let events = run_turn_raw(&mut ws, json!({"type": "response.cancel"})).await;
    assert_eq!(
        find(&events, "error")["error"]["code"],
        "invalid_request_error"
    );
}

/// Like [`run_turn`] but sends an arbitrary frame.
async fn run_turn_raw(ws: &mut Ws, frame: Value) -> Vec<Value> {
    ws.send(Message::Text(frame.to_string().into()))
        .await
        .unwrap();
    let mut events = Vec::new();
    loop {
        let message = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("timed out waiting for event")
            .expect("socket closed")
            .expect("socket error");
        let Message::Text(text) = message else {
            continue;
        };
        let event: Value = serde_json::from_str(&text).unwrap();
        let done = event["type"] == "error";
        events.push(event);
        if done {
            return events;
        }
    }
}

#[tokio::test]
async fn ping_is_answered_with_pong() {
    let (model, _calls) = recording_model();
    let addr = spawn_server_with_ws_config(model, Default::default()).await;
    let mut ws = connect(&addr).await;

    ws.send(Message::Ping(vec![1, 2, 3].into())).await.unwrap();
    loop {
        let message = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("timed out waiting for pong")
            .expect("socket closed")
            .expect("socket error");
        match message {
            Message::Pong(payload) => {
                assert_eq!(payload, vec![1, 2, 3]);
                return;
            }
            Message::Text(_) => continue,
            other => panic!("unexpected frame: {other:?}"),
        }
    }
}

#[tokio::test]
async fn connection_lifetime_limit_closes_the_socket() {
    let (model, _calls) = recording_model();
    let config = rig::websocket::WebSocketSessionConfig {
        connection_ttl: Duration::from_millis(50),
    };
    let addr = spawn_server_with_ws_config(model, config).await;
    let mut ws = connect(&addr).await;

    // Before the TTL the turn works.
    let events = run_turn(&mut ws, json!({"model": "m", "input": "early"})).await;
    assert!(
        events
            .iter()
            .any(|event| event["type"] == "response.completed")
    );

    tokio::time::sleep(Duration::from_millis(60)).await;

    // After the TTL the turn is refused, the envelope carries the codex
    // retryable code, and the server closes the socket.
    ws.send(Message::Text(
        json!({"type": "response.create", "response": {"model": "m", "input": "late"}})
            .to_string()
            .into(),
    ))
    .await
    .unwrap();

    let mut saw_limit_error = false;
    loop {
        let message = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("timed out waiting for close");
        let Some(message) = message else {
            assert!(saw_limit_error, "socket closed without the error envelope");
            return;
        };
        let message = message.expect("socket error");
        match message {
            Message::Text(text) => {
                let event: Value = serde_json::from_str(&text).unwrap();
                if event["type"] == "error" {
                    assert_eq!(event["error"]["code"], "websocket_connection_limit_reached");
                    assert_eq!(event["status_code"], 429);
                    saw_limit_error = true;
                }
            }
            Message::Close(_) => {
                assert!(saw_limit_error);
                return;
            }
            _ => {}
        }
    }
}

// HTTP integration tests for the test rig: SSE streaming, stateful
// chaining, and error envelopes. (The basic JSON turn lives at the top of
// this file.)

#[tokio::test]
async fn stateful_chaining_resolves_previous_response_id() {
    let (model, calls) = recording_model();
    let addr = spawn_server(model).await;
    let url = format!("http://{addr}/v1/responses");

    let first: Value = client()
        .post(&url)
        .json(&json!({"model": "m", "instructions": "sys", "input": "first"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let first_id = first["id"].as_str().unwrap().to_string();

    let second: Value = client()
        .post(&url)
        .json(&json!({
            "model": "m",
            "previous_response_id": first_id,
            "instructions": "new sys",
            "input": "second",
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(second["status"], "completed");

    // Turn 1: instructions + prompt. Turn 2: replaced instructions + first
    // turn history + second prompt.
    let calls = calls.lock().unwrap();
    assert_eq!(*calls, vec![2, 4]);
}

#[tokio::test]
async fn sse_stream_frames_and_done_sentinel() {
    let (model, _calls) = recording_model();
    let addr = spawn_server(model).await;

    let response = client()
        .post(format!("http://{addr}/v1/responses"))
        .json(&json!({"model": "m", "input": "hello", "stream": true}))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let body = response.text().await.unwrap();
    assert!(body.starts_with("data: {"));
    assert!(body.contains("\"type\":\"response.created\""));
    assert!(body.contains("\"type\":\"response.completed\""));
    assert!(body.ends_with("data: [DONE]\n\n"));

    // Sequence numbers restart per event stream and are contiguous.
    let sequences: Vec<u64> = body
        .lines()
        .filter(|line| line.starts_with("data: {"))
        .filter_map(|line| serde_json::from_str::<Value>(&line[6..]).ok())
        .filter_map(|event| event["sequence_number"].as_u64())
        .collect();
    let expected: Vec<u64> = (0..sequences.len() as u64).collect();
    assert_eq!(sequences, expected);
}

#[tokio::test]
async fn malformed_body_returns_invalid_request_envelope() {
    let (model, _calls) = recording_model();
    let addr = spawn_server(model).await;

    let response = client()
        .post(format!("http://{addr}/v1/responses"))
        .header("content-type", "application/json")
        .body("{not json")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], "invalid_request_error");
}

#[tokio::test]
async fn unknown_previous_response_id_is_404_with_code() {
    let (model, _calls) = recording_model();
    let addr = spawn_server(model).await;

    let response = client()
        .post(format!("http://{addr}/v1/responses"))
        .json(&json!({"model": "m", "previous_response_id": "resp_missing", "input": "x"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 404);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], "previous_response_not_found");
}

#[tokio::test]
async fn background_mode_is_rejected() {
    let (model, _calls) = recording_model();
    let addr = spawn_server(model).await;

    let response = client()
        .post(format!("http://{addr}/v1/responses"))
        .json(&json!({"model": "m", "input": "x", "background": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], "invalid_request_error");
}

#[tokio::test]
async fn builtin_tools_are_rejected() {
    let (model, _calls) = recording_model();
    let addr = spawn_server(model).await;

    let response = client()
        .post(format!("http://{addr}/v1/responses"))
        .json(&json!({
            "model": "m",
            "input": "x",
            "tools": [{"type": "web_search_preview"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], "invalid_request_error");
}

#[tokio::test]
async fn health_check() {
    let (model, _calls) = recording_model();
    let addr = spawn_server(model).await;

    let response = client()
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.text().await.unwrap(), "ok");
}

#[tokio::test]
async fn unknown_tool_output_is_rejected_without_invoking_model() {
    let (model, calls) = recording_model();
    let addr = spawn_server(model).await;
    let response = client().post(format!("http://{addr}/v1/responses"))
        .json(&json!({"model":"m","input":[{"type":"function_call_output","call_id":"missing","output":"x"}]}))
        .send().await.unwrap();
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unknown call_id 'missing'")
    );
    assert!(calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn replacing_instructions_preserves_input_system_messages_and_order() {
    use rig::engine::ResponsesEngine;
    use serdes_ai_models::mock::FunctionModel;
    let engine = ResponsesEngine::new(std::sync::Arc::new(FunctionModel::constant_text("ok")));
    let request = serde_json::from_value(json!({"model":"m", "instructions":"initial", "input":[
        {"role":"system", "content":"input system"},
        {"role":"user", "content":"first"},
        {"role":"developer", "content":"mid-conversation developer"}
    ]}))
    .unwrap();
    let turn = engine.prepare(&request, None).await.unwrap();
    let input_history = serde_json::to_value(&turn.history[1..]).unwrap();
    let output = engine.execute(turn).await.unwrap();
    engine.persist(&request, None, &output).await;
    let mut previous = output.response.id;
    for instructions in [None, Some("replacement"), None, Some("another replacement")] {
        let request = serde_json::from_value(json!({"model":"m", "instructions":instructions, "previous_response_id":previous, "input":"next"})).unwrap();
        let turn = engine.prepare(&request, None).await.unwrap();
        assert_eq!(
            serde_json::to_value(&turn.history[1..4]).unwrap(),
            input_history
        );
        if let Some(expected) = instructions {
            assert_eq!(
                serde_json::to_value(&turn.history[0]).unwrap()["parts"][0]["content"],
                expected
            );
        }
        let output = engine.execute(turn).await.unwrap();
        engine.persist(&request, None, &output).await;
        previous = output.response.id;
    }
}

#[tokio::test]
async fn rig_preserves_encrypted_reasoning_in_json_stream_and_stored_history() {
    use serdes_ai_core::messages::*;
    use serdes_ai_models::mock::FunctionModel;
    let thinking = ThinkingPart::new("summary").with_provider_details(
        [("encrypted_content".to_string(), json!("initial-ciphertext"))]
            .into_iter()
            .collect(),
    );
    let original = thinking.clone();
    let model = FunctionModel::with_both(
        move |_, _| ModelResponse::with_parts(vec![ModelResponsePart::Thinking(original.clone())]),
        move |_, _| {
            Box::pin(futures::stream::iter(vec![
                Ok(ModelResponseStreamEvent::PartStart(PartStartEvent::new(
                    0,
                    ModelResponsePart::Thinking(thinking.clone()),
                ))),
                Ok(ModelResponseStreamEvent::PartDelta(PartDeltaEvent::new(
                    0,
                    ModelResponsePartDelta::Thinking(ThinkingPartDelta {
                        content_delta: " extended".into(),
                        signature_delta: None,
                        provider_name: None,
                        provider_details: Some(
                            [("encrypted_content".to_string(), json!("final-ciphertext"))]
                                .into_iter()
                                .collect(),
                        ),
                    }),
                ))),
                Ok(ModelResponseStreamEvent::PartEnd(PartEndEvent::new(0))),
                Ok(ModelResponseStreamEvent::StreamComplete(
                    StreamCompleteEvent::new(serdes_ai_core::FinishReason::Stop),
                )),
            ]))
        },
    );
    let engine = rig::engine::ResponsesEngine::new(std::sync::Arc::new(model));
    let request = serde_json::from_value(json!({"model":"m", "input":"reason"})).unwrap();
    let output = engine
        .execute(engine.prepare(&request, None).await.unwrap())
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_value(&output.response).unwrap()["output"][0]["encrypted_content"],
        "initial-ciphertext"
    );
    let mut events = Vec::new();
    let output = engine
        .execute_streaming(
            engine.prepare(&request, None).await.unwrap(),
            &mut |event| {
                events.push(serde_json::to_value(event).unwrap());
                std::future::ready(Ok(()))
            },
        )
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_value(&output.response).unwrap()["output"][0]["encrypted_content"],
        "final-ciphertext"
    );
    assert_eq!(
        events
            .iter()
            .find(|e| e["type"] == "response.output_item.done")
            .unwrap()["item"]["encrypted_content"],
        "final-ciphertext"
    );
    engine.persist(&request, None, &output).await;
    let stored = engine.get_response(&output.response.id).await.unwrap();
    let ModelRequestPart::ModelResponse(response) = &stored.history.last().unwrap().parts[0] else {
        panic!("assistant history");
    };
    let ModelResponsePart::Thinking(thinking) = &response.parts[0] else {
        panic!("thinking");
    };
    assert_eq!(thinking.content, "summary extended");
    assert_eq!(
        thinking.provider_details.as_ref().unwrap()["encrypted_content"],
        "final-ciphertext"
    );
}

struct PendingModelStream(std::sync::Arc<tokio::sync::Notify>);
impl futures::Stream for PendingModelStream {
    type Item =
        Result<serdes_ai_core::messages::ModelResponseStreamEvent, serdes_ai_models::ModelError>;
    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::task::Poll::Pending
    }
}
impl Drop for PendingModelStream {
    fn drop(&mut self) {
        self.0.notify_one();
    }
}

#[tokio::test]
async fn disconnected_clients_cancel_stalled_rig_streams_without_persisting() {
    for websocket in [false, true] {
        let dropped = std::sync::Arc::new(tokio::sync::Notify::new());
        let signal = dropped.clone();
        let model = serdes_ai_models::mock::FunctionModel::with_stream(move |_, _| {
            Box::pin(PendingModelStream(signal.clone()))
        });
        let engine = std::sync::Arc::new(rig::engine::ResponsesEngine::new(std::sync::Arc::new(
            model,
        )));
        let router = rig::server::ResponsesServer::from_engine(engine.clone()).router();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        let created: Value = if websocket {
            let mut ws = connect(&addr).await;
            ws.send(Message::text(
                json!({"type":"response.create", "model":"m", "input":"stall"}).to_string(),
            ))
            .await
            .unwrap();
            let Message::Text(text) = ws.next().await.unwrap().unwrap() else {
                panic!("created frame");
            };
            let created = serde_json::from_str(&text).unwrap();
            ws.close(None).await.unwrap();
            created
        } else {
            let mut body = client()
                .post(format!("http://{addr}/v1/responses"))
                .json(&json!({"model":"m", "input":"stall", "stream":true}))
                .send()
                .await
                .unwrap()
                .bytes_stream();
            let chunk = body.next().await.unwrap().unwrap();
            serde_json::from_str(
                std::str::from_utf8(&chunk)
                    .unwrap()
                    .lines()
                    .next()
                    .unwrap()
                    .strip_prefix("data: ")
                    .unwrap(),
            )
            .unwrap()
        };
        tokio::time::timeout(Duration::from_secs(5), dropped.notified())
            .await
            .expect("model stream must drop after disconnect");
        assert!(
            engine
                .get_response(created["response"]["id"].as_str().unwrap())
                .await
                .is_none()
        );
        server.abort();
        let _ = server.await;
    }
}

#[tokio::test]
async fn saturated_websocket_input_cancels_stalled_turn_without_persisting() {
    let dropped = std::sync::Arc::new(tokio::sync::Notify::new());
    let signal = dropped.clone();
    let started = std::sync::Arc::new(tokio::sync::Notify::new());
    let start_signal = started.clone();
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let call_count = calls.clone();
    let model = serdes_ai_models::mock::FunctionModel::with_stream(move |_, _| {
        call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        start_signal.notify_one();
        Box::pin(PendingModelStream(signal.clone()))
    });
    let engine = std::sync::Arc::new(rig::engine::ResponsesEngine::new(std::sync::Arc::new(
        model,
    )));
    let router = rig::server::ResponsesServer::from_engine(engine.clone()).router();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        let mut ws = connect(&addr).await;
        let create = Message::text(
            json!({"type":"response.create", "model":"m", "input":"stall"}).to_string(),
        );
        ws.send(create.clone()).await.unwrap();
        started.notified().await;
        let Message::Text(text) = ws.next().await.unwrap().unwrap() else {
            panic!("created frame");
        };
        let created: Value = serde_json::from_str(&text).unwrap();
        // Buffer all seventeen frames and Close before flushing, so the peer
        // sees a full queue even though overflow itself will close the socket.
        for _ in 0..17 {
            ws.feed(create.clone()).await.unwrap();
        }
        ws.feed(Message::Close(None)).await.unwrap();
        ws.flush().await.unwrap();
        dropped.notified().await;
        assert!(
            engine
                .get_response(created["response"]["id"].as_str().unwrap())
                .await
                .is_none()
        );
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    })
    .await
    .expect("overflow must cancel the stalled turn and close the socket");
    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn idle_rig_socket_expires_without_another_create_frame() {
    let (model, _) = recording_model();
    let addr = spawn_server_with_ws_config(
        model,
        rig::websocket::WebSocketSessionConfig {
            connection_ttl: Duration::from_millis(50),
        },
    )
    .await;
    let mut ws = connect(&addr).await;
    let event = tokio::time::timeout(Duration::from_secs(5), ws.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let Message::Text(text) = event else {
        panic!("limit envelope");
    };
    let error: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(error["error"]["code"], "websocket_connection_limit_reached");
    assert!(matches!(ws.next().await, Some(Ok(Message::Close(_)))));
}

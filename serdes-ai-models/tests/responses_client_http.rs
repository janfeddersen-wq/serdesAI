//! HTTP-transport tests for the responses model: rig-driven coverage of
//! terminal-last SSE streaming and the server-side history lens chained
//! turns reconstruct, plus scripted fake servers that pin the wire
//! contracts the rig cannot observe (`store: true`, delta-only chained
//! input, stale-continuation replay, error-envelope mapping) and the SSE
//! terminal-event contract. Websocket-transport coverage lives in
//! `responses_client_ws.rs`.

mod rig;
mod ws_fakes_common;

use std::sync::Arc;

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use futures::StreamExt;
use rig::{recording_model, spawn_malformed_sse_server, spawn_server};
use serdes_ai_core::messages::{
    ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart, ModelResponseStreamEvent,
    TextPart,
};
use serdes_ai_models::ModelError;
use serdes_ai_models::model::Model;
use serdes_ai_models::openai::responses::wire::codes;
use serdes_ai_models::openai::responses::{OpenAIResponsesModel, Transport};
use ws_fakes_common::{params, response_turn, settings, system_turn, text_of, user_turn};

/// An HTTP-transport model pointed at the rig. The HTTP transport appends
/// `/responses` to the base URL, so this is the API root, not the endpoint.
fn http_client(addr: std::net::SocketAddr) -> OpenAIResponsesModel {
    OpenAIResponsesModel::new("test-model", "test-key")
        .with_base_url(format!("http://{addr}/v1"))
        .with_transport(Transport::Http)
        .with_session_chaining(true)
}

/// A minimal assistant response for extending history in tests.
fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        parts: vec![ModelResponsePart::Text(TextPart::new(text))],
        model_name: None,
        timestamp: chrono::Utc::now(),
        finish_reason: None,
        usage: None,
        vendor_id: None,
        vendor_details: None,
        kind: "response".to_string(),
    }
}

#[tokio::test]
async fn http_stateful_chaining_across_turns() {
    let (model, calls) = recording_model();
    let addr = spawn_server(model).await;
    let client = http_client(addr);

    let mut history = vec![system_turn("be brief"), user_turn("first")];
    let first = client
        .request(&history, &settings(), &params())
        .await
        .expect("first turn");
    assert_eq!(text_of(&first), "ok");

    history.push(ModelRequest::with_parts(vec![
        ModelRequestPart::ModelResponse(Box::new(first)),
    ]));
    history.push(user_turn("second"));
    let second = client
        .request(&history, &settings(), &params())
        .await
        .expect("second turn");

    // Turn 1 sees instructions + prompt. Turn 2 chains onto the recorded
    // response and sends only the new user item; the rig reconstructs the
    // full server-side history (instructions + prompt + prior response +
    // prompt = 4). The delta-only wire shape is pinned by
    // `http_chained_turns_send_store_true_and_delta_only_items` below.
    assert_eq!(text_of(&second), "ok");
    assert_eq!(*calls.lock().unwrap(), vec![2, 4]);
}

#[tokio::test]
async fn http_stream_yields_events_with_terminal_last() {
    let (model, _calls) = recording_model();
    let addr = spawn_server(model).await;
    let client = http_client(addr);

    let history = vec![user_turn("hello")];
    let mut stream = client
        .request_stream(&history, &settings(), &params())
        .await
        .expect("stream starts");

    let mut saw_terminal = false;
    let mut count = 0;
    while let Some(event) = stream.next().await {
        let event = event.expect("event ok");
        count += 1;
        match &event {
            ModelResponseStreamEvent::StreamComplete(_) => saw_terminal = true,
            _ => assert!(!saw_terminal, "terminal event must be last"),
        }
    }
    assert!(saw_terminal);
    assert!(count >= 2);
}

#[tokio::test]
async fn http_stream_chained_turns_send_only_new_items() {
    let (model, calls) = recording_model();
    let addr = spawn_server(model).await;
    let client = http_client(addr);

    let drain = |client: OpenAIResponsesModel, history: Vec<ModelRequest>| async move {
        let mut stream = client
            .request_stream(&history, &settings(), &params())
            .await
            .expect("stream starts");
        let mut text = String::new();
        let mut terminal = false;
        while let Some(item) = stream.next().await {
            match item.expect("event ok") {
                ModelResponseStreamEvent::PartDelta(delta) => {
                    if let serdes_ai_core::messages::ModelResponsePartDelta::Text(text_delta) =
                        delta.delta
                    {
                        text.push_str(&text_delta.content_delta);
                    }
                }
                ModelResponseStreamEvent::StreamComplete(_) => terminal = true,
                _ => {}
            }
        }
        assert!(terminal, "stream must end with a terminal event");
        text
    };

    let mut history = vec![system_turn("be brief"), user_turn("first")];
    let first_text = drain(client.clone(), history.clone()).await;
    assert_eq!(first_text, "ok");

    history.push(ModelRequest::with_parts(vec![
        ModelRequestPart::ModelResponse(Box::new(text_response(&first_text))),
    ]));
    history.push(user_turn("second"));
    let second_text = drain(client, history).await;
    assert_eq!(second_text, "ok");

    // Turn 2 must send only the new user item; a client that replayed the
    // prior ModelResponse would show 5 instead of 4.
    assert_eq!(*calls.lock().unwrap(), vec![2, 4]);
}

#[tokio::test]
async fn http_conversations_stay_isolated() {
    let (model, calls) = recording_model();
    let addr = spawn_server(model).await;
    let client = http_client(addr);

    let mut convo_a = vec![system_turn("be brief"), user_turn("first")];
    let first = client
        .request(&convo_a, &settings(), &params())
        .await
        .expect("conversation A turn 1");
    assert_eq!(text_of(&first), "ok");

    // A different first request is a different conversation even through
    // the same model instance.
    let convo_b = vec![system_turn("other brief"), user_turn("b")];
    let b = client
        .request(&convo_b, &settings(), &params())
        .await
        .expect("conversation B turn 1");
    assert_eq!(text_of(&b), "ok");

    // Conversation A's second turn chains onto its own last response.
    convo_a.push(response_turn(first));
    convo_a.push(user_turn("second"));
    let second = client
        .request(&convo_a, &settings(), &params())
        .await
        .expect("conversation A turn 2");
    assert_eq!(text_of(&second), "ok");

    // A1 sees 2; B1 sees 2 (a fresh conversation, not A's chain); A2 sees
    // the chained 4. A client that keyed chains per model instead of per
    // conversation would show B1 chaining onto A's response.
    assert_eq!(*calls.lock().unwrap(), vec![2, 2, 4]);
}

// ---------------------------------------------------------------------------
// Wire contracts against scripted fake servers
// ---------------------------------------------------------------------------

/// Behavior script for the scripted fake responses server.
#[derive(Debug, Clone, Copy)]
enum FakeMode {
    /// Accept unchained turns; reject turns carrying a
    /// `previous_response_id` with `previous_response_not_found`.
    StaleChain,
    /// Accept the first request, then reject the continuation and full replay.
    StaleReplayFails,
    /// Reject every turn with a `model_error` envelope.
    EnvelopeEveryTurn,
    /// Reject every turn with a 500 non-JSON body.
    PlainTextEveryTurn,
}

/// Request bodies the fake server received, in order.
type CapturedBodies = Arc<std::sync::Mutex<Vec<serde_json::Value>>>;

/// Spawn a scripted single-route responses server; returns its address and
/// the request bodies it received.
async fn spawn_fake_server(mode: FakeMode) -> (std::net::SocketAddr, CapturedBodies) {
    let bodies: CapturedBodies = Arc::default();
    let router = Router::new()
        .route(
            "/v1/responses",
            post(handle_fake_turn).with_state((mode, Arc::clone(&bodies))),
        )
        .with_state((mode, Arc::clone(&bodies)));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    (addr, bodies)
}

/// The completed-response body the fake server answers accepted turns
/// with; `id` follows the number of requests seen so far.
fn completed_body(id: &str, request: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "response",
        "created_at": 1,
        "model": request["model"],
        "status": "completed",
        "output": [{
            "type": "message",
            "id": format!("msg_{id}"),
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }],
    })
}

async fn handle_fake_turn(
    State((mode, bodies)): State<(FakeMode, CapturedBodies)>,
    body: Bytes,
) -> Response {
    let request: serde_json::Value = serde_json::from_slice(&body).expect("json body");
    bodies.lock().unwrap().push(request.clone());
    let chained = request
        .get("previous_response_id")
        .is_some_and(|value| !value.is_null());

    let rejection = match mode {
        FakeMode::StaleChain => {
            chained.then_some((StatusCode::NOT_FOUND, codes::PREVIOUS_RESPONSE_NOT_FOUND))
        }
        FakeMode::EnvelopeEveryTurn => Some((StatusCode::BAD_GATEWAY, "model_error")),
        FakeMode::StaleReplayFails => (bodies.lock().unwrap().len() > 1)
            .then_some((StatusCode::NOT_FOUND, codes::PREVIOUS_RESPONSE_NOT_FOUND)),
        FakeMode::PlainTextEveryTurn => None,
    };
    if let Some((status, code)) = rejection {
        let envelope = serde_json::json!({
            "error": {"code": code, "message": format!("rejected: {code}")},
        });
        return (status, axum::Json(envelope)).into_response();
    }
    if matches!(mode, FakeMode::PlainTextEveryTurn) {
        return (StatusCode::INTERNAL_SERVER_ERROR, "backend exploded").into_response();
    }

    let id = format!("resp_{}", bodies.lock().unwrap().len());
    axum::Json(completed_body(&id, &request)).into_response()
}

#[tokio::test]
async fn http_chained_turns_send_store_true_and_delta_only_items() {
    let (addr, bodies) = spawn_fake_server(FakeMode::StaleChain).await;
    let client = http_client(addr);

    let mut history = vec![system_turn("be brief"), user_turn("first")];
    let first = client
        .request(&history, &settings(), &params())
        .await
        .expect("first turn");
    history.push(response_turn(first));
    history.push(user_turn("second"));

    // The fake rejects the chained turn with previous_response_not_found;
    // the client must clear the chain and replay the full input.
    let second = client
        .request(&history, &settings(), &params())
        .await
        .expect("second turn after stale-continuation replay");
    assert_eq!(text_of(&second), "ok");

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 3, "the chained turn was rejected once");
    // Turn 1: full input, persisted for chaining.
    assert_eq!(bodies[0]["store"], serde_json::Value::Bool(true));
    assert!(bodies[0]["previous_response_id"].is_null());
    assert_eq!(bodies[0]["input"].as_array().unwrap().len(), 1);
    // Turn 2 chains with store:true and only the new user item.
    assert_eq!(bodies[1]["previous_response_id"], "resp_1");
    assert_eq!(bodies[1]["store"], serde_json::Value::Bool(true));
    assert_eq!(
        bodies[1]["input"].as_array().unwrap().len(),
        1,
        "chained turn sends only the new input item"
    );
    // The stale-continuation replay sends the full input without the id.
    assert!(bodies[2]["previous_response_id"].is_null());
    assert_eq!(
        bodies[2]["input"].as_array().unwrap().len(),
        3,
        "replay carries the full input (user, assistant echo, user)"
    );
}

#[tokio::test]
async fn http_error_envelope_surfaces_as_provider_error() {
    let (addr, _bodies) = spawn_fake_server(FakeMode::EnvelopeEveryTurn).await;
    let client = http_client(addr);

    let error = client
        .request(&[user_turn("hello")], &settings(), &params())
        .await
        .expect_err("turn must fail");

    match &error {
        ModelError::Provider {
            provider,
            code,
            status,
            ..
        } => {
            assert_eq!(provider, "openai");
            assert_eq!(code, "model_error");
            assert_eq!(*status, Some(502));
        }
        other => panic!("expected a provider error, got {other:?}"),
    }
}

#[tokio::test]
async fn http_non_envelope_error_stays_transport_error() {
    let (addr, _bodies) = spawn_fake_server(FakeMode::PlainTextEveryTurn).await;
    let client = http_client(addr);

    let error = client
        .request(&[user_turn("hello")], &settings(), &params())
        .await
        .expect_err("turn must fail");

    match &error {
        ModelError::Http { status, .. } => assert_eq!(*status, 500),
        other => panic!("expected a transport error, got {other:?}"),
    }
}

#[tokio::test]
async fn http_done_without_terminal_event_fails_the_stream() {
    let addr = spawn_malformed_sse_server().await;
    let client = http_client(addr);

    let history = vec![user_turn("hello")];
    let mut stream = client
        .request_stream(&history, &settings(), &params())
        .await
        .expect("stream starts");

    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(item);
    }

    // The malformed route streams a delta and then the [DONE] sentinel
    // with no terminal event: the delta reaches the caller, no terminal
    // event may be synthesized, and the sentinel surfaces as an error.
    let deltas = events
        .iter()
        .filter(|item| matches!(item, Ok(ModelResponseStreamEvent::PartDelta(_))))
        .count();
    assert_eq!(deltas, 1, "the streamed delta reached the caller");
    assert!(
        !events
            .iter()
            .any(|item| matches!(item, Ok(ModelResponseStreamEvent::StreamComplete(_)))),
        "no terminal event may follow a stream without one"
    );
    assert!(
        matches!(events.last(), Some(Err(_))),
        "error is the last item"
    );
    match &events.last() {
        Some(Err(ModelError::IncompleteStream(message))) => {
            assert!(message.contains("[DONE]"), "got: {message}");
        }
        other => panic!("expected an incomplete-stream error, got {other:?}"),
    }
}

#[tokio::test]
async fn custom_headers_reach_all_http_modes() {
    check_http_headers(false).await;
}

#[tokio::test]
async fn explicit_headers_override_defaults_in_all_http_modes() {
    check_http_headers(true).await;
}

async fn check_http_headers(explicit_overrides: bool) {
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    let expected = if explicit_overrides {
        [
            ("authorization", "Bearer explicit-key"),
            ("openai-organization", "org-explicit"),
            ("openai-project", "project-explicit"),
            ("x-gateway-key", "gateway-explicit"),
            ("content-type", "application/vnd.test+json"),
        ]
    } else {
        [
            ("authorization", "Bearer test-key"),
            ("openai-organization", "org-test"),
            ("openai-project", "project-test"),
            ("x-gateway-key", "gateway-test"),
            ("content-type", "application/json"),
        ]
    };
    for chaining in [false, true] {
        for streaming in [false, true] {
            let server = MockServer::start().await;
            let response =
                completed_body("resp_headers", &serde_json::json!({"model":"test-model"}));
            let template = if chaining && streaming {
                ResponseTemplate::new(200).set_body_string(format!(
                    "data: {}\n\n", serde_json::json!({"type":"response.completed", "sequence_number":0,"response":response})
                )).insert_header("content-type", "text/event-stream")
            } else {
                ResponseTemplate::new(200).set_body_json(response)
            };
            Mock::given(method("POST"))
                .and(path("/responses"))
                .respond_with(template)
                .expect(1)
                .mount(&server)
                .await;
            let mut client = OpenAIResponsesModel::new("test-model", "test-key")
                .with_base_url(server.uri())
                .with_session_chaining(chaining)
                .with_organization("org-test")
                .with_project("project-test")
                .with_header("x-gateway-key", "gateway-test");
            if explicit_overrides {
                client = client
                    .with_header("Authorization", "Bearer superseded")
                    .with_header("aUtHoRiZaTiOn", "Bearer explicit-key")
                    .with_header("OPENAI-ORGANIZATION", "org-explicit")
                    .with_header("OpenAI-Project", "project-explicit")
                    .with_header("X-Gateway-Key", "gateway-explicit")
                    .with_header("Content-Type", "application/vnd.test+json");
            }
            if streaming {
                let events: Vec<_> = client
                    .request_stream(&[user_turn("headers")], &settings(), &params())
                    .await
                    .unwrap()
                    .collect()
                    .await;
                assert!(events.iter().all(Result::is_ok));
                assert!(matches!(
                    events.last(),
                    Some(Ok(ModelResponseStreamEvent::StreamComplete(_)))
                ));
            } else {
                assert_eq!(
                    text_of(
                        &client
                            .request(&[user_turn("headers")], &settings(), &params())
                            .await
                            .unwrap()
                    ),
                    "ok"
                );
            }
            let requests = server.received_requests().await.unwrap();
            assert_eq!(requests.len(), 1);
            for (name, value) in expected {
                let values: Vec<_> = requests[0].headers.get_all(name).iter().collect();
                assert_eq!(values.len(), 1, "exactly one {name} header");
                assert_eq!(values[0], value);
            }
        }
    }
}

async fn sse_events(payload: String) -> Vec<Result<ModelResponseStreamEvent, ModelError>> {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(payload),
        )
        .mount(&server)
        .await;
    OpenAIResponsesModel::new("test-model", "test-key")
        .with_base_url(server.uri())
        .with_session_chaining(true)
        .request_stream(&[user_turn("events")], &settings(), &params())
        .await
        .unwrap()
        .collect()
        .await
}

#[tokio::test]
async fn sse_unknown_events_are_ignored_and_terminal_ends_the_stream() {
    let response = completed_body("resp_terminal", &serde_json::json!({"model":"test-model"}));
    let terminal =
        serde_json::json!({"type":"response.completed", "sequence_number":1, "response":response});
    for suffix in [
        "",
        "data: [DONE]\n\n",
        "data: {bad JSON after terminal}\n\n",
    ] {
        let events = sse_events(format!(
            "data: {{\"type\":\"response.output_text.annotation.added\",\"annotation\":{{}}}}\n\ndata: {terminal}\n\n{suffix}"
        )).await;
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Ok(ModelResponseStreamEvent::StreamComplete(_))
        ));
    }
}

#[tokio::test]
async fn sse_malformed_known_events_and_invalid_json_fail() {
    for event in [
        r#"{"type":"response.output_text.delta","sequence_number":0}"#,
        "{bad json}",
        r#"{"delta":"missing type"}"#,
    ] {
        let events = sse_events(format!("data: {event}\n\ndata: [DONE]\n\n")).await;
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], Err(ModelError::InvalidResponse(_))));
    }
}

#[tokio::test]
async fn shared_system_prefixes_keep_http_conversations_independent() {
    let (model, calls) = recording_model();
    let client = http_client(spawn_server(model).await);
    let system = system_turn("shared instructions");
    let mut a = vec![system.clone(), user_turn("a")];
    let b = vec![system.clone(), user_turn("b")];
    // System-only requests must not become a chain for the first user turn.
    client
        .request(&[system], &settings(), &params())
        .await
        .unwrap();
    let first = client.request(&a, &settings(), &params()).await.unwrap();
    client.request(&b, &settings(), &params()).await.unwrap();
    a.extend([response_turn(first), user_turn("a next")]);
    client.request(&a, &settings(), &params()).await.unwrap();
    assert_eq!(*calls.lock().unwrap(), vec![1, 2, 2, 4]);
}

#[tokio::test]
async fn stale_full_replay_rejection_keeps_provider_code_and_stops_retrying() {
    let (addr, bodies) = spawn_fake_server(FakeMode::StaleReplayFails).await;
    let client = http_client(addr);
    let mut history = vec![user_turn("first")];
    let first = client
        .request(&history, &settings(), &params())
        .await
        .unwrap();
    history.extend([response_turn(first), user_turn("second")]);
    let error = client
        .request(&history, &settings(), &params())
        .await
        .unwrap_err();
    assert!(
        matches!(error, ModelError::Provider { code, status: Some(404), .. } if code == codes::PREVIOUS_RESPONSE_NOT_FOUND)
    );
    let bodies = bodies.lock().unwrap();
    assert_eq!(
        bodies.len(),
        3,
        "one initial turn, one continuation, one full replay"
    );
    assert!(bodies[2]["previous_response_id"].is_null());
}

#[tokio::test]
async fn sse_errors_survive_a_full_output_channel() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};
    let server = MockServer::start().await;
    let mut payload = String::new();
    for sequence in 0..64 {
        payload.push_str(&format!("data: {}\n\n", serde_json::json!({
            "type":"response.output_text.delta", "sequence_number":sequence, "item_id":"msg", "output_index":0, "content_index":0, "delta":"x"
        })));
    }
    payload.push_str("data: {bad json}\n\n");
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(payload),
        )
        .mount(&server)
        .await;
    let client = OpenAIResponsesModel::new("m", "key")
        .with_base_url(server.uri())
        .with_session_chaining(true);
    let stream = client
        .request_stream(&[user_turn("backpressure")], &settings(), &params())
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let events: Vec<_> = stream.collect().await;
    assert_eq!(events.len(), 65);
    assert!(matches!(
        events.last(),
        Some(Err(ModelError::InvalidResponse(_)))
    ));
}

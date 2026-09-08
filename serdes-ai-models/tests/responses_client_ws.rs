//! End-to-end tests for the responses model's websocket transport against
//! the local rig: response mapping, delta-only chained sends, terminal-last
//! streaming, and TTL-triggered reconnect. The fake servers here pin the
//! wire contracts the rig cannot observe: function-tool type tags,
//! conversation isolation, chain resets on mutated or replayed history,
//! idle conversation eviction, the close handshake on model drop, and
//! concurrent conversations.
//!
//! The four scripted-fake recovery tests (stale continuation, connection
//! limit, hard error, mid-stream no-replay) live in `responses_ws_fakes.rs`.
#![cfg(feature = "responses-ws")]

mod rig;
mod ws_fakes_common;

use futures::StreamExt;
use rig::websocket::WebSocketSessionConfig;
use rig::{recording_model, spawn_server, spawn_server_with_ws_config};
use serdes_ai_core::messages::{ModelRequest, ModelRequestPart, ModelResponseStreamEvent};
use serdes_ai_models::model::{Model, ModelRequestParameters};
use serdes_ai_models::openai::responses::wire::{CreateResponseRequest, ResponseInput};
use serdes_ai_tools::ToolDefinition;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;
use ws_fakes_common::{
    accept_ws, params, read_turn, response_turn, send_completed_turn, settings, system_turn,
    text_of, user_turn, ws_client,
};

/// Number of input items a `response.create` request carries.
fn input_len(request: &CreateResponseRequest) -> usize {
    match &request.input {
        ResponseInput::Items(items) => items.len(),
        other => panic!("expected items input, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// End-to-end against the rig
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ws_turns_map_responses_and_send_only_new_items() {
    let (model, calls) = recording_model();
    let addr = spawn_server(model).await;
    let client = ws_client(addr).with_session_chaining(true);

    let mut history = vec![system_turn("be brief"), user_turn("first")];
    let first = client
        .request(&history, &settings(), &params())
        .await
        .expect("first turn");

    assert_eq!(text_of(&first), "ok");
    assert!(
        first
            .vendor_id
            .as_deref()
            .unwrap_or("")
            .starts_with("resp_")
    );
    let first_id = first.vendor_id.clone().unwrap();

    history.push(ModelRequest::with_parts(vec![
        ModelRequestPart::ModelResponse(Box::new(first)),
    ]));
    history.push(user_turn("second"));
    let second = client
        .request(&history, &settings(), &params())
        .await
        .expect("second turn");

    // Turn 1 sees instructions + prompt. Turn 2 chains onto the recorded
    // response and sends only the new item; the rig reconstructs the full
    // server-side history (instructions + prompt + prior response + prompt
    // = 4). The delta-only wire shape is pinned by
    // `stale_continuation_clears_chain_and_replays_full_input` in
    // `responses_ws_fakes.rs`.
    assert_eq!(*calls.lock().unwrap(), vec![2, 4]);
    assert_eq!(text_of(&second), "ok");
    assert_ne!(second.vendor_id.as_deref(), Some(first_id.as_str()));
}

#[tokio::test]
async fn ws_stream_emits_events_with_terminal_last() {
    let (model, _calls) = recording_model();
    let addr = spawn_server(model).await;
    let client = ws_client(addr);

    let history = vec![user_turn("hello")];
    let mut stream = client
        .request_stream(&history, &settings(), &params())
        .await
        .expect("stream starts");

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.expect("event ok"));
    }

    let terminals = events
        .iter()
        .filter(|event| matches!(event, ModelResponseStreamEvent::StreamComplete(_)))
        .count();
    assert_eq!(terminals, 1, "exactly one terminal event");
    assert!(matches!(
        events.last(),
        Some(ModelResponseStreamEvent::StreamComplete(_))
    ));
    assert!(matches!(
        events.first(),
        Some(ModelResponseStreamEvent::PartStart(_))
    ));
}

#[tokio::test]
async fn ws_connection_limit_reconnects_and_replays() {
    let (model, calls) = recording_model();
    let config = WebSocketSessionConfig {
        connection_ttl: Duration::ZERO,
    };
    let addr = spawn_server_with_ws_config(model, config).await;
    let client = ws_client(addr).with_session_chaining(true);

    let mut history = vec![system_turn("sys"), user_turn("first")];
    let first = client
        .request(&history, &settings(), &params())
        .await
        .expect("first turn");
    history.push(ModelRequest::with_parts(vec![
        ModelRequestPart::ModelResponse(Box::new(first)),
    ]));
    history.push(user_turn("second"));

    // The rig refuses the second turn on the aged socket
    // (websocket_connection_limit_reached); the client must reconnect,
    // start a fresh session, and replay the full input.
    let second = client
        .request(&history, &settings(), &params())
        .await
        .expect("second turn after reconnect");
    assert_eq!(text_of(&second), "ok");
    assert_eq!(*calls.lock().unwrap(), vec![2, 4]);
}

// ---------------------------------------------------------------------------
// Wire contracts against scripted fake servers
// ---------------------------------------------------------------------------

#[tokio::test]
async fn client_sends_function_tools_with_wire_type_tag() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let mut ws = accept_ws(&listener).await;
        // Capture the raw frame: the type tag must be present on the wire,
        // not just in the parsed form.
        let value = loop {
            match ws.next().await.unwrap().unwrap() {
                Message::Text(text) => {
                    break serde_json::from_str::<serde_json::Value>(&text).unwrap();
                }
                Message::Close(_) => panic!("client closed before sending a turn"),
                _ => continue,
            }
        };
        assert_eq!(value["type"], "response.create");
        let tools = value["tools"].as_array().expect("tools on the wire");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "get_weather");
        assert_eq!(tools[0]["strict"], true);
        assert_eq!(tools[0]["parameters"]["type"], "object");

        // Codex frames are flat: strip `type`, the rest is the payload.
        let mut payload = value;
        payload
            .as_object_mut()
            .expect("frame object")
            .remove("type");
        let request = serde_json::from_value::<CreateResponseRequest>(payload).unwrap();
        send_completed_turn(&mut ws, "resp_1", &request).await;
    });

    let client = ws_client(addr);
    let tool = ToolDefinition {
        name: "get_weather".to_string(),
        description: "Look up weather".to_string(),
        parameters_json_schema: serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}}
        }),
        strict: Some(true),
        outer_typed_dict_key: None,
    };
    let params = ModelRequestParameters::new().with_tools(vec![tool]);
    let response = client
        .request(&[user_turn("weather in NYC?")], &settings(), &params)
        .await
        .expect("turn with tools");
    assert_eq!(text_of(&response), "ok");
    server.await.unwrap();
}

// ---------------------------------------------------------------------------
// Conversation keying: one model instance, many conversations
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ws_independent_conversations_stay_isolated() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        // Conversation A keeps its own socket and chains across turns.
        let mut ws_a = accept_ws(&listener).await;
        let request = read_turn(&mut ws_a).await;
        assert!(request.previous_response_id.is_none());
        send_completed_turn(&mut ws_a, "resp_a1", &request).await;

        let request = read_turn(&mut ws_a).await;
        assert_eq!(request.previous_response_id.as_deref(), Some("resp_a1"));
        assert_eq!(input_len(&request), 1, "chained turn sends only new input");
        send_completed_turn(&mut ws_a, "resp_a2", &request).await;

        // A different first user request starts a second conversation despite
        // shared system prompts: its own socket, no continuation id, full input.
        let mut ws_b = accept_ws(&listener).await;
        let request_b = read_turn(&mut ws_b).await;
        assert!(
            request_b.previous_response_id.is_none(),
            "a fresh conversation must not chain onto conversation A"
        );
        assert_eq!(
            input_len(&request_b),
            2,
            "fresh conversation sends full input"
        );

        // The client awaits B's turn before sending A's next one, so B is
        // completed here first.
        send_completed_turn(&mut ws_b, "resp_b1", &request_b).await;

        // Conversation A is unaffected by its sibling: it still chains on
        // its own last response.
        let request = read_turn(&mut ws_a).await;
        assert_eq!(request.previous_response_id.as_deref(), Some("resp_a2"));
        send_completed_turn(&mut ws_a, "resp_a3", &request).await;
    });

    let client = ws_client(addr).with_session_chaining(true);

    let mut convo_a = vec![system_turn("sys a"), user_turn("first")];
    let first = client
        .request(&convo_a, &settings(), &params())
        .await
        .expect("conversation A turn 1");
    convo_a.push(response_turn(first));
    convo_a.push(user_turn("second"));
    let second = client
        .request(&convo_a, &settings(), &params())
        .await
        .expect("conversation A turn 2");

    let convo_b = vec![convo_a[0].clone(), user_turn("b1"), user_turn("b2")];
    client
        .request(&convo_b, &settings(), &params())
        .await
        .expect("conversation B turn 1");

    convo_a.push(response_turn(second));
    convo_a.push(user_turn("third"));
    let third = client
        .request(&convo_a, &settings(), &params())
        .await
        .expect("conversation A turn 3");
    assert_eq!(text_of(&third), "ok");
    server.await.unwrap();
}

#[tokio::test]
async fn ws_mutated_history_restarts_the_chain() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let mut ws = accept_ws(&listener).await;

        // Turn 1: full input, no continuation.
        let request = read_turn(&mut ws).await;
        assert!(request.previous_response_id.is_none());
        send_completed_turn(&mut ws, "resp_1", &request).await;

        // Turn 2: chains onto resp_1 with only the new item.
        let request = read_turn(&mut ws).await;
        assert_eq!(request.previous_response_id.as_deref(), Some("resp_1"));
        assert_eq!(input_len(&request), 1);
        send_completed_turn(&mut ws, "resp_2", &request).await;

        // Turn 3: the caller mutated the conversation history (same first
        // request, different later turn). The recorded chain no longer
        // matches, so the client must drop the continuation id and replay
        // the full input.
        let request = read_turn(&mut ws).await;
        assert!(
            request.previous_response_id.is_none(),
            "mutated history must drop the stale chain"
        );
        assert_eq!(input_len(&request), 3, "restart replays the full input");
        send_completed_turn(&mut ws, "resp_3", &request).await;

        // The restart happens in place: no extra connection may appear.
        let extra = tokio::time::timeout(Duration::from_millis(300), listener.accept()).await;
        assert!(
            extra.is_err(),
            "a chain reset must reuse the socket, not reconnect"
        );
    });

    let client = ws_client(addr).with_session_chaining(true);
    let mut history = vec![system_turn("sys"), user_turn("first")];
    let first = client
        .request(&history, &settings(), &params())
        .await
        .expect("turn 1");
    history.push(response_turn(first));
    history.push(user_turn("second"));
    let second = client
        .request(&history, &settings(), &params())
        .await
        .expect("turn 2");
    assert_eq!(text_of(&second), "ok");

    // Same conversation, mutated history: replace the second user turn.
    let mut mutated = history.clone();
    mutated.pop();
    mutated.push(user_turn("second, edited"));
    let third = tokio::time::timeout(
        Duration::from_secs(5),
        client.request(&mutated, &settings(), &params()),
    )
    .await
    .expect("turn 3 must not hang")
    .expect("turn 3 succeeds after the chain reset");
    assert_eq!(text_of(&third), "ok");
    server.await.unwrap();
}

#[tokio::test]
async fn ws_concurrent_conversations_run_in_parallel() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        // Accept the first conversation and hold its turn open. The second
        // conversation must still connect and send while the first is in
        // flight; a whole-model turn lock would deadlock here.
        let mut ws_a = accept_ws(&listener).await;
        let request_a = read_turn(&mut ws_a).await;
        let mut ws_b = accept_ws(&listener).await;
        let request_b = read_turn(&mut ws_b).await;
        assert!(request_a.previous_response_id.is_none());
        assert!(request_b.previous_response_id.is_none());
        send_completed_turn(&mut ws_a, "resp_a", &request_a).await;
        send_completed_turn(&mut ws_b, "resp_b", &request_b).await;
    });

    let client = ws_client(addr).with_session_chaining(true);
    let system = system_turn("shared instructions");
    let turn_a = [system.clone(), user_turn("hello a")];
    let turn_b = [system, user_turn("hello b")];
    let settings = settings();
    let params = params();
    let (a, b) = tokio::time::timeout(Duration::from_secs(10), async {
        tokio::join!(
            client.request(&turn_a, &settings, &params),
            client.request(&turn_b, &settings, &params),
        )
    })
    .await
    .expect("concurrent conversations must not deadlock");
    assert_eq!(text_of(&a.expect("conversation A")), "ok");
    assert_eq!(text_of(&b.expect("conversation B")), "ok");
    server.await.unwrap();
}

// ---------------------------------------------------------------------------
// Chain reset on replayed history, idle eviction, and close handshakes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ws_identical_history_restarts_the_chain() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let mut ws = accept_ws(&listener).await;

        // Turn 1: full input (the system turn maps to instructions), no
        // continuation.
        let request = read_turn(&mut ws).await;
        assert!(request.previous_response_id.is_none());
        assert_eq!(input_len(&request), 1);
        send_completed_turn(&mut ws, "resp_1", &request).await;

        // Turn 2: an extension chains as usual, sending only the new item.
        let request = read_turn(&mut ws).await;
        assert_eq!(request.previous_response_id.as_deref(), Some("resp_1"));
        assert_eq!(input_len(&request), 1);
        send_completed_turn(&mut ws, "resp_2", &request).await;

        // Turn 3: the exact same history arrives again. A continuation
        // must add material, so the chain resets: no continuation id and
        // the full input, not a chained turn with empty input.
        let request = read_turn(&mut ws).await;
        assert!(
            request.previous_response_id.is_none(),
            "an identical history must reset the chain, not chain onto it"
        );
        assert_eq!(input_len(&request), 3, "the replay re-sends the full input");
        send_completed_turn(&mut ws, "resp_3", &request).await;

        // Turn 4: extending the history chains again from the replay.
        let request = read_turn(&mut ws).await;
        assert_eq!(request.previous_response_id.as_deref(), Some("resp_3"));
        assert_eq!(input_len(&request), 1);
        send_completed_turn(&mut ws, "resp_4", &request).await;

        // The reset happens in place: no extra connection may appear.
        let extra = tokio::time::timeout(Duration::from_millis(300), listener.accept()).await;
        assert!(
            extra.is_err(),
            "a replay reset must reuse the socket, not reconnect"
        );
    });

    let client = ws_client(addr).with_session_chaining(true);
    let mut history = vec![system_turn("sys"), user_turn("first")];
    let first = client
        .request(&history, &settings(), &params())
        .await
        .expect("turn 1");
    history.push(response_turn(first));
    history.push(user_turn("second"));
    let second = client
        .request(&history, &settings(), &params())
        .await
        .expect("turn 2");
    assert_eq!(text_of(&second), "ok");

    // Replay the identical history.
    let replay = client
        .request(&history, &settings(), &params())
        .await
        .expect("turn 3 (replayed history)");
    assert_eq!(text_of(&replay), "ok");

    // Chained turns still work after the reset.
    history.push(response_turn(replay));
    history.push(user_turn("third"));
    let fourth = client
        .request(&history, &settings(), &params())
        .await
        .expect("turn 4 after the reset");
    assert_eq!(text_of(&fourth), "ok");
    server.await.unwrap();
}

#[tokio::test]
async fn ws_idle_conversations_are_evicted_with_a_clean_close() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        // Turn 1 completes; the conversation then goes idle past the TTL.
        let mut ws = accept_ws(&listener).await;
        let request = read_turn(&mut ws).await;
        assert!(request.previous_response_id.is_none());
        send_completed_turn(&mut ws, "resp_1", &request).await;

        // Eviction closes the idle socket before the client reconnects.
        let frame = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("timed out waiting for the eviction close");
        match frame {
            Some(Ok(Message::Close(_))) => {}
            other => panic!("expected a Close frame on eviction, got {other:?}"),
        }

        // The same conversation resumes on a fresh socket: full input, no
        // continuation id.
        let mut ws = accept_ws(&listener).await;
        let request = read_turn(&mut ws).await;
        assert!(
            request.previous_response_id.is_none(),
            "an evicted conversation must not chain"
        );
        assert_eq!(input_len(&request), 1, "full input after eviction");
        send_completed_turn(&mut ws, "resp_2", &request).await;
    });

    let client = ws_client(addr)
        .with_session_chaining(true)
        .with_conversation_idle_ttl(Duration::from_millis(50));
    let history = vec![system_turn("sys"), user_turn("hello")];
    client
        .request(&history, &settings(), &params())
        .await
        .expect("turn 1");
    tokio::time::sleep(Duration::from_millis(80)).await;
    client
        .request(&history, &settings(), &params())
        .await
        .expect("turn 2 after eviction");
    server.await.unwrap();
}

#[tokio::test]
async fn dropped_model_closes_the_socket_with_a_handshake() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let mut ws = accept_ws(&listener).await;
        let request = read_turn(&mut ws).await;
        send_completed_turn(&mut ws, "resp_1", &request).await;

        // The model is dropped after the turn completes; the next frame
        // must be a Close handshake, not a TCP reset.
        let frame = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("timed out waiting for the close on drop");
        match frame {
            Some(Ok(Message::Close(_))) => {}
            other => panic!("expected a Close frame on model drop, got {other:?}"),
        }
    });

    let client = ws_client(addr);
    client
        .request(&[user_turn("hello")], &settings(), &params())
        .await
        .expect("turn ok");
    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn long_turn_gets_a_full_idle_window_after_completion() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let mut ws = accept_ws(&listener).await;
        let first = read_turn(&mut ws).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        send_completed_turn(&mut ws, "resp_long", &first).await;
        let second = read_turn(&mut ws).await;
        assert_eq!(second.previous_response_id.as_deref(), Some("resp_long"));
        send_completed_turn(&mut ws, "resp_next", &second).await;
    });
    let client = ws_client(addr)
        .with_session_chaining(true)
        .with_conversation_idle_ttl(Duration::from_millis(100));
    let mut history = vec![user_turn("long")];
    let first = client
        .request(&history, &settings(), &params())
        .await
        .unwrap();
    history.extend([response_turn(first), user_turn("next")]);
    tokio::time::timeout(
        Duration::from_secs(5),
        client.request(&history, &settings(), &params()),
    )
    .await
    .unwrap()
    .unwrap();
    server.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_last_model_drops_close_the_socket() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let mut ws = accept_ws(&listener).await;
        let request = read_turn(&mut ws).await;
        send_completed_turn(&mut ws, "resp_drop", &request).await;
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(5), ws.next())
                .await
                .unwrap(),
            Some(Ok(Message::Close(_)))
        ));
    });
    let client = ws_client(addr);
    client
        .request(&[user_turn("drop")], &settings(), &params())
        .await
        .unwrap();
    let sibling = client.clone();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let other = barrier.clone();
    let a = tokio::task::spawn_blocking(move || {
        barrier.wait();
        drop(client);
    });
    let b = tokio::task::spawn_blocking(move || {
        other.wait();
        drop(sibling);
    });
    a.await.unwrap();
    b.await.unwrap();
    server.await.unwrap();
}

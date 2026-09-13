#![cfg(feature = "openai")]
use futures::StreamExt;
use serde_json::{Value, json};
use serdes_ai_core::{FinishReason, ModelResponseStreamEvent as Event, ModelSettings};
use serdes_ai_models::{Model, ModelError, ModelRequestParameters, openai::OpenAIResponsesModel};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
fn frame(value: Value) -> String {
    format!("data: {value}\r\n\r\n")
}
fn message(text: &str) -> Value {
    json!({"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":text}]})
}
fn terminal(status: &str, reason: Option<&str>, output: Value) -> String {
    frame(
        json!({"type":format!("response.{status}"),"response":{"id":"resp_1","status":status,"output":output,
        "incomplete_details":reason.map(|r| json!({"reason":r})),"usage":{"input_tokens":7,"output_tokens":2}}}),
    )
}
async fn native(body: String) -> Vec<Result<Event, ModelError>> {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = vec![0; 65536];
        socket.read(&mut request).await.unwrap();
        socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n").await.unwrap();
        // Force every UTF-8 and JSON byte boundary over real chunked HTTP.
        for byte in body.as_bytes() {
            if socket
                .write_all(&[b'1', b'\r', b'\n', *byte, b'\r', b'\n'])
                .await
                .is_err()
            {
                return;
            }
        }
        let _ = socket.write_all(b"0\r\n\r\n").await;
    });
    OpenAIResponsesModel::new("local", "test")
        .with_base_url(url)
        .request_stream(&[], &ModelSettings::new(), &ModelRequestParameters::new())
        .await
        .unwrap()
        .collect()
        .await
}
fn prefix() -> String {
    frame(json!({"type":"response.created","response":{"id":"resp_1"}}))
        + &frame(
            json!({"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","content":[]}}),
        )
        + &frame(
            json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"café"}),
        )
}
#[tokio::test]
async fn native_chunked_completed_and_incomplete() {
    for (status, reason, finish) in [
        ("completed", None, FinishReason::Stop),
        (
            "incomplete",
            Some("max_output_tokens"),
            FinishReason::Length,
        ),
        (
            "incomplete",
            Some("content_filter"),
            FinishReason::ContentFilter,
        ),
    ] {
        let events = native(prefix() + &terminal(status, reason, json!([message("café")]))).await;
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        let Some(Ok(Event::StreamComplete(last))) = events.last() else {
            panic!("missing terminal")
        };
        assert_eq!(last.finish_reason, finish);
        assert_eq!(last.input_tokens, Some(7));
        assert!(
            serde_json::to_string(
                &events
                    .iter()
                    .filter_map(|r| r.as_ref().ok())
                    .collect::<Vec<_>>()
            )
            .unwrap()
            .contains("café")
        );
    }
}
#[tokio::test]
async fn native_errors_keep_partials_and_never_complete() {
    for suffix in [
        String::new(),
        "data: {broken\n\n".into(),
        frame(
            json!({"type":"response.failed","response":{"id":"resp_1","error":{"code":"server_error"}}}),
        ),
        frame(json!({"type":"response.cancelled","response":{"id":"resp_1"}})),
    ] {
        let events = native(prefix() + &suffix).await;
        assert!(events.last().unwrap().is_err());
        assert!(events.iter().any(|e| matches!(e, Ok(Event::PartDelta(_)))));
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(Event::StreamComplete(_))))
        );
    }
}
#[tokio::test]
async fn native_reasoning_and_function_items() {
    let reasoning =
        json!({"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"cipher"});
    let function = json!({"type":"function_call","id":"fc_1","call_id":"call_1","name":"next","arguments":"{}"});
    let body = frame(json!({"type":"response.created","response":{"id":"resp_1"}}))
        + &frame(json!({"type":"response.output_item.added","output_index":0,"item":reasoning}))
        + &frame(json!({"type":"response.output_item.done","output_index":0,"item":reasoning}))
        + &frame(
            json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"next","arguments":""}}),
        )
        + &frame(
            json!({"type":"response.function_call_arguments.delta","output_index":1,"delta":"{"}),
        )
        + &frame(
            json!({"type":"response.function_call_arguments.delta","output_index":1,"delta":"}"}),
        )
        + &terminal("completed", None, json!([reasoning, function]));
    let events = native(body).await;
    assert!(events.iter().all(Result::is_ok), "{events:?}");
    assert!(
        matches!(events.last(), Some(Ok(Event::StreamComplete(e))) if e.finish_reason == FinishReason::ToolCall)
    );
    let mut signature = String::new();
    for event in events.into_iter().map(Result::unwrap) {
        if let Event::PartDelta(delta) = event {
            if let serdes_ai_core::ModelResponsePartDelta::Thinking(delta) = delta.delta {
                signature.push_str(delta.signature_delta.as_deref().unwrap_or_default());
            }
        }
    }
    assert_eq!(signature, "cipher");
}

#[tokio::test]
async fn terminal_metadata_empty_and_opaque_records() {
    for output in [
        json!([]),
        json!([{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"private query"}}]),
        json!([{"type":"message","id":"msg_1","content":[{"type":"refusal","refusal":"private refusal"}]}]),
    ] {
        let response = json!({"id":"resp_metadata","model":"actual-model","status":"completed","output":output,
            "usage":{"input_tokens":12,"output_tokens":4,"total_tokens":16,"input_tokens_details":{"cached_tokens":3},"output_tokens_details":{"reasoning_tokens":2}}});
        let events = native(frame(
            json!({"type":"response.completed","response":response}),
        ))
        .await;
        let Some(Ok(Event::StreamComplete(complete))) = events.last() else {
            panic!("{events:?}")
        };
        let metadata = complete.metadata.as_ref().unwrap();
        assert_eq!(metadata.response_id.as_deref(), Some("resp_metadata"));
        assert_eq!(
            metadata.details.as_ref().unwrap()["output_records"]
                .as_array()
                .unwrap()
                .len(),
            output.as_array().unwrap().len()
        );
        assert_eq!(
            complete.request_usage().unwrap().details.unwrap()["output_tokens_details"]["reasoning_tokens"],
            2
        );
        assert!(!format!("{complete:?}").contains("private"));
        let mut manager = serdes_ai_streaming::ModelResponsePartsManager::new();
        manager.handle_stream_complete(complete.clone());
        assert_eq!(
            manager.get_response().vendor_id.as_deref(),
            Some("resp_metadata")
        );
    }
}
#[tokio::test]
async fn interleaved_slots_fail_explicitly_not_concatenated() {
    let events = native(prefix() + &frame(json!({"type":"response.output_text.delta","output_index":0,"content_index":1,"delta":"second"}))).await;
    assert!(events.last().unwrap().is_err());
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Ok(Event::StreamComplete(_))))
    );
}

#[tokio::test]
async fn interleaved_message_slots_keep_native_order() {
    let body = frame(
        json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","content":[]}}),
    ) + &frame(
        json!({"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}),
    ) + &frame(
        json!({"type":"response.content_part.added","output_index":0,"content_index":1,"part":{"type":"output_text","text":"","annotations":[]}}),
    ) + &frame(
        json!({"type":"response.output_text.delta","output_index":0,"content_index":1,"delta":"second"}),
    ) + &frame(
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"first"}),
    ) + &terminal(
        "completed",
        None,
        json!([{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"first","annotations":[]},{"type":"output_text","text":"second","annotations":[]}]}]),
    );
    let events = native(body).await;
    let mut parts = Vec::new();
    for event in events {
        match event.unwrap() {
            Event::PartStart(start) => {
                assert_eq!(start.index, parts.len());
                parts.push(start.part);
            }
            Event::PartDelta(delta) => {
                assert!(delta.delta.apply(&mut parts[delta.index]));
            }
            _ => {}
        }
    }
    assert!(
        matches!(&parts[0], serdes_ai_core::ModelResponsePart::Text(t) if t.content == "first")
    );
    assert!(
        matches!(&parts[1], serdes_ai_core::ModelResponsePart::Text(t) if t.content == "second")
    );
}

#[tokio::test]
async fn cross_item_out_of_order_fails_after_preserving_emitted_partial() {
    let body = frame(
        json!({"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"later","content":[]}}),
    ) + &frame(
        json!({"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"preserved"}),
    ) + &frame(
        json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"f","call_id":"c","name":"never","arguments":"{}"}}),
    );
    let events = native(body).await;
    assert!(events.iter().any(|e| matches!(e, Ok(Event::PartDelta(_)))));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Err(serdes_ai_models::ModelError::InvalidResponse(_))))
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Ok(Event::StreamComplete(_))))
    );
    assert!(!events.iter().any(|e| matches!(e, Ok(Event::PartStart(p)) if matches!(p.part, serdes_ai_core::ModelResponsePart::ToolCall(_)))));
}

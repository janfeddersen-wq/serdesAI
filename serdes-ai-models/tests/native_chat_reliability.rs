#![cfg(feature = "openai")]
use futures::StreamExt;
use serde_json::{Value, json};
use serdes_ai_core::{FinishReason, ModelRequest, ModelResponseStreamEvent, ModelSettings};
use serdes_ai_models::error::ModelError;
use serdes_ai_models::openai::{OpenAIChatModel, stream::OpenAIStreamParser};
use serdes_ai_models::{Model, ModelRequestParameters};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

fn frame(delta: Value, finish: Value) -> String {
    format!(
        "data:{}\r\n\r\n",
        json!({"id":"test", "object":"chat.completion.chunk", "created":1,
        "model":"alias", "choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
    )
}

async fn native(
    body: String,
    model: &str,
    override_key: Option<bool>,
    usage: bool,
) -> (Vec<Result<ModelResponseStreamEvent, ModelError>>, Value) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;
    let mut model = OpenAIChatModel::new(model, "local-test").with_base_url(server.uri());
    if let Some(enabled) = override_key {
        model = model.with_max_completion_tokens(enabled);
    }
    let params = ModelRequestParameters {
        stream_usage: usage,
        ..Default::default()
    };
    let events = model
        .request_stream(
            &[ModelRequest::new()],
            &ModelSettings::new().max_tokens(17),
            &params,
        )
        .await
        .unwrap()
        .collect()
        .await;
    let requests = server.received_requests().await.unwrap();
    (events, serde_json::from_slice(&requests[0].body).unwrap())
}

#[tokio::test]
async fn native_eof_preserves_partial_tools_but_errors() {
    let body = frame(
        json!({"content":"partial", "tool_calls":[{"index":0,"id":"call-1",
        "type":"function","function":{"name":"danger","arguments":"{\"x\":"}}]}),
        Value::Null,
    );
    let (events, _) = native(body, "gpt-4o", None, false).await;
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, Ok(ModelResponseStreamEvent::PartStart(_))))
            .count(),
        2
    );
    assert!(matches!(
        events.last(),
        Some(Err(ModelError::IncompleteStream(_)))
    ));
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Ok(ModelResponseStreamEvent::StreamComplete(_))))
    );
}

#[tokio::test]
async fn native_length_usage_and_cap_keys() {
    for (name, override_key, completion, usage) in [
        ("o3", None, true, true),
        ("gpt-5-custom", None, true, false),
        ("alias", Some(true), true, true),
        ("o3", Some(false), false, false),
        ("gpt-4o", None, false, true),
    ] {
        let body = frame(json!({"content":"valid partial"}), json!("length"))
            + &format!(
                "data: {}\n\n",
                json!({"id":"test","object":"chat.completion.chunk","created":1,
                "model":name,"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}})
            )
            + "data: [DONE]\n\ndata: [DONE]\n\n";
        let (events, request) = native(body, name, override_key, usage).await;
        assert!(events.iter().all(Result::is_ok));
        assert_eq!(
            request[if completion {
                "max_completion_tokens"
            } else {
                "max_tokens"
            }],
            17
        );
        assert!(
            request
                .get(if completion {
                    "max_tokens"
                } else {
                    "max_completion_tokens"
                })
                .is_none()
        );
        assert_eq!(request["stream_options"]["include_usage"], usage);
        let terminals: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Ok(ModelResponseStreamEvent::StreamComplete(event)) => Some(event),
                _ => None,
            })
            .collect();
        assert_eq!(terminals.len(), 1);
        assert_eq!(terminals[0].finish_reason, FinishReason::Length);
        assert_eq!(terminals[0].input_tokens, Some(12));
    }
}

#[tokio::test]
async fn native_provider_and_malformed_errors_are_terminal() {
    for data in [
        "{broken",
        r#"{"error":{"message":"failed","code":"overloaded"}}"#,
    ] {
        let (events, _) = native(
            frame(json!({"content":"kept"}), Value::Null)
                + &format!("data: {data}\n\ndata: [DONE]\n\n"),
            "alias",
            None,
            false,
        )
        .await;
        assert!(matches!(
            events.first(),
            Some(Ok(ModelResponseStreamEvent::PartStart(_)))
        ));
        assert!(events.last().unwrap().is_err());
        assert_eq!(events.len(), 2);
    }
}

#[tokio::test]
async fn every_byte_boundary_preserves_utf8_and_finish_in_same_frame() {
    let body = frame(json!({"content":" café"}), json!("stop")) + "data: [DONE]";
    for split in 0..=body.len() {
        let bytes = body.as_bytes();
        let input = futures::stream::iter(vec![
            Ok(bytes::Bytes::copy_from_slice(&bytes[..split])),
            Ok(bytes::Bytes::copy_from_slice(&bytes[split..])),
        ]);
        let events: Vec<_> = OpenAIStreamParser::new(input).collect().await;
        assert!(events.iter().all(Result::is_ok), "split {split}");
        assert_eq!(events.len(), 3);
    }
}

#[tokio::test]
async fn finish_frame_eof_requires_explicit_endpoint_opt_in() {
    let body = frame(json!({"content":"complete"}), json!("stop"));
    for enabled in [false, true] {
        let input = futures::stream::iter(vec![Ok(bytes::Bytes::from(body.clone()))]);
        let events: Vec<_> = OpenAIStreamParser::new(input)
            .with_finish_reason_terminal(enabled)
            .collect()
            .await;
        assert_eq!(events.last().unwrap().is_ok(), enabled);
    }
}

#[tokio::test]
async fn dropping_stream_cancels_without_synthesizing_success() {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    struct Pending {
        dropped: Arc<AtomicBool>,
    }
    impl futures::Stream for Pending {
        type Item = Result<bytes::Bytes, reqwest::Error>;
        fn poll_next(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            std::task::Poll::Pending
        }
    }
    impl Drop for Pending {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }
    let dropped = Arc::new(AtomicBool::new(false));
    let parser = OpenAIStreamParser::new(Pending {
        dropped: dropped.clone(),
    });
    drop(parser);
    assert!(dropped.load(Ordering::SeqCst));
}

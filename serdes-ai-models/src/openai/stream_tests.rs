use super::*;
use futures::StreamExt;
use futures::stream;
use serdes_ai_core::ModelResponsePartDelta;

fn make_chunk_bytes(data: &str) -> Bytes {
    Bytes::from(format!("data: {}\n\n", data))
}

#[tokio::test]
async fn test_parse_text_chunk() {
    // Test multi-chunk text streaming
    let chunk1 = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"}}]}"#;
    let chunk2 = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"}}]}"#;
    let chunk3 = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" World"}}]}"#;
    let bytes = vec![
        Ok(make_chunk_bytes(chunk1)),
        Ok(make_chunk_bytes(chunk2)),
        Ok(make_chunk_bytes(chunk3)),
    ];
    let stream = stream::iter(
        bytes
            .into_iter()
            .chain(std::iter::once(Ok(make_chunk_bytes("[DONE]")))),
    );
    let mut parser = OpenAIStreamParser::new(stream);

    // Collect all events
    let mut events = Vec::new();
    while let Some(result) = parser.next().await {
        events.push(result.unwrap());
    }

    // Should have: PartStart(Hello), PartDelta( World)
    assert!(
        events.len() >= 2,
        "Expected at least 2 events, got {}: {:?}",
        events.len(),
        events
    );

    // First should be PartStart with "Hello" content
    if let ModelResponseStreamEvent::PartStart(start) = &events[0] {
        if let ModelResponsePart::Text(text) = &start.part {
            assert_eq!(text.content, "Hello", "PartStart should contain 'Hello'");
        } else {
            panic!("Expected Text part in PartStart, got {:?}", start.part);
        }
    } else {
        panic!("First event should be PartStart, got {:?}", events[0]);
    }

    // Second should be PartDelta with " World"
    if let ModelResponseStreamEvent::PartDelta(delta) = &events[1] {
        if let ModelResponsePartDelta::Text(text) = &delta.delta {
            assert_eq!(
                text.content_delta, " World",
                "Delta should contain ' World'"
            );
        } else {
            panic!("Expected Text delta, got {:?}", delta.delta);
        }
    } else {
        panic!("Second event should be PartDelta, got {:?}", events[1]);
    }
}

/// [DONE] without a usage chunk emits exactly one terminal event with the
/// default finish reason and all token fields None; nothing follows it.
#[tokio::test]
async fn test_parse_done() {
    let bytes = vec![Ok(Bytes::from("data: [DONE]\n\n"))];
    let stream = stream::iter(
        bytes
            .into_iter()
            .chain(std::iter::once(Ok(make_chunk_bytes("[DONE]")))),
    );
    let mut parser = OpenAIStreamParser::new(stream);

    let event = parser.next().await.unwrap().unwrap();
    match event {
        ModelResponseStreamEvent::StreamComplete(complete) => {
            assert_eq!(complete.finish_reason, FinishReason::Stop);
            assert_eq!(complete.input_tokens, None);
            assert_eq!(complete.output_tokens, None);
            assert_eq!(complete.cache_creation_tokens, None);
            assert_eq!(complete.cache_read_tokens, None);
        }
        other => panic!("expected StreamComplete, got {:?}", other),
    }

    // The terminal event is the final event of the stream.
    assert!(parser.next().await.is_none());
}

#[tokio::test]
async fn test_eof_without_done_emits_no_terminal_event() {
    let chunk = r#"{"id":"123","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"}}]}"#;
    let mut parser = OpenAIStreamParser::new(stream::iter(vec![Ok(make_chunk_bytes(chunk))]));
    assert!(matches!(
        parser.next().await,
        Some(Ok(ModelResponseStreamEvent::PartStart(_)))
    ));
    assert!(matches!(
        parser.next().await,
        Some(Err(ModelError::IncompleteStream(_)))
    ));
    assert!(parser.next().await.is_none());
}

/// A transport error mid-stream ends the stream with the error and never
/// emits the terminal event afterwards.
#[tokio::test]
async fn stream_error_suppresses_terminal_event() {
    // reqwest::Error has no public constructor; a refused connection to a
    // closed loopback port is the cheapest real one to obtain.
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let err = client
        .get("http://127.0.0.1:1")
        .send()
        .await
        .expect_err("connection to closed loopback port must fail");

    let text_chunk = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"}}]}"#;
    let bytes = vec![Ok(make_chunk_bytes(text_chunk)), Err(err)];
    let stream = stream::iter(
        bytes
            .into_iter()
            .chain(std::iter::once(Ok(make_chunk_bytes("[DONE]")))),
    );
    let mut parser = OpenAIStreamParser::new(stream);

    let mut events = Vec::new();
    while let Some(result) = parser.next().await {
        events.push(result);
    }

    assert_eq!(
        events.len(),
        2,
        "expected only the part event and the error: {:?}",
        events
    );
    assert!(matches!(
        events[0],
        Ok(ModelResponseStreamEvent::PartStart(_))
    ));
    assert!(events[1].is_err());
}

#[tokio::test]
async fn test_parse_finish_reason() {
    let chunk = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
    let bytes = vec![Ok(make_chunk_bytes(chunk))];
    let stream = stream::iter(
        bytes
            .into_iter()
            .chain(std::iter::once(Ok(make_chunk_bytes("[DONE]")))),
    );
    let mut parser = OpenAIStreamParser::new(stream);

    // A finish frame plus [DONE] emits exactly one terminal event.
    let event = parser.next().await;
    assert!(matches!(
        event,
        Some(Ok(ModelResponseStreamEvent::StreamComplete(_)))
    ));
}

/// The include_usage chunk is buffered and produces no part events; its
/// counts are carried on the terminal event at [DONE].
#[tokio::test]
async fn usage_chunk_then_done_populates_terminal_event() {
    let text_chunk = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"}}]}"#;
    let finish_chunk = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
    let usage_chunk = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":7}}}"#;

    let bytes = vec![
        Ok(make_chunk_bytes(text_chunk)),
        Ok(make_chunk_bytes(finish_chunk)),
        Ok(make_chunk_bytes(usage_chunk)),
        Ok(Bytes::from("data: [DONE]\n\n")),
    ];
    let stream = stream::iter(
        bytes
            .into_iter()
            .chain(std::iter::once(Ok(make_chunk_bytes("[DONE]")))),
    );
    let mut parser = OpenAIStreamParser::new(stream);

    let mut events = Vec::new();
    while let Some(result) = parser.next().await {
        events.push(result.unwrap());
    }

    // The usage chunk is ignored for part events: only PartStart and
    // PartEnd precede the terminal event.
    assert_eq!(
        events.len(),
        3,
        "expected PartStart, PartEnd, StreamComplete: {:?}",
        events
    );
    assert!(matches!(events[0], ModelResponseStreamEvent::PartStart(_)));
    assert!(matches!(events[1], ModelResponseStreamEvent::PartEnd(_)));

    match events.last() {
        Some(ModelResponseStreamEvent::StreamComplete(complete)) => {
            assert_eq!(complete.finish_reason, FinishReason::Stop);
            assert_eq!(complete.input_tokens, Some(10));
            assert_eq!(complete.output_tokens, Some(5));
            assert_eq!(complete.cache_creation_tokens, None);
            assert_eq!(complete.cache_read_tokens, Some(7));
        }
        other => panic!("expected terminal StreamComplete, got {:?}", other),
    }
}

/// Finish reason strings map onto the terminal event; unknown reasons
/// default to Stop.
#[tokio::test]
async fn finish_reason_maps_onto_terminal_event() {
    let cases = [
        ("stop", FinishReason::Stop),
        ("length", FinishReason::Length),
        ("content_filter", FinishReason::ContentFilter),
        ("tool_calls", FinishReason::ToolCall),
        ("function_call", FinishReason::Stop),
        ("unknown_reason", FinishReason::Stop),
    ];

    for (wire_reason, expected) in cases {
        let finish_chunk = format!(
            r#"{{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{{"index":0,"delta":{{}},"finish_reason":"{wire_reason}"}}]}}"#
        );
        let bytes = vec![
            Ok(make_chunk_bytes(&finish_chunk)),
            Ok(Bytes::from("data: [DONE]\n\n")),
        ];
        let stream = stream::iter(
            bytes
                .into_iter()
                .chain(std::iter::once(Ok(make_chunk_bytes("[DONE]")))),
        );
        let mut parser = OpenAIStreamParser::new(stream);

        let mut terminals = 0;
        let mut finish_reason = None;
        while let Some(result) = parser.next().await {
            if let ModelResponseStreamEvent::StreamComplete(complete) = result.unwrap() {
                terminals += 1;
                finish_reason = Some(complete.finish_reason);
            }
        }

        assert_eq!(
            terminals, 1,
            "expected one terminal event for {wire_reason}"
        );
        assert_eq!(
            finish_reason,
            Some(expected),
            "finish reason mapping for {wire_reason}"
        );
    }
}

#[tokio::test]
async fn test_parse_tool_call() {
    let chunk1 = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search","arguments":""}}]}}]}"#;
    let chunk2 = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]}}]}"#;
    let chunk3 = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"test\"}"}}]}}]}"#;

    let bytes = vec![
        Ok(make_chunk_bytes(chunk1)),
        Ok(make_chunk_bytes(chunk2)),
        Ok(make_chunk_bytes(chunk3)),
    ];
    let stream = stream::iter(
        bytes
            .into_iter()
            .chain(std::iter::once(Ok(make_chunk_bytes("[DONE]")))),
    );
    let mut parser = OpenAIStreamParser::new(stream);

    // First event should be PartStart for tool call
    let event = parser.next().await.unwrap().unwrap();
    assert!(matches!(event, ModelResponseStreamEvent::PartStart(_)));

    // Subsequent events should be deltas
    let event = parser.next().await.unwrap().unwrap();
    assert!(matches!(event, ModelResponseStreamEvent::PartDelta(_)));
}

/// Regression test: when finish_reason is received with multiple open parts
/// (text + tool calls), ALL PartEnd events must be emitted, not just the
/// first. The terminal event follows the queued part ends and is emitted
/// exactly once, last.
#[tokio::test]
async fn test_multiple_part_ends_on_finish() {
    // Scenario: text part starts, then 2 tool calls start, then finish_reason
    // We should get 3 PartEnd events (one for text, two for tool calls)
    let text_chunk = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"}}]}"#;
    let tool1_chunk = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search","arguments":"{\"q\":\"test\"}"}}]}}]}"#;
    let tool2_chunk = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"lookup","arguments":"{\"id\":1}"}}]}}]}"#;
    let finish_chunk = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#;
    let usage_chunk = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":7}}}"#;

    let bytes = vec![
        Ok(make_chunk_bytes(text_chunk)),
        Ok(make_chunk_bytes(tool1_chunk)),
        Ok(make_chunk_bytes(tool2_chunk)),
        Ok(make_chunk_bytes(finish_chunk)),
        Ok(make_chunk_bytes(usage_chunk)),
        Ok(Bytes::from("data: [DONE]\n\n")),
    ];
    let stream = stream::iter(
        bytes
            .into_iter()
            .chain(std::iter::once(Ok(make_chunk_bytes("[DONE]")))),
    );
    let mut parser = OpenAIStreamParser::new(stream);

    // Collect all events
    let mut events = Vec::new();
    while let Some(result) = parser.next().await {
        events.push(result.unwrap());
    }

    // Should have:
    // - 1 PartStart for text (index 0)
    // - 1 PartStart for tool call 1 (index 1)
    // - 1 PartStart for tool call 2 (index 2)
    // - 3 PartEnd events (for indices 0, 1, 2)
    // - 1 StreamComplete terminal event
    let part_starts: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, ModelResponseStreamEvent::PartStart(_)))
        .collect();
    let part_ends: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, ModelResponseStreamEvent::PartEnd(_)))
        .collect();

    assert_eq!(
        part_starts.len(),
        3,
        "Expected 3 PartStart events, got {}: {:?}",
        part_starts.len(),
        part_starts
    );
    assert_eq!(
        part_ends.len(),
        3,
        "Expected 3 PartEnd events (regression: bug caused only 1 to be emitted), got {}: {:?}",
        part_ends.len(),
        part_ends
    );

    // Verify all part indices are closed
    let mut closed_indices: Vec<usize> = part_ends
        .iter()
        .filter_map(|e| {
            if let ModelResponseStreamEvent::PartEnd(end) = e {
                Some(end.index)
            } else {
                None
            }
        })
        .collect();
    closed_indices.sort();
    assert_eq!(
        closed_indices,
        vec![0, 1, 2],
        "All part indices should be closed"
    );

    // Exactly one terminal event, carrying the buffered usage, emitted
    // after every queued part end.
    let terminals: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, ModelResponseStreamEvent::StreamComplete(_)))
        .collect();
    assert_eq!(terminals.len(), 1, "expected exactly one terminal event");
    assert!(
        matches!(
            events.last(),
            Some(ModelResponseStreamEvent::StreamComplete(_))
        ),
        "terminal event must be emitted last, got {:?}",
        events.last()
    );
    if let Some(ModelResponseStreamEvent::StreamComplete(complete)) = events.last() {
        assert_eq!(complete.finish_reason, FinishReason::ToolCall);
        assert_eq!(complete.input_tokens, Some(10));
        assert_eq!(complete.output_tokens, Some(5));
        assert_eq!(complete.cache_creation_tokens, None);
        assert_eq!(complete.cache_read_tokens, Some(7));
    }
}

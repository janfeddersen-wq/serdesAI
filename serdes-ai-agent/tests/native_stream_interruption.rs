use futures::StreamExt;
use serdes_ai_agent::AgentBuilder;
use serdes_ai_models::openai::OpenAIChatModel;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

#[tokio::test]
async fn interrupted_native_tool_call_is_never_dispatched() {
    let server = MockServer::start().await;
    // Even valid JSON arguments must not run before terminal evidence.
    let body = format!(
        "data: {}\n\n",
        serde_json::json!({"id":"test","object":"chat.completion.chunk",
        "created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{
            "index":0,"id":"call_1","type":"function","function":{"name":"danger","arguments":"{}"}
        }]}}]})
    );
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let agent = AgentBuilder::<(), String>::new(
        OpenAIChatModel::new("gpt-4o", "local").with_base_url(server.uri()),
    )
    .tool_fn("danger", "must not run", move |_, _: serde_json::Value| {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(serdes_ai_tools::ToolReturn::text("oops"))
    })
    .build();
    let mut stream = agent.run_stream("test", ()).await.unwrap();
    let mut failed = false;
    while let Some(event) = stream.next().await {
        failed |= event.is_err();
    }
    assert!(failed);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn token_limit_is_terminal_and_does_not_dispatch_partial_tools() {
    let server = MockServer::start().await;
    let body = format!(
        "data: {}\n\ndata: [DONE]\n\n",
        serde_json::json!({"id":"test","object":"chat.completion.chunk",
        "created":1,"model":"gpt-4o","choices":[{"index":0,"finish_reason":"length","delta":{
            "content":"valid partial", "tool_calls":[{"index":0,"id":"call_1","type":"function",
                "function":{"name":"danger","arguments":"{\"unfinished\":"}}]}}]})
    );
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let agent = AgentBuilder::<(), String>::new(
        OpenAIChatModel::new("gpt-4o", "local").with_base_url(server.uri()),
    )
    .tool_fn("danger", "must not run", move |_, _: serde_json::Value| {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(serdes_ai_tools::ToolReturn::text("oops"))
    })
    .build();
    let mut stream = agent.run_stream("test", ()).await.unwrap();
    while let Some(event) = stream.next().await {
        assert!(event.is_ok());
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

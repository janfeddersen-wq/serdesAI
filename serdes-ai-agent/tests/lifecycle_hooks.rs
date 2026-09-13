use async_trait::async_trait;
use futures::{StreamExt, stream};
use serdes_ai_agent::*;
use serdes_ai_core::{
    FinishReason, ModelRequest, ModelResponse, ModelResponsePart, ModelResponseStreamEvent as Event,
};
use serdes_ai_models::FunctionModel;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone)]
struct Sink {
    records: Arc<Mutex<Vec<AgentCheckpoint>>>,
    fail: Option<CheckpointBoundary>,
}
#[async_trait]
impl CheckpointSink for Sink {
    async fn save(&self, checkpoint: &AgentCheckpoint) -> Result<(), String> {
        self.records.lock().unwrap().push(checkpoint.clone());
        if self.fail.as_ref() == Some(&checkpoint.boundary) {
            Err("durability unavailable".into())
        } else {
            Ok(())
        }
    }
}
struct Compact(Arc<AtomicUsize>);
#[async_trait]
impl HistoryProcessor<()> for Compact {
    async fn process(&self, _: &RunContext<()>, messages: Vec<ModelRequest>) -> Vec<ModelRequest> {
        self.0.fetch_add(1, Ordering::SeqCst);
        // This fixture has no tools; retain only the current user request.
        messages.into_iter().rev().take(1).collect()
    }
}
struct FailingPolicy;
#[async_trait]
impl ContextPolicy<()> for FailingPolicy {
    async fn prepare(
        &self,
        input: ContextPolicyInput<'_, ()>,
        _: Vec<ModelRequest>,
    ) -> Result<Vec<ModelRequest>, String> {
        assert_eq!(input.settings, &input.context.model_settings);
        Err("summary model unavailable".into())
    }
}
fn sink(fail: Option<CheckpointBoundary>) -> Sink {
    Sink {
        records: Default::default(),
        fail,
    }
}
fn model() -> FunctionModel {
    FunctionModel::with_both(
        |_, _| ModelResponse::text("ok"),
        |_, _| {
            Box::pin(stream::iter(vec![
                Ok(Event::part_start(0, ModelResponsePart::text("ok"))),
                Ok(Event::StreamComplete(
                    serdes_ai_core::messages::StreamCompleteEvent::new(FinishReason::Stop),
                )),
            ]))
        },
    )
}
#[tokio::test]
async fn canonical_processor_and_checkpoint_parity() {
    for streaming in [false, true] {
        let sink = sink(None);
        let calls = Arc::new(AtomicUsize::new(0));
        let agent = AgentBuilder::<(), String>::new(model())
            .system_prompt("discard me")
            .history_processor(Compact(calls.clone()))
            .checkpoint_sink(sink.clone())
            .build();
        if streaming {
            let mut stream = agent.run_stream("test", ()).await.unwrap();
            while let Some(event) = stream.next().await {
                event.unwrap();
            }
        } else {
            agent.run("test", ()).await.unwrap();
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let records = sink.records.lock().unwrap();
        assert_eq!(records[0].boundary, CheckpointBoundary::BeforeRequest);
        assert_eq!(records[0].messages.len(), 1);
        assert_eq!(
            records.last().unwrap().boundary,
            CheckpointBoundary::Terminal
        );
        let json = serde_json::to_string(records.last().unwrap()).unwrap();
        let restored = AgentCheckpoint::from_json(&json).unwrap();
        assert!(restored.response.is_some());
        assert!(!format!("{restored:?}").contains("discard me"));
    }
}
#[tokio::test]
async fn summary_failure_is_not_silent_truncation() {
    for failure in [
        ContextFailurePolicy::Stop,
        ContextFailurePolicy::KeepHistory,
    ] {
        let sink = sink(None);
        let agent = AgentBuilder::<(), String>::new(model())
            .context_policy(FailingPolicy)
            .context_failure_policy(failure)
            .checkpoint_sink(sink.clone())
            .build();
        let mut stream = agent.run_stream("test", ()).await.unwrap();
        let mut failed = false;
        while let Some(event) = stream.next().await {
            failed |= matches!(event, Err(AgentRunError::ContextPolicy(_)));
        }
        assert_eq!(failed, matches!(failure, ContextFailurePolicy::Stop));
        assert_eq!(
            sink.records.lock().unwrap().last().unwrap().boundary == CheckpointBoundary::Failed,
            failed
        );
    }
}
#[tokio::test]
async fn sink_failure_before_tool_dispatch_stops_side_effects() {
    let sink = sink(Some(CheckpointBoundary::AfterResponse));
    let calls = Arc::new(AtomicUsize::new(0));
    let count = calls.clone();
    let model = FunctionModel::with_stream(|_, _| {
        Box::pin(stream::iter(vec![
            Ok(Event::part_start(
                0,
                ModelResponsePart::tool_call("danger", serde_json::json!({})),
            )),
            Ok(Event::StreamComplete(
                serdes_ai_core::messages::StreamCompleteEvent::new(FinishReason::ToolCall),
            )),
        ]))
    });
    let agent = AgentBuilder::<(), String>::new(model)
        .checkpoint_sink(sink.clone())
        .tool_fn("danger", "danger", move |_, _: serde_json::Value| {
            count.fetch_add(1, Ordering::SeqCst);
            Ok(serdes_ai_tools::ToolReturn::text("done"))
        })
        .build();
    let mut stream = agent.run_stream("test", ()).await.unwrap();
    let mut failed = false;
    while let Some(event) = stream.next().await {
        failed |= matches!(event, Err(AgentRunError::Checkpoint(_)));
    }
    assert!(failed);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
#[tokio::test]
async fn cancellation_during_stream_saves_partial_native_parts() {
    let sink = sink(None);
    let model = FunctionModel::with_stream(|_, _| {
        Box::pin(
            stream::iter(vec![Ok(Event::part_start(
                0,
                ModelResponsePart::Thinking(
                    serdes_ai_core::ThinkingPart::new("private").with_signature("secret"),
                ),
            ))])
            .chain(stream::pending()),
        )
    });
    let agent = AgentBuilder::<(), String>::new(model)
        .checkpoint_sink(sink.clone())
        .build();
    let token = CancellationToken::new();
    let mut stream = AgentStream::new_with_cancel(
        &agent,
        "test".into(),
        (),
        RunOptions::default(),
        token.clone(),
    )
    .await
    .unwrap();
    while let Some(event) = stream.next().await {
        if matches!(event, Ok(AgentStreamEvent::ThinkingDelta { .. })) {
            token.cancel();
            break;
        }
    }
    while let Some(event) = stream.next().await {
        if event.is_err() {
            break;
        }
    }
    let records = sink.records.lock().unwrap();
    let last = records.last().unwrap();
    assert_eq!(last.boundary, CheckpointBoundary::Cancelled);
    assert!(serde_json::to_string(last).unwrap().contains("secret"));
    assert!(!format!("{last:?}").contains("secret"));
    assert!(last.response.as_ref().unwrap().usage.is_none());
}

struct KeepRecent(Arc<AtomicUsize>);
#[async_trait]
impl HistoryProcessor<()> for KeepRecent {
    async fn process(
        &self,
        _: &RunContext<()>,
        mut messages: Vec<ModelRequest>,
    ) -> Vec<ModelRequest> {
        self.0.fetch_add(1, Ordering::SeqCst);
        if messages.len() > 3 {
            messages.drain(..messages.len() - 2);
        }
        messages
    }
}
#[tokio::test]
async fn processors_run_for_each_tool_iteration_and_active_history_shrinks() {
    let requests = Arc::new(AtomicUsize::new(0));
    let count = requests.clone();
    let model = FunctionModel::with_stream(move |_, _| {
        let step = count.fetch_add(1, Ordering::SeqCst);
        let (part, reason) = if step < 4 {
            (
                ModelResponsePart::ToolCall(
                    serdes_ai_core::ToolCallPart::new(
                        "next",
                        serdes_ai_core::messages::ToolCallArgs::string("{}"),
                    )
                    .with_tool_call_id(format!("call_{step}")),
                ),
                FinishReason::ToolCall,
            )
        } else {
            (ModelResponsePart::text("done"), FinishReason::Stop)
        };
        Box::pin(stream::iter(vec![
            Ok(Event::part_start(0, part)),
            Ok(Event::StreamComplete(
                serdes_ai_core::messages::StreamCompleteEvent::new(reason),
            )),
        ]))
    });
    let calls = Arc::new(AtomicUsize::new(0));
    let sink = sink(None);
    let agent = AgentBuilder::<(), String>::new(model)
        .history_processor(KeepRecent(calls.clone()))
        .checkpoint_sink(sink.clone())
        .tool_fn("next", "next", |_, _: serde_json::Value| {
            Ok(serdes_ai_tools::ToolReturn::text("ok"))
        })
        .build();
    let mut stream = agent.run_stream("start", ()).await.unwrap();
    while let Some(event) = stream.next().await {
        event.unwrap();
    }
    assert_eq!(calls.load(Ordering::SeqCst), 5);
    let records = sink.records.lock().unwrap();
    assert!(
        records
            .iter()
            .filter(|r| r.boundary == CheckpointBoundary::BeforeRequest)
            .all(|r| r.messages.len() <= 3)
    );
    assert_eq!(
        records
            .iter()
            .filter(|r| r.boundary == CheckpointBoundary::AfterTools)
            .count(),
        4
    );
}

#[tokio::test]
async fn provider_failure_snapshot_preserves_partial_arguments() {
    let sink = sink(None);
    let model = FunctionModel::with_stream(|_, _| {
        Box::pin(stream::iter(vec![
            Ok(Event::part_start(
                0,
                ModelResponsePart::ToolCall(serdes_ai_core::ToolCallPart::new(
                    "unfinished",
                    serdes_ai_core::messages::ToolCallArgs::string("{\"partial\":"),
                )),
            )),
            Err(serdes_ai_models::ModelError::incomplete_stream(
                "fixture EOF",
            )),
        ]))
    });
    let agent = AgentBuilder::<(), String>::new(model)
        .checkpoint_sink(sink.clone())
        .build();
    let mut stream = agent.run_stream("test", ()).await.unwrap();
    while let Some(event) = stream.next().await {
        if event.is_err() {
            break;
        }
    }
    let records = sink.records.lock().unwrap();
    let last = records.last().unwrap();
    assert_eq!(
        last.boundary,
        CheckpointBoundary::ModelFailed(serdes_ai_core::ModelFailureKind::IncompleteStream)
    );
    assert!(serde_json::to_string(last).unwrap().contains("partial"));
}

#[tokio::test]
async fn cancellation_interrupts_request_establishment() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_delay(std::time::Duration::from_secs(30)))
        .mount(&server)
        .await;
    let sink = sink(None);
    let agent = AgentBuilder::<(), String>::new(
        serdes_ai_models::openai::OpenAIChatModel::new("local", "test").with_base_url(server.uri()),
    )
    .checkpoint_sink(sink.clone())
    .build();
    let token = CancellationToken::new();
    let mut stream = AgentStream::new_with_cancel(
        &agent,
        "test".into(),
        (),
        RunOptions::default(),
        token.clone(),
    )
    .await
    .unwrap();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        token.cancel();
    });
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while let Some(event) = stream.next().await {
            if event.is_err() {
                break;
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(
        sink.records.lock().unwrap().last().unwrap().boundary,
        CheckpointBoundary::Cancelled
    );
}

#[tokio::test]
async fn cancellation_drops_pending_tool_and_never_records_success() {
    let token = CancellationToken::new();
    let cancel = token.clone();
    let dropped = Arc::new(AtomicUsize::new(0));
    let cleanup = dropped.clone();
    struct Guard(Arc<AtomicUsize>);
    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let sink = sink(None);
    let model = FunctionModel::with_stream(|_, _| {
        Box::pin(stream::iter(vec![
            Ok(Event::part_start(
                0,
                ModelResponsePart::tool_call("wait", serde_json::json!({})),
            )),
            Ok(Event::StreamComplete(
                serdes_ai_core::messages::StreamCompleteEvent::new(FinishReason::ToolCall),
            )),
        ]))
    });
    let agent = AgentBuilder::<(), String>::new(model)
        .checkpoint_sink(sink.clone())
        .tool_fn_async("wait", "wait", move |_, _: serde_json::Value| {
            let cleanup = cleanup.clone();
            let cancel = cancel.clone();
            async move {
                let _guard = Guard(cleanup);
                cancel.cancel();
                std::future::pending().await
            }
        })
        .build();
    let mut stream =
        AgentStream::new_with_cancel(&agent, "test".into(), (), RunOptions::default(), token)
            .await
            .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while let Some(event) = stream.next().await {
            assert!(!matches!(
                event,
                Ok(AgentStreamEvent::ToolExecuted { success: true, .. })
            ));
            if event.is_err() {
                break;
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    let records = sink.records.lock().unwrap();
    assert_eq!(
        records.last().unwrap().boundary,
        CheckpointBoundary::Cancelled
    );
    assert!(
        !records
            .iter()
            .any(|r| r.boundary == CheckpointBoundary::AfterTools)
    );
}

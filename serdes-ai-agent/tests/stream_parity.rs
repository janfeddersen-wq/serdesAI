use async_trait::async_trait;
use futures::{StreamExt, stream};
use serdes_ai_agent::*;
use serdes_ai_core::{FinishReason, ModelResponsePart, ModelResponseStreamEvent as Event};
use serdes_ai_models::FunctionModel;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;
use tokio::sync::Notify;
#[derive(Clone, Default)]
struct Sink {
    records: Arc<Mutex<Vec<AgentCheckpoint>>>,
    notify: Arc<Notify>,
}
#[async_trait]
impl CheckpointSink for Sink {
    async fn save(&self, cp: &AgentCheckpoint) -> Result<(), String> {
        self.records.lock().unwrap().push(cp.clone());
        self.notify.notify_one();
        Ok(())
    }
}
impl Sink {
    async fn terminal(&self) -> CheckpointBoundary {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let waiting = self.notify.notified();
                if let Some(cp) = self.records.lock().unwrap().iter().find(|c| {
                    matches!(
                        c.boundary,
                        CheckpointBoundary::Terminal
                            | CheckpointBoundary::Cancelled
                            | CheckpointBoundary::ConsumerDetached
                            | CheckpointBoundary::ValidationFailed
                    )
                }) {
                    return cp.boundary.clone();
                }
                waiting.await;
            }
        })
        .await
        .unwrap()
    }
}
fn text(value: &str) -> Vec<Result<Event, serdes_ai_models::ModelError>> {
    vec![
        Ok(Event::part_start(0, ModelResponsePart::text(value))),
        Ok(Event::StreamComplete(
            serdes_ai_core::messages::StreamCompleteEvent::new(FinishReason::Stop),
        )),
    ]
}
struct Validator(Arc<AtomicUsize>);
#[async_trait]
impl OutputValidator<String, String> for Validator {
    async fn validate(
        &self,
        output: String,
        ctx: &RunContext<String>,
    ) -> Result<String, OutputValidationError> {
        tokio::task::yield_now().await;
        assert_eq!(ctx.deps.as_str(), "dependency");
        self.0.fetch_add(1, Ordering::SeqCst);
        if output == "valid" {
            Ok(output)
        } else {
            Err(OutputValidationError::failed("private validation reason"))
        }
    }
}
#[tokio::test]
async fn async_validation_retries_and_dynamic_prompts() {
    let requests = Arc::new(AtomicUsize::new(0));
    let count = requests.clone();
    let calls = Arc::new(AtomicUsize::new(0));
    let sink = Sink::default();
    let model = FunctionModel::with_stream(move |messages, _| {
        let request = count.fetch_add(1, Ordering::SeqCst);
        let encoded = serde_json::to_string(messages).unwrap();
        assert!(encoded.contains("dynamic-dependency"));
        if request > 0 {
            assert!(encoded.contains("retry-prompt") || encoded.contains("retry_prompt"));
        }
        Box::pin(stream::iter(text(if request == 0 {
            "invalid"
        } else {
            "valid"
        })))
    });
    let agent = AgentBuilder::<String, String>::new(model)
        .system_prompt_fn(|ctx| {
            let text = format!("dynamic-{}", ctx.deps);
            async move { Some(text) }
        })
        .output_validator(Validator(calls.clone()))
        .checkpoint_sink(sink.clone())
        .build();
    let mut events = agent.run_stream("test", "dependency".into()).await.unwrap();
    let mut ready = 0;
    while let Some(event) = events.next().await {
        let event = event.unwrap();
        if matches!(event, AgentStreamEvent::OutputReady) {
            ready += 1;
        }
    }
    assert_eq!(ready, 1);
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(sink.terminal().await, CheckpointBoundary::Terminal);
}
#[tokio::test]
async fn structured_schema_exhaustion_never_signals_output_ready() {
    let sink = Sink::default();
    let agent = AgentBuilder::<(), String>::new(FunctionModel::with_stream(|_, _| {
        Box::pin(stream::iter(text("not json")))
    }))
    .output_type::<serde_json::Value>()
    .max_output_retries(1)
    .checkpoint_sink(sink.clone())
    .build();
    let mut events = agent.run_stream("test", ()).await.unwrap();
    let mut failed = false;
    while let Some(event) = events.next().await {
        assert!(!matches!(
            event,
            Ok(AgentStreamEvent::OutputReady | AgentStreamEvent::RunComplete { .. })
        ));
        failed |= matches!(event, Err(AgentRunError::OutputValidationFailed(_)));
    }
    assert!(failed);
    assert_eq!(sink.terminal().await, CheckpointBoundary::ValidationFailed);
}
#[tokio::test]
async fn detach_idle_native_partial_and_cancel_before_start() {
    for before_start in [true, false] {
        let sink = Sink::default();
        let calls = Arc::new(AtomicUsize::new(0));
        let count = calls.clone();
        let model = FunctionModel::with_stream(move |_, _| {
            count.fetch_add(1, Ordering::SeqCst);
            Box::pin(
                stream::iter(vec![Ok(Event::part_start(
                    0,
                    ModelResponsePart::text("retained"),
                ))])
                .chain(stream::pending()),
            )
        });
        let agent = AgentBuilder::<(), String>::new(model)
            .checkpoint_sink(sink.clone())
            .build();
        let token = CancellationToken::new();
        if before_start {
            token.cancel();
        }
        let mut response =
            AgentStream::new_with_cancel(&agent, "test".into(), (), RunOptions::default(), token)
                .await
                .unwrap();
        if !before_start {
            while let Some(event) = response.next().await {
                if matches!(event, Ok(AgentStreamEvent::TextDelta { .. })) {
                    break;
                }
            }
        }
        drop(response);
        assert_eq!(sink.terminal().await, CheckpointBoundary::ConsumerDetached);
        if before_start {
            assert_eq!(calls.load(Ordering::SeqCst), 0);
        } else {
            assert_eq!(
                sink.records
                    .lock()
                    .unwrap()
                    .last()
                    .unwrap()
                    .response
                    .as_ref()
                    .unwrap()
                    .parts
                    .len(),
                1
            );
        }
    }
}
#[tokio::test]
async fn committed_terminal_wins_receiver_detach() {
    let sink = Sink::default();
    let agent = AgentBuilder::<(), String>::new(FunctionModel::with_stream(|_, _| {
        Box::pin(stream::iter(text("valid")))
    }))
    .checkpoint_sink(sink.clone())
    .build();
    let response = agent.run_stream("test", ()).await.unwrap();
    assert_eq!(sink.terminal().await, CheckpointBoundary::Terminal);
    drop(response);
    tokio::task::yield_now().await;
    assert_eq!(
        sink.records
            .lock()
            .unwrap()
            .iter()
            .filter(|c| matches!(
                c.boundary,
                CheckpointBoundary::Terminal
                    | CheckpointBoundary::ConsumerDetached
                    | CheckpointBoundary::Cancelled
            ))
            .count(),
        1
    );
}

struct PendingModel {
    entered: Arc<Notify>,
    dropped: Arc<Notify>,
    summary: bool,
}
struct Cleanup(Arc<Notify>);
impl Drop for Cleanup {
    fn drop(&mut self) {
        self.0.notify_one();
    }
}
#[async_trait]
impl serdes_ai_models::Model for PendingModel {
    fn name(&self) -> &str {
        "pending"
    }
    fn system(&self) -> &str {
        "test"
    }
    fn profile(&self) -> &serdes_ai_models::ModelProfile {
        static PROFILE: std::sync::OnceLock<serdes_ai_models::ModelProfile> =
            std::sync::OnceLock::new();
        PROFILE.get_or_init(|| serdes_ai_models::ModelProfile {
            context_window: Some(1000),
            ..Default::default()
        })
    }
    async fn request(
        &self,
        _: &[serdes_ai_core::ModelRequest],
        _: &serdes_ai_core::ModelSettings,
        _: &serdes_ai_models::ModelRequestParameters,
    ) -> Result<serdes_ai_core::ModelResponse, serdes_ai_models::ModelError> {
        assert!(self.summary);
        let _cleanup = Cleanup(self.dropped.clone());
        self.entered.notify_one();
        std::future::pending().await
    }
    async fn request_stream(
        &self,
        _: &[serdes_ai_core::ModelRequest],
        _: &serdes_ai_core::ModelSettings,
        _: &serdes_ai_models::ModelRequestParameters,
    ) -> Result<serdes_ai_models::StreamedResponse, serdes_ai_models::ModelError> {
        assert!(!self.summary);
        let _cleanup = Cleanup(self.dropped.clone());
        self.entered.notify_one();
        std::future::pending().await
    }
}
#[tokio::test]
async fn detach_request_and_legacy_summary_drops_inflight_future() {
    for summary in [false, true] {
        let entered = Arc::new(Notify::new());
        let dropped = Arc::new(Notify::new());
        let sink = Sink::default();
        let agent = AgentBuilder::<(), String>::new(PendingModel {
            entered: entered.clone(),
            dropped: dropped.clone(),
            summary,
        })
        .checkpoint_sink(sink.clone())
        .build();
        let mut options = RunOptions::default();
        if summary {
            options.compression = Some(serdes_ai_agent::run::ContextCompression {
                strategy: serdes_ai_agent::run::CompressionStrategy::Summarize,
                threshold: 0.0,
                target_tokens: 10,
            });
            options.message_history = Some(
                (0..8)
                    .map(|_| {
                        let mut message = serdes_ai_core::ModelRequest::new();
                        message.add_user_prompt("history".repeat(100));
                        message
                    })
                    .collect(),
            );
        }
        let response = AgentStream::new(&agent, "test".into(), (), options)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), entered.notified())
            .await
            .unwrap();
        drop(response);
        assert_eq!(sink.terminal().await, CheckpointBoundary::ConsumerDetached);
        tokio::time::timeout(Duration::from_secs(2), dropped.notified())
            .await
            .unwrap();
    }
}
#[tokio::test]
async fn detach_async_tool_cancels_without_followup_request() {
    let entered = Arc::new(Notify::new());
    let dropped = Arc::new(Notify::new());
    let sink = Sink::default();
    let calls = Arc::new(AtomicUsize::new(0));
    let count = calls.clone();
    let model = FunctionModel::with_stream(move |_, _| {
        count.fetch_add(1, Ordering::SeqCst);
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
    let start = entered.clone();
    let cleanup = dropped.clone();
    let agent = AgentBuilder::<(), String>::new(model)
        .checkpoint_sink(sink.clone())
        .tool_fn_async("wait", "wait", move |_, _: serde_json::Value| {
            let start = start.clone();
            let cleanup = cleanup.clone();
            async move {
                let _cleanup = Cleanup(cleanup);
                start.notify_one();
                std::future::pending::<
                    Result<serdes_ai_tools::ToolReturn, serdes_ai_tools::ToolError>,
                >()
                .await
            }
        })
        .build();
    let response = agent.run_stream("test", ()).await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), entered.notified())
        .await
        .unwrap();
    drop(response);
    assert_eq!(sink.terminal().await, CheckpointBoundary::ConsumerDetached);
    tokio::time::timeout(Duration::from_secs(2), dropped.notified())
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

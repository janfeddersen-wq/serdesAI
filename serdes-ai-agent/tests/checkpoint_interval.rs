use async_trait::async_trait;
use futures::{StreamExt, stream};
use serdes_ai_agent::*;
use serdes_ai_core::{ModelRequest, ModelResponsePart, ModelResponseStreamEvent as Event};
use serdes_ai_models::FunctionModel;
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};
#[derive(Clone, Default)]
struct Sink(Arc<Mutex<Vec<AgentCheckpoint>>>);
#[async_trait]
impl CheckpointSink for Sink {
    fn partial_interval(&self) -> Option<Duration> {
        Some(Duration::from_millis(10))
    }
    async fn save(&self, value: &AgentCheckpoint) -> Result<(), String> {
        self.0.lock().unwrap().push(value.clone());
        Ok(())
    }
}
#[tokio::test]
async fn idle_stream_periodic_snapshot_and_full_channel_cancel() {
    for flood in [false, true] {
        let sink = Sink::default();
        let model = FunctionModel::with_stream(move |_, _| {
            let first = stream::iter(vec![Ok(Event::part_start(
                0,
                ModelResponsePart::text("partial"),
            ))]);
            if flood {
                Box::pin(first.chain(stream::repeat_with(|| Ok(Event::text_delta(0, "x")))))
            } else {
                Box::pin(first.chain(stream::pending()))
            }
        });
        let agent = AgentBuilder::<(), String>::new(model)
            .checkpoint_sink(sink.clone())
            .build();
        let token = CancellationToken::new();
        let mut response = AgentStream::new_with_cancel(
            &agent,
            "test".into(),
            (),
            RunOptions::default(),
            token.clone(),
        )
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        if !flood {
            assert!(
                sink.0
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|c| c.boundary == CheckpointBoundary::Partial)
            );
        }
        token.cancel();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if sink
                    .0
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|c| c.boundary == CheckpointBoundary::Cancelled)
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        while response.next().await.is_some() {}
    }
}
struct PendingPolicy;
#[async_trait]
impl ContextPolicy<()> for PendingPolicy {
    async fn prepare(
        &self,
        _: ContextPolicyInput<'_, ()>,
        _: Vec<ModelRequest>,
    ) -> Result<Vec<ModelRequest>, String> {
        std::future::pending().await
    }
}
#[tokio::test]
async fn cancel_policy_preparation() {
    let sink = Sink::default();
    let token = CancellationToken::new();
    let agent = AgentBuilder::<(), String>::new(FunctionModel::constant_text("unused"))
        .context_policy(PendingPolicy)
        .checkpoint_sink(sink.clone())
        .build();
    let mut response = AgentStream::new_with_cancel(
        &agent,
        "test".into(),
        (),
        RunOptions::default(),
        token.clone(),
    )
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    token.cancel();
    tokio::time::timeout(Duration::from_secs(2), async {
        while response.next().await.is_some() {}
    })
    .await
    .unwrap();
    assert_eq!(
        sink.0.lock().unwrap().last().unwrap().boundary,
        CheckpointBoundary::Cancelled
    );
}

#[derive(Clone)]
struct StalledSink(Arc<std::sync::atomic::AtomicUsize>);
#[async_trait]
impl CheckpointSink for StalledSink {
    fn save_timeout(&self) -> Duration {
        Duration::from_millis(20)
    }
    async fn save(&self, _: &AgentCheckpoint) -> Result<(), String> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        std::future::pending().await
    }
}
#[tokio::test]
async fn sink_deadline_stops_without_recursive_save_or_model_call() {
    let saves = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let model_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count = model_calls.clone();
    let model = FunctionModel::with_stream(move |_, _| {
        count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::pin(stream::empty())
    });
    let agent = AgentBuilder::<(), String>::new(model)
        .checkpoint_sink(StalledSink(saves.clone()))
        .build();
    let mut response = agent.run_stream("test", ()).await.unwrap();
    let mut failed = false;
    while let Some(event) = response.next().await {
        failed |= matches!(event, Err(AgentRunError::Checkpoint(_)));
    }
    assert!(failed);
    assert_eq!(saves.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(model_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[derive(Clone)]
struct SlowSave {
    entered: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
    records: Arc<Mutex<Vec<CheckpointBoundary>>>,
}
#[async_trait]
impl CheckpointSink for SlowSave {
    fn save_timeout(&self) -> Duration {
        Duration::from_millis(100)
    }
    async fn save(&self, checkpoint: &AgentCheckpoint) -> Result<(), String> {
        self.records
            .lock()
            .unwrap()
            .push(checkpoint.boundary.clone());
        if checkpoint.boundary == CheckpointBoundary::BeforeRequest {
            self.entered.notify_one();
            self.release.notified().await;
        }
        Ok(())
    }
}
#[tokio::test]
async fn cancel_during_save_waits_without_starting_model() {
    let sink = SlowSave {
        entered: Arc::new(tokio::sync::Notify::new()),
        release: Arc::new(tokio::sync::Notify::new()),
        records: Arc::new(Mutex::new(Vec::new())),
    };
    let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls = count.clone();
    let agent = AgentBuilder::<(), String>::new(FunctionModel::with_stream(move |_, _| {
        calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::pin(stream::empty())
    }))
    .checkpoint_sink(sink.clone())
    .build();
    let token = CancellationToken::new();
    let mut events = AgentStream::new_with_cancel(
        &agent,
        "test".into(),
        (),
        RunOptions::default(),
        token.clone(),
    )
    .await
    .unwrap();
    sink.entered.notified().await;
    token.cancel();
    tokio::task::yield_now().await;
    sink.release.notify_one();
    tokio::time::timeout(Duration::from_secs(1), async {
        while events.next().await.is_some() {}
    })
    .await
    .unwrap();
    assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(
        sink.records.lock().unwrap().as_slice(),
        &[
            CheckpointBoundary::BeforeRequest,
            CheckpointBoundary::Cancelled
        ]
    );
}

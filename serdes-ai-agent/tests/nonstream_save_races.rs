use async_trait::async_trait;
use serdes_ai_agent::*;
use serdes_ai_models::FunctionModel;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
#[derive(Clone)]
struct Sink {
    boundary: CheckpointBoundary,
    entered: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
    attempts: Arc<Mutex<Vec<CheckpointBoundary>>>,
    outcome: u8,
}
#[async_trait]
impl CheckpointSink for Sink {
    fn save_timeout(&self) -> Duration {
        Duration::from_millis(50)
    }
    async fn save(&self, checkpoint: &AgentCheckpoint) -> Result<(), String> {
        self.attempts
            .lock()
            .unwrap()
            .push(checkpoint.boundary.clone());
        if checkpoint.boundary == self.boundary {
            self.entered.notify_one();
            self.release.notified().await;
            if self.outcome == 1 {
                return Err("rejected".into());
            }
        }
        Ok(())
    }
}
#[tokio::test]
async fn cooperative_nonstream_saves_have_one_outcome() {
    for boundary in [
        CheckpointBoundary::BeforeRequest,
        CheckpointBoundary::Terminal,
    ] {
        for outcome in 0..3 {
            let sink = Sink {
                boundary: boundary.clone(),
                entered: Arc::new(tokio::sync::Notify::new()),
                release: Arc::new(tokio::sync::Notify::new()),
                attempts: Arc::new(Mutex::new(Vec::new())),
                outcome,
            };
            let calls = Arc::new(AtomicUsize::new(0));
            let counter = calls.clone();
            let agent = AgentBuilder::<(), String>::new(FunctionModel::new(move |_, _| {
                counter.fetch_add(1, Ordering::SeqCst);
                serdes_ai_core::ModelResponse::text("done")
            }))
            .checkpoint_sink(sink.clone())
            .build();
            let token = CancellationToken::new();
            let mut run = AgentRun::new_with_cancel(
                &agent,
                "test".into(),
                (),
                RunOptions::default(),
                token.clone(),
            )
            .await
            .unwrap();
            let controller = async {
                sink.entered.notified().await;
                token.cancel();
                tokio::task::yield_now().await;
                if outcome != 2 {
                    sink.release.notify_one();
                }
            };
            let (result, ()) = tokio::join!(run.step(), controller);
            if outcome != 0 {
                assert!(matches!(result, Err(AgentRunError::Checkpoint(_))));
            } else if boundary == CheckpointBoundary::Terminal {
                assert!(result.is_ok());
            } else {
                assert!(matches!(result, Err(AgentRunError::Cancelled)));
            }
            let attempts = sink.attempts.lock().unwrap().clone();
            assert_eq!(
                calls.load(Ordering::SeqCst),
                usize::from(boundary == CheckpointBoundary::Terminal)
            );
            assert_eq!(attempts.iter().filter(|b| **b == boundary).count(), 1);
            assert_eq!(
                attempts
                    .iter()
                    .filter(|b| **b == CheckpointBoundary::Cancelled)
                    .count(),
                usize::from(outcome == 0 && boundary == CheckpointBoundary::BeforeRequest)
            );
            let _ = run.step().await;
            assert_eq!(*sink.attempts.lock().unwrap(), attempts);
        }
    }
}

//! Owns the lifetime of a streaming task, including receiver detach during idle work.
use crate::{
    AgentCheckpoint, AgentRunError, AgentStreamEvent, CheckpointBoundary, CheckpointSink, RunUsage,
};
use async_trait::async_trait;
use std::{
    future::Future,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::{Mutex as AsyncMutex, mpsc};
use tokio_util::sync::CancellationToken;

pub(crate) struct StreamLifecycle {
    sink: Option<Arc<dyn CheckpointSink>>,
    snapshot: Mutex<AgentCheckpoint>,
    gate: AsyncMutex<()>,
    terminal: AtomicBool,
    failed: AtomicBool,
    interrupted: AtomicBool,
}
impl StreamLifecycle {
    pub fn new(sink: Option<Arc<dyn CheckpointSink>>, id: String) -> Arc<Self> {
        Arc::new(Self {
            sink,
            snapshot: Mutex::new(AgentCheckpoint {
                version: 1,
                run_id: id,
                step: 0,
                boundary: CheckpointBoundary::BeforeRequest,
                messages: Vec::new(),
                response: None,
                usage: RunUsage::new(),
            }),
            gate: AsyncMutex::new(()),
            terminal: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            interrupted: AtomicBool::new(false),
        })
    }
    pub fn update(
        &self,
        messages: &[serdes_ai_core::ModelRequest],
        response: Option<&serdes_ai_core::ModelResponse>,
        step: u32,
        usage: &RunUsage,
    ) {
        let mut snapshot = self.snapshot.lock().unwrap();
        snapshot.messages = messages.to_vec();
        snapshot.response = response.cloned();
        snapshot.step = step;
        snapshot.usage = usage.clone();
    }
    pub async fn supervise(
        &self,
        future: impl Future<Output = ()>,
        tx: mpsc::Sender<Result<AgentStreamEvent, AgentRunError>>,
        token: CancellationToken,
    ) {
        tokio::pin!(future);
        let boundary = tokio::select! {
            biased;
            _ = tx.closed() => Some(CheckpointBoundary::ConsumerDetached),
            _ = token.cancelled() => Some(CheckpointBoundary::Cancelled),
            _ = &mut future => Some(if tx.is_closed() { CheckpointBoundary::ConsumerDetached } else { CheckpointBoundary::Failed }),
        };
        if let Some(boundary) = boundary {
            // A save in progress is part of `future`. Poll it until its bounded
            // transaction completes; never cancel a persistence operation midway.
            self.interrupted.store(true, Ordering::SeqCst);
            if self.gate.try_lock().is_err() {
                tokio::select! { _ = self.gate.lock() => {}, _ = &mut future => {} }
            }
            self.interrupted.store(false, Ordering::SeqCst);
            if !self.terminal.load(Ordering::SeqCst) && !self.failed.load(Ordering::SeqCst) {
                let mut snapshot = self.snapshot.lock().unwrap().clone();
                snapshot.boundary = boundary;
                let _ = self.save(&snapshot).await;
            }
            if matches!(
                self.snapshot.lock().unwrap().boundary,
                CheckpointBoundary::Cancelled | CheckpointBoundary::ConsumerDetached
            ) {
                let _ = tx.try_send(Err(AgentRunError::Cancelled));
            }
        }
        // Dropping future cancels request/summary/validator/tool futures here.
    }
}
fn terminal(boundary: &CheckpointBoundary) -> bool {
    matches!(
        boundary,
        CheckpointBoundary::Terminal
            | CheckpointBoundary::Cancelled
            | CheckpointBoundary::ConsumerDetached
            | CheckpointBoundary::Failed
            | CheckpointBoundary::ModelFailed(_)
            | CheckpointBoundary::ValidationFailed
    )
}
#[async_trait]
impl CheckpointSink for StreamLifecycle {
    fn partial_interval(&self) -> Option<std::time::Duration> {
        self.sink.as_ref().and_then(|sink| sink.partial_interval())
    }
    // The inner operation enforces the application's timeout. Allow its timeout
    // result to propagate before the outer lifecycle helper's own deadline.
    fn save_timeout(&self) -> std::time::Duration {
        self.sink
            .as_ref()
            .map(|s| s.save_timeout())
            .unwrap_or_default()
            .saturating_add(std::time::Duration::from_secs(1))
    }
    async fn save(&self, checkpoint: &AgentCheckpoint) -> Result<(), String> {
        let _guard = self.gate.lock().await;
        if self.terminal.load(Ordering::SeqCst) || self.failed.load(Ordering::SeqCst) {
            return Ok(());
        }
        *self.snapshot.lock().unwrap() = checkpoint.clone();
        if let Some(sink) = &self.sink {
            let result = tokio::time::timeout(sink.save_timeout(), sink.save(checkpoint)).await;
            match result {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    self.failed.store(true, Ordering::SeqCst);
                    return Err(error);
                }
                Err(_) => {
                    self.failed.store(true, Ordering::SeqCst);
                    return Err("checkpoint deadline exceeded; reconcile storage".into());
                }
            }
        }
        if terminal(&checkpoint.boundary) {
            self.terminal.store(true, Ordering::SeqCst);
        }
        drop(_guard);
        if self.interrupted.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn terminal_checkpoint_wins_blocked_final_delivery() {
        let state = StreamLifecycle::new(None, "test".into());
        let (tx, _rx) = mpsc::channel(1);
        let token = CancellationToken::new();
        let work = async {
            tx.send(Ok(AgentStreamEvent::OutputReady)).await.unwrap();
            let mut checkpoint = state.snapshot.lock().unwrap().clone();
            checkpoint.boundary = CheckpointBoundary::Terminal;
            state.save(&checkpoint).await.unwrap();
            token.cancel();
            let _ = tx.send(Ok(AgentStreamEvent::OutputReady)).await;
        };
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            state.supervise(work, tx.clone(), token.clone()),
        )
        .await
        .unwrap();
        assert_eq!(
            state.snapshot.lock().unwrap().boundary,
            CheckpointBoundary::Terminal
        );
    }
}

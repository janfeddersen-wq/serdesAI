//! Worker for processing A2A tasks.
//!
//! The worker polls the broker for tasks and processes them using the agent.

use crate::broker::Broker;
use crate::schema::Message;
use crate::storage::Storage;
use crate::task::{Task, TaskResult, TaskStatus};
use serdes_ai_agent::Agent;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Handle for controlling a running worker.
#[derive(Debug)]
pub struct WorkerHandle {
    /// Handle to the worker task.
    handle: JoinHandle<()>,
    /// Channel to signal shutdown.
    shutdown_tx: mpsc::Sender<()>,
    /// Flag indicating if worker is running.
    running: Arc<AtomicBool>,
}

impl WorkerHandle {
    /// Check if the worker is still running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Signal the worker to stop.
    pub async fn stop(&self) {
        let _ = self.shutdown_tx.send(()).await;
    }

    /// Wait for the worker to finish.
    pub async fn wait(self) {
        let _ = self.handle.await;
    }

    /// Stop the worker and wait for it to finish.
    pub async fn shutdown(self) {
        self.stop().await;
        self.wait().await;
    }
}

/// Worker that processes tasks from a broker.
///
/// The worker continuously polls the broker for tasks and processes them
/// using the provided agent.
pub struct AgentWorker<Deps, Output> {
    agent: Arc<Agent<Deps, Output>>,
    broker: Arc<dyn Broker>,
    storage: Arc<dyn Storage>,
    deps_factory: Arc<dyn Fn() -> Deps + Send + Sync>,
}

impl<Deps, Output> AgentWorker<Deps, Output>
where
    Deps: Send + Sync + 'static,
    Output: Send + Sync + ToString + 'static,
{
    /// Create a new worker.
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to use for processing tasks.
    /// * `broker` - The broker to poll for tasks.
    /// * `storage` - The storage for persisting task state.
    /// * `deps_factory` - A factory function to create dependencies for each task.
    pub fn new<F>(
        agent: Arc<Agent<Deps, Output>>,
        broker: Arc<dyn Broker>,
        storage: Arc<dyn Storage>,
        deps_factory: F,
    ) -> Self
    where
        F: Fn() -> Deps + Send + Sync + 'static,
    {
        Self {
            agent,
            broker,
            storage,
            deps_factory: Arc::new(deps_factory),
        }
    }

    /// Start the worker in a background task.
    ///
    /// Returns a handle that can be used to control the worker.
    pub fn spawn(self) -> WorkerHandle {
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&running);

        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        break;
                    }
                    task = self.broker.wait_for_task() => {
                        match task {
                            Some(task) => {
                                self.process_task(task).await;
                            }
                            None => {
                                // Broker shut down
                                break;
                            }
                        }
                    }
                }
            }
            running_clone.store(false, Ordering::Relaxed);
        });

        WorkerHandle {
            handle,
            shutdown_tx,
            running,
        }
    }

    /// Process a single task.
    pub async fn process_task(&self, mut task: Task) -> TaskResult {
        let start = Instant::now();

        // Check if task was already cancelled before we start
        if task.status == TaskStatus::Cancelled {
            return TaskResult {
                task_id: task.id,
                status: TaskStatus::Cancelled,
                messages: Vec::new(),
                artifacts: Vec::new(),
                error: None,
                duration_ms: Some(0),
            };
        }

        // Get cancellation token for this task
        let cancellation_token = self
            .broker
            .get_cancellation_token(&task.id)
            .await
            .unwrap_or_else(CancellationToken::new);

        // Mark task as running
        if let Err(e) = task.start() {
            return TaskResult::failure(&task.id, format!("Failed to start task: {}", e));
        }
        if let Err(e) = self.storage.update_task(&task).await {
            return TaskResult::failure(&task.id, format!("Failed to update task: {}", e));
        }

        // Extract user message content
        let prompt = task.message.text_content();

        // Create dependencies for this task
        let deps = (self.deps_factory)();

        // Run the agent with cancellation support
        let result = tokio::select! {
            result = self.agent.run(prompt, deps) => result,
            _ = cancellation_token.cancelled() => {
                // Task was cancelled during execution
                task.force_status(TaskStatus::Cancelled);
                if let Err(e) = self.storage.update_task(&task).await {
                    return TaskResult::failure(&task.id, format!("Failed to save cancellation: {}", e));
                }
                return TaskResult {
                    task_id: task.id,
                    status: TaskStatus::Cancelled,
                    messages: Vec::new(),
                    artifacts: Vec::new(),
                    error: None,
                    duration_ms: Some(start.elapsed().as_millis() as u64),
                };
            }
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(agent_result) => {
                // Check for cancellation one more time before completing
                if cancellation_token.is_cancelled() {
                    task.force_status(TaskStatus::Cancelled);
                    if let Err(e) = self.storage.update_task(&task).await {
                        return TaskResult::failure(
                            &task.id,
                            format!("Failed to save cancellation: {}", e),
                        );
                    }
                    return TaskResult {
                        task_id: task.id,
                        status: TaskStatus::Cancelled,
                        messages: Vec::new(),
                        artifacts: Vec::new(),
                        error: None,
                        duration_ms: Some(duration_ms),
                    };
                }

                // Create response message
                let output_text = agent_result.output.to_string();
                let response_message = Message::agent(output_text);

                task.add_message(response_message.clone());
                if let Err(e) = task.complete() {
                    return TaskResult::failure(
                        &task.id,
                        format!("Failed to complete task: {}", e),
                    );
                }

                if let Err(e) = self.storage.update_task(&task).await {
                    return TaskResult::failure(&task.id, format!("Failed to save result: {}", e));
                }

                TaskResult::success(&task.id, vec![response_message])
                    .with_artifacts(task.artifacts)
                    .with_duration(duration_ms)
            }
            Err(e) => {
                let error_msg = format!("Agent error: {}", e);
                // Use force_status since fail() requires Running state
                // but we might be in a weird state after an error
                task.force_status(TaskStatus::Failed);
                task.error = Some(error_msg.clone());

                if let Err(storage_err) = self.storage.update_task(&task).await {
                    return TaskResult::failure(
                        &task.id,
                        format!("Agent failed: {}. Also failed to save: {}", e, storage_err),
                    );
                }

                TaskResult::failure(&task.id, error_msg).with_duration(duration_ms)
            }
        }
    }
}

/// Builder for creating an AgentWorker with unit dependencies.
impl<Output> AgentWorker<(), Output>
where
    Output: Send + Sync + ToString + 'static,
{
    /// Create a worker with no dependencies.
    pub fn without_deps(
        agent: Arc<Agent<(), Output>>,
        broker: Arc<dyn Broker>,
        storage: Arc<dyn Storage>,
    ) -> Self {
        Self::new(agent, broker, storage, || ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests would require a mock agent
    // For now, we test the handle functionality

    #[tokio::test]
    async fn test_worker_handle_creation() {
        let (tx, _rx) = mpsc::channel(1);
        let running = Arc::new(AtomicBool::new(true));

        let handle = WorkerHandle {
            handle: tokio::spawn(async {}),
            shutdown_tx: tx,
            running: Arc::clone(&running),
        };

        assert!(handle.is_running());
    }

    #[tokio::test]
    async fn test_task_result_builder() {
        let result = TaskResult::success("task-1", vec![Message::agent("Done!")]);
        assert!(result.is_success());
        assert!(result.duration_ms.is_none());

        let with_duration =
            TaskResult::success("task-2", vec![Message::agent("Done!")]).with_duration(100);
        assert_eq!(with_duration.duration_ms, Some(100));
    }
}

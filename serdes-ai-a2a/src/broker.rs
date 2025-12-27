//! Broker abstraction for task distribution.
//!
//! The broker handles the queue of tasks waiting to be processed by workers.

use crate::task::Task;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{Mutex, Notify};

/// Errors that can occur during broker operations.
#[derive(Debug, Error)]
pub enum BrokerError {
    /// Failed to submit task.
    #[error("Failed to submit task: {0}")]
    SubmitFailed(String),

    /// Queue is full.
    #[error("Queue is full (max: {0})")]
    QueueFull(usize),

    /// Broker is shut down.
    #[error("Broker is shut down")]
    ShutDown,

    /// Backend error.
    #[error("Broker backend error: {0}")]
    Backend(String),
}

/// Broker trait for task distribution.
///
/// The broker is responsible for queuing tasks and distributing them
/// to workers for processing.
#[async_trait]
pub trait Broker: Send + Sync {
    /// Submit a task to the queue.
    async fn submit_task(&self, task: Task) -> Result<(), BrokerError>;

    /// Poll for the next task to process.
    ///
    /// Returns `None` if no task is available.
    async fn poll_task(&self) -> Option<Task>;

    /// Wait for a task to become available.
    ///
    /// This will block until a task is available or the broker is shut down.
    async fn wait_for_task(&self) -> Option<Task>;

    /// Get the number of pending tasks.
    async fn pending_count(&self) -> usize;

    /// Check if the broker is empty.
    async fn is_empty(&self) -> bool;
}

/// In-memory broker implementation.
///
/// A simple FIFO queue for task distribution. Suitable for single-node
/// deployments. For distributed systems, consider using a message queue
/// like Redis or RabbitMQ.
#[derive(Debug)]
pub struct InMemoryBroker {
    queue: Arc<Mutex<VecDeque<Task>>>,
    notify: Arc<Notify>,
    max_size: Option<usize>,
    shutdown: Arc<Mutex<bool>>,
}

impl Default for InMemoryBroker {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryBroker {
    /// Create a new in-memory broker.
    pub fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            notify: Arc::new(Notify::new()),
            max_size: None,
            shutdown: Arc::new(Mutex::new(false)),
        }
    }

    /// Create a broker with a maximum queue size.
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            notify: Arc::new(Notify::new()),
            max_size: Some(max_size),
            shutdown: Arc::new(Mutex::new(false)),
        }
    }

    /// Shut down the broker.
    ///
    /// This will cause `wait_for_task` to return `None` for all waiting workers.
    pub async fn shutdown(&self) {
        let mut shutdown = self.shutdown.lock().await;
        *shutdown = true;
        // Notify all waiting workers
        self.notify.notify_waiters();
    }

    /// Check if the broker is shut down.
    pub async fn is_shutdown(&self) -> bool {
        *self.shutdown.lock().await
    }

    /// Clear all pending tasks.
    pub async fn clear(&self) {
        let mut queue = self.queue.lock().await;
        queue.clear();
    }
}

#[async_trait]
impl Broker for InMemoryBroker {
    async fn submit_task(&self, task: Task) -> Result<(), BrokerError> {
        if *self.shutdown.lock().await {
            return Err(BrokerError::ShutDown);
        }

        let mut queue = self.queue.lock().await;

        // Check max size
        if let Some(max) = self.max_size {
            if queue.len() >= max {
                return Err(BrokerError::QueueFull(max));
            }
        }

        queue.push_back(task);
        drop(queue); // Release lock before notifying

        self.notify.notify_one();
        Ok(())
    }

    async fn poll_task(&self) -> Option<Task> {
        let mut queue = self.queue.lock().await;
        queue.pop_front()
    }

    async fn wait_for_task(&self) -> Option<Task> {
        loop {
            // Check for shutdown
            if *self.shutdown.lock().await {
                return None;
            }

            // Try to get a task
            if let Some(task) = self.poll_task().await {
                return Some(task);
            }

            // Wait for notification
            self.notify.notified().await;
        }
    }

    async fn pending_count(&self) -> usize {
        self.queue.lock().await.len()
    }

    async fn is_empty(&self) -> bool {
        self.queue.lock().await.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Message;

    #[tokio::test]
    async fn test_submit_and_poll() {
        let broker = InMemoryBroker::new();
        let task = Task::new("thread-1", Message::user("Hello"));
        let task_id = task.id.clone();

        broker.submit_task(task).await.unwrap();

        let polled = broker.poll_task().await;
        assert!(polled.is_some());
        assert_eq!(polled.unwrap().id, task_id);
    }

    #[tokio::test]
    async fn test_fifo_order() {
        let broker = InMemoryBroker::new();

        let task1 = Task::new("thread-1", Message::user("First"));
        let task2 = Task::new("thread-1", Message::user("Second"));
        let task3 = Task::new("thread-1", Message::user("Third"));

        let id1 = task1.id.clone();
        let id2 = task2.id.clone();
        let id3 = task3.id.clone();

        broker.submit_task(task1).await.unwrap();
        broker.submit_task(task2).await.unwrap();
        broker.submit_task(task3).await.unwrap();

        assert_eq!(broker.poll_task().await.unwrap().id, id1);
        assert_eq!(broker.poll_task().await.unwrap().id, id2);
        assert_eq!(broker.poll_task().await.unwrap().id, id3);
        assert!(broker.poll_task().await.is_none());
    }

    #[tokio::test]
    async fn test_max_size() {
        let broker = InMemoryBroker::with_max_size(2);

        broker
            .submit_task(Task::new("t1", Message::user("1")))
            .await
            .unwrap();
        broker
            .submit_task(Task::new("t2", Message::user("2")))
            .await
            .unwrap();

        let result = broker
            .submit_task(Task::new("t3", Message::user("3")))
            .await;
        assert!(matches!(result, Err(BrokerError::QueueFull(2))));
    }

    #[tokio::test]
    async fn test_pending_count() {
        let broker = InMemoryBroker::new();

        assert_eq!(broker.pending_count().await, 0);
        assert!(broker.is_empty().await);

        broker
            .submit_task(Task::new("t1", Message::user("1")))
            .await
            .unwrap();
        broker
            .submit_task(Task::new("t2", Message::user("2")))
            .await
            .unwrap();

        assert_eq!(broker.pending_count().await, 2);
        assert!(!broker.is_empty().await);

        broker.poll_task().await;
        assert_eq!(broker.pending_count().await, 1);
    }

    #[tokio::test]
    async fn test_shutdown() {
        let broker = InMemoryBroker::new();

        assert!(!broker.is_shutdown().await);

        broker.shutdown().await;

        assert!(broker.is_shutdown().await);

        let result = broker
            .submit_task(Task::new("t1", Message::user("1")))
            .await;
        assert!(matches!(result, Err(BrokerError::ShutDown)));
    }

    #[tokio::test]
    async fn test_wait_for_task() {
        let broker = Arc::new(InMemoryBroker::new());
        let broker_clone = Arc::clone(&broker);

        // Spawn a task that waits for work
        let handle = tokio::spawn(async move { broker_clone.wait_for_task().await });

        // Give the spawned task time to start waiting
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Submit a task
        let task = Task::new("thread-1", Message::user("Hello"));
        let task_id = task.id.clone();
        broker.submit_task(task).await.unwrap();

        // Wait for the spawned task to complete
        let result = handle.await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, task_id);
    }

    #[tokio::test]
    async fn test_clear() {
        let broker = InMemoryBroker::new();

        broker
            .submit_task(Task::new("t1", Message::user("1")))
            .await
            .unwrap();
        broker
            .submit_task(Task::new("t2", Message::user("2")))
            .await
            .unwrap();

        assert_eq!(broker.pending_count().await, 2);

        broker.clear().await;

        assert!(broker.is_empty().await);
    }
}

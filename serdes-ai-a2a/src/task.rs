//! Task types for the A2A protocol.
//!
//! Tasks represent units of work submitted to an agent.

use crate::schema::{Artifact, Message};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors related to task operations.
#[derive(Debug, Error)]
pub enum TaskError {
    /// Invalid state transition attempted.
    #[error("Invalid state transition from {from} to {to}")]
    InvalidStateTransition {
        from: TaskStatus,
        to: TaskStatus,
    },
}

/// Unique identifier for a task.
pub type TaskId = String;

/// A task in the A2A protocol.
///
/// Tasks are the primary unit of work in A2A. They are submitted
/// by clients and processed by agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique identifier for this task.
    pub id: TaskId,
    /// Thread ID for grouping related tasks.
    pub thread_id: String,
    /// Current status of the task.
    pub status: TaskStatus,
    /// The original message that created this task.
    pub message: Message,
    /// Messages in the conversation.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Artifacts produced by the task.
    #[serde(default)]
    pub artifacts: Vec<Artifact>,
    /// When the task was created.
    pub created_at: DateTime<Utc>,
    /// When the task was last updated.
    pub updated_at: DateTime<Utc>,
    /// Optional error message if the task failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Optional metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl Task {
    /// Create a new task.
    ///
    /// The original message is automatically added to the messages list.
    pub fn new(thread_id: impl Into<String>, message: Message) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            thread_id: thread_id.into(),
            status: TaskStatus::Pending,
            message: message.clone(),
            messages: vec![message], // Include the original message
            artifacts: Vec::new(),
            created_at: now,
            updated_at: now,
            error: None,
            metadata: None,
        }
    }

    /// Create a new task with a specific ID.
    ///
    /// The original message is automatically added to the messages list.
    pub fn with_id(id: impl Into<String>, thread_id: impl Into<String>, message: Message) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            thread_id: thread_id.into(),
            status: TaskStatus::Pending,
            message: message.clone(),
            messages: vec![message], // Include the original message
            artifacts: Vec::new(),
            created_at: now,
            updated_at: now,
            error: None,
            metadata: None,
        }
    }

    /// Check if the task is pending.
    pub fn is_pending(&self) -> bool {
        self.status == TaskStatus::Pending
    }

    /// Check if the task is running.
    pub fn is_running(&self) -> bool {
        self.status == TaskStatus::Running
    }

    /// Check if the task is completed (success or failure).
    pub fn is_completed(&self) -> bool {
        matches!(
            self.status,
            TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Cancelled
        )
    }

    /// Check if the task succeeded.
    pub fn is_success(&self) -> bool {
        self.status == TaskStatus::Completed
    }

    /// Check if the task failed.
    pub fn is_failed(&self) -> bool {
        self.status == TaskStatus::Failed
    }

    /// Mark the task as running.
    ///
    /// # Errors
    ///
    /// Returns `TaskError::InvalidStateTransition` if the task is not pending.
    pub fn start(&mut self) -> Result<(), TaskError> {
        if self.status != TaskStatus::Pending {
            return Err(TaskError::InvalidStateTransition {
                from: self.status,
                to: TaskStatus::Running,
            });
        }
        self.status = TaskStatus::Running;
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Mark the task as completed.
    ///
    /// # Errors
    ///
    /// Returns `TaskError::InvalidStateTransition` if the task is not running.
    pub fn complete(&mut self) -> Result<(), TaskError> {
        if self.status != TaskStatus::Running {
            return Err(TaskError::InvalidStateTransition {
                from: self.status,
                to: TaskStatus::Completed,
            });
        }
        self.status = TaskStatus::Completed;
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Mark the task as failed.
    ///
    /// # Errors
    ///
    /// Returns `TaskError::InvalidStateTransition` if the task is not running.
    pub fn fail(&mut self, error: impl Into<String>) -> Result<(), TaskError> {
        if self.status != TaskStatus::Running {
            return Err(TaskError::InvalidStateTransition {
                from: self.status,
                to: TaskStatus::Failed,
            });
        }
        self.status = TaskStatus::Failed;
        self.error = Some(error.into());
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Mark the task as cancelled.
    ///
    /// Cancellation is allowed from pending or running states.
    ///
    /// # Errors
    ///
    /// Returns `TaskError::InvalidStateTransition` if the task is already completed.
    pub fn cancel(&mut self) -> Result<(), TaskError> {
        if self.is_completed() {
            return Err(TaskError::InvalidStateTransition {
                from: self.status,
                to: TaskStatus::Cancelled,
            });
        }
        self.status = TaskStatus::Cancelled;
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Force set status without validation (for internal/recovery use).
    ///
    /// Use with caution - this bypasses state transition validation.
    pub fn force_status(&mut self, status: TaskStatus) {
        self.status = status;
        self.updated_at = Utc::now();
    }

    /// Add a message to the task.
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
        self.updated_at = Utc::now();
    }

    /// Add an artifact to the task.
    pub fn add_artifact(&mut self, artifact: Artifact) {
        self.artifacts.push(artifact);
        self.updated_at = Utc::now();
    }

    /// Set metadata on the task.
    pub fn set_metadata(&mut self, metadata: serde_json::Value) {
        self.metadata = Some(metadata);
        self.updated_at = Utc::now();
    }
}

/// Status of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    /// Task is waiting to be processed.
    Pending,
    /// Task is currently being processed.
    Running,
    /// Task completed successfully.
    Completed,
    /// Task failed.
    Failed,
    /// Task was cancelled.
    Cancelled,
}

impl Default for TaskStatus {
    fn default() -> Self {
        Self::Pending
    }
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskStatus::Pending => write!(f, "pending"),
            TaskStatus::Running => write!(f, "running"),
            TaskStatus::Completed => write!(f, "completed"),
            TaskStatus::Failed => write!(f, "failed"),
            TaskStatus::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Result of a task execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// The task ID.
    pub task_id: TaskId,
    /// Final status.
    pub status: TaskStatus,
    /// Response messages.
    pub messages: Vec<Message>,
    /// Produced artifacts.
    pub artifacts: Vec<Artifact>,
    /// Error message if failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Execution duration in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

impl TaskResult {
    /// Create a successful result.
    pub fn success(task_id: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            task_id: task_id.into(),
            status: TaskStatus::Completed,
            messages,
            artifacts: Vec::new(),
            error: None,
            duration_ms: None,
        }
    }

    /// Create a failed result.
    pub fn failure(task_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            status: TaskStatus::Failed,
            messages: Vec::new(),
            artifacts: Vec::new(),
            error: Some(error.into()),
            duration_ms: None,
        }
    }

    /// Add artifacts to the result.
    pub fn with_artifacts(mut self, artifacts: Vec<Artifact>) -> Self {
        self.artifacts = artifacts;
        self
    }

    /// Set the duration.
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    /// Check if the result is successful.
    pub fn is_success(&self) -> bool {
        self.status == TaskStatus::Completed
    }

    /// Check if the result is a failure.
    pub fn is_failure(&self) -> bool {
        self.status == TaskStatus::Failed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Message;

    #[test]
    fn test_task_creation() {
        let task = Task::new("thread-1", Message::user("Hello"));
        assert!(task.is_pending());
        assert!(!task.is_running());
        assert!(!task.is_completed());
    }

    #[test]
    fn test_task_includes_original_message() {
        let task = Task::new("thread-1", Message::user("Hello"));
        
        // Original message should be in the messages list
        assert_eq!(task.messages.len(), 1);
        assert_eq!(task.messages[0].text_content(), "Hello");
        assert_eq!(task.message.text_content(), "Hello");
    }

    #[test]
    fn test_task_with_id_includes_original_message() {
        let task = Task::with_id("my-id", "thread-1", Message::user("Test"));
        
        assert_eq!(task.id, "my-id");
        assert_eq!(task.messages.len(), 1);
        assert_eq!(task.messages[0].text_content(), "Test");
    }

    #[test]
    fn test_task_lifecycle() {
        let mut task = Task::new("thread-1", Message::user("Hello"));

        task.start().unwrap();
        assert!(task.is_running());

        task.complete().unwrap();
        assert!(task.is_completed());
        assert!(task.is_success());
    }

    #[test]
    fn test_task_failure() {
        let mut task = Task::new("thread-1", Message::user("Hello"));

        task.start().unwrap();
        task.fail("Something went wrong").unwrap();

        assert!(task.is_failed());
        assert_eq!(task.error, Some("Something went wrong".to_string()));
    }

    #[test]
    fn test_invalid_state_transition_start_from_running() {
        let mut task = Task::new("thread-1", Message::user("Hello"));
        task.start().unwrap();
        
        let result = task.start();
        assert!(result.is_err());
        match result {
            Err(TaskError::InvalidStateTransition { from, to }) => {
                assert_eq!(from, TaskStatus::Running);
                assert_eq!(to, TaskStatus::Running);
            }
            _ => panic!("Expected InvalidStateTransition error"),
        }
    }

    #[test]
    fn test_invalid_state_transition_complete_from_pending() {
        let mut task = Task::new("thread-1", Message::user("Hello"));
        
        let result = task.complete();
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_state_transition_complete_after_cancel() {
        let mut task = Task::new("thread-1", Message::user("Hello"));
        task.cancel().unwrap();
        
        // Can't complete an already cancelled task
        let result = task.complete();
        assert!(result.is_err());
    }

    #[test]
    fn test_cancel_from_pending() {
        let mut task = Task::new("thread-1", Message::user("Hello"));
        assert!(task.cancel().is_ok());
        assert_eq!(task.status, TaskStatus::Cancelled);
    }

    #[test]
    fn test_cancel_from_running() {
        let mut task = Task::new("thread-1", Message::user("Hello"));
        task.start().unwrap();
        assert!(task.cancel().is_ok());
        assert_eq!(task.status, TaskStatus::Cancelled);
    }

    #[test]
    fn test_cannot_cancel_completed_task() {
        let mut task = Task::new("thread-1", Message::user("Hello"));
        task.start().unwrap();
        task.complete().unwrap();
        
        let result = task.cancel();
        assert!(result.is_err());
    }

    #[test]
    fn test_force_status() {
        let mut task = Task::new("thread-1", Message::user("Hello"));
        task.force_status(TaskStatus::Completed);
        assert_eq!(task.status, TaskStatus::Completed);
    }

    #[test]
    fn test_task_result_success() {
        let result = TaskResult::success("task-1", vec![Message::agent("Done!")]);
        assert!(result.is_success());
        assert!(!result.is_failure());
    }

    #[test]
    fn test_task_result_failure() {
        let result = TaskResult::failure("task-1", "Error occurred");
        assert!(result.is_failure());
        assert_eq!(result.error, Some("Error occurred".to_string()));
    }

    #[test]
    fn test_status_display() {
        assert_eq!(TaskStatus::Pending.to_string(), "pending");
        assert_eq!(TaskStatus::Running.to_string(), "running");
        assert_eq!(TaskStatus::Completed.to_string(), "completed");
        assert_eq!(TaskStatus::Failed.to_string(), "failed");
        assert_eq!(TaskStatus::Cancelled.to_string(), "cancelled");
    }
}

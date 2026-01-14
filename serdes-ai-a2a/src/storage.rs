//! Storage abstraction for A2A tasks.
//!
//! This module provides the storage trait and implementations for persisting tasks.

use crate::task::{Task, TaskId};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

/// Errors that can occur during storage operations.
#[derive(Debug, Error)]
pub enum StorageError {
    /// Task was not found.
    #[error("Task not found: {0}")]
    NotFound(TaskId),

    /// Task already exists.
    #[error("Task already exists: {0}")]
    AlreadyExists(TaskId),

    /// Storage backend error.
    #[error("Storage error: {0}")]
    Backend(String),

    /// Serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Storage trait for task persistence.
///
/// Implementations of this trait handle storing and retrieving tasks.
/// The default implementation uses an in-memory HashMap.
#[async_trait]
pub trait Storage: Send + Sync {
    /// Get a task by ID.
    async fn get_task(&self, task_id: &str) -> Result<Option<Task>, StorageError>;

    /// Save a new task.
    async fn save_task(&self, task: &Task) -> Result<(), StorageError>;

    /// Update an existing task.
    async fn update_task(&self, task: &Task) -> Result<(), StorageError>;

    /// Delete a task.
    async fn delete_task(&self, task_id: &str) -> Result<(), StorageError>;

    /// List tasks by thread ID.
    async fn list_tasks_by_thread(&self, thread_id: &str) -> Result<Vec<Task>, StorageError>;

    /// List all tasks (with optional limit).
    async fn list_tasks(&self, limit: Option<usize>) -> Result<Vec<Task>, StorageError>;
}

/// In-memory storage implementation.
///
/// Suitable for development and testing. Not recommended for production
/// as data is lost when the process terminates.
#[derive(Debug, Default)]
pub struct InMemoryStorage {
    tasks: Arc<RwLock<HashMap<TaskId, Task>>>,
}

impl InMemoryStorage {
    /// Create a new in-memory storage.
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get the number of stored tasks.
    pub async fn len(&self) -> usize {
        self.tasks.read().await.len()
    }

    /// Check if storage is empty.
    pub async fn is_empty(&self) -> bool {
        self.tasks.read().await.is_empty()
    }

    /// Clear all tasks.
    pub async fn clear(&self) {
        self.tasks.write().await.clear();
    }
}

#[async_trait]
impl Storage for InMemoryStorage {
    async fn get_task(&self, task_id: &str) -> Result<Option<Task>, StorageError> {
        let tasks = self.tasks.read().await;
        Ok(tasks.get(task_id).cloned())
    }

    async fn save_task(&self, task: &Task) -> Result<(), StorageError> {
        let mut tasks = self.tasks.write().await;
        if tasks.contains_key(&task.id) {
            return Err(StorageError::AlreadyExists(task.id.clone()));
        }
        tasks.insert(task.id.clone(), task.clone());
        Ok(())
    }

    async fn update_task(&self, task: &Task) -> Result<(), StorageError> {
        let mut tasks = self.tasks.write().await;
        if !tasks.contains_key(&task.id) {
            return Err(StorageError::NotFound(task.id.clone()));
        }
        tasks.insert(task.id.clone(), task.clone());
        Ok(())
    }

    async fn delete_task(&self, task_id: &str) -> Result<(), StorageError> {
        let mut tasks = self.tasks.write().await;
        if tasks.remove(task_id).is_none() {
            return Err(StorageError::NotFound(task_id.to_string()));
        }
        Ok(())
    }

    async fn list_tasks_by_thread(&self, thread_id: &str) -> Result<Vec<Task>, StorageError> {
        let tasks = self.tasks.read().await;
        let thread_tasks: Vec<Task> = tasks
            .values()
            .filter(|t| t.thread_id == thread_id)
            .cloned()
            .collect();
        Ok(thread_tasks)
    }

    async fn list_tasks(&self, limit: Option<usize>) -> Result<Vec<Task>, StorageError> {
        let tasks = self.tasks.read().await;
        let mut all_tasks: Vec<Task> = tasks.values().cloned().collect();

        // Sort by created_at descending (newest first)
        all_tasks.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        if let Some(limit) = limit {
            all_tasks.truncate(limit);
        }

        Ok(all_tasks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Message;

    #[tokio::test]
    async fn test_save_and_get_task() {
        let storage = InMemoryStorage::new();
        let task = Task::new("thread-1", Message::user("Hello"));
        let task_id = task.id.clone();

        storage.save_task(&task).await.unwrap();

        let retrieved = storage.get_task(&task_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, task_id);
    }

    #[tokio::test]
    async fn test_save_duplicate_task() {
        let storage = InMemoryStorage::new();
        let task = Task::new("thread-1", Message::user("Hello"));

        storage.save_task(&task).await.unwrap();
        let result = storage.save_task(&task).await;

        assert!(matches!(result, Err(StorageError::AlreadyExists(_))));
    }

    #[tokio::test]
    async fn test_update_task() {
        let storage = InMemoryStorage::new();
        let mut task = Task::new("thread-1", Message::user("Hello"));

        storage.save_task(&task).await.unwrap();

        let _ = task.start();
        storage.update_task(&task).await.unwrap();

        let retrieved = storage.get_task(&task.id).await.unwrap().unwrap();
        assert!(retrieved.is_running());
    }

    #[tokio::test]
    async fn test_update_nonexistent_task() {
        let storage = InMemoryStorage::new();
        let task = Task::new("thread-1", Message::user("Hello"));

        let result = storage.update_task(&task).await;
        assert!(matches!(result, Err(StorageError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_delete_task() {
        let storage = InMemoryStorage::new();
        let task = Task::new("thread-1", Message::user("Hello"));
        let task_id = task.id.clone();

        storage.save_task(&task).await.unwrap();
        storage.delete_task(&task_id).await.unwrap();

        let retrieved = storage.get_task(&task_id).await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_list_tasks_by_thread() {
        let storage = InMemoryStorage::new();

        let task1 = Task::new("thread-1", Message::user("Hello"));
        let task2 = Task::new("thread-1", Message::user("World"));
        let task3 = Task::new("thread-2", Message::user("Other"));

        storage.save_task(&task1).await.unwrap();
        storage.save_task(&task2).await.unwrap();
        storage.save_task(&task3).await.unwrap();

        let thread1_tasks = storage.list_tasks_by_thread("thread-1").await.unwrap();
        assert_eq!(thread1_tasks.len(), 2);

        let thread2_tasks = storage.list_tasks_by_thread("thread-2").await.unwrap();
        assert_eq!(thread2_tasks.len(), 1);
    }

    #[tokio::test]
    async fn test_list_tasks_with_limit() {
        let storage = InMemoryStorage::new();

        for i in 0..10 {
            let task = Task::new("thread-1", Message::user(format!("Task {}", i)));
            storage.save_task(&task).await.unwrap();
        }

        let limited = storage.list_tasks(Some(5)).await.unwrap();
        assert_eq!(limited.len(), 5);

        let unlimited = storage.list_tasks(None).await.unwrap();
        assert_eq!(unlimited.len(), 10);
    }

    #[tokio::test]
    async fn test_clear_storage() {
        let storage = InMemoryStorage::new();

        let task = Task::new("thread-1", Message::user("Hello"));
        storage.save_task(&task).await.unwrap();

        assert!(!storage.is_empty().await);

        storage.clear().await;
        assert!(storage.is_empty().await);
    }
}

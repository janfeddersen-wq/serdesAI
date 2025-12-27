//! State persistence for graph execution.

use crate::error::{GraphError, GraphResult};
use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Error during persistence operations.
#[derive(Debug, thiserror::Error)]
pub enum PersistenceError {
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// State not found.
    #[error("State not found for run: {0}")]
    NotFound(String),

    /// Other error.
    #[error("{0}")]
    Other(String),
}

impl From<PersistenceError> for GraphError {
    fn from(e: PersistenceError) -> Self {
        GraphError::Persistence(e.to_string())
    }
}

/// Trait for persisting graph state.
#[async_trait]
pub trait StatePersistence<State, End>: Send + Sync {
    /// Save state for a run.
    async fn save_state(
        &self,
        run_id: &str,
        state: &State,
        step: u32,
    ) -> Result<(), PersistenceError>;

    /// Load state for a run.
    async fn load_state(
        &self,
        run_id: &str,
    ) -> Result<Option<(State, u32)>, PersistenceError>;

    /// Save the final result.
    async fn save_result(
        &self,
        run_id: &str,
        result: &End,
    ) -> Result<(), PersistenceError>;

    /// Load the final result.
    async fn load_result(
        &self,
        run_id: &str,
    ) -> Result<Option<End>, PersistenceError>;

    /// Delete state for a run.
    async fn delete(&self, run_id: &str) -> Result<(), PersistenceError>;

    /// List all stored run IDs.
    async fn list_runs(&self) -> Result<Vec<String>, PersistenceError>;
}

/// In-memory state persistence.
#[derive(Clone)]
pub struct InMemoryPersistence<State, End> {
    states: Arc<RwLock<HashMap<String, (State, u32)>>>,
    results: Arc<RwLock<HashMap<String, End>>>,
}

impl<State, End> InMemoryPersistence<State, End> {
    /// Create a new in-memory persistence store.
    pub fn new() -> Self {
        Self {
            states: Arc::new(RwLock::new(HashMap::new())),
            results: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Clear all stored data.
    pub fn clear(&self) {
        self.states.write().clear();
        self.results.write().clear();
    }

    /// Get the number of stored states.
    pub fn state_count(&self) -> usize {
        self.states.read().len()
    }

    /// Get the number of stored results.
    pub fn result_count(&self) -> usize {
        self.results.read().len()
    }
}

impl<State, End> Default for InMemoryPersistence<State, End> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<State, End> StatePersistence<State, End> for InMemoryPersistence<State, End>
where
    State: Clone + Send + Sync + 'static,
    End: Clone + Send + Sync + 'static,
{
    async fn save_state(
        &self,
        run_id: &str,
        state: &State,
        step: u32,
    ) -> Result<(), PersistenceError> {
        self.states
            .write()
            .insert(run_id.to_string(), (state.clone(), step));
        Ok(())
    }

    async fn load_state(
        &self,
        run_id: &str,
    ) -> Result<Option<(State, u32)>, PersistenceError> {
        Ok(self.states.read().get(run_id).cloned())
    }

    async fn save_result(
        &self,
        run_id: &str,
        result: &End,
    ) -> Result<(), PersistenceError> {
        self.results.write().insert(run_id.to_string(), result.clone());
        Ok(())
    }

    async fn load_result(
        &self,
        run_id: &str,
    ) -> Result<Option<End>, PersistenceError> {
        Ok(self.results.read().get(run_id).cloned())
    }

    async fn delete(&self, run_id: &str) -> Result<(), PersistenceError> {
        self.states.write().remove(run_id);
        self.results.write().remove(run_id);
        Ok(())
    }

    async fn list_runs(&self) -> Result<Vec<String>, PersistenceError> {
        let state_keys: std::collections::HashSet<_> = 
            self.states.read().keys().cloned().collect();
        let result_keys: std::collections::HashSet<_> = 
            self.results.read().keys().cloned().collect();
        Ok(state_keys.union(&result_keys).cloned().collect())
    }
}

/// File-based state persistence.
pub struct FilePersistence {
    directory: PathBuf,
}

impl FilePersistence {
    /// Create a new file-based persistence store.
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    /// Ensure the directory exists.
    pub async fn ensure_dir(&self) -> Result<(), PersistenceError> {
        tokio::fs::create_dir_all(&self.directory).await?;
        Ok(())
    }

    fn state_path(&self, run_id: &str) -> PathBuf {
        self.directory.join(format!("{}_state.json", run_id))
    }

    fn result_path(&self, run_id: &str) -> PathBuf {
        self.directory.join(format!("{}_result.json", run_id))
    }
}

#[async_trait]
impl<State, End> StatePersistence<State, End> for FilePersistence
where
    State: Serialize + DeserializeOwned + Send + Sync + 'static,
    End: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    async fn save_state(
        &self,
        run_id: &str,
        state: &State,
        step: u32,
    ) -> Result<(), PersistenceError> {
        self.ensure_dir().await?;
        let path = self.state_path(run_id);
        let data = serde_json::json!({
            "state": state,
            "step": step
        });
        let content = serde_json::to_string_pretty(&data)?;
        tokio::fs::write(&path, content).await?;
        Ok(())
    }

    async fn load_state(
        &self,
        run_id: &str,
    ) -> Result<Option<(State, u32)>, PersistenceError> {
        let path = self.state_path(run_id);
        if !path.exists() {
            return Ok(None);
        }

        let content = tokio::fs::read_to_string(&path).await?;
        let value: serde_json::Value = serde_json::from_str(&content)?;
        let state: State = serde_json::from_value(value["state"].clone())?;
        let step = value["step"].as_u64().unwrap_or(0) as u32;
        Ok(Some((state, step)))
    }

    async fn save_result(
        &self,
        run_id: &str,
        result: &End,
    ) -> Result<(), PersistenceError> {
        self.ensure_dir().await?;
        let path = self.result_path(run_id);
        let content = serde_json::to_string_pretty(result)?;
        tokio::fs::write(&path, content).await?;
        Ok(())
    }

    async fn load_result(
        &self,
        run_id: &str,
    ) -> Result<Option<End>, PersistenceError> {
        let path = self.result_path(run_id);
        if !path.exists() {
            return Ok(None);
        }

        let content = tokio::fs::read_to_string(&path).await?;
        let result: End = serde_json::from_str(&content)?;
        Ok(Some(result))
    }

    async fn delete(&self, run_id: &str) -> Result<(), PersistenceError> {
        let state_path = self.state_path(run_id);
        let result_path = self.result_path(run_id);

        if state_path.exists() {
            tokio::fs::remove_file(&state_path).await?;
        }
        if result_path.exists() {
            tokio::fs::remove_file(&result_path).await?;
        }
        Ok(())
    }

    async fn list_runs(&self) -> Result<Vec<String>, PersistenceError> {
        if !self.directory.exists() {
            return Ok(Vec::new());
        }

        let mut runs = std::collections::HashSet::new();
        let mut entries = tokio::fs::read_dir(&self.directory).await?;

        while let Some(entry) = entries.next_entry().await? {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(run_id) = name
                .strip_suffix("_state.json")
                .or_else(|| name.strip_suffix("_result.json"))
            {
                runs.insert(run_id.to_string());
            }
        }

        Ok(runs.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestState {
        value: i32,
    }

    #[tokio::test]
    async fn test_in_memory_persistence() {
        let persistence: InMemoryPersistence<TestState, String> = InMemoryPersistence::new();

        let state = TestState { value: 42 };
        persistence.save_state("run1", &state, 5).await.unwrap();

        let loaded = persistence.load_state("run1").await.unwrap();
        assert!(loaded.is_some());
        let (loaded_state, step) = loaded.unwrap();
        assert_eq!(loaded_state.value, 42);
        assert_eq!(step, 5);
    }

    #[tokio::test]
    async fn test_in_memory_result() {
        let persistence: InMemoryPersistence<TestState, String> = InMemoryPersistence::new();

        persistence
            .save_result("run1", &"success".to_string())
            .await
            .unwrap();

        let loaded = persistence.load_result("run1").await.unwrap();
        assert_eq!(loaded, Some("success".to_string()));
    }

    #[tokio::test]
    async fn test_in_memory_delete() {
        let persistence: InMemoryPersistence<TestState, String> = InMemoryPersistence::new();

        let state = TestState { value: 1 };
        persistence.save_state("run1", &state, 1).await.unwrap();
        persistence.delete("run1").await.unwrap();

        let loaded = persistence.load_state("run1").await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn test_in_memory_list_runs() {
        let persistence: InMemoryPersistence<TestState, String> = InMemoryPersistence::new();

        let state = TestState { value: 1 };
        persistence.save_state("run1", &state, 1).await.unwrap();
        persistence.save_state("run2", &state, 1).await.unwrap();

        let runs = persistence.list_runs().await.unwrap();
        assert_eq!(runs.len(), 2);
    }

    #[tokio::test]
    async fn test_file_persistence() {
        let temp_dir = std::env::temp_dir().join("serdes_ai_test");
        let persistence = FilePersistence::new(&temp_dir);

        let state = TestState { value: 42 };
        StatePersistence::<TestState, String>::save_state(&persistence, "test_run", &state, 5).await.unwrap();

        let loaded: Option<(TestState, u32)> = StatePersistence::<TestState, String>::load_state(&persistence, "test_run").await.unwrap();
        assert!(loaded.is_some());
        let (loaded_state, step) = loaded.unwrap();
        assert_eq!(loaded_state.value, 42);
        assert_eq!(step, 5);

        // Cleanup
        let _ = StatePersistence::<TestState, String>::delete(&persistence, "test_run").await;
    }
}

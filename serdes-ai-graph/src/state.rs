//! Graph state types.

use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;

/// Trait for graph state types.
///
/// State must be clonable, sendable, and debuggable.
/// For persistence, it should also be serializable.
pub trait GraphState: Clone + Send + Sync + Debug + 'static {}

/// Blanket implementation for all compatible types.
impl<T> GraphState for T where T: Clone + Send + Sync + Debug + 'static {}

/// Context passed to nodes during execution.
#[derive(Debug, Clone)]
pub struct GraphRunContext<State, Deps = ()> {
    /// Current state.
    pub state: State,
    /// Dependencies.
    pub deps: Deps,
    /// Current step number.
    pub step: u32,
    /// Unique run identifier.
    pub run_id: String,
    /// Maximum steps allowed.
    pub max_steps: u32,
}

impl<State, Deps> GraphRunContext<State, Deps> {
    /// Create a new context.
    pub fn new(state: State, deps: Deps, run_id: impl Into<String>) -> Self {
        Self {
            state,
            deps,
            step: 0,
            run_id: run_id.into(),
            max_steps: 100,
        }
    }

    /// Set maximum steps.
    pub fn with_max_steps(mut self, max: u32) -> Self {
        self.max_steps = max;
        self
    }

    /// Increment step counter.
    pub fn increment_step(&mut self) {
        self.step += 1;
    }

    /// Check if max steps reached.
    pub fn is_max_steps_reached(&self) -> bool {
        self.step >= self.max_steps
    }
}

impl<State: Default, Deps: Default> Default for GraphRunContext<State, Deps> {
    fn default() -> Self {
        Self {
            state: State::default(),
            deps: Deps::default(),
            step: 0,
            run_id: generate_run_id(),
            max_steps: 100,
        }
    }
}

/// Result of a graph run.
#[derive(Debug, Clone)]
pub struct GraphRunResult<State, End = ()> {
    /// Final result value.
    pub result: End,
    /// Final state.
    pub state: State,
    /// Number of steps executed.
    pub steps: u32,
    /// History of node names visited.
    pub history: Vec<String>,
    /// Run ID.
    pub run_id: String,
}

impl<State, End> GraphRunResult<State, End> {
    /// Create a new result.
    pub fn new(result: End, state: State, steps: u32, run_id: impl Into<String>) -> Self {
        Self {
            result,
            state,
            steps,
            history: Vec::new(),
            run_id: run_id.into(),
        }
    }

    /// Add history.
    pub fn with_history(mut self, history: Vec<String>) -> Self {
        self.history = history;
        self
    }
}

/// Generate a unique run ID.
pub fn generate_run_id() -> String {
    use std::time::SystemTime;
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("run-{:x}", timestamp)
}

/// Trait for serializable state (for persistence).
pub trait PersistableState: GraphState + Serialize + DeserializeOwned {}

impl<T> PersistableState for T where T: GraphState + Serialize + DeserializeOwned {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Default)]
    struct TestState {
        value: i32,
    }

    #[test]
    fn test_graph_state_trait() {
        let state = TestState { value: 42 };
        let cloned = state.clone();
        assert_eq!(cloned.value, 42);
    }

    #[test]
    fn test_run_context() {
        let mut ctx = GraphRunContext::new(
            TestState { value: 0 },
            (),
            "test-run",
        );

        assert_eq!(ctx.step, 0);
        ctx.increment_step();
        assert_eq!(ctx.step, 1);
    }

    #[test]
    fn test_max_steps() {
        let ctx = GraphRunContext::new(
            TestState::default(),
            (),
            "test",
        ).with_max_steps(5);

        assert_eq!(ctx.max_steps, 5);
    }

    #[test]
    fn test_generate_run_id() {
        let id1 = generate_run_id();
        let id2 = generate_run_id();
        assert!(id1.starts_with("run-"));
        // IDs might be same if generated in same nanosecond, but that's rare
        assert!(!id1.is_empty());
        assert!(!id2.is_empty());
    }
}

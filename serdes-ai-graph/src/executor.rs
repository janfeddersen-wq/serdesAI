//! Graph execution engine.

use crate::error::{GraphError, GraphResult};
use crate::graph::Graph;
use crate::persistence::StatePersistence;
use crate::state::{generate_run_id, GraphRunContext, GraphRunResult, GraphState};
use std::sync::Arc;
use tracing::{debug, info, span, Level};

/// Graph executor with optional persistence and instrumentation.
pub struct GraphExecutor<State, Deps, End, P = NoPersistence>
where
    State: GraphState,
{
    _persistence_type: std::marker::PhantomData<P>,
    graph: Arc<Graph<State, Deps, End>>,
    persistence: Option<Arc<P>>,
    auto_save: bool,
    max_steps: u32,
    instrumentation: bool,
}

/// Marker type for no persistence.
#[derive(Debug, Clone, Copy)]
pub struct NoPersistence;

impl<State, Deps, End> GraphExecutor<State, Deps, End, NoPersistence>
where
    State: GraphState,
    Deps: Clone + Send + Sync + 'static,
    End: Clone + Send + Sync + 'static,
{
    /// Create a new executor without persistence.
    pub fn new(graph: Graph<State, Deps, End>) -> Self {
        Self {
            _persistence_type: std::marker::PhantomData,
            graph: Arc::new(graph),
            persistence: None,
            auto_save: false,
            max_steps: 100,
            instrumentation: true,
        }
    }
}

impl<State, Deps, End, P> GraphExecutor<State, Deps, End, P>
where
    State: GraphState,
    Deps: Clone + Send + Sync + 'static,
    End: Clone + Send + Sync + 'static,
    P: StatePersistence<State, End> + 'static,
{
    /// Create an executor with persistence.
    pub fn with_persistence(
        graph: Graph<State, Deps, End>,
        persistence: P,
    ) -> Self {
        Self {
            _persistence_type: std::marker::PhantomData,
            graph: Arc::new(graph),
            persistence: Some(Arc::new(persistence)),
            auto_save: true,
            max_steps: 100,
            instrumentation: true,
        }
    }

    /// Set whether to automatically save state.
    pub fn auto_save(mut self, enabled: bool) -> Self {
        self.auto_save = enabled;
        self
    }

    /// Set maximum steps.
    pub fn max_steps(mut self, max: u32) -> Self {
        self.max_steps = max;
        self
    }

    /// Disable instrumentation.
    pub fn without_instrumentation(mut self) -> Self {
        self.instrumentation = false;
        self
    }

    /// Get a reference to the graph.
    pub fn graph(&self) -> &Graph<State, Deps, End> {
        &self.graph
    }

    /// Run the graph.
    pub async fn run(
        &self,
        state: State,
        deps: Deps,
    ) -> GraphResult<GraphRunResult<State, End>> {
        let run_id = generate_run_id();

        if self.instrumentation {
            let _span = span!(Level::INFO, "graph_run", run_id = %run_id).entered();
            info!("Starting graph execution");
        }

        self.graph.run(state, deps).await
    }

    /// Resume a previous run.
    pub async fn resume(
        &self,
        run_id: &str,
        deps: Deps,
    ) -> GraphResult<Option<GraphRunResult<State, End>>> {
        let Some(ref persistence) = self.persistence else {
            return Err(GraphError::persistence("No persistence configured"));
        };

        let Some((state, _step)) = persistence.load_state(run_id).await? else {
            return Ok(None);
        };

        // Resume from the loaded state
        let result = self.graph.run(state, deps).await?;

        // Save final result
        if self.auto_save {
            persistence.save_result(run_id, &result.result).await?;
        }

        Ok(Some(result))
    }

    /// Get a saved result.
    pub async fn get_result(&self, run_id: &str) -> GraphResult<Option<End>> {
        let Some(ref persistence) = self.persistence else {
            return Err(GraphError::persistence("No persistence configured"));
        };

        Ok(persistence.load_result(run_id).await?)
    }

    /// List all saved runs.
    pub async fn list_runs(&self) -> GraphResult<Vec<String>> {
        let Some(ref persistence) = self.persistence else {
            return Err(GraphError::persistence("No persistence configured"));
        };

        Ok(persistence.list_runs().await?)
    }
}

/// Execution options.
#[derive(Debug, Clone)]
pub struct ExecutionOptions {
    /// Maximum steps.
    pub max_steps: u32,
    /// Enable tracing.
    pub tracing: bool,
    /// Checkpoint interval.
    pub checkpoint_interval: Option<u32>,
    /// Custom run ID.
    pub run_id: Option<String>,
}

impl Default for ExecutionOptions {
    fn default() -> Self {
        Self {
            max_steps: 100,
            tracing: true,
            checkpoint_interval: None,
            run_id: None,
        }
    }
}

impl ExecutionOptions {
    /// Create new options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max steps.
    pub fn max_steps(mut self, max: u32) -> Self {
        self.max_steps = max;
        self
    }

    /// Enable or disable tracing.
    pub fn tracing(mut self, enabled: bool) -> Self {
        self.tracing = enabled;
        self
    }

    /// Set checkpoint interval.
    pub fn checkpoint_every(mut self, steps: u32) -> Self {
        self.checkpoint_interval = Some(steps);
        self
    }

    /// Set custom run ID.
    pub fn run_id(mut self, id: impl Into<String>) -> Self {
        self.run_id = Some(id.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_options() {
        let opts = ExecutionOptions::new()
            .max_steps(50)
            .tracing(false)
            .checkpoint_every(10)
            .run_id("custom-run");

        assert_eq!(opts.max_steps, 50);
        assert!(!opts.tracing);
        assert_eq!(opts.checkpoint_interval, Some(10));
        assert_eq!(opts.run_id, Some("custom-run".to_string()));
    }
}

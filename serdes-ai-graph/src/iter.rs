//! Graph iteration support.

use crate::error::GraphError;
use crate::node::{BaseNode, NodeResult};
use crate::state::{generate_run_id, GraphRunContext, GraphRunResult, GraphState};
use std::marker::PhantomData;

/// Iterator for stepping through a graph.
pub struct GraphIter<'a, State, Deps, End>
where
    State: GraphState,
    Deps: Clone + Send + Sync + 'static,
    End: Clone + Send + Sync + 'static,
{
    ctx: GraphRunContext<State, Deps>,
    current: Option<Box<dyn BaseNode<State, Deps, End>>>,
    finished: bool,
    result: Option<End>,
    history: Vec<String>,
    _phantom: PhantomData<&'a ()>,
}

impl<'a, State, Deps, End> GraphIter<'a, State, Deps, End>
where
    State: GraphState,
    Deps: Clone + Send + Sync + 'static,
    End: Clone + Send + Sync + 'static,
{
    /// Create a new graph iterator.
    pub fn new<N: BaseNode<State, Deps, End> + Clone + 'static>(
        start: N,
        state: State,
        deps: Deps,
    ) -> Self {
        let run_id = generate_run_id();
        Self {
            ctx: GraphRunContext::new(state, deps, run_id),
            current: Some(Box::new(start)),
            finished: false,
            result: None,
            history: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Execute the next step.
    pub async fn step(&mut self) -> Option<StepResult<State>> {
        if self.finished {
            return None;
        }

        let current = self.current.take()?;
        self.ctx.increment_step();

        let node_name = current.name().to_string();
        self.history.push(node_name.clone());

        match current.run(&mut self.ctx).await {
            Ok(NodeResult::Next(next)) => {
                self.current = Some(next);
                Some(StepResult::Continue { node: node_name })
            }
            Ok(NodeResult::NextNamed(name)) => {
                // Named transitions require external graph lookup
                self.finished = true;
                Some(StepResult::NamedTransition {
                    node: node_name,
                    next: name,
                })
            }
            Ok(NodeResult::End(_end)) => {
                self.finished = true;
                Some(StepResult::Finished { node: node_name })
            }
            Err(e) => {
                self.finished = true;
                Some(StepResult::Error(e))
            }
        }
    }

    /// Get the current state.
    pub fn state(&self) -> &State {
        &self.ctx.state
    }

    /// Get mutable state.
    pub fn state_mut(&mut self) -> &mut State {
        &mut self.ctx.state
    }

    /// Get the current step number.
    pub fn step_count(&self) -> u32 {
        self.ctx.step
    }

    /// Check if finished.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Get the history of visited nodes.
    pub fn history(&self) -> &[String] {
        &self.history
    }

    /// Consume and get the result.
    pub fn into_result(self) -> Option<GraphRunResult<State, End>> {
        self.result.map(|r| {
            GraphRunResult::new(r, self.ctx.state, self.ctx.step, self.ctx.run_id)
                .with_history(self.history)
        })
    }
}

/// Result of a single step.
#[derive(Debug)]
pub enum StepResult<State> {
    /// Graph continues to next node.
    Continue {
        /// Node that was executed.
        node: String,
    },
    /// Named transition (requires lookup).
    NamedTransition {
        /// Node that was executed.
        node: String,
        /// Name of next node.
        next: String,
    },
    /// Graph finished.
    Finished {
        /// Final node.
        node: String,
    },
    /// Error occurred.
    Error(GraphError),
    /// Phantom state type holder.
    #[doc(hidden)]
    _State(PhantomData<State>),
}

impl<State> StepResult<State> {
    /// Check if this step finished the graph.
    pub fn is_finished(&self) -> bool {
        matches!(self, Self::Finished { .. })
    }

    /// Check if this step had an error.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// Get the node name if applicable.
    pub fn node(&self) -> Option<&str> {
        match self {
            Self::Continue { node } => Some(node),
            Self::NamedTransition { node, .. } => Some(node),
            Self::Finished { node } => Some(node),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Default)]
    struct TestState {
        _value: i32,
    }

    #[test]
    fn test_step_result_is_finished() {
        let result: StepResult<TestState> = StepResult::Finished {
            node: "test".to_string(),
        };
        assert!(result.is_finished());
    }

    #[test]
    fn test_step_result_is_error() {
        let result: StepResult<TestState> = StepResult::Error(GraphError::NoEntryNode);
        assert!(result.is_error());
    }

    #[test]
    fn test_step_result_node() {
        let result: StepResult<TestState> = StepResult::Continue {
            node: "my_node".to_string(),
        };
        assert_eq!(result.node(), Some("my_node"));
    }
}

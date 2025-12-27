//! # serdes-ai-graph
//!
//! Graph-based execution and multi-agent orchestration for serdes-ai.
//!
//! This crate provides a powerful graph execution engine for building
//! complex, multi-step AI workflows with conditional branching and
//! state management.
//!
//! ## Core Concepts
//!
//! - **[`Graph`]**: The main graph type, generic over state, deps, and end result
//! - **[`BaseNode`]**: Trait for nodes that can execute within a graph
//! - **[`NodeResult`]**: Enum indicating next step or termination
//! - **[`Edge`]**: Conditional transitions between nodes
//!
//! ## Node Types
//!
//! - **[`FunctionNode`]**: Execute an async function
//! - **[`AgentNode`]**: Run an agent and update state
//! - **[`RouterNode`]**: Dynamic routing based on state
//! - **[`ConditionalNode`]**: Branch based on condition
//!
//! ## State Persistence
//!
//! - **[`StatePersistence`]**: Trait for saving/loading state
//! - **[`InMemoryPersistence`]**: In-memory state storage
//! - **[`FilePersistence`]**: File-based state storage
//!
//! ## Example
//!
//! ```ignore
//! use serdes_ai_graph::{Graph, BaseNode, NodeResult, GraphRunContext};
//! use async_trait::async_trait;
//!
//! #[derive(Debug, Clone, Default)]
//! struct WorkflowState {
//!     query: String,
//!     response: Option<String>,
//! }
//!
//! struct ProcessNode;
//!
//! #[async_trait]
//! impl BaseNode<WorkflowState, (), String> for ProcessNode {
//!     fn name(&self) -> &str { "process" }
//!
//!     async fn run(
//!         &self,
//!         ctx: &mut GraphRunContext<WorkflowState, ()>,
//!     ) -> Result<NodeResult<WorkflowState, (), String>, GraphError> {
//!         ctx.state.response = Some(format!("Processed: {}", ctx.state.query));
//!         Ok(NodeResult::end(ctx.state.response.clone().unwrap()))
//!     }
//! }
//!
//! let graph = Graph::new()
//!     .node("process", ProcessNode)
//!     .entry("process")
//!     .build()?;
//!
//! let result = graph.run(WorkflowState::default(), ()).await?;
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod edge;
pub mod error;
pub mod executor;
pub mod graph;
pub mod iter;
pub mod mermaid;
pub mod node;
pub mod persistence;
pub mod state;

// Re-exports
pub use edge::{Edge, EdgeBuilder};
pub use error::{GraphError, GraphResult};
pub use executor::{ExecutionOptions, GraphExecutor, NoPersistence};
pub use graph::{Graph, SimpleGraph};
pub use iter::GraphIter;
pub use iter::StepResult;
pub use mermaid::{generate_flowchart, generate_mermaid, MermaidBuilder, MermaidDirection, MermaidOptions};
pub use node::{
    AgentNode, BaseNode, ConditionalNode, End, FunctionNode, Node, NodeDef, NodeResult, RouterNode,
};
pub use persistence::{FilePersistence, InMemoryPersistence, PersistenceError, StatePersistence};
pub use state::{generate_run_id, GraphRunContext, GraphRunResult, GraphState, PersistableState};

/// Prelude for common imports.
pub mod prelude {
    pub use crate::{
        BaseNode, Edge, End, Graph, GraphError, GraphExecutor, GraphResult, GraphRunContext,
        GraphRunResult, GraphState, NodeResult,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    #[derive(Debug, Clone, Default)]
    struct TestState {
        value: i32,
    }

    struct TestNode;

    #[async_trait]
    impl BaseNode<TestState, (), i32> for TestNode {
        fn name(&self) -> &str {
            "test"
        }

        async fn run(
            &self,
            ctx: &mut GraphRunContext<TestState, ()>,
        ) -> GraphResult<NodeResult<TestState, (), i32>> {
            ctx.state.value += 1;
            Ok(NodeResult::end(ctx.state.value))
        }
    }

    #[tokio::test]
    async fn test_simple_graph() {
        let graph = Graph::new()
            .node("test", TestNode)
            .entry("test")
            .build()
            .unwrap();

        let result = graph.run(TestState::default(), ()).await.unwrap();
        assert_eq!(result.result, 1);
    }

    #[test]
    fn test_edge_builder() {
        let edge = EdgeBuilder::<TestState>::from("a")
            .to("b")
            .always();

        assert_eq!(edge.from, "a");
        assert_eq!(edge.to, "b");
    }

    #[test]
    fn test_mermaid_generation() {
        let diagram = generate_flowchart(
            &["start", "end"],
            &[("start", "end", None)],
            &MermaidOptions::new(),
        );

        assert!(diagram.contains("flowchart"));
    }
}

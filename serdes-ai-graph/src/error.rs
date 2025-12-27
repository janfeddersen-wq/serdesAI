//! Graph error types.

use thiserror::Error;

/// Errors that can occur during graph execution.
#[derive(Error, Debug)]
pub enum GraphError {
    /// No entry node defined.
    #[error("No entry node defined")]
    NoEntryNode,

    /// Node not found.
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    /// Cycle detected.
    #[error("Cycle detected in graph")]
    CycleDetected,

    /// Node execution failed.
    #[error("Node '{node}' execution failed: {message}")]
    ExecutionFailed {
        /// Node name.
        node: String,
        /// Error message.
        message: String,
    },

    /// Maximum steps exceeded.
    #[error("Maximum steps exceeded: {0}")]
    MaxStepsExceeded(u32),

    /// Invalid graph structure.
    #[error("Invalid graph: {0}")]
    InvalidGraph(String),

    /// Persistence error.
    #[error("Persistence error: {0}")]
    Persistence(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Other error.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

impl GraphError {
    /// Create a node not found error.
    pub fn node_not_found(name: impl Into<String>) -> Self {
        Self::NodeNotFound(name.into())
    }

    /// Create an execution failed error.
    pub fn execution_failed(node: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ExecutionFailed {
            node: node.into(),
            message: message.into(),
        }
    }

    /// Create a persistence error.
    pub fn persistence(msg: impl Into<String>) -> Self {
        Self::Persistence(msg.into())
    }
}

/// Result type for graph operations.
pub type GraphResult<T> = Result<T, GraphError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = GraphError::node_not_found("test");
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_execution_failed() {
        let err = GraphError::execution_failed("node1", "failed");
        assert!(err.to_string().contains("node1"));
        assert!(err.to_string().contains("failed"));
    }
}

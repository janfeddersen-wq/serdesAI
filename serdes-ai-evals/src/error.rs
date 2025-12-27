//! Evaluation errors.

use thiserror::Error;

/// Errors that can occur during evaluation.
#[derive(Debug, Error)]
pub enum EvalError {
    /// Dataset loading error.
    #[error("Failed to load dataset: {0}")]
    DatasetLoad(String),

    /// Dataset serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Task execution error.
    #[error("Task execution failed: {0}")]
    TaskFailed(String),

    /// Evaluator error.
    #[error("Evaluator '{evaluator}' failed: {message}")]
    EvaluatorFailed {
        /// Evaluator name.
        evaluator: String,
        /// Error message.
        message: String,
    },

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// YAML error.
    #[error("YAML error: {0}")]
    Yaml(String),

    /// Other error.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

impl EvalError {
    /// Create a dataset load error.
    pub fn dataset_load(msg: impl Into<String>) -> Self {
        Self::DatasetLoad(msg.into())
    }

    /// Create an evaluator failed error.
    pub fn evaluator_failed(evaluator: impl Into<String>, message: impl Into<String>) -> Self {
        Self::EvaluatorFailed {
            evaluator: evaluator.into(),
            message: message.into(),
        }
    }

    /// Create a task failed error.
    pub fn task_failed(msg: impl Into<String>) -> Self {
        Self::TaskFailed(msg.into())
    }
}

/// Result type for evaluation operations.
pub type EvalResult<T> = Result<T, EvalError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = EvalError::dataset_load("file not found");
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn test_evaluator_failed() {
        let err = EvalError::evaluator_failed("ExactMatch", "no expected output");
        let s = err.to_string();
        assert!(s.contains("ExactMatch"));
        assert!(s.contains("no expected output"));
    }
}

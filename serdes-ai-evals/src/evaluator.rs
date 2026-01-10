//! Evaluator traits and implementations.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Result of an evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvaluationResult {
    /// Passed with optional score and message.
    Pass {
        /// Numeric score (0.0 to 1.0).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        score: Option<f64>,
        /// Optional message.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
    /// Failed with reason.
    Fail {
        /// Failure reason.
        reason: String,
        /// Additional details.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<serde_json::Value>,
    },
    /// Skipped (e.g., no expected output).
    Skip {
        /// Skip reason.
        reason: String,
    },
    /// Error during evaluation.
    Error {
        /// Error message.
        error: String,
    },
}

impl EvaluationResult {
    /// Create a pass result.
    pub fn pass() -> Self {
        Self::Pass {
            score: None,
            message: None,
        }
    }

    /// Create a pass result with score.
    pub fn pass_with_score(score: f64) -> Self {
        Self::Pass {
            score: Some(score),
            message: None,
        }
    }

    /// Create a pass result with message.
    pub fn pass_with_message(message: impl Into<String>) -> Self {
        Self::Pass {
            score: None,
            message: Some(message.into()),
        }
    }

    /// Create a pass with score and message.
    pub fn pass_full(score: f64, message: impl Into<String>) -> Self {
        Self::Pass {
            score: Some(score),
            message: Some(message.into()),
        }
    }

    /// Create a fail result.
    pub fn fail(reason: impl Into<String>) -> Self {
        Self::Fail {
            reason: reason.into(),
            details: None,
        }
    }

    /// Create a fail result with details.
    pub fn fail_with_details(reason: impl Into<String>, details: serde_json::Value) -> Self {
        Self::Fail {
            reason: reason.into(),
            details: Some(details),
        }
    }

    /// Create a skip result.
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip {
            reason: reason.into(),
        }
    }

    /// Create an error result.
    pub fn error(error: impl Into<String>) -> Self {
        Self::Error {
            error: error.into(),
        }
    }

    /// Check if passed.
    pub fn is_pass(&self) -> bool {
        matches!(self, Self::Pass { .. })
    }

    /// Check if failed.
    pub fn is_fail(&self) -> bool {
        matches!(self, Self::Fail { .. })
    }

    /// Check if skipped.
    pub fn is_skip(&self) -> bool {
        matches!(self, Self::Skip { .. })
    }

    /// Check if error.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Get the score if present.
    pub fn score(&self) -> Option<f64> {
        match self {
            Self::Pass { score, .. } => *score,
            _ => None,
        }
    }

    /// Convert to numeric (1.0 for pass, 0.0 otherwise).
    pub fn to_numeric(&self) -> f64 {
        match self {
            Self::Pass { score, .. } => score.unwrap_or(1.0),
            _ => 0.0,
        }
    }
}

impl fmt::Display for EvaluationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pass { score, message } => {
                write!(f, "✅ PASS")?;
                if let Some(s) = score {
                    write!(f, " (score: {:.2})", s)?;
                }
                if let Some(m) = message {
                    write!(f, " - {}", m)?;
                }
                Ok(())
            }
            Self::Fail { reason, .. } => write!(f, "❌ FAIL: {}", reason),
            Self::Skip { reason } => write!(f, "⏭️ SKIP: {}", reason),
            Self::Error { error } => write!(f, "⚠️ ERROR: {}", error),
        }
    }
}

/// Context passed to evaluators.
#[derive(Debug)]
pub struct EvaluatorContext<'a, Inputs, TaskOutput, ExpectedOutput = (), Metadata = ()> {
    /// The inputs that were given to the task.
    pub inputs: &'a Inputs,
    /// The output produced by the task.
    pub output: &'a TaskOutput,
    /// The expected output, if any.
    pub expected_output: Option<&'a ExpectedOutput>,
    /// Additional metadata.
    pub metadata: Option<&'a Metadata>,
}

impl<'a, Inputs, TaskOutput, ExpectedOutput, Metadata>
    EvaluatorContext<'a, Inputs, TaskOutput, ExpectedOutput, Metadata>
{
    /// Create a new context.
    pub fn new(
        inputs: &'a Inputs,
        output: &'a TaskOutput,
        expected_output: Option<&'a ExpectedOutput>,
        metadata: Option<&'a Metadata>,
    ) -> Self {
        Self {
            inputs,
            output,
            expected_output,
            metadata,
        }
    }

    /// Check if expected output is available.
    pub fn has_expected(&self) -> bool {
        self.expected_output.is_some()
    }

    /// Check if metadata is available.
    pub fn has_metadata(&self) -> bool {
        self.metadata.is_some()
    }
}

/// Core evaluator trait.
#[async_trait]
pub trait Evaluator: Send + Sync {
    /// Evaluator name.
    fn name(&self) -> &str;

    /// Run the evaluation on string output.
    async fn evaluate_str(&self, output: &str, expected: Option<&str>) -> EvaluationResult;
}

/// Generic evaluator trait with type parameters.
#[async_trait]
pub trait TypedEvaluator<Inputs, TaskOutput, ExpectedOutput = (), Metadata = ()>:
    Send + Sync
{
    /// Evaluator name.
    fn name(&self) -> &str;

    /// Run the evaluation.
    async fn evaluate(
        &self,
        ctx: &EvaluatorContext<'_, Inputs, TaskOutput, ExpectedOutput, Metadata>,
    ) -> EvaluationResult;
}

/// Boxed evaluator for dynamic dispatch.
pub type BoxedEvaluator = Box<dyn Evaluator>;

/// Named evaluation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedEvaluationResult {
    /// Evaluator name.
    pub evaluator: String,
    /// Result.
    pub result: EvaluationResult,
}

impl NamedEvaluationResult {
    /// Create a new named result.
    pub fn new(evaluator: impl Into<String>, result: EvaluationResult) -> Self {
        Self {
            evaluator: evaluator.into(),
            result,
        }
    }
}

/// Collection of evaluators.
#[derive(Default)]
pub struct EvaluatorSet {
    evaluators: Vec<BoxedEvaluator>,
}

impl EvaluatorSet {
    /// Create an empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an evaluator.
    pub fn with_evaluator<E: Evaluator + 'static>(mut self, evaluator: E) -> Self {
        self.evaluators.push(Box::new(evaluator));
        self
    }

    /// Run all evaluators.
    pub async fn evaluate(
        &self,
        output: &str,
        expected: Option<&str>,
    ) -> Vec<NamedEvaluationResult> {
        let mut results = Vec::new();
        for evaluator in &self.evaluators {
            let result = evaluator.evaluate_str(output, expected).await;
            results.push(NamedEvaluationResult::new(evaluator.name(), result));
        }
        results
    }

    /// Get the number of evaluators.
    pub fn len(&self) -> usize {
        self.evaluators.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.evaluators.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation_result_pass() {
        let result = EvaluationResult::pass();
        assert!(result.is_pass());
        assert!(!result.is_fail());
        assert_eq!(result.to_numeric(), 1.0);
    }

    #[test]
    fn test_evaluation_result_pass_with_score() {
        let result = EvaluationResult::pass_with_score(0.85);
        assert!(result.is_pass());
        assert_eq!(result.score(), Some(0.85));
        assert_eq!(result.to_numeric(), 0.85);
    }

    #[test]
    fn test_evaluation_result_fail() {
        let result = EvaluationResult::fail("mismatch");
        assert!(result.is_fail());
        assert_eq!(result.to_numeric(), 0.0);
    }

    #[test]
    fn test_evaluation_result_display() {
        assert!(EvaluationResult::pass().to_string().contains("PASS"));
        assert!(EvaluationResult::fail("bad").to_string().contains("FAIL"));
        assert!(EvaluationResult::skip("no expected")
            .to_string()
            .contains("SKIP"));
    }

    #[test]
    fn test_evaluator_context() {
        let inputs = "test input";
        let output = "test output";
        let expected = "expected";

        let ctx = EvaluatorContext::new(&inputs, &output, Some(&expected), None::<&()>);

        assert!(ctx.has_expected());
        assert!(!ctx.has_metadata());
    }

    #[test]
    fn test_named_evaluation_result() {
        let result = NamedEvaluationResult::new("ExactMatch", EvaluationResult::pass());
        assert_eq!(result.evaluator, "ExactMatch");
        assert!(result.result.is_pass());
    }
}

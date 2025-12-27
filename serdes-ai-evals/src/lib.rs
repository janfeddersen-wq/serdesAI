//! # serdes-ai-evals
//!
//! Evaluation framework for testing and benchmarking serdes-ai agents.
//!
//! This crate provides tools for systematically evaluating agent performance
//! across test cases, measuring accuracy, latency, and cost.
//!
//! ## Core Concepts
//!
//! - **[`Case`] / [`EvalCase`]**: Individual test cases with inputs and expected outputs
//! - **[`Dataset`] / [`EvalSuite`]**: Collections of test cases
//! - **[`Evaluator`]**: Trait for implementing custom evaluators
//! - **[`EvalRunner`]**: Runs evaluations and collects results
//! - **[`EvaluationReport`]**: Detailed results with statistics
//!
//! ## Built-in Evaluators
//!
//! - **[`ExactMatchScorer`]**: Output must match expected exactly
//! - **[`ContainsScorer`]**: Output must contain expected substring
//! - **[`RegexScorer`]**: Output must match regex pattern
//! - **[`LengthScorer`]**: Output must meet length constraints
//! - **[`FunctionScorer`]**: Custom evaluation function
//!
//! ## Example
//!
//! ```ignore
//! use serdes_ai_evals::{EvalRunner, EvalCase, EvalSuite, ExactMatchScorer, ContainsScorer};
//!
//! let suite = EvalSuite::new("weather_agent_tests")
//!     .add_case(EvalCase::new()
//!         .input("What's the weather in NYC?")
//!         .expected_contains("New York")
//!         .expected_contains("temperature"))
//!     .add_case(EvalCase::new()
//!         .input("Weather in London")
//!         .expected_contains("London"));
//!
//! let runner = EvalRunner::new()
//!     .evaluator(ContainsScorer::new("weather"));
//!
//! // Would run with actual agent
//! ```
//!
//! ## Quick Evaluation
//!
//! ```ignore
//! use serdes_ai_evals::quick_eval;
//!
//! let report = quick_eval(
//!     vec![
//!         ("What is 2+2?", Some("4")),
//!         ("What is 3+3?", Some("6")),
//!     ],
//!     |input| async move { calculate(input) },
//! ).await?;
//!
//! println!("Pass rate: {:.1}%", report.summary.pass_rate * 100.0);
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod case;
pub mod dataset;
pub mod error;
pub mod evaluator;
pub mod metrics;
pub mod report;
pub mod result;
pub mod runner;
pub mod scorers;
pub mod suite;

// Re-exports
pub use case::{Case, EvalCase, Expected};
pub use dataset::{Dataset, DatasetBuilder};
pub use error::{EvalError, EvalResult};
pub use evaluator::{
    BoxedEvaluator, EvaluationResult, Evaluator, EvaluatorContext, EvaluatorSet,
    NamedEvaluationResult, TypedEvaluator,
};
pub use metrics::{AggregateMetrics, EvalMetrics, TokenUsage};
pub use report::{CaseResult, EvaluationReport, EvaluatorStats, ReportSummary};
pub use result::EvalResult as LegacyEvalResult;
pub use runner::{quick_eval, EvalOptions, EvalRunner};
pub use scorers::{
    AlwaysFailScorer, AlwaysPassScorer, ContainsScorer, ExactMatchScorer, FunctionScorer,
    LengthScorer, LlmJudgeScorer, NotContainsScorer, RegexScorer, Scorer,
};
pub use suite::EvalSuite;

/// Prelude for common imports.
pub mod prelude {
    pub use crate::{
        quick_eval, Case, ContainsScorer, Dataset, EvalCase, EvalOptions, EvalRunner,
        EvalSuite, EvaluationReport, EvaluationResult, Evaluator, ExactMatchScorer,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;

        let case: Case<String> = Case::new("test".to_string());
        assert_eq!(case.inputs, "test");
    }

    #[tokio::test]
    async fn test_basic_evaluation() {
        let runner = EvalRunner::new()
            .evaluator(ExactMatchScorer::new());

        let cases = vec![
            ("a".to_string(), Some("a".to_string())),
            ("b".to_string(), Some("c".to_string())),
        ];

        let report = runner
            .run_simple(&cases, |s| {
                let s = s.to_string();
                async move { s }
            })
            .await
            .unwrap();

        assert_eq!(report.summary.total_cases, 2);
        assert_eq!(report.summary.passed, 1);
    }
}

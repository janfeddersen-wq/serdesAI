//! Evaluation runner.

use crate::case::Case;
use crate::dataset::Dataset;
use crate::error::{EvalError, EvalResult};
use crate::evaluator::{EvaluationResult, Evaluator, EvaluatorSet, NamedEvaluationResult};
use crate::report::{CaseResult, EvaluationReport};
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::time::timeout;

/// Options for running evaluations.
#[derive(Debug, Clone)]
pub struct EvalOptions {
    /// Maximum concurrent evaluations.
    pub concurrency: usize,
    /// Timeout per case.
    pub timeout: Option<Duration>,
    /// Whether to stop on first failure.
    pub fail_fast: bool,
    /// Skip cases without expected output.
    pub skip_without_expected: bool,
    /// Verbose output.
    pub verbose: bool,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            concurrency: 4,
            timeout: None,
            fail_fast: false,
            skip_without_expected: false,
            verbose: false,
        }
    }
}

impl EvalOptions {
    /// Create new options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set concurrency.
    pub fn concurrency(mut self, n: usize) -> Self {
        self.concurrency = n.max(1);
        self
    }

    /// Set timeout per case.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Enable fail-fast mode.
    pub fn fail_fast(mut self) -> Self {
        self.fail_fast = true;
        self
    }

    /// Skip cases without expected output.
    pub fn skip_without_expected(mut self) -> Self {
        self.skip_without_expected = true;
        self
    }

    /// Enable verbose output.
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

/// Evaluation runner.
pub struct EvalRunner {
    evaluators: EvaluatorSet,
    options: EvalOptions,
}

impl EvalRunner {
    /// Create a new runner.
    pub fn new() -> Self {
        Self {
            evaluators: EvaluatorSet::new(),
            options: EvalOptions::default(),
        }
    }

    /// Add an evaluator.
    pub fn evaluator<E: Evaluator + 'static>(mut self, evaluator: E) -> Self {
        self.evaluators = self.evaluators.add(evaluator);
        self
    }

    /// Set options.
    pub fn options(mut self, options: EvalOptions) -> Self {
        self.options = options;
        self
    }

    /// Run evaluation on a dataset.
    pub async fn run_dataset<Inputs, Output, Metadata, F, Fut, TaskOutput>(
        &self,
        dataset: &Dataset<Inputs, Output, Metadata>,
        task: F,
    ) -> EvalResult<EvaluationReport<TaskOutput>>
    where
        Inputs: Send + Sync,
        Output: AsRef<str> + Send + Sync,
        Metadata: Send + Sync,
        F: Fn(&Inputs) -> Fut + Send + Sync,
        Fut: Future<Output = TaskOutput> + Send,
        TaskOutput: AsRef<str> + Clone + Send + Sync,
    {
        let options = &self.options;
        let evaluators = &self.evaluators;
        let task = &task;

        let mut results = Vec::new();

        if options.fail_fast {
            for (idx, case) in dataset.cases.iter().enumerate() {
                let result = self
                    .run_single_case(idx, case, task, options, evaluators)
                    .await;
                let failed = result.failed();
                results.push(result);
                if failed {
                    break;
                }
            }
        } else {
            let semaphore = Arc::new(Semaphore::new(options.concurrency.max(1)));

            let tasks: Vec<_> = dataset
                .cases
                .iter()
                .enumerate()
                .map(|(idx, case)| {
                    let sem = semaphore.clone();
                    async move {
                        let _permit = sem.acquire().await.expect("Semaphore closed");
                        self.run_single_case(idx, case, task, options, evaluators)
                            .await
                    }
                })
                .collect();

            results = futures::future::join_all(tasks).await;
        }

        Ok(EvaluationReport::new(results))
    }

    /// Helper to run a single case evaluation.
    async fn run_single_case<Inputs, Output, Metadata, F, Fut, TaskOutput>(
        &self,
        idx: usize,
        case: &Case<Inputs, Output, Metadata>,
        task: &F,
        options: &EvalOptions,
        evaluators: &EvaluatorSet,
    ) -> CaseResult<TaskOutput>
    where
        Inputs: Send + Sync,
        Output: AsRef<str> + Send + Sync,
        Metadata: Send + Sync,
        F: Fn(&Inputs) -> Fut + Send + Sync,
        Fut: Future<Output = TaskOutput> + Send,
        TaskOutput: AsRef<str> + Clone + Send + Sync,
    {
        let name = case.display_name(idx);
        let start = Instant::now();

        let output = task(&case.inputs).await;

        if options.skip_without_expected && case.expected_output.is_none() {
            let duration = start.elapsed();
            return CaseResult::new(name, idx, output, Vec::new(), duration);
        }

        let expected_str = case.expected_output.as_ref().map(|e| e.as_ref());
        let eval_future = evaluators.evaluate(output.as_ref(), expected_str);

        let evaluations = if let Some(timeout_duration) = options.timeout {
            match timeout(timeout_duration, eval_future).await {
                Ok(results) => results,
                Err(_) => vec![NamedEvaluationResult::new(
                    "Timeout",
                    EvaluationResult::Error {
                        error: format!(
                            "Evaluation exceeded timeout of {:?}",
                            timeout_duration
                        ),
                    },
                )],
            }
        } else {
            eval_future.await
        };

        let duration = start.elapsed();
        CaseResult::new(name, idx, output, evaluations, duration)
    }

    /// Run evaluation on string inputs/outputs.
    pub async fn run_simple<F, Fut>(
        &self,
        cases: &[(String, Option<String>)],
        task: F,
    ) -> EvalResult<EvaluationReport<String>>
    where
        F: Fn(&str) -> Fut + Send + Sync,
        Fut: Future<Output = String> + Send,
    {
        let options = &self.options;
        let evaluators = &self.evaluators;
        let task = &task;

        let mut results = Vec::new();

        if options.fail_fast {
            for (idx, (input, expected)) in cases.iter().enumerate() {
                let result = self
                    .run_simple_case(idx, input, expected, task, options, evaluators)
                    .await;
                let failed = result.failed();
                results.push(result);
                if failed {
                    break;
                }
            }
        } else {
            let semaphore = Arc::new(Semaphore::new(options.concurrency.max(1)));

            let tasks: Vec<_> = cases
                .iter()
                .enumerate()
                .map(|(idx, (input, expected))| {
                    let sem = semaphore.clone();
                    async move {
                        let _permit = sem.acquire().await.expect("Semaphore closed");
                        self.run_simple_case(idx, input, expected, task, options, evaluators)
                            .await
                    }
                })
                .collect();

            results = futures::future::join_all(tasks).await;
        }

        Ok(EvaluationReport::new(results))
    }

    /// Helper to run a single simple case evaluation.
    async fn run_simple_case<F, Fut>(
        &self,
        idx: usize,
        input: &str,
        expected: &Option<String>,
        task: &F,
        options: &EvalOptions,
        evaluators: &EvaluatorSet,
    ) -> CaseResult<String>
    where
        F: Fn(&str) -> Fut + Send + Sync,
        Fut: Future<Output = String> + Send,
    {
        let name = format!("case_{}", idx);
        let start = Instant::now();

        let output = task(input).await;

        if options.skip_without_expected && expected.is_none() {
            let duration = start.elapsed();
            return CaseResult::new(name, idx, output, Vec::new(), duration);
        }

        let eval_future = evaluators.evaluate(&output, expected.as_deref());

        let evaluations = if let Some(timeout_duration) = options.timeout {
            match timeout(timeout_duration, eval_future).await {
                Ok(results) => results,
                Err(_) => vec![NamedEvaluationResult::new(
                    "Timeout",
                    EvaluationResult::Error {
                        error: format!(
                            "Evaluation exceeded timeout of {:?}",
                            timeout_duration
                        ),
                    },
                )],
            }
        } else {
            eval_future.await
        };

        let duration = start.elapsed();
        CaseResult::new(name, idx, output, evaluations, duration)
    }
}

impl Default for EvalRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Run a quick evaluation with default settings.
pub async fn quick_eval<F, Fut>(
    cases: Vec<(&str, Option<&str>)>,
    task: F,
) -> EvalResult<EvaluationReport<String>>
where
    F: Fn(&str) -> Fut + Send + Sync,
    Fut: Future<Output = String> + Send,
{
    use crate::scorers::ExactMatchScorer;

    let cases: Vec<_> = cases
        .into_iter()
        .map(|(i, e)| (i.to_string(), e.map(|s| s.to_string())))
        .collect();

    EvalRunner::new()
        .evaluator(ExactMatchScorer::new())
        .run_simple(&cases, task)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scorers::{ContainsScorer, ExactMatchScorer};

    #[tokio::test]
    async fn test_eval_runner_simple() {
        let runner = EvalRunner::new()
            .evaluator(ExactMatchScorer::new());

        let cases = vec![
            ("hello".to_string(), Some("HELLO".to_string())),
            ("world".to_string(), Some("world".to_string())),
        ];

        let report = runner
            .run_simple(&cases, |input| {
                let input = input.to_string();
                async move { input }
            })
            .await
            .unwrap();

        assert_eq!(report.summary.total_cases, 2);
        assert_eq!(report.summary.passed, 1); // Only "world" matches
    }

    #[tokio::test]
    async fn test_eval_runner_multiple_evaluators() {
        let runner = EvalRunner::new()
            .evaluator(ExactMatchScorer::new())
            .evaluator(ContainsScorer::new("hello"));

        let cases = vec![(
            "test".to_string(),
            Some("hello world".to_string()),
        )];

        let report = runner
            .run_simple(&cases, |_| async move { "hello world".to_string() })
            .await
            .unwrap();

        // ExactMatch should pass, Contains should pass
        assert_eq!(report.cases[0].evaluations.len(), 2);
    }

    #[tokio::test]
    async fn test_eval_options() {
        let options = EvalOptions::new()
            .concurrency(8)
            .timeout(Duration::from_secs(30))
            .fail_fast()
            .verbose();

        assert_eq!(options.concurrency, 8);
        assert!(options.fail_fast);
        assert!(options.verbose);
    }

    #[tokio::test]
    async fn test_quick_eval() {
        let report = quick_eval(
            vec![
                ("a", Some("a")),
                ("b", Some("c")),
            ],
            |s| {
                let s = s.to_string();
                async move { s }
            },
        )
        .await
        .unwrap();

        assert_eq!(report.summary.passed, 1);
        assert_eq!(report.summary.failed, 1);
    }
}

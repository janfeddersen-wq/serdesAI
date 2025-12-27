//! Evaluation result types.

use crate::metrics::{AggregateMetrics, EvalMetrics};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Result of an evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// Name of the evaluation suite.
    pub name: Option<String>,
    /// Individual case metrics.
    pub cases: Vec<EvalMetrics>,
    /// Aggregate metrics.
    pub aggregate: AggregateMetrics,
    /// Timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl EvalResult {
    /// Create a new result.
    pub fn new(name: Option<String>, cases: Vec<EvalMetrics>) -> Self {
        let aggregate = AggregateMetrics::from_metrics(&cases);
        Self {
            name,
            cases,
            aggregate,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Create an empty result.
    pub fn empty() -> Self {
        Self::new(None, Vec::new())
    }

    /// Get pass rate.
    pub fn pass_rate(&self) -> f64 {
        self.aggregate.pass_rate
    }

    /// Get average latency.
    pub fn avg_latency(&self) -> Duration {
        self.aggregate.average_duration
    }

    /// Get total duration.
    pub fn total_duration(&self) -> Duration {
        self.aggregate.total_duration
    }

    /// Get number of passed cases.
    pub fn passed(&self) -> usize {
        self.aggregate.passed
    }

    /// Get total number of cases.
    pub fn total(&self) -> usize {
        self.aggregate.count
    }

    /// Get failed cases.
    pub fn failed(&self) -> usize {
        self.aggregate.count - self.aggregate.passed
    }

    /// Check if all cases passed.
    pub fn all_passed(&self) -> bool {
        self.aggregate.passed == self.aggregate.count && self.aggregate.count > 0
    }

    /// Get average score.
    pub fn average_score(&self) -> Option<f64> {
        self.aggregate.average_score
    }

    /// Print summary to stdout.
    pub fn print_summary(&self) {
        println!("\n\u{1F4CA} Evaluation Summary");
        println!("═══════════════════");
        if let Some(ref name) = self.name {
            println!("Suite: {}", name);
        }
        println!("Total Cases: {}", self.total());
        println!(
            "✅ Passed: {} ({:.1}%)",
            self.passed(),
            self.pass_rate() * 100.0
        );
        println!("❌ Failed: {}", self.failed());
        if let Some(score) = self.average_score() {
            println!("\n\u{1F4C8} Average Score: {:.2}", score);
        }
        println!("\n⏱️ Total Duration: {:?}", self.total_duration());
        println!("⏱️ Average Latency: {:?}", self.avg_latency());
    }
}

impl Default for EvalResult {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_result_empty() {
        let result = EvalResult::empty();
        assert_eq!(result.total(), 0);
        assert!(!result.all_passed());
    }

    #[test]
    fn test_eval_result_with_cases() {
        let cases = vec![
            EvalMetrics::new(true).with_duration(Duration::from_millis(100)),
            EvalMetrics::new(true).with_duration(Duration::from_millis(200)),
            EvalMetrics::new(false).with_duration(Duration::from_millis(150)),
        ];

        let result = EvalResult::new(Some("test".to_string()), cases);

        assert_eq!(result.total(), 3);
        assert_eq!(result.passed(), 2);
        assert_eq!(result.failed(), 1);
        assert!(!result.all_passed());
    }

    #[test]
    fn test_eval_result_all_passed() {
        let cases = vec![
            EvalMetrics::new(true),
            EvalMetrics::new(true),
        ];

        let result = EvalResult::new(None, cases);
        assert!(result.all_passed());
    }

    #[test]
    fn test_pass_rate() {
        let cases = vec![
            EvalMetrics::new(true),
            EvalMetrics::new(false),
        ];

        let result = EvalResult::new(None, cases);
        assert!((result.pass_rate() - 0.5).abs() < 0.01);
    }
}
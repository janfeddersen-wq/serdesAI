//! Evaluation reports and summaries.

use crate::evaluator::NamedEvaluationResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// Result for a single case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseResult<TaskOutput> {
    /// Case name.
    pub name: String,
    /// Case index.
    pub index: usize,
    /// Task output.
    pub output: TaskOutput,
    /// Evaluator results.
    pub evaluations: Vec<NamedEvaluationResult>,
    /// Execution duration.
    #[serde(with = "duration_serde")]
    pub duration: Duration,
}

impl<TaskOutput> CaseResult<TaskOutput> {
    /// Create a new case result.
    pub fn new(
        name: impl Into<String>,
        index: usize,
        output: TaskOutput,
        evaluations: Vec<NamedEvaluationResult>,
        duration: Duration,
    ) -> Self {
        Self {
            name: name.into(),
            index,
            output,
            evaluations,
            duration,
        }
    }

    /// Check if all evaluations passed.
    pub fn passed(&self) -> bool {
        !self.evaluations.is_empty() && self.evaluations.iter().all(|e| e.result.is_pass())
    }

    /// Check if any evaluation failed.
    pub fn failed(&self) -> bool {
        self.evaluations.iter().any(|e| e.result.is_fail())
    }

    /// Check if any evaluation errored.
    pub fn errored(&self) -> bool {
        self.evaluations.iter().any(|e| e.result.is_error())
    }

    /// Get the average score.
    pub fn average_score(&self) -> Option<f64> {
        let scores: Vec<f64> = self
            .evaluations
            .iter()
            .filter_map(|e| e.result.score())
            .collect();

        if scores.is_empty() {
            None
        } else {
            Some(scores.iter().sum::<f64>() / scores.len() as f64)
        }
    }

    /// Get count of passed evaluations.
    pub fn pass_count(&self) -> usize {
        self.evaluations
            .iter()
            .filter(|e| e.result.is_pass())
            .count()
    }

    /// Get count of failed evaluations.
    pub fn fail_count(&self) -> usize {
        self.evaluations
            .iter()
            .filter(|e| e.result.is_fail())
            .count()
    }
}

impl<TaskOutput: fmt::Display> fmt::Display for CaseResult<TaskOutput> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.passed() {
            "✅"
        } else if self.failed() {
            "❌"
        } else {
            "⏭️"
        };

        writeln!(f, "{} {} ({:?})", status, self.name, self.duration)?;

        for eval in &self.evaluations {
            writeln!(f, "    {} - {}", eval.evaluator, eval.result)?;
        }

        Ok(())
    }
}

/// Summary statistics for an evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Total number of cases.
    pub total_cases: usize,
    /// Number of passed cases.
    pub passed: usize,
    /// Number of failed cases.
    pub failed: usize,
    /// Number of skipped cases.
    pub skipped: usize,
    /// Number of errored cases.
    pub errors: usize,
    /// Pass rate (0.0 to 1.0).
    pub pass_rate: f64,
    /// Average score (if scores available).
    pub average_score: Option<f64>,
    /// Total execution duration.
    #[serde(with = "duration_serde")]
    pub total_duration: Duration,
    /// Per-evaluator statistics.
    pub evaluator_stats: HashMap<String, EvaluatorStats>,
}

/// Statistics for a single evaluator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatorStats {
    /// Evaluator name.
    pub name: String,
    /// Total evaluations.
    pub total: usize,
    /// Passed evaluations.
    pub passed: usize,
    /// Failed evaluations.
    pub failed: usize,
    /// Pass rate.
    pub pass_rate: f64,
    /// Average score.
    pub average_score: Option<f64>,
}

/// Full evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport<TaskOutput> {
    /// Report name.
    pub name: Option<String>,
    /// Case results.
    pub cases: Vec<CaseResult<TaskOutput>>,
    /// Summary statistics.
    pub summary: ReportSummary,
    /// Timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl<TaskOutput> EvaluationReport<TaskOutput> {
    /// Create a new report from case results.
    pub fn new(cases: Vec<CaseResult<TaskOutput>>) -> Self {
        let summary = Self::compute_summary(&cases);
        Self {
            name: None,
            cases,
            summary,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Set the report name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    fn compute_summary(cases: &[CaseResult<TaskOutput>]) -> ReportSummary {
        let total_cases = cases.len();
        let passed = cases.iter().filter(|c| c.passed()).count();
        let failed = cases.iter().filter(|c| c.failed()).count();
        let errors = cases.iter().filter(|c| c.errored()).count();
        let skipped = total_cases - passed - failed - errors;

        let pass_rate = if total_cases > 0 {
            passed as f64 / total_cases as f64
        } else {
            0.0
        };

        // Collect all scores
        let scores: Vec<f64> = cases
            .iter()
            .flat_map(|c| c.evaluations.iter())
            .filter_map(|e| e.result.score())
            .collect();

        let average_score = if scores.is_empty() {
            None
        } else {
            Some(scores.iter().sum::<f64>() / scores.len() as f64)
        };

        let total_duration = cases.iter().map(|c| c.duration).sum();

        // Per-evaluator stats
        let mut evaluator_stats: HashMap<String, EvaluatorStats> = HashMap::new();
        let mut evaluator_scores: HashMap<String, (f64, usize)> = HashMap::new();

        for case in cases {
            for eval in &case.evaluations {
                let stats = evaluator_stats
                    .entry(eval.evaluator.clone())
                    .or_insert_with(|| EvaluatorStats {
                        name: eval.evaluator.clone(),
                        total: 0,
                        passed: 0,
                        failed: 0,
                        pass_rate: 0.0,
                        average_score: None,
                    });

                stats.total += 1;
                if eval.result.is_pass() {
                    stats.passed += 1;
                }
                if eval.result.is_fail() {
                    stats.failed += 1;
                }

                if let Some(score) = eval.result.score() {
                    let entry = evaluator_scores
                        .entry(eval.evaluator.clone())
                        .or_insert((0.0, 0));
                    entry.0 += score;
                    entry.1 += 1;
                }
            }
        }

        // Calculate pass rates and averages
        for stats in evaluator_stats.values_mut() {
            stats.pass_rate = if stats.total > 0 {
                stats.passed as f64 / stats.total as f64
            } else {
                0.0
            };

            if let Some((sum, count)) = evaluator_scores.get(&stats.name) {
                if *count > 0 {
                    stats.average_score = Some(sum / *count as f64);
                }
            }
        }

        ReportSummary {
            total_cases,
            passed,
            failed,
            skipped,
            errors,
            pass_rate,
            average_score,
            total_duration,
            evaluator_stats,
        }
    }

    /// Get passed cases.
    pub fn passed_cases(&self) -> impl Iterator<Item = &CaseResult<TaskOutput>> {
        self.cases.iter().filter(|c| c.passed())
    }

    /// Get failed cases.
    pub fn failed_cases(&self) -> impl Iterator<Item = &CaseResult<TaskOutput>> {
        self.cases.iter().filter(|c| c.failed())
    }

    /// Render as text.
    pub fn to_text(&self) -> String
    where
        TaskOutput: fmt::Display,
    {
        let mut output = String::new();

        output.push_str("\n\u{1F4CA} Evaluation Report\n");
        output.push_str("═══════════════════\n\n");

        if let Some(ref name) = self.name {
            output.push_str(&format!("Name: {}\n", name));
        }
        output.push_str(&format!("Timestamp: {}\n\n", self.timestamp));

        output.push_str(&format!("Total Cases: {}\n", self.summary.total_cases));
        output.push_str(&format!(
            "✅ Passed: {} ({:.1}%)\n",
            self.summary.passed,
            self.summary.pass_rate * 100.0
        ));
        output.push_str(&format!("❌ Failed: {}\n", self.summary.failed));

        if self.summary.skipped > 0 {
            output.push_str(&format!("⏭️ Skipped: {}\n", self.summary.skipped));
        }
        if self.summary.errors > 0 {
            output.push_str(&format!("⚠️ Errors: {}\n", self.summary.errors));
        }

        if let Some(avg) = self.summary.average_score {
            output.push_str(&format!("\n\u{1F4C8} Average Score: {:.2}\n", avg));
        }

        output.push_str(&format!(
            "\n⏱️ Duration: {:?}\n",
            self.summary.total_duration
        ));

        // Per-evaluator breakdown
        if !self.summary.evaluator_stats.is_empty() {
            output.push_str("\nEvaluator Breakdown:\n");
            for (name, stats) in &self.summary.evaluator_stats {
                output.push_str(&format!(
                    "  {}: {}/{} ({:.1}%)\n",
                    name,
                    stats.passed,
                    stats.total,
                    stats.pass_rate * 100.0
                ));
            }
        }

        // Failed cases
        let failed: Vec<_> = self.failed_cases().collect();
        if !failed.is_empty() {
            output.push_str("\nFailed Cases:\n");
            for case in failed.iter().take(10) {
                output.push_str(&format!("  - {}\n", case.name));
            }
            if failed.len() > 10 {
                output.push_str(&format!("  ... and {} more\n", failed.len() - 10));
            }
        }

        output
    }

    /// Render as JSON.
    pub fn to_json(&self) -> serde_json::Result<String>
    where
        TaskOutput: Serialize,
    {
        serde_json::to_string_pretty(self)
    }
}

impl<TaskOutput: fmt::Display> fmt::Display for EvaluationReport<TaskOutput> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

/// Serde helper for Duration.
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs_f64().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(deserializer)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::EvaluationResult;

    fn make_case(name: &str, passed: bool) -> CaseResult<String> {
        let result = if passed {
            EvaluationResult::pass()
        } else {
            EvaluationResult::fail("failed")
        };

        CaseResult::new(
            name,
            0,
            "output".to_string(),
            vec![NamedEvaluationResult::new("test", result)],
            Duration::from_millis(100),
        )
    }

    #[test]
    fn test_case_result_passed() {
        let case = make_case("test", true);
        assert!(case.passed());
        assert!(!case.failed());
    }

    #[test]
    fn test_case_result_failed() {
        let case = make_case("test", false);
        assert!(!case.passed());
        assert!(case.failed());
    }

    #[test]
    fn test_report_summary() {
        let cases = vec![
            make_case("case1", true),
            make_case("case2", true),
            make_case("case3", false),
        ];

        let report = EvaluationReport::new(cases);

        assert_eq!(report.summary.total_cases, 3);
        assert_eq!(report.summary.passed, 2);
        assert_eq!(report.summary.failed, 1);
        assert!((report.summary.pass_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_report_text() {
        let cases = vec![make_case("test", true)];
        let report = EvaluationReport::new(cases).with_name("Test Report");
        let text = report.to_text();

        assert!(text.contains("Test Report"));
        assert!(text.contains("100.0%"));
    }

    #[test]
    fn test_report_json() {
        let cases = vec![make_case("test", true)];
        let report = EvaluationReport::new(cases);
        let json = report.to_json().unwrap();

        assert!(json.contains("total_cases"));
        assert!(json.contains("pass_rate"));
    }

    #[test]
    fn test_evaluator_stats() {
        let cases = vec![CaseResult::new(
            "case1",
            0,
            "out".to_string(),
            vec![
                NamedEvaluationResult::new("ExactMatch", EvaluationResult::pass()),
                NamedEvaluationResult::new("Contains", EvaluationResult::fail("no")),
            ],
            Duration::from_millis(10),
        )];

        let report = EvaluationReport::new(cases);
        let stats = &report.summary.evaluator_stats;

        assert!(stats.contains_key("ExactMatch"));
        assert!(stats.contains_key("Contains"));
        assert_eq!(stats["ExactMatch"].passed, 1);
        assert_eq!(stats["Contains"].failed, 1);
    }
}

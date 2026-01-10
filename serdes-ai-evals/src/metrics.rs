//! Evaluation metrics and statistics.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Metrics for a single evaluation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalMetrics {
    /// Whether the evaluation passed.
    pub passed: bool,
    /// Numeric score (0.0 to 1.0).
    pub score: Option<f64>,
    /// Execution duration.
    #[serde(with = "duration_serde")]
    pub duration: Duration,
    /// Token count (if applicable).
    pub tokens: Option<TokenUsage>,
    /// Cost (if applicable).
    pub cost: Option<f64>,
    /// Custom metrics.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub custom: std::collections::HashMap<String, serde_json::Value>,
}

impl EvalMetrics {
    /// Create new metrics.
    pub fn new(passed: bool) -> Self {
        Self {
            passed,
            ..Default::default()
        }
    }

    /// Set the score.
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = Some(score);
        self
    }

    /// Set the duration.
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Set token usage.
    pub fn with_tokens(mut self, tokens: TokenUsage) -> Self {
        self.tokens = Some(tokens);
        self
    }

    /// Set cost.
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = Some(cost);
        self
    }

    /// Add a custom metric.
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        self.custom.insert(
            key.into(),
            serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
        );
        self
    }
}

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input/prompt tokens.
    pub input_tokens: u64,
    /// Output/completion tokens.
    pub output_tokens: u64,
    /// Total tokens.
    pub total_tokens: u64,
}

impl TokenUsage {
    /// Create new token usage.
    pub fn new(input: u64, output: u64) -> Self {
        Self {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
        }
    }
}

/// Aggregate metrics across multiple evaluations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregateMetrics {
    /// Total evaluations.
    pub count: usize,
    /// Passed evaluations.
    pub passed: usize,
    /// Pass rate.
    pub pass_rate: f64,
    /// Average score.
    pub average_score: Option<f64>,
    /// Min score.
    pub min_score: Option<f64>,
    /// Max score.
    pub max_score: Option<f64>,
    /// Total duration.
    #[serde(with = "duration_serde")]
    pub total_duration: Duration,
    /// Average duration.
    #[serde(with = "duration_serde")]
    pub average_duration: Duration,
    /// Total tokens.
    pub total_tokens: Option<TokenUsage>,
    /// Total cost.
    pub total_cost: Option<f64>,
}

impl AggregateMetrics {
    /// Compute aggregate metrics from a list of individual metrics.
    pub fn from_metrics(metrics: &[EvalMetrics]) -> Self {
        if metrics.is_empty() {
            return Self::default();
        }

        let count = metrics.len();
        let passed = metrics.iter().filter(|m| m.passed).count();
        let pass_rate = passed as f64 / count as f64;

        let scores: Vec<f64> = metrics.iter().filter_map(|m| m.score).collect();
        let average_score = if scores.is_empty() {
            None
        } else {
            Some(scores.iter().sum::<f64>() / scores.len() as f64)
        };
        let min_score = scores
            .iter()
            .cloned()
            .fold(None, |min, s| Some(min.map_or(s, |m: f64| m.min(s))));
        let max_score = scores
            .iter()
            .cloned()
            .fold(None, |max, s| Some(max.map_or(s, |m: f64| m.max(s))));

        let total_duration: Duration = metrics.iter().map(|m| m.duration).sum();
        let average_duration = total_duration / count as u32;

        let total_tokens = {
            let input: u64 = metrics
                .iter()
                .filter_map(|m| m.tokens.as_ref())
                .map(|t| t.input_tokens)
                .sum();
            let output: u64 = metrics
                .iter()
                .filter_map(|m| m.tokens.as_ref())
                .map(|t| t.output_tokens)
                .sum();
            if input > 0 || output > 0 {
                Some(TokenUsage::new(input, output))
            } else {
                None
            }
        };

        let total_cost: f64 = metrics.iter().filter_map(|m| m.cost).sum();
        let total_cost = if total_cost > 0.0 {
            Some(total_cost)
        } else {
            None
        };

        Self {
            count,
            passed,
            pass_rate,
            average_score,
            min_score,
            max_score,
            total_duration,
            average_duration,
            total_tokens,
            total_cost,
        }
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

    #[test]
    fn test_eval_metrics_new() {
        let metrics = EvalMetrics::new(true)
            .with_score(0.95)
            .with_duration(Duration::from_millis(100));

        assert!(metrics.passed);
        assert_eq!(metrics.score, Some(0.95));
    }

    #[test]
    fn test_token_usage() {
        let usage = TokenUsage::new(100, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_aggregate_metrics() {
        let metrics = vec![
            EvalMetrics::new(true)
                .with_score(0.8)
                .with_duration(Duration::from_millis(100)),
            EvalMetrics::new(true)
                .with_score(0.9)
                .with_duration(Duration::from_millis(200)),
            EvalMetrics::new(false)
                .with_score(0.5)
                .with_duration(Duration::from_millis(150)),
        ];

        let agg = AggregateMetrics::from_metrics(&metrics);

        assert_eq!(agg.count, 3);
        assert_eq!(agg.passed, 2);
        assert!((agg.pass_rate - 0.666).abs() < 0.01);
        assert!(agg.average_score.is_some());
        assert_eq!(agg.min_score, Some(0.5));
        assert_eq!(agg.max_score, Some(0.9));
    }

    #[test]
    fn test_aggregate_empty() {
        let agg = AggregateMetrics::from_metrics(&[]);
        assert_eq!(agg.count, 0);
        assert_eq!(agg.pass_rate, 0.0);
    }
}

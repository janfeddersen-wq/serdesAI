//! Common evaluator implementations (scorers).

use crate::evaluator::{EvaluationResult, Evaluator};
use async_trait::async_trait;
use regex::Regex;

/// Re-export Scorer as alias for Evaluator.
pub use crate::evaluator::Evaluator as Scorer;

/// Evaluator that checks for exact string match.
#[derive(Debug, Clone, Default)]
pub struct ExactMatchScorer {
    /// Whether to ignore case.
    pub ignore_case: bool,
    /// Whether to trim whitespace.
    pub trim: bool,
}

impl ExactMatchScorer {
    /// Create a new exact match scorer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ignore case when comparing.
    pub fn ignore_case(mut self) -> Self {
        self.ignore_case = true;
        self
    }

    /// Trim whitespace before comparing.
    pub fn trim(mut self) -> Self {
        self.trim = true;
        self
    }
}

#[async_trait]
impl Evaluator for ExactMatchScorer {
    fn name(&self) -> &str {
        "ExactMatch"
    }

    async fn evaluate_str(&self, output: &str, expected: Option<&str>) -> EvaluationResult {
        let Some(expected) = expected else {
            return EvaluationResult::skip("No expected output provided");
        };

        let (out, exp) = if self.trim {
            (output.trim(), expected.trim())
        } else {
            (output, expected)
        };

        let matches = if self.ignore_case {
            out.eq_ignore_ascii_case(exp)
        } else {
            out == exp
        };

        if matches {
            EvaluationResult::pass()
        } else {
            EvaluationResult::fail_with_details(
                "Output does not match expected",
                serde_json::json!({
                    "expected": expected,
                    "actual": output
                }),
            )
        }
    }
}

/// Evaluator that checks if output contains a substring.
#[derive(Debug, Clone)]
pub struct ContainsScorer {
    /// Pattern to look for.
    pub pattern: String,
    /// Whether to ignore case.
    pub ignore_case: bool,
}

impl ContainsScorer {
    /// Create a new contains scorer.
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            ignore_case: false,
        }
    }

    /// Ignore case when searching.
    pub fn ignore_case(mut self) -> Self {
        self.ignore_case = true;
        self
    }
}

#[async_trait]
impl Evaluator for ContainsScorer {
    fn name(&self) -> &str {
        "Contains"
    }

    async fn evaluate_str(&self, output: &str, _expected: Option<&str>) -> EvaluationResult {
        let contains = if self.ignore_case {
            output.to_lowercase().contains(&self.pattern.to_lowercase())
        } else {
            output.contains(&self.pattern)
        };

        if contains {
            EvaluationResult::pass()
        } else {
            EvaluationResult::fail(format!("Output does not contain '{}'", self.pattern))
        }
    }
}

/// Evaluator that checks if output does NOT contain a substring.
#[derive(Debug, Clone)]
pub struct NotContainsScorer {
    /// Pattern that should not appear.
    pub pattern: String,
    /// Whether to ignore case.
    pub ignore_case: bool,
}

impl NotContainsScorer {
    /// Create a new not-contains scorer.
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            ignore_case: false,
        }
    }

    /// Ignore case when searching.
    pub fn ignore_case(mut self) -> Self {
        self.ignore_case = true;
        self
    }
}

#[async_trait]
impl Evaluator for NotContainsScorer {
    fn name(&self) -> &str {
        "NotContains"
    }

    async fn evaluate_str(&self, output: &str, _expected: Option<&str>) -> EvaluationResult {
        let contains = if self.ignore_case {
            output.to_lowercase().contains(&self.pattern.to_lowercase())
        } else {
            output.contains(&self.pattern)
        };

        if contains {
            EvaluationResult::fail(format!("Output should not contain '{}'", self.pattern))
        } else {
            EvaluationResult::pass()
        }
    }
}

/// Evaluator that checks regex pattern match.
#[derive(Debug, Clone)]
pub struct RegexScorer {
    /// Regex pattern.
    pattern: String,
    /// Compiled regex.
    regex: Regex,
}

impl RegexScorer {
    /// Create a new regex scorer.
    pub fn new(pattern: impl Into<String>) -> Result<Self, regex::Error> {
        let pattern = pattern.into();
        let regex = Regex::new(&pattern)?;
        Ok(Self { pattern, regex })
    }

    /// Create with an already-compiled regex.
    pub fn from_regex(regex: Regex) -> Self {
        Self {
            pattern: regex.as_str().to_string(),
            regex,
        }
    }
}

#[async_trait]
impl Evaluator for RegexScorer {
    fn name(&self) -> &str {
        "Regex"
    }

    async fn evaluate_str(&self, output: &str, _expected: Option<&str>) -> EvaluationResult {
        if self.regex.is_match(output) {
            EvaluationResult::pass()
        } else {
            EvaluationResult::fail(format!("Output does not match pattern '{}'", self.pattern))
        }
    }
}

/// Evaluator that checks length constraints.
#[derive(Debug, Clone, Default)]
pub struct LengthScorer {
    /// Minimum length.
    pub min: Option<usize>,
    /// Maximum length.
    pub max: Option<usize>,
    /// Count characters (default) or words.
    pub count_words: bool,
}

impl LengthScorer {
    /// Create a new length scorer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum length.
    pub fn min(mut self, min: usize) -> Self {
        self.min = Some(min);
        self
    }

    /// Set maximum length.
    pub fn max(mut self, max: usize) -> Self {
        self.max = Some(max);
        self
    }

    /// Set both min and max.
    pub fn between(mut self, min: usize, max: usize) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }

    /// Count words instead of characters.
    pub fn words(mut self) -> Self {
        self.count_words = true;
        self
    }
}

#[async_trait]
impl Evaluator for LengthScorer {
    fn name(&self) -> &str {
        "Length"
    }

    async fn evaluate_str(&self, output: &str, _expected: Option<&str>) -> EvaluationResult {
        let len = if self.count_words {
            output.split_whitespace().count()
        } else {
            output.chars().count()
        };

        let unit = if self.count_words { "words" } else { "chars" };

        if let Some(min) = self.min {
            if len < min {
                return EvaluationResult::fail(format!(
                    "Output too short: {} {} (min: {})",
                    len, unit, min
                ));
            }
        }

        if let Some(max) = self.max {
            if len > max {
                return EvaluationResult::fail(format!(
                    "Output too long: {} {} (max: {})",
                    len, unit, max
                ));
            }
        }

        EvaluationResult::pass()
    }
}

/// Evaluator that always passes (for testing).
#[derive(Debug, Clone, Copy, Default)]
pub struct AlwaysPassScorer;

#[async_trait]
impl Evaluator for AlwaysPassScorer {
    fn name(&self) -> &str {
        "AlwaysPass"
    }

    async fn evaluate_str(&self, _output: &str, _expected: Option<&str>) -> EvaluationResult {
        EvaluationResult::pass()
    }
}

/// Evaluator that always fails (for testing).
#[derive(Debug, Clone)]
pub struct AlwaysFailScorer {
    /// Failure reason.
    pub reason: String,
}

impl AlwaysFailScorer {
    /// Create a new always-fail scorer.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl Default for AlwaysFailScorer {
    fn default() -> Self {
        Self::new("Always fails")
    }
}

#[async_trait]
impl Evaluator for AlwaysFailScorer {
    fn name(&self) -> &str {
        "AlwaysFail"
    }

    async fn evaluate_str(&self, _output: &str, _expected: Option<&str>) -> EvaluationResult {
        EvaluationResult::fail(&self.reason)
    }
}

/// Function-based evaluator.
pub struct FunctionScorer<F> {
    name: String,
    func: F,
}

impl<F> FunctionScorer<F>
where
    F: Fn(&str, Option<&str>) -> EvaluationResult + Send + Sync,
{
    /// Create a new function-based scorer.
    pub fn new(name: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            func,
        }
    }
}

#[async_trait]
impl<F> Evaluator for FunctionScorer<F>
where
    F: Fn(&str, Option<&str>) -> EvaluationResult + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    async fn evaluate_str(&self, output: &str, expected: Option<&str>) -> EvaluationResult {
        (self.func)(output, expected)
    }
}

/// LLM-as-judge placeholder (would need agent integration).
#[derive(Debug, Clone)]
pub struct LlmJudgeScorer {
    /// Judge name.
    pub name: String,
    /// Prompt template.
    pub prompt_template: String,
}

impl LlmJudgeScorer {
    /// Create a new LLM judge scorer.
    pub fn new(prompt_template: impl Into<String>) -> Self {
        Self {
            name: "LlmJudge".to_string(),
            prompt_template: prompt_template.into(),
        }
    }

    /// Set the name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

#[async_trait]
impl Evaluator for LlmJudgeScorer {
    fn name(&self) -> &str {
        &self.name
    }

    async fn evaluate_str(&self, _output: &str, _expected: Option<&str>) -> EvaluationResult {
        // Would call an agent here
        EvaluationResult::skip("LLM judge requires agent integration")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_exact_match_pass() {
        let scorer = ExactMatchScorer::new();
        let result = scorer.evaluate_str("hello", Some("hello")).await;
        assert!(result.is_pass());
    }

    #[tokio::test]
    async fn test_exact_match_fail() {
        let scorer = ExactMatchScorer::new();
        let result = scorer.evaluate_str("hello", Some("world")).await;
        assert!(result.is_fail());
    }

    #[tokio::test]
    async fn test_exact_match_ignore_case() {
        let scorer = ExactMatchScorer::new().ignore_case();
        let result = scorer.evaluate_str("HELLO", Some("hello")).await;
        assert!(result.is_pass());
    }

    #[tokio::test]
    async fn test_exact_match_trim() {
        let scorer = ExactMatchScorer::new().trim();
        let result = scorer.evaluate_str("  hello  ", Some("hello")).await;
        assert!(result.is_pass());
    }

    #[tokio::test]
    async fn test_exact_match_no_expected() {
        let scorer = ExactMatchScorer::new();
        let result = scorer.evaluate_str("hello", None).await;
        assert!(result.is_skip());
    }

    #[tokio::test]
    async fn test_contains_pass() {
        let scorer = ContainsScorer::new("world");
        let result = scorer.evaluate_str("hello world", None).await;
        assert!(result.is_pass());
    }

    #[tokio::test]
    async fn test_contains_fail() {
        let scorer = ContainsScorer::new("foo");
        let result = scorer.evaluate_str("hello world", None).await;
        assert!(result.is_fail());
    }

    #[tokio::test]
    async fn test_contains_ignore_case() {
        let scorer = ContainsScorer::new("WORLD").ignore_case();
        let result = scorer.evaluate_str("hello world", None).await;
        assert!(result.is_pass());
    }

    #[tokio::test]
    async fn test_not_contains_pass() {
        let scorer = NotContainsScorer::new("foo");
        let result = scorer.evaluate_str("hello world", None).await;
        assert!(result.is_pass());
    }

    #[tokio::test]
    async fn test_not_contains_fail() {
        let scorer = NotContainsScorer::new("hello");
        let result = scorer.evaluate_str("hello world", None).await;
        assert!(result.is_fail());
    }

    #[tokio::test]
    async fn test_regex_pass() {
        let scorer = RegexScorer::new(r"\d{3}-\d{4}").unwrap();
        let result = scorer.evaluate_str("Call 555-1234", None).await;
        assert!(result.is_pass());
    }

    #[tokio::test]
    async fn test_regex_fail() {
        let scorer = RegexScorer::new(r"\d{3}-\d{4}").unwrap();
        let result = scorer.evaluate_str("No phone here", None).await;
        assert!(result.is_fail());
    }

    #[tokio::test]
    async fn test_length_min() {
        let scorer = LengthScorer::new().min(10);
        assert!(scorer.evaluate_str("hello world!", None).await.is_pass());
        assert!(scorer.evaluate_str("hi", None).await.is_fail());
    }

    #[tokio::test]
    async fn test_length_max() {
        let scorer = LengthScorer::new().max(10);
        assert!(scorer.evaluate_str("hello", None).await.is_pass());
        assert!(scorer.evaluate_str("hello world!", None).await.is_fail());
    }

    #[tokio::test]
    async fn test_length_words() {
        let scorer = LengthScorer::new().words().between(2, 5);
        assert!(scorer.evaluate_str("hello world", None).await.is_pass());
        assert!(scorer.evaluate_str("hi", None).await.is_fail());
    }

    #[tokio::test]
    async fn test_always_pass() {
        let scorer = AlwaysPassScorer;
        assert!(scorer.evaluate_str("anything", None).await.is_pass());
    }

    #[tokio::test]
    async fn test_always_fail() {
        let scorer = AlwaysFailScorer::new("test failure");
        let result = scorer.evaluate_str("anything", None).await;
        assert!(result.is_fail());
    }

    #[tokio::test]
    async fn test_function_scorer() {
        let scorer = FunctionScorer::new("custom", |output, _| {
            if output.len() > 5 {
                EvaluationResult::pass()
            } else {
                EvaluationResult::fail("too short")
            }
        });

        assert!(scorer.evaluate_str("hello world", None).await.is_pass());
        assert!(scorer.evaluate_str("hi", None).await.is_fail());
    }
}

//! Evaluation case definitions.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A single evaluation test case.
#[derive(Debug, Clone)]
pub struct Case<Inputs, Output = (), Metadata = ()> {
    /// Test case name.
    pub name: Option<String>,
    /// Inputs to the task.
    pub inputs: Inputs,
    /// Expected output (if known).
    pub expected_output: Option<Output>,
    /// Metadata for evaluators.
    pub metadata: Option<Metadata>,
    /// Tags for filtering.
    pub tags: Vec<String>,
}

impl<Inputs, Output, Metadata> Case<Inputs, Output, Metadata> {
    /// Create a new case with inputs.
    pub fn new(inputs: Inputs) -> Self {
        Self {
            name: None,
            inputs,
            expected_output: None,
            metadata: None,
            tags: Vec::new(),
        }
    }

    /// Set the case name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the expected output.
    pub fn with_expected_output(mut self, output: Output) -> Self {
        self.expected_output = Some(output);
        self
    }

    /// Set the metadata.
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add multiple tags.
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(Into::into));
        self
    }

    /// Get the display name.
    pub fn display_name(&self, index: usize) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| format!("case_{}", index))
    }

    /// Check if case has a tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

impl<Inputs: Default, Output, Metadata> Default for Case<Inputs, Output, Metadata> {
    fn default() -> Self {
        Self::new(Inputs::default())
    }
}

/// Legacy eval case for backward compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalCase {
    /// Test case name.
    #[serde(default)]
    pub name: Option<String>,
    /// Input prompt.
    pub input: String,
    /// Expected output patterns.
    #[serde(default)]
    pub expected: Vec<Expected>,
    /// Tags for filtering.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Expected output criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Expected {
    /// Exact match.
    Exact {
        /// Expected value.
        value: String,
    },
    /// Contains substring.
    Contains {
        /// Pattern to find.
        pattern: String,
    },
    /// Matches regex pattern.
    Regex {
        /// Regex pattern.
        pattern: String,
    },
    /// Semantic similarity above threshold.
    Semantic {
        /// Expected text.
        text: String,
        /// Minimum similarity score.
        threshold: f32,
    },
    /// Custom check function name.
    Custom {
        /// Function name.
        name: String,
    },
}

impl Expected {
    /// Create an exact match expectation.
    pub fn exact(s: impl Into<String>) -> Self {
        Self::Exact { value: s.into() }
    }

    /// Create a contains expectation.
    pub fn contains(s: impl Into<String>) -> Self {
        Self::Contains { pattern: s.into() }
    }

    /// Create a regex expectation.
    pub fn regex(pattern: impl Into<String>) -> Self {
        Self::Regex {
            pattern: pattern.into(),
        }
    }

    /// Create a semantic similarity expectation.
    pub fn semantic(text: impl Into<String>, threshold: f32) -> Self {
        Self::Semantic {
            text: text.into(),
            threshold,
        }
    }
}

impl fmt::Display for Expected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact { value } => write!(f, "exact({})", value),
            Self::Contains { pattern } => write!(f, "contains({})", pattern),
            Self::Regex { pattern } => write!(f, "regex({})", pattern),
            Self::Semantic { text, threshold } => {
                write!(f, "semantic({}, threshold={})", text, threshold)
            }
            Self::Custom { name } => write!(f, "custom({})", name),
        }
    }
}

impl EvalCase {
    /// Create a new eval case.
    pub fn new() -> Self {
        Self {
            name: None,
            input: String::new(),
            expected: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Set the input.
    pub fn input(mut self, input: impl Into<String>) -> Self {
        self.input = input.into();
        self
    }

    /// Add an exact match expectation.
    pub fn expected_exact(mut self, s: impl Into<String>) -> Self {
        self.expected.push(Expected::exact(s));
        self
    }

    /// Add a contains expectation.
    pub fn expected_contains(mut self, s: impl Into<String>) -> Self {
        self.expected.push(Expected::contains(s));
        self
    }

    /// Add a regex expectation.
    pub fn expected_regex(mut self, pattern: impl Into<String>) -> Self {
        self.expected.push(Expected::regex(pattern));
        self
    }

    /// Add a semantic similarity expectation.
    pub fn expected_semantic(mut self, text: impl Into<String>, threshold: f32) -> Self {
        self.expected.push(Expected::semantic(text, threshold));
        self
    }

    /// Set the name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Add a tag.
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Check if all expectations are satisfied.
    pub fn check(&self, output: &str) -> Vec<(&Expected, bool)> {
        self.expected
            .iter()
            .map(|exp| {
                let passed = match exp {
                    Expected::Exact { value } => output == value,
                    Expected::Contains { pattern } => output.contains(pattern),
                    Expected::Regex { pattern } => {
                        regex::Regex::new(pattern)
                            .map(|re| re.is_match(output))
                            .unwrap_or(false)
                    }
                    Expected::Semantic { .. } => false, // Requires embedding model
                    Expected::Custom { .. } => false,   // Requires external handler
                };
                (exp, passed)
            })
            .collect()
    }

    /// Check if all expectations pass.
    pub fn all_pass(&self, output: &str) -> bool {
        self.check(output).iter().all(|(_, passed)| *passed)
    }
}

impl Default for EvalCase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_new() {
        let case: Case<String, String, ()> = Case::new("test input".to_string())
            .with_name("test case")
            .with_expected_output("expected".to_string())
            .with_tag("unit");

        assert_eq!(case.name, Some("test case".to_string()));
        assert_eq!(case.inputs, "test input");
        assert!(case.has_tag("unit"));
    }

    #[test]
    fn test_case_display_name() {
        let case: Case<String> = Case::new("input".to_string());
        assert_eq!(case.display_name(0), "case_0");

        let named = case.with_name("my_case");
        assert_eq!(named.display_name(0), "my_case");
    }

    #[test]
    fn test_eval_case_check() {
        let case = EvalCase::new()
            .input("What is 2+2?")
            .expected_contains("4")
            .expected_contains("four");

        let results = case.check("The answer is 4");
        assert_eq!(results.len(), 2);
        assert!(results[0].1); // contains "4"
        assert!(!results[1].1); // doesn't contain "four"
    }

    #[test]
    fn test_eval_case_exact() {
        let case = EvalCase::new().input("test").expected_exact("hello");

        assert!(case.all_pass("hello"));
        assert!(!case.all_pass("hello world"));
    }

    #[test]
    fn test_expected_display() {
        assert_eq!(Expected::exact("foo").to_string(), "exact(foo)");
        assert_eq!(Expected::contains("bar").to_string(), "contains(bar)");
    }

    #[test]
    fn test_eval_case_serialize() {
        let case = EvalCase::new()
            .input("hello")
            .expected_contains("world");

        let json = serde_json::to_string(&case).unwrap();
        assert!(json.contains("hello"));
        assert!(json.contains("contains"));
    }
}

//! Evaluation suite definitions.

use crate::case::EvalCase;
use serde::{Deserialize, Serialize};

/// A collection of evaluation test cases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSuite {
    /// Suite name.
    pub name: String,
    /// Description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Test cases.
    pub cases: Vec<EvalCase>,
    /// Tags for filtering.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

impl EvalSuite {
    /// Create a new evaluation suite.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            cases: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Set the description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a test case.
    pub fn add_case(mut self, case: EvalCase) -> Self {
        self.cases.push(case);
        self
    }

    /// Add multiple cases.
    pub fn add_cases(mut self, cases: impl IntoIterator<Item = EvalCase>) -> Self {
        self.cases.extend(cases);
        self
    }

    /// Add a tag.
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Get number of cases.
    pub fn len(&self) -> usize {
        self.cases.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }

    /// Filter cases by tag.
    pub fn filter_by_tag(&self, tag: &str) -> Vec<&EvalCase> {
        self.cases.iter().filter(|c| c.tags.contains(&tag.to_string())).collect()
    }

    /// Get a subset of cases by indices.
    pub fn subset(&self, indices: &[usize]) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            cases: indices
                .iter()
                .filter_map(|&i| self.cases.get(i).cloned())
                .collect(),
            tags: self.tags.clone(),
        }
    }

    /// Take first N cases.
    pub fn take(&self, n: usize) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            cases: self.cases.iter().take(n).cloned().collect(),
            tags: self.tags.clone(),
        }
    }

    /// Load from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suite_new() {
        let suite = EvalSuite::new("test")
            .description("Test suite");

        assert_eq!(suite.name, "test");
        assert_eq!(suite.description, Some("Test suite".to_string()));
        assert!(suite.is_empty());
    }

    #[test]
    fn test_suite_add_cases() {
        let suite = EvalSuite::new("test")
            .add_case(EvalCase::new().input("input1"))
            .add_case(EvalCase::new().input("input2"));

        assert_eq!(suite.len(), 2);
    }

    #[test]
    fn test_suite_filter_by_tag() {
        let suite = EvalSuite::new("test")
            .add_case(EvalCase::new().input("a").tag("unit"))
            .add_case(EvalCase::new().input("b").tag("integration"))
            .add_case(EvalCase::new().input("c").tag("unit"));

        let unit_cases = suite.filter_by_tag("unit");
        assert_eq!(unit_cases.len(), 2);
    }

    #[test]
    fn test_suite_take() {
        let suite = EvalSuite::new("test")
            .add_case(EvalCase::new().input("a"))
            .add_case(EvalCase::new().input("b"))
            .add_case(EvalCase::new().input("c"));

        let subset = suite.take(2);
        assert_eq!(subset.len(), 2);
    }

    #[test]
    fn test_suite_json_roundtrip() {
        let suite = EvalSuite::new("test")
            .add_case(EvalCase::new().input("hello").expected_contains("world"));

        let json = suite.to_json().unwrap();
        let loaded = EvalSuite::from_json(&json).unwrap();

        assert_eq!(loaded.name, "test");
        assert_eq!(loaded.len(), 1);
    }
}

//! Dataset management for evaluation cases.

use crate::case::Case;
use crate::error::{EvalError, EvalResult};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::path::Path;

/// A collection of test cases.
#[derive(Debug, Clone)]
pub struct Dataset<Inputs, Output = (), Metadata = ()> {
    /// Dataset name.
    pub name: Option<String>,
    /// Description.
    pub description: Option<String>,
    /// Test cases.
    pub cases: Vec<Case<Inputs, Output, Metadata>>,
}

impl<Inputs, Output, Metadata> Dataset<Inputs, Output, Metadata> {
    /// Create a new empty dataset.
    pub fn new() -> Self {
        Self {
            name: None,
            description: None,
            cases: Vec::new(),
        }
    }

    /// Set the dataset name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a case.
    pub fn case(mut self, case: Case<Inputs, Output, Metadata>) -> Self {
        self.cases.push(case);
        self
    }

    /// Add multiple cases.
    pub fn cases(
        mut self,
        cases: impl IntoIterator<Item = Case<Inputs, Output, Metadata>>,
    ) -> Self {
        self.cases.extend(cases);
        self
    }

    /// Get the number of cases.
    pub fn len(&self) -> usize {
        self.cases.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }

    /// Filter cases by tag.
    pub fn filter_by_tag(&self, tag: &str) -> Vec<&Case<Inputs, Output, Metadata>> {
        self.cases.iter().filter(|c| c.has_tag(tag)).collect()
    }

    /// Filter cases by predicate.
    pub fn filter<F>(&self, predicate: F) -> Vec<&Case<Inputs, Output, Metadata>>
    where
        F: Fn(&Case<Inputs, Output, Metadata>) -> bool,
    {
        self.cases.iter().filter(|c| predicate(c)).collect()
    }

    /// Get a subset of cases.
    pub fn subset(&self, indices: &[usize]) -> Self
    where
        Inputs: Clone,
        Output: Clone,
        Metadata: Clone,
    {
        let cases = indices
            .iter()
            .filter_map(|&i| self.cases.get(i).cloned())
            .collect();
        Dataset {
            name: self.name.clone(),
            description: self.description.clone(),
            cases,
        }
    }

    /// Take first N cases.
    pub fn take(&self, n: usize) -> Self
    where
        Inputs: Clone,
        Output: Clone,
        Metadata: Clone,
    {
        Dataset {
            name: self.name.clone(),
            description: self.description.clone(),
            cases: self.cases.iter().take(n).cloned().collect(),
        }
    }

    /// Shuffle cases (deterministically with seed).
    pub fn shuffle(&self, seed: u64) -> Self
    where
        Inputs: Clone,
        Output: Clone,
        Metadata: Clone,
    {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut cases = self.cases.clone();
        let n = cases.len();

        for i in 0..n {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let j = (hasher.finish() as usize) % n;
            cases.swap(i, j);
        }

        Dataset {
            name: self.name.clone(),
            description: self.description.clone(),
            cases,
        }
    }
}

impl<Inputs, Output, Metadata> Default for Dataset<Inputs, Output, Metadata> {
    fn default() -> Self {
        Self::new()
    }
}

/// String-based dataset for easy serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringDataset {
    /// Dataset name.
    #[serde(default)]
    pub name: Option<String>,
    /// Description.
    #[serde(default)]
    pub description: Option<String>,
    /// Test cases as (input, expected_output) pairs.
    pub cases: Vec<StringCase>,
}

/// String-based case for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringCase {
    /// Case name.
    #[serde(default)]
    pub name: Option<String>,
    /// Input string.
    pub input: String,
    /// Expected output.
    #[serde(default)]
    pub expected: Option<String>,
    /// Tags.
    #[serde(default)]
    pub tags: Vec<String>,
}

impl StringDataset {
    /// Load from JSON file.
    pub fn from_json(path: impl AsRef<Path>) -> EvalResult<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Self::from_json_str(&content)
    }

    /// Load from JSON string.
    pub fn from_json_str(content: &str) -> EvalResult<Self> {
        serde_json::from_str(content).map_err(|e| EvalError::Serialization(e.to_string()))
    }

    /// Load from YAML file.
    pub fn from_yaml(path: impl AsRef<Path>) -> EvalResult<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Self::from_yaml_str(&content)
    }

    /// Load from YAML string.
    pub fn from_yaml_str(content: &str) -> EvalResult<Self> {
        serde_yaml::from_str(content).map_err(|e| EvalError::Yaml(e.to_string()))
    }

    /// Save to JSON file.
    pub fn to_json(&self, path: impl AsRef<Path>) -> EvalResult<()> {
        let content = self.to_json_string()?;
        std::fs::write(path.as_ref(), content)?;
        Ok(())
    }

    /// Serialize to JSON string.
    pub fn to_json_string(&self) -> EvalResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| EvalError::Serialization(e.to_string()))
    }

    /// Save to YAML file.
    pub fn to_yaml(&self, path: impl AsRef<Path>) -> EvalResult<()> {
        let content = self.to_yaml_string()?;
        std::fs::write(path.as_ref(), content)?;
        Ok(())
    }

    /// Serialize to YAML string.
    pub fn to_yaml_string(&self) -> EvalResult<String> {
        serde_yaml::to_string(self).map_err(|e| EvalError::Yaml(e.to_string()))
    }

    /// Convert to generic Dataset.
    pub fn to_dataset(&self) -> Dataset<String, String> {
        let cases = self
            .cases
            .iter()
            .map(|c| {
                Case::new(c.input.clone())
                    .with_name(c.name.clone().unwrap_or_default())
                    .with_tags(c.tags.clone())
                    .with_expected_output(c.expected.clone().unwrap_or_default())
            })
            .collect();

        Dataset {
            name: self.name.clone(),
            description: self.description.clone(),
            cases,
        }
    }
}

/// Builder for creating datasets programmatically.
#[derive(Debug)]
pub struct DatasetBuilder<Inputs, Output = (), Metadata = ()> {
    name: Option<String>,
    description: Option<String>,
    cases: Vec<Case<Inputs, Output, Metadata>>,
}

impl<Inputs, Output, Metadata> DatasetBuilder<Inputs, Output, Metadata> {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            name: None,
            description: None,
            cases: Vec::new(),
        }
    }

    /// Set the name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a case.
    pub fn case(mut self, case: Case<Inputs, Output, Metadata>) -> Self {
        self.cases.push(case);
        self
    }

    /// Build the dataset.
    pub fn build(self) -> Dataset<Inputs, Output, Metadata> {
        Dataset {
            name: self.name,
            description: self.description,
            cases: self.cases,
        }
    }
}

impl<Inputs, Output, Metadata> Default for DatasetBuilder<Inputs, Output, Metadata> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_new() {
        let dataset: Dataset<String> = Dataset::new()
            .with_name("test")
            .with_description("Test dataset");

        assert_eq!(dataset.name, Some("test".to_string()));
        assert!(dataset.is_empty());
    }

    #[test]
    fn test_dataset_add_cases() {
        let dataset: Dataset<String> = Dataset::new()
            .case(Case::new("input1".to_string()))
            .case(Case::new("input2".to_string()));

        assert_eq!(dataset.len(), 2);
    }

    #[test]
    fn test_dataset_filter_by_tag() {
        let dataset: Dataset<String> = Dataset::new()
            .case(Case::new("a".to_string()).with_tag("unit"))
            .case(Case::new("b".to_string()).with_tag("integration"))
            .case(Case::new("c".to_string()).with_tag("unit"));

        let unit_tests = dataset.filter_by_tag("unit");
        assert_eq!(unit_tests.len(), 2);
    }

    #[test]
    fn test_dataset_take() {
        let dataset: Dataset<String> = Dataset::new()
            .case(Case::new("a".to_string()))
            .case(Case::new("b".to_string()))
            .case(Case::new("c".to_string()));

        let subset = dataset.take(2);
        assert_eq!(subset.len(), 2);
    }

    #[test]
    fn test_string_dataset_json_roundtrip() {
        let dataset = StringDataset {
            name: Some("test".to_string()),
            description: None,
            cases: vec![
                StringCase {
                    name: Some("case1".to_string()),
                    input: "hello".to_string(),
                    expected: Some("world".to_string()),
                    tags: vec![],
                },
            ],
        };

        let json = dataset.to_json_string().unwrap();
        let loaded = StringDataset::from_json_str(&json).unwrap();

        assert_eq!(loaded.name, Some("test".to_string()));
        assert_eq!(loaded.cases.len(), 1);
    }

    #[test]
    fn test_dataset_builder() {
        let dataset: Dataset<String> = DatasetBuilder::new()
            .name("builder test")
            .case(Case::new("input".to_string()))
            .build();

        assert_eq!(dataset.name, Some("builder test".to_string()));
        assert_eq!(dataset.len(), 1);
    }
}

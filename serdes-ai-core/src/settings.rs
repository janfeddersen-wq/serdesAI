//! Model settings and configuration.
//!
//! This module provides the `ModelSettings` type for configuring model behavior,
//! including temperature, token limits, and other generation parameters.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Settings for model generation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelSettings {
    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,

    /// Sampling temperature (0.0 to 2.0 typically).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Top-p (nucleus) sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Top-k sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,

    /// Frequency penalty (-2.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,

    /// Presence penalty (-2.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,

    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// Random seed for reproducibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// Request timeout.
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "option_duration_serde"
    )]
    pub timeout: Option<Duration>,

    /// Whether to allow parallel tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Extra provider-specific settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

impl ModelSettings {
    /// Create new empty settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max tokens.
    #[must_use]
    pub fn max_tokens(mut self, tokens: u64) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set temperature.
    #[must_use]
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set top-p.
    #[must_use]
    pub fn top_p(mut self, p: f64) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Set top-k.
    #[must_use]
    pub fn top_k(mut self, k: u64) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Set frequency penalty.
    #[must_use]
    pub fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    /// Set presence penalty.
    #[must_use]
    pub fn presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    /// Set stop sequences.
    #[must_use]
    pub fn stop(mut self, sequences: Vec<String>) -> Self {
        self.stop = Some(sequences);
        self
    }

    /// Add a stop sequence.
    #[must_use]
    pub fn add_stop(mut self, sequence: impl Into<String>) -> Self {
        self.stop.get_or_insert_with(Vec::new).push(sequence.into());
        self
    }

    /// Set seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set timeout in seconds.
    #[must_use]
    pub fn timeout_secs(self, secs: u64) -> Self {
        self.timeout(Duration::from_secs(secs))
    }

    /// Set parallel tool calls.
    #[must_use]
    pub fn parallel_tool_calls(mut self, parallel: bool) -> Self {
        self.parallel_tool_calls = Some(parallel);
        self
    }

    /// Set extra settings.
    #[must_use]
    pub fn extra(mut self, extra: serde_json::Value) -> Self {
        self.extra = Some(extra);
        self
    }

    /// Merge with another settings, preferring values from `other`.
    ///
    /// Values in `other` override values in `self` when both are present.
    #[must_use]
    pub fn merge(&self, other: &ModelSettings) -> ModelSettings {
        ModelSettings {
            max_tokens: other.max_tokens.or(self.max_tokens),
            temperature: other.temperature.or(self.temperature),
            top_p: other.top_p.or(self.top_p),
            top_k: other.top_k.or(self.top_k),
            frequency_penalty: other.frequency_penalty.or(self.frequency_penalty),
            presence_penalty: other.presence_penalty.or(self.presence_penalty),
            stop: other.stop.clone().or_else(|| self.stop.clone()),
            seed: other.seed.or(self.seed),
            timeout: other.timeout.or(self.timeout),
            parallel_tool_calls: other.parallel_tool_calls.or(self.parallel_tool_calls),
            extra: match (&self.extra, &other.extra) {
                (Some(a), Some(b)) => Some(merge_json(a, b)),
                (_, Some(b)) => Some(b.clone()),
                (Some(a), None) => Some(a.clone()),
                (None, None) => None,
            },
        }
    }

    /// Check if all settings are None.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.max_tokens.is_none()
            && self.temperature.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
            && self.frequency_penalty.is_none()
            && self.presence_penalty.is_none()
            && self.stop.is_none()
            && self.seed.is_none()
            && self.timeout.is_none()
            && self.parallel_tool_calls.is_none()
            && self.extra.is_none()
    }
}

/// Merge two JSON values, with `b` taking precedence.
fn merge_json(a: &serde_json::Value, b: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match (a, b) {
        (Value::Object(a_obj), Value::Object(b_obj)) => {
            let mut result = a_obj.clone();
            for (k, v) in b_obj {
                result.insert(
                    k.clone(),
                    if let Some(existing) = a_obj.get(k) {
                        merge_json(existing, v)
                    } else {
                        v.clone()
                    },
                );
            }
            Value::Object(result)
        }
        (_, b) => b.clone(),
    }
}

/// Serde helper for optional Duration.
mod option_duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match duration {
            Some(d) => d.as_secs_f64().serialize(serializer),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<f64> = Option::deserialize(deserializer)?;
        Ok(opt.map(Duration::from_secs_f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_settings_new() {
        let settings = ModelSettings::new();
        assert!(settings.is_empty());
    }

    #[test]
    fn test_model_settings_builder() {
        let settings = ModelSettings::new()
            .max_tokens(1000)
            .temperature(0.7)
            .top_p(0.9)
            .seed(42);

        assert_eq!(settings.max_tokens, Some(1000));
        assert_eq!(settings.temperature, Some(0.7));
        assert_eq!(settings.top_p, Some(0.9));
        assert_eq!(settings.seed, Some(42));
    }

    #[test]
    fn test_model_settings_stop() {
        let settings = ModelSettings::new().add_stop("\n\n").add_stop("END");

        assert_eq!(
            settings.stop,
            Some(vec!["\n\n".to_string(), "END".to_string()])
        );
    }

    #[test]
    fn test_model_settings_merge() {
        let base = ModelSettings::new().max_tokens(1000).temperature(0.5);

        let override_settings = ModelSettings::new().temperature(0.8).top_p(0.9);

        let merged = base.merge(&override_settings);

        assert_eq!(merged.max_tokens, Some(1000)); // from base
        assert_eq!(merged.temperature, Some(0.8)); // overridden
        assert_eq!(merged.top_p, Some(0.9)); // from override
    }

    #[test]
    fn test_model_settings_timeout() {
        let settings = ModelSettings::new().timeout_secs(30);
        assert_eq!(settings.timeout, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_serde_roundtrip() {
        let settings = ModelSettings::new()
            .max_tokens(1000)
            .temperature(0.7)
            .timeout_secs(30);

        let json = serde_json::to_string(&settings).unwrap();
        let parsed: ModelSettings = serde_json::from_str(&json).unwrap();

        assert_eq!(settings.max_tokens, parsed.max_tokens);
        assert_eq!(settings.temperature, parsed.temperature);
        // Duration comparison (might have slight floating point differences)
        assert!(parsed.timeout.is_some());
    }
}

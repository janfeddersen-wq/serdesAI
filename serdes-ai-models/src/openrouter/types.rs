//! OpenRouter-specific types for provider routing and preferences.

use serde::{Deserialize, Serialize};

/// Provider routing preferences.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderPreferences {
    /// Ordered list of preferred providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
    /// Quantization preference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantizations: Option<Vec<Quantization>>,
    /// Whether to allow fallback to other providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    /// Data collection preference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<DataCollection>,
}

impl ProviderPreferences {
    /// Create new empty preferences.
    pub fn new() -> Self { Self::default() }
    /// Set provider order.
    #[must_use]
    pub fn with_order(mut self, order: Vec<String>) -> Self { self.order = Some(order); self }
    /// Set quantization preference.
    #[must_use]
    pub fn with_quantizations(mut self, q: Vec<Quantization>) -> Self { self.quantizations = Some(q); self }
    /// Set fallback behavior.
    #[must_use]
    pub fn with_allow_fallbacks(mut self, allow: bool) -> Self { self.allow_fallbacks = Some(allow); self }
}

/// Quantization preference levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    /// Full precision (bf16/fp16).
    #[serde(rename = "bf16")] Bf16,
    /// FP8 quantization.
    #[serde(rename = "fp8")] Fp8,
    /// INT8 quantization.
    #[serde(rename = "int8")] Int8,
    /// INT4 quantization.
    #[serde(rename = "int4")] Int4,
}

/// Data collection preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataCollection {
    /// Allow data collection.
    Allow,
    /// Deny data collection.
    Deny,
}

/// OpenRouter-specific request extensions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenRouterExtras {
    /// Provider routing preferences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderPreferences>,
    /// Message transforms (e.g., "middle-out").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transforms: Option<Vec<String>>,
    /// Route (model fallback chain).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route: Option<String>,
}

impl OpenRouterExtras {
    /// Create new empty extras.
    pub fn new() -> Self { Self::default() }
    /// Set provider preferences.
    #[must_use]
    pub fn with_provider(mut self, p: ProviderPreferences) -> Self { self.provider = Some(p); self }
    /// Set message transforms.
    #[must_use]
    pub fn with_transforms(mut self, t: Vec<String>) -> Self { self.transforms = Some(t); self }
}

/// Common OpenRouter model identifiers.
pub mod models {
    /// Claude 3.5 Sonnet.
    pub const CLAUDE_3_5_SONNET: &str = "anthropic/claude-3.5-sonnet";
    /// GPT-4o.
    pub const GPT_4O: &str = "openai/gpt-4o";
    /// Llama 3.1 70B.
    pub const LLAMA_3_1_70B: &str = "meta-llama/llama-3.1-70b-instruct";
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_provider_preferences() {
        assert!(serde_json::to_string(&ProviderPreferences::new().with_order(vec!["anthropic".into()])).unwrap().contains("anthropic"));
    }
}

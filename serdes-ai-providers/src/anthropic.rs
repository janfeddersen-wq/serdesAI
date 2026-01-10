//! Anthropic provider implementation.

use crate::provider::{Provider, ProviderConfig, ProviderError};
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use serdes_ai_models::profile::anthropic_claude_profile;
use serdes_ai_models::ModelProfile;

/// Anthropic provider.
#[derive(Debug)]
pub struct AnthropicProvider {
    config: ProviderConfig,
    client: Client,
    /// Anthropic API version.
    api_version: String,
}

impl AnthropicProvider {
    /// Default API version.
    pub const DEFAULT_API_VERSION: &'static str = "2023-06-01";

    /// Create a new Anthropic provider with an API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        let config = ProviderConfig::new().with_api_key(api_key);
        Self {
            client: config.build_client(),
            config,
            api_version: Self::DEFAULT_API_VERSION.to_string(),
        }
    }

    /// Create from configuration.
    pub fn from_config(config: ProviderConfig) -> Result<Self, ProviderError> {
        if config.api_key.is_none() {
            return Err(ProviderError::MissingApiKey("ANTHROPIC_API_KEY"));
        }
        Ok(Self {
            client: config.build_client(),
            config,
            api_version: Self::DEFAULT_API_VERSION.to_string(),
        })
    }

    /// Create from environment variables.
    pub fn from_env() -> Result<Self, ProviderError> {
        let config = ProviderConfig::from_env("ANTHROPIC");
        Self::from_config(config)
    }

    /// Set a custom API version.
    #[must_use]
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }
}

impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn base_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or("https://api.anthropic.com")
    }

    fn client(&self) -> &Client {
        &self.client
    }

    fn default_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();

        if let Some(key) = &self.config.api_key {
            if let Ok(value) = HeaderValue::from_str(key) {
                headers.insert("x-api-key", value);
            }
        }

        if let Ok(value) = HeaderValue::from_str(&self.api_version) {
            headers.insert("anthropic-version", value);
        }

        headers.insert("content-type", HeaderValue::from_static("application/json"));

        headers
    }

    fn model_profile(&self, model_name: &str) -> Option<ModelProfile> {
        match model_name {
            "claude-3-5-sonnet-20241022" | "claude-3-5-sonnet-latest" => {
                let mut profile = anthropic_claude_profile();
                profile.max_tokens = Some(8192);
                profile.context_window = Some(200000);
                profile.supports_documents = true;
                Some(profile)
            }
            "claude-3-5-haiku-20241022" | "claude-3-5-haiku-latest" => {
                let mut profile = anthropic_claude_profile();
                profile.max_tokens = Some(8192);
                profile.context_window = Some(200000);
                Some(profile)
            }
            "claude-3-opus-20240229" | "claude-3-opus-latest" => {
                let mut profile = anthropic_claude_profile();
                profile.max_tokens = Some(4096);
                profile.context_window = Some(200000);
                profile.supports_documents = true;
                Some(profile)
            }
            "claude-3-sonnet-20240229" => Some(anthropic_claude_profile()),
            "claude-3-haiku-20240307" => Some(anthropic_claude_profile()),
            _ if model_name.starts_with("claude-") => Some(anthropic_claude_profile()),
            _ => None,
        }
    }

    fn is_configured(&self) -> bool {
        self.config.api_key.is_some()
    }

    fn aliases(&self) -> &[&str] {
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_provider_new() {
        let provider = AnthropicProvider::new("sk-ant-test");
        assert_eq!(provider.name(), "anthropic");
        assert!(provider.is_configured());
    }

    #[test]
    fn test_anthropic_provider_base_url() {
        let provider = AnthropicProvider::new("key");
        assert_eq!(provider.base_url(), "https://api.anthropic.com");
    }

    #[test]
    fn test_anthropic_provider_headers() {
        let provider = AnthropicProvider::new("sk-ant-test");
        let headers = provider.default_headers();

        assert!(headers.contains_key("x-api-key"));
        assert!(headers.contains_key("anthropic-version"));
        assert!(headers.contains_key("content-type"));
    }

    #[test]
    fn test_anthropic_provider_custom_version() {
        let provider = AnthropicProvider::new("key").with_api_version("2024-01-01");

        let headers = provider.default_headers();
        let version = headers.get("anthropic-version").unwrap();
        assert_eq!(version.to_str().unwrap(), "2024-01-01");
    }

    #[test]
    fn test_anthropic_provider_model_profile() {
        let provider = AnthropicProvider::new("key");

        assert!(provider
            .model_profile("claude-3-5-sonnet-20241022")
            .is_some());
        assert!(provider.model_profile("claude-3-opus-20240229").is_some());
        assert!(provider.model_profile("unknown-model").is_none());
    }

    #[test]
    fn test_anthropic_provider_from_env_missing() {
        std::env::remove_var("ANTHROPIC_API_KEY");

        let result = AnthropicProvider::from_env();
        assert!(result.is_err());
    }
}

//! OpenAI provider implementation.

use crate::provider::{Provider, ProviderConfig, ProviderError};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use reqwest::Client;
use serdes_ai_models::profile::openai_gpt4o_profile;
use serdes_ai_models::ModelProfile;

/// OpenAI provider.
#[derive(Debug)]
pub struct OpenAIProvider {
    config: ProviderConfig,
    client: Client,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider with an API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        let config = ProviderConfig::new().with_api_key(api_key);
        Self {
            client: config.build_client(),
            config,
        }
    }

    /// Create from configuration.
    pub fn from_config(config: ProviderConfig) -> Result<Self, ProviderError> {
        if config.api_key.is_none() {
            return Err(ProviderError::MissingApiKey("OPENAI_API_KEY"));
        }
        Ok(Self {
            client: config.build_client(),
            config,
        })
    }

    /// Create from environment variables.
    pub fn from_env() -> Result<Self, ProviderError> {
        let config = ProviderConfig::from_env("OPENAI");
        Self::from_config(config)
    }
}

impl Provider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn base_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1")
    }

    fn client(&self) -> &Client {
        &self.client
    }

    fn default_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();

        if let Some(key) = &self.config.api_key {
            let auth_value = format!("Bearer {}", key);
            if let Ok(value) = HeaderValue::from_str(&auth_value) {
                headers.insert(AUTHORIZATION, value);
            }
        }

        if let Some(org) = &self.config.organization {
            if let Ok(value) = HeaderValue::from_str(org) {
                headers.insert("OpenAI-Organization", value);
            }
        }

        if let Some(project) = &self.config.project {
            if let Ok(value) = HeaderValue::from_str(project) {
                headers.insert("OpenAI-Project", value);
            }
        }

        headers
    }

    fn model_profile(&self, model_name: &str) -> Option<ModelProfile> {
        match model_name {
            "gpt-4o" | "gpt-4o-2024-11-20" | "gpt-4o-2024-08-06" | "gpt-4o-2024-05-13" => {
                Some(openai_gpt4o_profile())
            }
            "gpt-4o-mini" | "gpt-4o-mini-2024-07-18" => {
                let mut profile = openai_gpt4o_profile();
                profile.max_tokens = Some(16384);
                Some(profile)
            }
            "gpt-4-turbo" | "gpt-4-turbo-2024-04-09" | "gpt-4-turbo-preview" => {
                let mut profile = openai_gpt4o_profile();
                profile.context_window = Some(128000);
                Some(profile)
            }
            "o1" | "o1-2024-12-17" => {
                let mut profile = ModelProfile::default();
                profile.supports_reasoning = true;
                profile.supports_streaming = false; // o1 doesn't support streaming
                profile.context_window = Some(200000);
                profile.max_tokens = Some(100000);
                Some(profile)
            }
            "o1-preview" | "o1-mini" => {
                let mut profile = ModelProfile::default();
                profile.supports_reasoning = true;
                profile.supports_streaming = false;
                Some(profile)
            }
            "o3-mini" | "o3-mini-2025-01-31" => {
                let mut profile = ModelProfile::default();
                profile.supports_reasoning = true;
                profile.supports_streaming = false;
                profile.context_window = Some(200000);
                profile.max_tokens = Some(100000);
                Some(profile)
            }
            _ if model_name.starts_with("gpt-")
                || model_name.starts_with("o1")
                || model_name.starts_with("o3") =>
            {
                Some(openai_gpt4o_profile())
            }
            _ => None,
        }
    }

    fn is_configured(&self) -> bool {
        self.config.api_key.is_some()
    }

    fn aliases(&self) -> &[&str] {
        &["openai-chat", "openai-responses"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_provider_new() {
        let provider = OpenAIProvider::new("sk-test-key");
        assert_eq!(provider.name(), "openai");
        assert!(provider.is_configured());
    }

    #[test]
    fn test_openai_provider_base_url() {
        let provider = OpenAIProvider::new("key");
        assert_eq!(provider.base_url(), "https://api.openai.com/v1");
    }

    #[test]
    fn test_openai_provider_custom_base_url() {
        let config = ProviderConfig::new()
            .with_api_key("key")
            .with_base_url("https://custom.api.com");
        let provider = OpenAIProvider::from_config(config).unwrap();
        assert_eq!(provider.base_url(), "https://custom.api.com");
    }

    #[test]
    fn test_openai_provider_headers() {
        let config = ProviderConfig::new()
            .with_api_key("sk-test")
            .with_organization("org-123");
        let provider = OpenAIProvider::from_config(config).unwrap();

        let headers = provider.default_headers();
        assert!(headers.contains_key(AUTHORIZATION));
        assert!(headers.contains_key("OpenAI-Organization"));
    }

    #[test]
    fn test_openai_provider_model_profile() {
        let provider = OpenAIProvider::new("key");

        assert!(provider.model_profile("gpt-4o").is_some());
        assert!(provider.model_profile("gpt-4o-mini").is_some());
        assert!(provider.model_profile("o1").is_some());
        assert!(provider.model_profile("unknown-model").is_none());
    }

    #[test]
    fn test_openai_provider_aliases() {
        let provider = OpenAIProvider::new("key");
        let aliases = provider.aliases();
        assert!(aliases.contains(&"openai-chat"));
    }

    #[test]
    fn test_openai_provider_from_env_missing() {
        // Clear any existing env var
        std::env::remove_var("OPENAI_API_KEY");

        let result = OpenAIProvider::from_env();
        assert!(result.is_err());
    }
}

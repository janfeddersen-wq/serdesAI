//! Groq provider implementation.
//!
//! Groq provides ultra-fast inference for open models like Llama, Mixtral, etc.

use crate::provider::{Provider, ProviderConfig, ProviderError};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use reqwest::Client;
use serdes_ai_models::ModelProfile;

/// Groq provider.
#[derive(Debug)]
pub struct GroqProvider {
    config: ProviderConfig,
    client: Client,
}

impl GroqProvider {
    /// Create a new Groq provider with an API key.
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
            return Err(ProviderError::MissingApiKey("GROQ_API_KEY"));
        }
        Ok(Self {
            client: config.build_client(),
            config,
        })
    }

    /// Create from environment variables.
    pub fn from_env() -> Result<Self, ProviderError> {
        let config = ProviderConfig::from_env("GROQ");
        Self::from_config(config)
    }
}

impl Provider for GroqProvider {
    fn name(&self) -> &str {
        "groq"
    }

    fn base_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or("https://api.groq.com/openai/v1")
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

        headers.insert("content-type", HeaderValue::from_static("application/json"));

        headers
    }

    fn model_profile(&self, model_name: &str) -> Option<ModelProfile> {
        let mut profile = ModelProfile {
            supports_tools: true,
            supports_parallel_tools: true,
            supports_system_messages: true,
            supports_streaming: true,
            ..Default::default()
        };

        match model_name {
            "llama-3.3-70b-versatile" | "llama-3.1-70b-versatile" => {
                profile.context_window = Some(131072);
                profile.max_tokens = Some(8192);
            }
            "llama-3.1-8b-instant" | "llama-3.2-1b-preview" | "llama-3.2-3b-preview" => {
                profile.context_window = Some(131072);
                profile.max_tokens = Some(8192);
            }
            "llama3-70b-8192" | "llama3-8b-8192" => {
                profile.context_window = Some(8192);
                profile.max_tokens = Some(8192);
            }
            "mixtral-8x7b-32768" => {
                profile.context_window = Some(32768);
                profile.max_tokens = Some(8192);
            }
            "gemma-7b-it" | "gemma2-9b-it" => {
                profile.context_window = Some(8192);
                profile.max_tokens = Some(8192);
            }
            "deepseek-r1-distill-llama-70b" => {
                profile.context_window = Some(65536);
                profile.max_tokens = Some(8192);
                profile.supports_reasoning = true;
            }
            _ if model_name.starts_with("llama") || model_name.starts_with("mixtral") => {
                profile.context_window = Some(8192);
            }
            _ => return None,
        }

        Some(profile)
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
    fn test_groq_provider_new() {
        let provider = GroqProvider::new("gsk_test");
        assert_eq!(provider.name(), "groq");
        assert!(provider.is_configured());
    }

    #[test]
    fn test_groq_provider_base_url() {
        let provider = GroqProvider::new("key");
        assert_eq!(provider.base_url(), "https://api.groq.com/openai/v1");
    }

    #[test]
    fn test_groq_provider_model_profile() {
        let provider = GroqProvider::new("key");

        assert!(provider.model_profile("llama-3.3-70b-versatile").is_some());
        assert!(provider.model_profile("mixtral-8x7b-32768").is_some());
        assert!(provider.model_profile("unknown-model").is_none());
    }

    #[test]
    fn test_groq_provider_headers() {
        let provider = GroqProvider::new("gsk_test");
        let headers = provider.default_headers();

        assert!(headers.contains_key(AUTHORIZATION));
    }
}

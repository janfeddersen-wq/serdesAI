//! Mistral AI provider implementation.

use crate::provider::{Provider, ProviderConfig, ProviderError};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use reqwest::Client;
use serdes_ai_models::ModelProfile;

/// Mistral AI provider.
#[derive(Debug)]
pub struct MistralProvider {
    config: ProviderConfig,
    client: Client,
}

impl MistralProvider {
    /// Create a new Mistral provider with an API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        let config = ProviderConfig::new().with_api_key(api_key);
        Self {
            client: config.build_client(),
            config,
        }
    }

    /// Create from environment variables.
    pub fn from_env() -> Result<Self, ProviderError> {
        let config = ProviderConfig::from_env("MISTRAL");
        if config.api_key.is_none() {
            return Err(ProviderError::MissingApiKey("MISTRAL_API_KEY"));
        }
        Ok(Self {
            client: config.build_client(),
            config,
        })
    }
}

impl Provider for MistralProvider {
    fn name(&self) -> &str {
        "mistral"
    }

    fn base_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or("https://api.mistral.ai/v1")
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
            supports_system_messages: true,
            supports_streaming: true,
            ..Default::default()
        };

        match model_name {
            "mistral-large-latest" | "mistral-large-2411" => {
                profile.context_window = Some(131072);
                profile.max_tokens = Some(8192);
            }
            "mistral-small-latest" | "mistral-small-2503" => {
                profile.context_window = Some(32768);
                profile.max_tokens = Some(8192);
            }
            "codestral-latest" | "codestral-2501" => {
                profile.context_window = Some(256000);
                profile.max_tokens = Some(8192);
            }
            "pixtral-large-latest" | "pixtral-12b-2409" => {
                profile.context_window = Some(131072);
                profile.max_tokens = Some(8192);
                profile.supports_images = true;
            }
            "ministral-3b-latest" | "ministral-8b-latest" => {
                profile.context_window = Some(131072);
                profile.max_tokens = Some(8192);
            }
            _ if model_name.starts_with("mistral")
                || model_name.starts_with("codestral")
                || model_name.starts_with("pixtral") =>
            {
                profile.context_window = Some(32768);
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
    fn test_mistral_provider_new() {
        let provider = MistralProvider::new("key");
        assert_eq!(provider.name(), "mistral");
        assert!(provider.is_configured());
    }

    #[test]
    fn test_mistral_provider_base_url() {
        let provider = MistralProvider::new("key");
        assert_eq!(provider.base_url(), "https://api.mistral.ai/v1");
    }

    #[test]
    fn test_mistral_provider_model_profile() {
        let provider = MistralProvider::new("key");

        assert!(provider.model_profile("mistral-large-latest").is_some());
        assert!(provider.model_profile("codestral-latest").is_some());
        assert!(provider.model_profile("pixtral-12b-2409").is_some());
    }
}

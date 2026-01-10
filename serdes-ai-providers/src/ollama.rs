//! Ollama provider implementation.
//!
//! Ollama runs LLMs locally with an OpenAI-compatible API.

use crate::provider::{Provider, ProviderConfig, ProviderError};
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use serdes_ai_models::ModelProfile;

/// Ollama provider.
#[derive(Debug)]
pub struct OllamaProvider {
    config: ProviderConfig,
    client: Client,
}

impl OllamaProvider {
    /// Default Ollama host.
    pub const DEFAULT_HOST: &'static str = "http://localhost:11434";

    /// Create a new Ollama provider.
    pub fn new() -> Self {
        let config = ProviderConfig::new();
        Self {
            client: config.build_client(),
            config,
        }
    }

    /// Create with a custom host.
    pub fn with_host(host: impl Into<String>) -> Self {
        let config = ProviderConfig::new().with_base_url(host);
        Self {
            client: config.build_client(),
            config,
        }
    }

    /// Create from environment variables.
    pub fn from_env() -> Result<Self, ProviderError> {
        let config = ProviderConfig::from_env("OLLAMA");
        Ok(Self {
            client: config.build_client(),
            config,
        })
    }
}

impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    fn base_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or(Self::DEFAULT_HOST)
    }

    fn client(&self) -> &Client {
        &self.client
    }

    fn default_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        headers
    }

    fn model_profile(&self, model_name: &str) -> Option<ModelProfile> {
        // Ollama can run any model, so we provide sensible defaults
        let mut profile = ModelProfile::default();
        profile.supports_system_messages = true;
        profile.supports_streaming = true;

        // Enable tools for models that support it
        let model_lower = model_name.to_lowercase();
        if model_lower.contains("llama")
            || model_lower.contains("mistral")
            || model_lower.contains("mixtral")
            || model_lower.contains("qwen")
            || model_lower.contains("deepseek")
        {
            profile.supports_tools = true;
        }

        // Vision models
        if model_lower.contains("llava") || model_lower.contains("vision") {
            profile.supports_images = true;
        }

        // Context windows
        if model_lower.contains("llama3.2") || model_lower.contains("llama-3.2") {
            profile.context_window = Some(131072);
        } else if model_lower.contains("llama3") || model_lower.contains("llama-3") {
            profile.context_window = Some(8192);
        } else if model_lower.contains("mistral") {
            profile.context_window = Some(32768);
        } else if model_lower.contains("mixtral") {
            profile.context_window = Some(32768);
        } else if model_lower.contains("deepseek") {
            profile.context_window = Some(65536);
        } else {
            profile.context_window = Some(4096);
        }

        Some(profile)
    }

    fn is_configured(&self) -> bool {
        // Ollama doesn't require an API key
        true
    }

    fn aliases(&self) -> &[&str] {
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_provider_new() {
        let provider = OllamaProvider::new();
        assert_eq!(provider.name(), "ollama");
        assert!(provider.is_configured());
    }

    #[test]
    fn test_ollama_provider_default() {
        let provider = OllamaProvider::default();
        assert_eq!(provider.base_url(), "http://localhost:11434");
    }

    #[test]
    fn test_ollama_provider_custom_host() {
        let provider = OllamaProvider::with_host("http://192.168.1.100:11434");
        assert_eq!(provider.base_url(), "http://192.168.1.100:11434");
    }

    #[test]
    fn test_ollama_provider_model_profile() {
        let provider = OllamaProvider::new();

        let profile = provider.model_profile("llama3.2:latest").unwrap();
        assert!(profile.supports_tools);
        assert_eq!(profile.context_window, Some(131072));

        let profile = provider.model_profile("llava:latest").unwrap();
        assert!(profile.supports_images);
    }

    #[test]
    fn test_ollama_provider_headers() {
        let provider = OllamaProvider::new();
        let headers = provider.default_headers();

        assert!(headers.contains_key("content-type"));
    }
}

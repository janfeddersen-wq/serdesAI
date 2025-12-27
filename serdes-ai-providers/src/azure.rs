//! Azure OpenAI provider implementation.

use crate::provider::{Provider, ProviderConfig, ProviderError};
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use serdes_ai_models::profile::openai_gpt4o_profile;
use serdes_ai_models::ModelProfile;

/// Azure OpenAI provider.
#[derive(Debug)]
pub struct AzureProvider {
    config: ProviderConfig,
    client: Client,
    /// Azure resource name.
    resource_name: String,
    /// API version.
    api_version: String,
}

impl AzureProvider {
    /// Default API version.
    pub const DEFAULT_API_VERSION: &'static str = "2024-08-01-preview";

    /// Create a new Azure OpenAI provider.
    pub fn new(
        api_key: impl Into<String>,
        resource_name: impl Into<String>,
    ) -> Self {
        let config = ProviderConfig::new().with_api_key(api_key);
        Self {
            client: config.build_client(),
            config,
            resource_name: resource_name.into(),
            api_version: Self::DEFAULT_API_VERSION.to_string(),
        }
    }

    /// Create from environment variables.
    pub fn from_env() -> Result<Self, ProviderError> {
        let config = ProviderConfig::from_env("AZURE_OPENAI");
        let api_key = config.api_key.clone()
            .ok_or(ProviderError::MissingApiKey("AZURE_OPENAI_API_KEY"))?;

        let resource_name = std::env::var("AZURE_OPENAI_RESOURCE")
            .or_else(|_| std::env::var("AZURE_OPENAI_ENDPOINT").map(|e| {
                // Extract resource name from endpoint URL
                e.replace("https://", "")
                    .replace(".openai.azure.com", "")
                    .replace(".openai.azure.com/", "")
            }))
            .map_err(|_| ProviderError::MissingConfig("AZURE_OPENAI_RESOURCE".to_string()))?;

        Ok(Self::new(api_key, resource_name))
    }

    /// Set API version.
    #[must_use]
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// Get the endpoint URL for a deployment.
    pub fn deployment_url(&self, deployment_id: &str) -> String {
        format!(
            "https://{}.openai.azure.com/openai/deployments/{}/chat/completions?api-version={}",
            self.resource_name, deployment_id, self.api_version
        )
    }
}

impl Provider for AzureProvider {
    fn name(&self) -> &str {
        "azure"
    }

    fn base_url(&self) -> &str {
        // Azure uses per-deployment URLs, not a single base
        "https://openai.azure.com"
    }

    fn client(&self) -> &Client {
        &self.client
    }

    fn default_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();

        if let Some(key) = &self.config.api_key {
            if let Ok(value) = HeaderValue::from_str(key) {
                headers.insert("api-key", value);
            }
        }

        headers.insert(
            "content-type",
            HeaderValue::from_static("application/json"),
        );

        headers
    }

    fn model_profile(&self, model_name: &str) -> Option<ModelProfile> {
        // Azure deployments can use any model, so we guess based on name
        if model_name.contains("gpt-4o") || model_name.contains("gpt4o") {
            Some(openai_gpt4o_profile())
        } else if model_name.contains("gpt-4") || model_name.contains("gpt4") {
            let mut profile = openai_gpt4o_profile();
            profile.context_window = Some(128000);
            Some(profile)
        } else if model_name.contains("gpt-35") || model_name.contains("gpt-3.5") {
            let mut profile = openai_gpt4o_profile();
            profile.max_tokens = Some(4096);
            profile.context_window = Some(16385);
            Some(profile)
        } else {
            Some(openai_gpt4o_profile())
        }
    }

    fn is_configured(&self) -> bool {
        self.config.api_key.is_some() && !self.resource_name.is_empty()
    }

    fn aliases(&self) -> &[&str] {
        &["azure-openai"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_azure_provider_new() {
        let provider = AzureProvider::new("key", "my-resource");
        assert_eq!(provider.name(), "azure");
        assert!(provider.is_configured());
    }

    #[test]
    fn test_azure_provider_deployment_url() {
        let provider = AzureProvider::new("key", "my-resource");
        let url = provider.deployment_url("gpt-4o");

        assert!(url.contains("my-resource"));
        assert!(url.contains("gpt-4o"));
        assert!(url.contains("api-version"));
    }

    #[test]
    fn test_azure_provider_headers() {
        let provider = AzureProvider::new("test-key", "resource");
        let headers = provider.default_headers();

        assert!(headers.contains_key("api-key"));
        assert!(headers.contains_key("content-type"));
    }
}

//! Google AI / Vertex AI provider implementations.

use crate::provider::{Provider, ProviderConfig, ProviderError};
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use serdes_ai_models::ModelProfile;

/// Google AI (Generative Language API) provider.
#[derive(Debug)]
pub struct GoogleProvider {
    config: ProviderConfig,
    client: Client,
}

impl GoogleProvider {
    /// Create a new Google AI provider with an API key.
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
            return Err(ProviderError::MissingApiKey("GOOGLE_API_KEY"));
        }
        Ok(Self {
            client: config.build_client(),
            config,
        })
    }

    /// Create from environment variables.
    pub fn from_env() -> Result<Self, ProviderError> {
        let config = ProviderConfig::from_env("GOOGLE");
        Self::from_config(config)
    }

    fn create_profile(model: &str) -> ModelProfile {
        let mut profile = ModelProfile::default();
        profile.supports_tools = true;
        profile.supports_parallel_tools = true;
        profile.supports_system_messages = true;
        profile.supports_images = true;
        profile.supports_streaming = true;

        if model.contains("flash") {
            profile.max_tokens = Some(8192);
            profile.context_window = Some(1000000);
            if model.contains("thinking") {
                profile.supports_reasoning = true;
            }
        } else if model.contains("pro") {
            profile.max_tokens = Some(8192);
            profile.context_window = Some(2000000);
        }

        if model.contains("gemini-2") || model.contains("gemini-exp") {
            profile.supports_native_structured_output = true;
            profile.supports_audio = true;
            profile.supports_video = true;
        }

        profile
    }
}

impl Provider for GoogleProvider {
    fn name(&self) -> &str {
        "google"
    }

    fn base_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or("https://generativelanguage.googleapis.com")
    }

    fn client(&self) -> &Client {
        &self.client
    }

    fn default_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            "content-type",
            HeaderValue::from_static("application/json"),
        );
        headers
    }

    fn model_profile(&self, model_name: &str) -> Option<ModelProfile> {
        if model_name.starts_with("gemini") {
            Some(Self::create_profile(model_name))
        } else {
            None
        }
    }

    fn is_configured(&self) -> bool {
        self.config.api_key.is_some()
    }

    fn aliases(&self) -> &[&str] {
        &["google-gla", "gemini"]
    }
}

/// Vertex AI provider.
#[derive(Debug)]
#[allow(dead_code)]
pub struct VertexAIProvider {
    config: ProviderConfig,
    client: Client,
    project_id: String,
    location: String,
}

impl VertexAIProvider {
    /// Create a new Vertex AI provider.
    pub fn new(
        project_id: impl Into<String>,
        location: impl Into<String>,
    ) -> Self {
        let project_id = project_id.into();
        let location = location.into();
        let config = ProviderConfig::new();

        Self {
            client: config.build_client(),
            config,
            project_id,
            location,
        }
    }

    /// Create from environment variables.
    pub fn from_env() -> Result<Self, ProviderError> {
        let config = ProviderConfig::from_env("VERTEX");
        let project = config.project.clone()
            .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
            .or_else(|| std::env::var("GCLOUD_PROJECT").ok())
            .ok_or(ProviderError::MissingConfig("VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT".to_string()))?;

        let location = config.region.clone()
            .or_else(|| std::env::var("GOOGLE_CLOUD_REGION").ok())
            .unwrap_or_else(|| "us-central1".to_string());

        Ok(Self {
            client: config.build_client(),
            config,
            project_id: project,
            location,
        })
    }
}

impl Provider for VertexAIProvider {
    fn name(&self) -> &str {
        "google-vertex"
    }

    fn base_url(&self) -> &str {
        // Build dynamically based on location
        // This is a simplification - real impl would cache this
        "https://aiplatform.googleapis.com"
    }

    fn client(&self) -> &Client {
        &self.client
    }

    fn default_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            "content-type",
            HeaderValue::from_static("application/json"),
        );
        // Note: OAuth token would be added per-request
        headers
    }

    fn model_profile(&self, model_name: &str) -> Option<ModelProfile> {
        if model_name.starts_with("gemini") {
            Some(GoogleProvider::create_profile(model_name))
        } else {
            None
        }
    }

    fn is_configured(&self) -> bool {
        !self.project_id.is_empty()
    }

    fn aliases(&self) -> &[&str] {
        &["vertex"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_google_provider_new() {
        let provider = GoogleProvider::new("test-key");
        assert_eq!(provider.name(), "google");
        assert!(provider.is_configured());
    }

    #[test]
    fn test_google_provider_base_url() {
        let provider = GoogleProvider::new("key");
        assert_eq!(provider.base_url(), "https://generativelanguage.googleapis.com");
    }

    #[test]
    fn test_google_provider_model_profile() {
        let provider = GoogleProvider::new("key");

        assert!(provider.model_profile("gemini-2.0-flash").is_some());
        assert!(provider.model_profile("gemini-1.5-pro").is_some());
        assert!(provider.model_profile("unknown").is_none());
    }

    #[test]
    fn test_google_provider_aliases() {
        let provider = GoogleProvider::new("key");
        let aliases = provider.aliases();
        assert!(aliases.contains(&"gemini"));
    }

    #[test]
    fn test_vertex_provider_new() {
        let provider = VertexAIProvider::new("my-project", "us-central1");
        assert_eq!(provider.name(), "google-vertex");
        assert!(provider.is_configured());
    }

    #[test]
    fn test_vertex_provider_aliases() {
        let provider = VertexAIProvider::new("project", "region");
        let aliases = provider.aliases();
        assert!(aliases.contains(&"vertex"));
    }
}

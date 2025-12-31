//! Azure OpenAI model implementation.
//!
//! Azure OpenAI provides access to OpenAI models through Azure's cloud platform
//! with enterprise features like VNETs, managed identity, and regional deployment.
//!
//! ## Example
//!
//! ```ignore
//! use serdes_ai_models::azure::AzureOpenAIModel;
//!
//! let model = AzureOpenAIModel::new(
//!     "my-deployment",
//!     "https://my-resource.openai.azure.com",
//!     "2024-02-15-preview",
//!     "your-api-key",
//! );
//! ```
//!
//! ## Environment Variables
//!
//! - `AZURE_OPENAI_ENDPOINT` - Azure OpenAI endpoint URL
//! - `AZURE_OPENAI_API_KEY` - API key
//! - `AZURE_OPENAI_API_VERSION` - API version (default: 2024-02-15-preview)

use async_trait::async_trait;
use std::time::Duration;

use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse};
use crate::openai::OpenAIChatModel;
use crate::profile::ModelProfile;
use serdes_ai_core::{ModelRequest, ModelResponse, ModelSettings};

/// Azure OpenAI model client.
///
/// Azure OpenAI uses the same API format as OpenAI, but with
/// different authentication and endpoint structure.
#[derive(Debug, Clone)]
pub struct AzureOpenAIModel {
    /// Inner OpenAI-compatible model.
    inner: OpenAIChatModel,
    /// Deployment name (for identification).
    deployment_name: String,
}

impl AzureOpenAIModel {
    /// Default API version.
    pub const DEFAULT_API_VERSION: &'static str = "2024-02-15-preview";

    /// Create a new Azure OpenAI model.
    ///
    /// # Arguments
    ///
    /// * `deployment_name` - The deployment name in Azure
    /// * `endpoint` - Azure OpenAI endpoint (e.g., https://my-resource.openai.azure.com)
    /// * `api_version` - API version string
    /// * `api_key` - API key for authentication
    pub fn new(
        deployment_name: impl Into<String>,
        endpoint: impl Into<String>,
        api_version: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        let deployment_name = deployment_name.into();
        let endpoint = endpoint.into().trim_end_matches('/').to_string();
        let _api_version = api_version.into(); // TODO: Include in URL query params

        // Construct the Azure-specific URL
        let base_url = format!(
            "{}/openai/deployments/{}",
            endpoint, deployment_name
        );

        // Create inner model with Azure URL
        // Note: Azure uses api-key header instead of Authorization: Bearer
        let inner = OpenAIChatModel::new(&deployment_name, api_key)
            .with_base_url(base_url);

        Self {
            inner,
            deployment_name,
        }
    }

    /// Create from environment variables.
    ///
    /// Uses:
    /// - `AZURE_OPENAI_ENDPOINT`
    /// - `AZURE_OPENAI_API_KEY`
    /// - `AZURE_OPENAI_API_VERSION` (optional, defaults to 2024-02-15-preview)
    pub fn from_env(deployment_name: impl Into<String>) -> Result<Self, ModelError> {
        let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT")
            .map_err(|_| ModelError::configuration("AZURE_OPENAI_ENDPOINT not set"))?;
        let api_key = std::env::var("AZURE_OPENAI_API_KEY")
            .map_err(|_| ModelError::configuration("AZURE_OPENAI_API_KEY not set"))?;
        let api_version = std::env::var("AZURE_OPENAI_API_VERSION")
            .unwrap_or_else(|_| Self::DEFAULT_API_VERSION.to_string());

        Ok(Self::new(deployment_name, endpoint, api_version, api_key))
    }

    /// Set the default timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.inner = self.inner.with_timeout(timeout);
        self
    }

    /// Set a custom profile.
    pub fn with_profile(mut self, profile: ModelProfile) -> Self {
        self.inner = self.inner.with_profile(profile);
        self
    }

    /// Get the deployment name.
    pub fn deployment_name(&self) -> &str {
        &self.deployment_name
    }
}

#[async_trait]
impl Model for AzureOpenAIModel {
    fn name(&self) -> &str {
        &self.deployment_name
    }

    fn system(&self) -> &str {
        "azure"
    }

    fn profile(&self) -> &ModelProfile {
        self.inner.profile()
    }

    async fn request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        self.inner.request(messages, settings, params).await
    }

    async fn request_stream(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<StreamedResponse, ModelError> {
        self.inner.request_stream(messages, settings, params).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_azure_model_creation() {
        let model = AzureOpenAIModel::new(
            "gpt-4",
            "https://my-resource.openai.azure.com",
            "2024-02-15-preview",
            "test-key",
        );
        assert_eq!(model.name(), "gpt-4");
        assert_eq!(model.system(), "azure");
    }
}

//! Provider configuration.

use serde::{Deserialize, Serialize};
use url::Url;

/// Configuration for an LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// API key for authentication.
    #[serde(skip_serializing)]
    pub api_key: Option<String>,
    /// Base URL for the API.
    pub base_url: Option<Url>,
    /// Organization ID (if applicable).
    pub organization: Option<String>,
    /// Request timeout in seconds.
    pub timeout_seconds: u64,
    /// Maximum retries.
    pub max_retries: u32,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: None,
            organization: None,
            timeout_seconds: 60,
            max_retries: 3,
        }
    }
}

impl ProviderConfig {
    /// Create a new config with an API key.
    #[must_use]
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self {
            api_key: Some(api_key.into()),
            ..Default::default()
        }
    }

    /// Set the base URL.
    #[must_use]
    pub fn base_url(mut self, url: Url) -> Self {
        self.base_url = Some(url);
        self
    }

    /// Set the organization.
    #[must_use]
    pub fn organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }
}

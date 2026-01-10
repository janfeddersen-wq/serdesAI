//! Gateway provider for AI routing services.
//!
//! This provider allows routing requests through AI gateways like:
//!
//! - **Portkey** - AI Gateway with caching, fallbacks, and observability
//! - **LiteLLM Proxy** - Unified API for 100+ LLMs
//! - **Helicone** - LLM observability platform
//! - **AI Gateway** - Cloudflare AI Gateway
//! - **Custom proxies** - Any OpenAI-compatible proxy
//!
//! ## Example
//!
//! ```rust,ignore
//! use serdes_ai_providers::GatewayProvider;
//!
//! // Basic gateway
//! let gateway = GatewayProvider::new("https://gateway.example.com/v1")
//!     .with_api_key("your-api-key");
//!
//! // Portkey gateway
//! let portkey = GatewayProvider::portkey("your-portkey-api-key")
//!     .with_virtual_key("openai-key-alias");
//!
//! // LiteLLM Proxy
//! let litellm = GatewayProvider::litellm("http://localhost:4000")
//!     .with_api_key("sk-...");
//!
//! // With custom headers
//! let custom = GatewayProvider::new("https://my-gateway.com/v1")
//!     .with_header("X-Custom-Header", "value")
//!     .with_api_key("key");
//! ```
//!
//! ## Backend Model Specification
//!
//! Most gateways use the model name to determine routing. Some support
//! explicit backend specification:
//!
//! ```rust,ignore
//! let gateway = GatewayProvider::new("https://gateway.com/v1")
//!     .with_backend("openai")  // Optional: specify backend provider
//!     .with_api_key("key");
//! ```

use crate::provider::{Provider, ProviderError};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION};
use reqwest::Client;
use serdes_ai_models::ModelProfile;
use std::collections::HashMap;
use std::time::Duration;

/// Configuration for a gateway provider.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Gateway URL (e.g., "https://gateway.portkey.ai/v1")
    pub gateway_url: String,
    /// API key for gateway authentication
    pub api_key: Option<String>,
    /// Backend provider (e.g., "openai", "anthropic")
    pub backend: Option<String>,
    /// Virtual key for key aliasing (Portkey)
    pub virtual_key: Option<String>,
    /// Custom headers to include in requests
    pub custom_headers: HashMap<String, String>,
    /// Request timeout
    pub timeout: Duration,
    /// Provider name override
    pub name: String,
}

impl GatewayConfig {
    /// Create a new gateway configuration.
    pub fn new(gateway_url: impl Into<String>) -> Self {
        Self {
            gateway_url: gateway_url.into(),
            api_key: None,
            backend: None,
            virtual_key: None,
            custom_headers: HashMap::new(),
            timeout: Duration::from_secs(60),
            name: "gateway".to_string(),
        }
    }

    /// Set the API key.
    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the backend provider.
    #[must_use]
    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.backend = Some(backend.into());
        self
    }

    /// Set a virtual key (for Portkey).
    #[must_use]
    pub fn with_virtual_key(mut self, key: impl Into<String>) -> Self {
        self.virtual_key = Some(key.into());
        self
    }

    /// Add a custom header.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom_headers.insert(name.into(), value.into());
        self
    }

    /// Set the request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the provider name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

/// Gateway provider for AI routing services.
///
/// Routes requests through an AI gateway that provides features like:
/// - Multi-provider routing
/// - Caching and rate limiting
/// - Fallbacks and retries
/// - Observability and logging
/// - Key management and aliasing
#[derive(Debug)]
pub struct GatewayProvider {
    config: GatewayConfig,
    client: Client,
}

impl GatewayProvider {
    /// Create a new gateway provider with a custom URL.
    pub fn new(gateway_url: impl Into<String>) -> Self {
        Self::with_config(GatewayConfig::new(gateway_url))
    }

    /// Create a gateway provider with full configuration.
    pub fn with_config(config: GatewayConfig) -> Self {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .unwrap_or_default();

        Self { config, client }
    }

    /// Create a Portkey gateway provider.
    ///
    /// Portkey is an AI Gateway that provides caching, fallbacks,
    /// load balancing, and observability.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Your Portkey API key
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let portkey = GatewayProvider::portkey("your-portkey-key")
    ///     .with_virtual_key("openai-prod");  // Optional: use a virtual key
    /// ```
    pub fn portkey(api_key: impl Into<String>) -> Self {
        Self::with_config(
            GatewayConfig::new("https://api.portkey.ai/v1")
                .with_api_key(api_key)
                .with_name("portkey"),
        )
    }

    /// Create a Portkey gateway from environment variables.
    ///
    /// Looks for:
    /// - `PORTKEY_API_KEY` - Portkey API key
    /// - `PORTKEY_VIRTUAL_KEY` - Optional virtual key
    pub fn portkey_from_env() -> Result<Self, ProviderError> {
        let api_key = std::env::var("PORTKEY_API_KEY")
            .map_err(|_| ProviderError::MissingApiKey("PORTKEY_API_KEY"))?;

        let mut provider = Self::portkey(api_key);

        if let Ok(virtual_key) = std::env::var("PORTKEY_VIRTUAL_KEY") {
            provider.config.virtual_key = Some(virtual_key);
        }

        Ok(provider)
    }

    /// Create a LiteLLM Proxy gateway provider.
    ///
    /// LiteLLM Proxy provides a unified OpenAI-compatible API
    /// for 100+ LLM providers.
    ///
    /// # Arguments
    ///
    /// * `proxy_url` - The LiteLLM proxy URL (e.g., "http://localhost:4000")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let litellm = GatewayProvider::litellm("http://localhost:4000")
    ///     .with_api_key("sk-...");  // Optional: if proxy requires auth
    /// ```
    pub fn litellm(proxy_url: impl Into<String>) -> Self {
        let url = proxy_url.into();
        let base_url = if url.ends_with("/v1") {
            url
        } else {
            format!("{}/v1", url.trim_end_matches('/'))
        };

        Self::with_config(GatewayConfig::new(base_url).with_name("litellm"))
    }

    /// Create a LiteLLM gateway from environment variables.
    ///
    /// Looks for:
    /// - `LITELLM_PROXY_URL` or `LITELLM_BASE_URL` - Proxy URL
    /// - `LITELLM_API_KEY` - Optional API key
    pub fn litellm_from_env() -> Result<Self, ProviderError> {
        let proxy_url = std::env::var("LITELLM_PROXY_URL")
            .or_else(|_| std::env::var("LITELLM_BASE_URL"))
            .map_err(|_| ProviderError::MissingConfig("LITELLM_PROXY_URL".into()))?;

        let mut provider = Self::litellm(proxy_url);

        if let Ok(api_key) = std::env::var("LITELLM_API_KEY") {
            provider.config.api_key = Some(api_key);
        }

        Ok(provider)
    }

    /// Create a Helicone gateway provider.
    ///
    /// Helicone is an LLM observability platform that provides
    /// logging, caching, and analytics.
    ///
    /// # Arguments
    ///
    /// * `helicone_api_key` - Your Helicone API key
    /// * `target_base_url` - The target provider URL (e.g., OpenAI)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let helicone = GatewayProvider::helicone(
    ///     "your-helicone-key",
    ///     "https://api.openai.com/v1",
    /// ).with_api_key("sk-openai-key");
    /// ```
    pub fn helicone(
        helicone_api_key: impl Into<String>,
        target_base_url: impl Into<String>,
    ) -> Self {
        Self::with_config(
            GatewayConfig::new("https://oai.helicone.ai/v1")
                .with_header(
                    "Helicone-Auth",
                    format!("Bearer {}", helicone_api_key.into()),
                )
                .with_header("Helicone-Target-URL", target_base_url)
                .with_name("helicone"),
        )
    }

    /// Create a Helicone gateway from environment variables.
    ///
    /// Looks for:
    /// - `HELICONE_API_KEY` - Helicone API key
    /// - `OPENAI_API_KEY` - Target provider API key
    pub fn helicone_from_env() -> Result<Self, ProviderError> {
        let helicone_key = std::env::var("HELICONE_API_KEY")
            .map_err(|_| ProviderError::MissingApiKey("HELICONE_API_KEY"))?;

        let openai_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| ProviderError::MissingApiKey("OPENAI_API_KEY"))?;

        Ok(Self::helicone(helicone_key, "https://api.openai.com/v1").with_api_key(openai_key))
    }

    /// Create a Cloudflare AI Gateway provider.
    ///
    /// # Arguments
    ///
    /// * `account_id` - Your Cloudflare account ID
    /// * `gateway_id` - Your gateway ID
    /// * `provider` - The target provider (e.g., "openai", "anthropic")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let cf = GatewayProvider::cloudflare("account-id", "my-gateway", "openai")
    ///     .with_api_key("sk-openai-key");
    /// ```
    pub fn cloudflare(
        account_id: impl Into<String>,
        gateway_id: impl Into<String>,
        provider: impl Into<String>,
    ) -> Self {
        let url = format!(
            "https://gateway.ai.cloudflare.com/v1/{}/{}/{}",
            account_id.into(),
            gateway_id.into(),
            provider.into()
        );

        Self::with_config(GatewayConfig::new(url).with_name("cloudflare-ai-gateway"))
    }

    /// Create from generic environment variables.
    ///
    /// Looks for:
    /// - `AI_GATEWAY_URL` or `GATEWAY_URL` - Gateway URL
    /// - `AI_GATEWAY_API_KEY` or `GATEWAY_API_KEY` - API key
    /// - `AI_GATEWAY_BACKEND` - Optional backend provider
    pub fn from_env() -> Result<Self, ProviderError> {
        let gateway_url = std::env::var("AI_GATEWAY_URL")
            .or_else(|_| std::env::var("GATEWAY_URL"))
            .map_err(|_| ProviderError::MissingConfig("AI_GATEWAY_URL".into()))?;

        let mut config = GatewayConfig::new(gateway_url);

        if let Ok(api_key) =
            std::env::var("AI_GATEWAY_API_KEY").or_else(|_| std::env::var("GATEWAY_API_KEY"))
        {
            config.api_key = Some(api_key);
        }

        if let Ok(backend) = std::env::var("AI_GATEWAY_BACKEND") {
            config.backend = Some(backend);
        }

        Ok(Self::with_config(config))
    }

    /// Set the API key.
    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.config.api_key = Some(key.into());
        self
    }

    /// Set the backend provider.
    #[must_use]
    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.config.backend = Some(backend.into());
        self
    }

    /// Set a virtual key (for Portkey).
    #[must_use]
    pub fn with_virtual_key(mut self, key: impl Into<String>) -> Self {
        self.config.virtual_key = Some(key.into());
        self
    }

    /// Add a custom header.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.custom_headers.insert(name.into(), value.into());
        self
    }

    /// Set the request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        // Rebuild client with new timeout
        self.client = Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_default();
        self
    }

    /// Set the provider name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.config.name = name.into();
        self
    }

    /// Get the configured backend, if any.
    pub fn backend(&self) -> Option<&str> {
        self.config.backend.as_deref()
    }
}

impl Provider for GatewayProvider {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn base_url(&self) -> &str {
        &self.config.gateway_url
    }

    fn client(&self) -> &Client {
        &self.client
    }

    fn default_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();

        // Authorization header
        if let Some(key) = &self.config.api_key {
            let auth_value = format!("Bearer {}", key);
            if let Ok(value) = HeaderValue::from_str(&auth_value) {
                headers.insert(AUTHORIZATION, value);
            }
        }

        // Content-Type
        headers.insert("content-type", HeaderValue::from_static("application/json"));

        // Portkey-specific headers
        if self.config.name == "portkey" {
            // Portkey uses x-portkey-api-key for its own auth
            if let Some(key) = &self.config.api_key {
                if let Ok(value) = HeaderValue::from_str(key) {
                    headers.insert("x-portkey-api-key", value);
                }
            }

            // Virtual key for aliased provider keys
            if let Some(virtual_key) = &self.config.virtual_key {
                if let Ok(value) = HeaderValue::from_str(virtual_key) {
                    headers.insert("x-portkey-virtual-key", value);
                }
            }
        }

        // Backend header (if specified)
        if let Some(backend) = &self.config.backend {
            if let Ok(value) = HeaderValue::from_str(backend) {
                headers.insert("x-gateway-backend", value);
            }
        }

        // Custom headers
        for (name, value) in &self.config.custom_headers {
            let header_name: Result<HeaderName, _> = name.as_str().try_into();
            let header_value = HeaderValue::from_str(value);
            if let (Ok(hn), Ok(hv)) = (header_name, header_value) {
                headers.insert(hn, hv);
            }
        }

        headers
    }

    fn model_profile(&self, model_name: &str) -> Option<ModelProfile> {
        // Infer profile from model name since gateways can route to any model
        let mut profile = ModelProfile::default();
        profile.supports_tools = true;
        profile.supports_system_messages = true;
        profile.supports_streaming = true;

        let model_lower = model_name.to_lowercase();

        // Try to infer context window from model name
        if model_lower.contains("gpt-4") || model_lower.contains("gpt4") {
            profile.context_window = Some(128000);
            if model_lower.contains("o1") || model_lower.contains("o3") {
                profile.supports_reasoning = true;
            }
        } else if model_lower.contains("gpt-3.5") {
            profile.context_window = Some(16385);
        } else if model_lower.contains("claude-3") {
            profile.context_window = Some(200000);
        } else if model_lower.contains("claude") {
            profile.context_window = Some(100000);
        } else if model_lower.contains("gemini") {
            profile.context_window = Some(1000000);
        } else if model_lower.contains("llama-3.3") || model_lower.contains("llama-3.2") {
            profile.context_window = Some(131072);
        } else if model_lower.contains("llama") {
            profile.context_window = Some(8192);
        } else if model_lower.contains("mistral") {
            profile.context_window = Some(32768);
        } else if model_lower.contains("mixtral") {
            profile.context_window = Some(32768);
        } else if model_lower.contains("deepseek") {
            profile.context_window = Some(65536);
            if model_lower.contains("r1") {
                profile.supports_reasoning = true;
            }
        } else if model_lower.contains("qwen") {
            profile.context_window = Some(32768);
        } else {
            // Default for unknown models
            profile.context_window = Some(8192);
        }

        Some(profile)
    }

    fn is_configured(&self) -> bool {
        // Gateway is configured if we have a URL
        // API key may not be required for some gateways (like local LiteLLM)
        !self.config.gateway_url.is_empty()
    }

    fn aliases(&self) -> &[&str] {
        match self.config.name.as_str() {
            "portkey" => &["pk", "portkey-ai"],
            "litellm" => &["lite-llm", "litellm-proxy"],
            "helicone" => &[],
            "cloudflare-ai-gateway" => &["cf-gateway", "cloudflare"],
            _ => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_gateway() {
        let gateway =
            GatewayProvider::new("https://gateway.example.com/v1").with_api_key("sk-test");

        assert_eq!(gateway.name(), "gateway");
        assert_eq!(gateway.base_url(), "https://gateway.example.com/v1");
        assert!(gateway.is_configured());
    }

    #[test]
    fn test_portkey_gateway() {
        let portkey = GatewayProvider::portkey("pk-test").with_virtual_key("openai-prod");

        assert_eq!(portkey.name(), "portkey");
        assert_eq!(portkey.base_url(), "https://api.portkey.ai/v1");

        let headers = portkey.default_headers();
        assert!(headers.contains_key("x-portkey-api-key"));
        assert!(headers.contains_key("x-portkey-virtual-key"));
    }

    #[test]
    fn test_litellm_gateway() {
        let litellm = GatewayProvider::litellm("http://localhost:4000");

        assert_eq!(litellm.name(), "litellm");
        assert_eq!(litellm.base_url(), "http://localhost:4000/v1");
    }

    #[test]
    fn test_litellm_gateway_with_v1() {
        let litellm = GatewayProvider::litellm("http://localhost:4000/v1");
        assert_eq!(litellm.base_url(), "http://localhost:4000/v1");
    }

    #[test]
    fn test_helicone_gateway() {
        let helicone = GatewayProvider::helicone("hc-key", "https://api.openai.com/v1")
            .with_api_key("sk-openai");

        assert_eq!(helicone.name(), "helicone");
        assert_eq!(helicone.base_url(), "https://oai.helicone.ai/v1");

        let headers = helicone.default_headers();
        assert!(headers.contains_key("Helicone-Auth"));
        assert!(headers.contains_key("Helicone-Target-URL"));
    }

    #[test]
    fn test_cloudflare_gateway() {
        let cf = GatewayProvider::cloudflare("acc123", "my-gw", "openai").with_api_key("sk-test");

        assert_eq!(cf.name(), "cloudflare-ai-gateway");
        assert!(cf.base_url().contains("acc123"));
        assert!(cf.base_url().contains("my-gw"));
        assert!(cf.base_url().contains("openai"));
    }

    #[test]
    fn test_custom_headers() {
        let gateway = GatewayProvider::new("https://gateway.com/v1")
            .with_header("X-Custom-Header", "custom-value")
            .with_header("X-Another", "another-value");

        let headers = gateway.default_headers();
        assert!(headers.contains_key("x-custom-header"));
        assert!(headers.contains_key("x-another"));
    }

    #[test]
    fn test_backend_specification() {
        let gateway = GatewayProvider::new("https://gateway.com/v1").with_backend("anthropic");

        assert_eq!(gateway.backend(), Some("anthropic"));

        let headers = gateway.default_headers();
        assert!(headers.contains_key("x-gateway-backend"));
    }

    #[test]
    fn test_model_profile_inference() {
        let gateway = GatewayProvider::new("https://gateway.com/v1");

        let gpt4_profile = gateway.model_profile("gpt-4o").unwrap();
        assert_eq!(gpt4_profile.context_window, Some(128000));

        let claude_profile = gateway.model_profile("claude-3-sonnet").unwrap();
        assert_eq!(claude_profile.context_window, Some(200000));

        let deepseek_r1 = gateway.model_profile("deepseek-r1").unwrap();
        assert!(deepseek_r1.supports_reasoning);
    }

    #[test]
    fn test_timeout_configuration() {
        let gateway =
            GatewayProvider::new("https://gateway.com/v1").with_timeout(Duration::from_secs(120));

        assert_eq!(gateway.config.timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_provider_aliases() {
        let portkey = GatewayProvider::portkey("key");
        assert!(portkey.aliases().contains(&"pk"));

        let litellm = GatewayProvider::litellm("http://localhost:4000");
        assert!(litellm.aliases().contains(&"lite-llm"));
    }
}

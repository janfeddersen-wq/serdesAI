//! OpenAI-compatible provider implementations.
//!
//! These providers use the OpenAI API format but with different endpoints.

use crate::provider::{Provider, ProviderConfig, ProviderError};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use reqwest::Client;
use serdes_ai_models::ModelProfile;

// ============================================================================
// Base OpenAI-Compatible Provider
// ============================================================================

/// Base implementation for OpenAI-compatible providers.
#[derive(Debug)]
pub struct OpenAICompatibleProvider {
    name: String,
    config: ProviderConfig,
    client: Client,
    default_base_url: String,
    env_prefix: String,
    aliases: Vec<&'static str>,
}

impl OpenAICompatibleProvider {
    /// Create a new OpenAI-compatible provider.
    pub fn new(
        name: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        let config = ProviderConfig::new().with_api_key(api_key);
        let default_base_url = base_url.into();
        Self {
            name: name.into(),
            client: config.build_client(),
            config,
            default_base_url,
            env_prefix: String::new(),
            aliases: Vec::new(),
        }
    }

    fn with_env_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.env_prefix = prefix.into();
        self
    }

    fn with_aliases(mut self, aliases: Vec<&'static str>) -> Self {
        self.aliases = aliases;
        self
    }
}

impl Provider for OpenAICompatibleProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn base_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or(&self.default_base_url)
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

        headers.insert(
            "content-type",
            HeaderValue::from_static("application/json"),
        );

        headers
    }

    fn model_profile(&self, _model_name: &str) -> Option<ModelProfile> {
        // Generic profile for unknown models
        let mut profile = ModelProfile::default();
        profile.supports_tools = true;
        profile.supports_system_messages = true;
        profile.supports_streaming = true;
        Some(profile)
    }

    fn is_configured(&self) -> bool {
        self.config.api_key.is_some()
    }

    fn aliases(&self) -> &[&str] {
        &self.aliases
    }
}

// ============================================================================
// Together AI
// ============================================================================

/// Together AI provider.
#[derive(Debug)]
pub struct TogetherProvider {
    inner: OpenAICompatibleProvider,
}

impl TogetherProvider {
    /// Create a new Together AI provider.
    pub fn new(api_key: impl Into<String>) -> Self {
        let inner = OpenAICompatibleProvider::new(
            "together",
            api_key,
            "https://api.together.xyz/v1",
        )
        .with_env_prefix("TOGETHER")
        .with_aliases(vec!["together-ai"]);

        Self { inner }
    }

    /// Create from environment variables.
    pub fn from_env() -> Result<Self, ProviderError> {
        let config = ProviderConfig::from_env("TOGETHER");
        let api_key = config.api_key
            .ok_or(ProviderError::MissingApiKey("TOGETHER_API_KEY"))?;
        Ok(Self::new(api_key))
    }
}

impl Provider for TogetherProvider {
    fn name(&self) -> &str { self.inner.name() }
    fn base_url(&self) -> &str { self.inner.base_url() }
    fn client(&self) -> &Client { self.inner.client() }
    fn default_headers(&self) -> HeaderMap { self.inner.default_headers() }
    fn is_configured(&self) -> bool { self.inner.is_configured() }
    fn aliases(&self) -> &[&str] { self.inner.aliases() }

    fn model_profile(&self, model_name: &str) -> Option<ModelProfile> {
        let mut profile = ModelProfile::default();
        profile.supports_tools = true;
        profile.supports_system_messages = true;
        profile.supports_streaming = true;

        let model_lower = model_name.to_lowercase();
        if model_lower.contains("llama-3.3") || model_lower.contains("llama-3.2") {
            profile.context_window = Some(131072);
        } else if model_lower.contains("llama") {
            profile.context_window = Some(8192);
        } else if model_lower.contains("mixtral") {
            profile.context_window = Some(32768);
        } else if model_lower.contains("qwen") {
            profile.context_window = Some(32768);
        } else if model_lower.contains("deepseek") {
            profile.context_window = Some(65536);
            if model_lower.contains("r1") {
                profile.supports_reasoning = true;
            }
        }

        Some(profile)
    }
}

// ============================================================================
// Fireworks AI
// ============================================================================

/// Fireworks AI provider.
#[derive(Debug)]
pub struct FireworksProvider {
    inner: OpenAICompatibleProvider,
}

impl FireworksProvider {
    /// Create a new Fireworks AI provider.
    pub fn new(api_key: impl Into<String>) -> Self {
        let inner = OpenAICompatibleProvider::new(
            "fireworks",
            api_key,
            "https://api.fireworks.ai/inference/v1",
        )
        .with_env_prefix("FIREWORKS");

        Self { inner }
    }

    /// Create from environment variables.
    pub fn from_env() -> Result<Self, ProviderError> {
        let config = ProviderConfig::from_env("FIREWORKS");
        let api_key = config.api_key
            .ok_or(ProviderError::MissingApiKey("FIREWORKS_API_KEY"))?;
        Ok(Self::new(api_key))
    }
}

impl Provider for FireworksProvider {
    fn name(&self) -> &str { "fireworks" }
    fn base_url(&self) -> &str { self.inner.base_url() }
    fn client(&self) -> &Client { self.inner.client() }
    fn default_headers(&self) -> HeaderMap { self.inner.default_headers() }
    fn model_profile(&self, model_name: &str) -> Option<ModelProfile> { self.inner.model_profile(model_name) }
    fn is_configured(&self) -> bool { self.inner.is_configured() }
    fn aliases(&self) -> &[&str] { &[] }
}

// ============================================================================
// DeepSeek
// ============================================================================

/// DeepSeek provider.
#[derive(Debug)]
pub struct DeepSeekProvider {
    inner: OpenAICompatibleProvider,
}

impl DeepSeekProvider {
    /// Create a new DeepSeek provider.
    pub fn new(api_key: impl Into<String>) -> Self {
        let inner = OpenAICompatibleProvider::new(
            "deepseek",
            api_key,
            "https://api.deepseek.com/v1",
        )
        .with_env_prefix("DEEPSEEK");

        Self { inner }
    }

    /// Create from environment variables.
    pub fn from_env() -> Result<Self, ProviderError> {
        let config = ProviderConfig::from_env("DEEPSEEK");
        let api_key = config.api_key
            .ok_or(ProviderError::MissingApiKey("DEEPSEEK_API_KEY"))?;
        Ok(Self::new(api_key))
    }
}

impl Provider for DeepSeekProvider {
    fn name(&self) -> &str { "deepseek" }
    fn base_url(&self) -> &str { self.inner.base_url() }
    fn client(&self) -> &Client { self.inner.client() }
    fn default_headers(&self) -> HeaderMap { self.inner.default_headers() }
    fn is_configured(&self) -> bool { self.inner.is_configured() }
    fn aliases(&self) -> &[&str] { &[] }

    fn model_profile(&self, model_name: &str) -> Option<ModelProfile> {
        let mut profile = ModelProfile::default();
        profile.supports_tools = true;
        profile.supports_system_messages = true;
        profile.supports_streaming = true;

        match model_name {
            "deepseek-chat" => {
                profile.context_window = Some(65536);
                profile.max_tokens = Some(8192);
            }
            "deepseek-reasoner" | "deepseek-r1" => {
                profile.context_window = Some(65536);
                profile.max_tokens = Some(8192);
                profile.supports_reasoning = true;
            }
            "deepseek-coder" => {
                profile.context_window = Some(65536);
                profile.max_tokens = Some(8192);
            }
            _ => {
                profile.context_window = Some(65536);
            }
        }

        Some(profile)
    }
}

// ============================================================================
// OpenRouter
// ============================================================================

/// OpenRouter provider.
#[derive(Debug)]
pub struct OpenRouterProvider {
    inner: OpenAICompatibleProvider,
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider.
    pub fn new(api_key: impl Into<String>) -> Self {
        let inner = OpenAICompatibleProvider::new(
            "openrouter",
            api_key,
            "https://openrouter.ai/api/v1",
        )
        .with_env_prefix("OPENROUTER");

        Self { inner }
    }

    /// Create from environment variables.
    pub fn from_env() -> Result<Self, ProviderError> {
        let config = ProviderConfig::from_env("OPENROUTER");
        let api_key = config.api_key
            .ok_or(ProviderError::MissingApiKey("OPENROUTER_API_KEY"))?;
        Ok(Self::new(api_key))
    }
}

impl Provider for OpenRouterProvider {
    fn name(&self) -> &str { "openrouter" }
    fn base_url(&self) -> &str { self.inner.base_url() }
    fn client(&self) -> &Client { self.inner.client() }
    fn is_configured(&self) -> bool { self.inner.is_configured() }
    fn aliases(&self) -> &[&str] { &[] }

    fn default_headers(&self) -> HeaderMap {
        let mut headers = self.inner.default_headers();
        // OpenRouter recommends these headers
        headers.insert(
            "HTTP-Referer",
            HeaderValue::from_static("https://github.com/serdes-ai"),
        );
        headers.insert(
            "X-Title",
            HeaderValue::from_static("serdes-ai"),
        );
        headers
    }

    fn model_profile(&self, model_name: &str) -> Option<ModelProfile> {
        // OpenRouter routes to many models, so we try to infer from name
        let mut profile = ModelProfile::default();
        profile.supports_tools = true;
        profile.supports_system_messages = true;
        profile.supports_streaming = true;

        let model_lower = model_name.to_lowercase();
        if model_lower.contains("gpt-4") || model_lower.contains("openai") {
            profile.context_window = Some(128000);
        } else if model_lower.contains("claude") || model_lower.contains("anthropic") {
            profile.context_window = Some(200000);
        } else if model_lower.contains("gemini") || model_lower.contains("google") {
            profile.context_window = Some(1000000);
        } else if model_lower.contains("llama") || model_lower.contains("meta") {
            profile.context_window = Some(131072);
        } else {
            profile.context_window = Some(8192);
        }

        Some(profile)
    }
}

// ============================================================================
// Cohere
// ============================================================================

/// Cohere provider.
#[derive(Debug)]
pub struct CohereProvider {
    config: ProviderConfig,
    client: Client,
}

impl CohereProvider {
    /// Create a new Cohere provider.
    pub fn new(api_key: impl Into<String>) -> Self {
        let config = ProviderConfig::new().with_api_key(api_key);
        Self {
            client: config.build_client(),
            config,
        }
    }

    /// Create from environment variables.
    pub fn from_env() -> Result<Self, ProviderError> {
        let config = ProviderConfig::from_env("COHERE");
        let api_key = config.api_key
            .ok_or(ProviderError::MissingApiKey("COHERE_API_KEY"))?;
        Ok(Self::new(api_key))
    }
}

impl Provider for CohereProvider {
    fn name(&self) -> &str { "cohere" }

    fn base_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or("https://api.cohere.ai/v2")
    }

    fn client(&self) -> &Client { &self.client }

    fn default_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();

        if let Some(key) = &self.config.api_key {
            let auth_value = format!("Bearer {}", key);
            if let Ok(value) = HeaderValue::from_str(&auth_value) {
                headers.insert(AUTHORIZATION, value);
            }
        }

        headers.insert(
            "content-type",
            HeaderValue::from_static("application/json"),
        );

        headers
    }

    fn model_profile(&self, model_name: &str) -> Option<ModelProfile> {
        let mut profile = ModelProfile::default();
        profile.supports_tools = true;
        profile.supports_system_messages = true;
        profile.supports_streaming = true;

        match model_name {
            "command-r-plus" | "command-r-plus-08-2024" => {
                profile.context_window = Some(128000);
                profile.max_tokens = Some(4096);
            }
            "command-r" | "command-r-08-2024" => {
                profile.context_window = Some(128000);
                profile.max_tokens = Some(4096);
            }
            "command" | "command-light" => {
                profile.context_window = Some(4096);
                profile.max_tokens = Some(4096);
            }
            _ if model_name.starts_with("command") => {
                profile.context_window = Some(4096);
            }
            _ => return None,
        }

        Some(profile)
    }

    fn is_configured(&self) -> bool { self.config.api_key.is_some() }
    fn aliases(&self) -> &[&str] { &[] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_together_provider() {
        let provider = TogetherProvider::new("key");
        assert_eq!(provider.name(), "together");
        assert_eq!(provider.base_url(), "https://api.together.xyz/v1");
    }

    #[test]
    fn test_fireworks_provider() {
        let provider = FireworksProvider::new("key");
        assert_eq!(provider.name(), "fireworks");
        assert_eq!(provider.base_url(), "https://api.fireworks.ai/inference/v1");
    }

    #[test]
    fn test_deepseek_provider() {
        let provider = DeepSeekProvider::new("key");
        assert_eq!(provider.name(), "deepseek");
        assert_eq!(provider.base_url(), "https://api.deepseek.com/v1");

        let profile = provider.model_profile("deepseek-reasoner").unwrap();
        assert!(profile.supports_reasoning);
    }

    #[test]
    fn test_openrouter_provider() {
        let provider = OpenRouterProvider::new("key");
        assert_eq!(provider.name(), "openrouter");
        assert_eq!(provider.base_url(), "https://openrouter.ai/api/v1");

        let headers = provider.default_headers();
        assert!(headers.contains_key("HTTP-Referer"));
    }

    #[test]
    fn test_cohere_provider() {
        let provider = CohereProvider::new("key");
        assert_eq!(provider.name(), "cohere");
        assert_eq!(provider.base_url(), "https://api.cohere.ai/v2");

        assert!(provider.model_profile("command-r-plus").is_some());
    }
}

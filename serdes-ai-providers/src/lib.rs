//! Provider abstractions for serdes-ai.
//!
//! This crate provides a unified interface for different AI providers:
//!
//! - **OpenAI** - GPT-4o, GPT-4 Turbo, o1, o3-mini
//! - **Anthropic** - Claude 3.5 Sonnet, Claude 3 Opus
//! - **Google** - Gemini 2.0 Flash, Gemini Pro
//! - **Azure** - Azure OpenAI Service
//! - **Groq** - Ultra-fast Llama, Mixtral
//! - **Mistral** - Mistral Large, Codestral
//! - **Ollama** - Local models
//! - **Together AI** - Open models
//! - **Fireworks** - Fast inference
//! - **DeepSeek** - DeepSeek Chat, DeepSeek R1
//! - **OpenRouter** - Multi-provider routing
//! - **Cohere** - Command R+
//! - **Gateway** - AI gateways (Portkey, LiteLLM, Helicone, Cloudflare)
//!
//! ## Example
//!
//! ```rust,ignore
//! use serdes_ai_providers::{ProviderRegistry, OpenAIProvider};
//! use std::sync::Arc;
//!
//! // Create a registry
//! let registry = ProviderRegistry::new();
//!
//! // Register providers
//! registry.register(Arc::new(OpenAIProvider::new("sk-...")));
//!
//! // Infer provider from model string
//! let (provider, model) = registry.infer_provider("openai:gpt-4o")?;
//! ```
//!
//! ## Model Strings
//!
//! Models can be specified with provider prefixes:
//!
//! - `openai:gpt-4o` - OpenAI GPT-4o
//! - `anthropic:claude-3-5-sonnet-20241022` - Anthropic Claude
//! - `google:gemini-2.0-flash` - Google Gemini
//! - `groq:llama-3.3-70b-versatile` - Groq Llama
//! - `ollama:llama3.2` - Local Ollama model
//!
//! Or models can be inferred from their names:
//!
//! - `gpt-4o` → OpenAI
//! - `claude-3-5-sonnet-20241022` → Anthropic
//! - `gemini-2.0-flash` → Google

mod provider;
mod registry;

// Provider implementations
#[cfg(feature = "anthropic")]
mod anthropic;
#[cfg(feature = "azure")]
mod azure;
#[cfg(feature = "google")]
mod google;
#[cfg(feature = "groq")]
mod groq;
#[cfg(feature = "mistral")]
mod mistral;
#[cfg(feature = "ollama")]
mod ollama;
#[cfg(feature = "openai")]
mod openai;

// OpenAI-compatible providers
mod compatible;

// Gateway providers
mod gateway;

// OAuth utilities
pub mod oauth;
pub use oauth::config::{chatgpt_oauth_config, claude_code_oauth_config};
pub use oauth::{
    refresh_token, run_pkce_flow, OAuthConfig, OAuthContext, OAuthError, TokenResponse,
};

// Re-exports
pub use provider::*;
pub use registry::*;

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicProvider;
#[cfg(feature = "azure")]
pub use azure::AzureProvider;
#[cfg(feature = "google")]
pub use google::{GoogleProvider, VertexAIProvider};
#[cfg(feature = "groq")]
pub use groq::GroqProvider;
#[cfg(feature = "mistral")]
pub use mistral::MistralProvider;
#[cfg(feature = "ollama")]
pub use ollama::OllamaProvider;
#[cfg(feature = "openai")]
pub use openai::OpenAIProvider;

// Compatible providers
pub use compatible::{
    CohereProvider, DeepSeekProvider, FireworksProvider, OpenRouterProvider, TogetherProvider,
};

// Gateway providers
pub use gateway::{GatewayConfig, GatewayProvider};

use std::sync::Arc;

/// Create a provider registry configured from environment variables.
///
/// This will check for API keys and create providers for each configured service.
pub fn from_env() -> ProviderRegistry {
    let registry = ProviderRegistry::new();

    #[cfg(feature = "openai")]
    if let Ok(provider) = OpenAIProvider::from_env() {
        registry.register(Arc::new(provider));
    }

    #[cfg(feature = "anthropic")]
    if let Ok(provider) = AnthropicProvider::from_env() {
        registry.register(Arc::new(provider));
    }

    #[cfg(feature = "google")]
    if let Ok(provider) = GoogleProvider::from_env() {
        registry.register(Arc::new(provider));
    }

    #[cfg(feature = "azure")]
    if let Ok(provider) = AzureProvider::from_env() {
        registry.register(Arc::new(provider));
    }

    #[cfg(feature = "groq")]
    if let Ok(provider) = GroqProvider::from_env() {
        registry.register(Arc::new(provider));
    }

    #[cfg(feature = "mistral")]
    if let Ok(provider) = MistralProvider::from_env() {
        registry.register(Arc::new(provider));
    }

    #[cfg(feature = "ollama")]
    if let Ok(provider) = OllamaProvider::from_env() {
        registry.register(Arc::new(provider));
    }

    // Compatible providers
    if let Ok(provider) = TogetherProvider::from_env() {
        registry.register(Arc::new(provider));
    }

    if let Ok(provider) = FireworksProvider::from_env() {
        registry.register(Arc::new(provider));
    }

    if let Ok(provider) = DeepSeekProvider::from_env() {
        registry.register(Arc::new(provider));
    }

    if let Ok(provider) = OpenRouterProvider::from_env() {
        registry.register(Arc::new(provider));
    }

    if let Ok(provider) = CohereProvider::from_env() {
        registry.register(Arc::new(provider));
    }

    // Gateway providers
    if let Ok(provider) = GatewayProvider::portkey_from_env() {
        registry.register(Arc::new(provider));
    }

    if let Ok(provider) = GatewayProvider::litellm_from_env() {
        registry.register(Arc::new(provider));
    }

    if let Ok(provider) = GatewayProvider::helicone_from_env() {
        registry.register(Arc::new(provider));
    }

    if let Ok(provider) = GatewayProvider::from_env() {
        registry.register(Arc::new(provider));
    }

    registry
}

/// Infer provider and model from a model string.
///
/// Supports formats like:
/// - `openai:gpt-4o` (explicit provider)
/// - `gpt-4o` (inferred from model name)
///
/// Returns a tuple of (provider, model_name).
pub fn infer(model_string: &str) -> Result<(BoxedProvider, String), ProviderError> {
    let registry = from_env();
    registry.infer_provider(model_string)
}

/// Prelude for common imports.
pub mod prelude {
    pub use crate::{
        from_env, global_registry, infer, BoxedProvider, GatewayProvider, Provider, ProviderConfig,
        ProviderError, ProviderRegistry,
    };

    #[cfg(feature = "anthropic")]
    pub use crate::AnthropicProvider;
    #[cfg(feature = "groq")]
    pub use crate::GroqProvider;
    #[cfg(feature = "ollama")]
    pub use crate::OllamaProvider;
    #[cfg(feature = "openai")]
    pub use crate::OpenAIProvider;
    #[cfg(feature = "google")]
    pub use crate::{GoogleProvider, VertexAIProvider};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_env_creates_registry() {
        let registry = from_env();
        // Should have at least the ollama provider (no API key needed)
        let providers = registry.list();
        // The exact providers depend on env vars, but registry should exist
        // Just verify we can list providers (count depends on env)
        let _ = providers;
    }

    #[test]
    fn test_global_registry() {
        let registry = global_registry();
        // Just verify global registry is accessible
        let _ = registry.list();
    }
}

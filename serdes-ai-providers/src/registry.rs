//! Provider registry for lookup and inference.
//!
//! The registry maintains a collection of configured providers and supports:
//! - Lookup by name
//! - Model string inference (e.g., "openai:gpt-4o")
//! - Auto-configuration from environment variables

use crate::provider::{BoxedProvider, ProviderError};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Registry for looking up providers by name.
#[derive(Debug, Default)]
pub struct ProviderRegistry {
    providers: RwLock<HashMap<String, BoxedProvider>>,
}

impl ProviderRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
        }
    }

    /// Register a provider.
    pub fn register(&self, provider: BoxedProvider) {
        let mut providers = self.providers.write();
        let name = provider.name().to_string();

        // Register main name
        providers.insert(name, Arc::clone(&provider));

        // Register aliases
        for alias in provider.aliases() {
            providers.insert((*alias).to_string(), Arc::clone(&provider));
        }
    }

    /// Get a provider by name.
    pub fn get(&self, name: &str) -> Option<BoxedProvider> {
        let providers = self.providers.read();
        providers.get(name).cloned()
    }

    /// Check if a provider exists.
    pub fn contains(&self, name: &str) -> bool {
        let providers = self.providers.read();
        providers.contains_key(name)
    }

    /// List all registered provider names.
    pub fn list(&self) -> Vec<String> {
        let providers = self.providers.read();
        providers.keys().cloned().collect()
    }

    /// Infer provider from model string (e.g., "openai:gpt-4o").
    ///
    /// Returns the provider and the model name without the prefix.
    pub fn infer_provider(&self, model: &str) -> Result<(BoxedProvider, String), ProviderError> {
        // Check for explicit prefix
        if let Some((provider_name, model_name)) = model.split_once(':') {
            if let Some(provider) = self.get(provider_name) {
                return Ok((provider, model_name.to_string()));
            } else {
                return Err(ProviderError::UnknownProvider(provider_name.to_string()));
            }
        }

        // Try to infer from model name
        let inferred = infer_provider_from_model_name(model);
        if let Some(provider_name) = inferred {
            if let Some(provider) = self.get(provider_name) {
                return Ok((provider, model.to_string()));
            }
        }

        Err(ProviderError::InvalidModelString(model.to_string()))
    }

    /// Remove a provider.
    pub fn remove(&self, name: &str) -> Option<BoxedProvider> {
        let mut providers = self.providers.write();
        providers.remove(name)
    }

    /// Clear all providers.
    pub fn clear(&self) {
        let mut providers = self.providers.write();
        providers.clear();
    }
}

/// Infer provider from model name.
///
/// Returns the provider name if it can be inferred.
pub fn infer_provider_from_model_name(model: &str) -> Option<&'static str> {
    let model_lower = model.to_lowercase();

    // OpenAI models
    if model_lower.starts_with("gpt-")
        || model_lower.starts_with("o1")
        || model_lower.starts_with("o3")
    {
        return Some("openai");
    }

    // Anthropic models
    if model_lower.starts_with("claude") {
        return Some("anthropic");
    }

    // Google models
    if model_lower.starts_with("gemini") {
        return Some("google");
    }

    // Mistral models
    if model_lower.starts_with("mistral")
        || model_lower.starts_with("codestral")
        || model_lower.starts_with("pixtral")
    {
        return Some("mistral");
    }

    // Cohere models
    if model_lower.starts_with("command") {
        return Some("cohere");
    }

    // Llama (typically via various providers)
    if model_lower.starts_with("llama") || model_lower.starts_with("meta-llama") {
        // Could be Groq, Together, etc. - prefer Together
        return Some("together");
    }

    // DeepSeek
    if model_lower.starts_with("deepseek") {
        return Some("deepseek");
    }

    None
}

/// Global default registry.
static GLOBAL_REGISTRY: std::sync::OnceLock<ProviderRegistry> = std::sync::OnceLock::new();

/// Get the global provider registry.
pub fn global_registry() -> &'static ProviderRegistry {
    GLOBAL_REGISTRY.get_or_init(ProviderRegistry::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Provider;
    use reqwest::header::HeaderMap;
    use reqwest::Client;
    use serdes_ai_models::ModelProfile;

    #[derive(Debug)]
    struct MockProvider {
        name: &'static str,
        client: Client,
    }

    impl MockProvider {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                client: Client::new(),
            }
        }
    }

    impl Provider for MockProvider {
        fn name(&self) -> &str {
            self.name
        }

        fn base_url(&self) -> &str {
            "https://api.example.com"
        }

        fn client(&self) -> &Client {
            &self.client
        }

        fn default_headers(&self) -> HeaderMap {
            HeaderMap::new()
        }

        fn model_profile(&self, _model_name: &str) -> Option<ModelProfile> {
            None
        }

        fn aliases(&self) -> &[&str] {
            if self.name == "openai" {
                &["openai-chat"]
            } else {
                &[]
            }
        }
    }

    #[test]
    fn test_registry_register_and_get() {
        let registry = ProviderRegistry::new();
        let provider: BoxedProvider = Arc::new(MockProvider::new("test"));

        registry.register(provider);

        assert!(registry.contains("test"));
        assert!(registry.get("test").is_some());
    }

    #[test]
    fn test_registry_aliases() {
        let registry = ProviderRegistry::new();
        let provider: BoxedProvider = Arc::new(MockProvider::new("openai"));

        registry.register(provider);

        assert!(registry.contains("openai"));
        assert!(registry.contains("openai-chat"));
    }

    #[test]
    fn test_registry_list() {
        let registry = ProviderRegistry::new();
        registry.register(Arc::new(MockProvider::new("openai")));
        registry.register(Arc::new(MockProvider::new("anthropic")));

        let list = registry.list();
        assert!(list.contains(&"openai".to_string()));
        assert!(list.contains(&"anthropic".to_string()));
    }

    #[test]
    fn test_infer_provider_explicit() {
        let registry = ProviderRegistry::new();
        registry.register(Arc::new(MockProvider::new("openai")));

        let result = registry.infer_provider("openai:gpt-4o");
        assert!(result.is_ok());

        let (provider, model) = result.unwrap();
        assert_eq!(provider.name(), "openai");
        assert_eq!(model, "gpt-4o");
    }

    #[test]
    fn test_infer_provider_from_model() {
        let registry = ProviderRegistry::new();
        registry.register(Arc::new(MockProvider::new("openai")));

        let result = registry.infer_provider("gpt-4o");
        assert!(result.is_ok());

        let (provider, model) = result.unwrap();
        assert_eq!(provider.name(), "openai");
        assert_eq!(model, "gpt-4o");
    }

    #[test]
    fn test_infer_provider_unknown() {
        let registry = ProviderRegistry::new();

        let result = registry.infer_provider("unknown:model");
        assert!(matches!(result, Err(ProviderError::UnknownProvider(_))));
    }

    #[test]
    fn test_infer_provider_from_model_name() {
        assert_eq!(infer_provider_from_model_name("gpt-4o"), Some("openai"));
        assert_eq!(
            infer_provider_from_model_name("gpt-4-turbo"),
            Some("openai")
        );
        assert_eq!(infer_provider_from_model_name("o1-preview"), Some("openai"));
        assert_eq!(
            infer_provider_from_model_name("claude-3-5-sonnet-20241022"),
            Some("anthropic")
        );
        assert_eq!(
            infer_provider_from_model_name("gemini-2.0-flash"),
            Some("google")
        );
        assert_eq!(
            infer_provider_from_model_name("mistral-large"),
            Some("mistral")
        );
        assert_eq!(
            infer_provider_from_model_name("deepseek-chat"),
            Some("deepseek")
        );
        assert_eq!(infer_provider_from_model_name("unknown-model"), None);
    }
}

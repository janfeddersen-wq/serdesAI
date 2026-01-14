//! # serdes-ai-models
//!
//! Model trait and provider implementations for serdes-ai.
//!
//! This crate provides the core `Model` trait and implementations for
//! various LLM providers:
//!
//! - **OpenAI**: GPT-4o, GPT-4, GPT-3.5, o1, o3 (feature: `openai`)
//! - **Anthropic**: Claude 3.5, Claude 3 (feature: `anthropic`)
//! - **Google**: Gemini Pro, Gemini Flash (feature: `gemini`)
//! - **Groq**: Llama, Mixtral, Gemma (feature: `groq`)
//! - **Mistral**: Mistral Large, Small, Codestral (feature: `mistral`)
//! - **Ollama**: Local models (feature: `ollama`)
//! - **Bedrock**: AWS-hosted models (feature: `bedrock`)
//!
//! ## Feature Flags
//!
//! Each provider is behind a feature flag:
//!
//! ```toml
//! [dependencies]
//! serdes-ai-models = { version = "0.1", features = ["openai", "anthropic"] }
//! ```
//!
//! Available features:
//! - `openai` (default): OpenAI API support
//! - `anthropic`: Anthropic Claude support
//! - `gemini`: Google Gemini support
//! - `groq`: Groq fast inference
//! - `mistral`: Mistral AI support
//! - `ollama`: Ollama local models
//! - `bedrock`: AWS Bedrock support
//! - `azure`: Azure OpenAI support
//! - `full`: Enable all providers
//!
//! ## Example
//!
//! ```rust,ignore
//! use serdes_ai_models::{Model, ModelRequestParameters};
//! use serdes_ai_models::openai::OpenAIChatModel;
//! use serdes_ai_core::{ModelRequest, ModelSettings};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let model = OpenAIChatModel::new("gpt-4o", std::env::var("OPENAI_API_KEY")?);
//!
//!     let mut request = ModelRequest::new();
//!     request.add_user_prompt("Hello!");
//!     let settings = ModelSettings::new().temperature(0.7);
//!
//!     let response = model.request(request, Some(settings)).await?;
//!     println!("Response: {:?}", response);
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod error;
pub mod fallback;
pub mod model;
pub mod profile;
pub mod schema_transformer;

// Provider modules (feature-gated)

/// OpenAI models (GPT-4, GPT-4o, o1, etc.).
#[cfg(feature = "openai")]
#[cfg_attr(docsrs, doc(cfg(feature = "openai")))]
pub mod openai;

/// Anthropic Claude models.
#[cfg(feature = "anthropic")]
#[cfg_attr(docsrs, doc(cfg(feature = "anthropic")))]
pub mod anthropic;

/// Google Gemini models.
#[cfg(any(feature = "gemini", feature = "google"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "gemini", feature = "google"))))]
pub mod google;

/// Groq fast inference (Llama, Mixtral, Gemma).
#[cfg(feature = "groq")]
#[cfg_attr(docsrs, doc(cfg(feature = "groq")))]
pub mod groq;

/// Mistral AI models.
#[cfg(feature = "mistral")]
#[cfg_attr(docsrs, doc(cfg(feature = "mistral")))]
pub mod mistral;

/// Ollama local models.
#[cfg(feature = "ollama")]
#[cfg_attr(docsrs, doc(cfg(feature = "ollama")))]
pub mod ollama;

/// AWS Bedrock models.
#[cfg(feature = "bedrock")]
#[cfg_attr(docsrs, doc(cfg(feature = "bedrock")))]
pub mod bedrock;

/// Azure OpenAI models.
#[cfg(feature = "azure")]
#[cfg_attr(docsrs, doc(cfg(feature = "azure")))]
pub mod azure;

/// OpenRouter models (multi-provider routing).
#[cfg(feature = "openrouter")]
#[cfg_attr(docsrs, doc(cfg(feature = "openrouter")))]
pub mod openrouter;

/// HuggingFace Inference API models.
#[cfg(feature = "huggingface")]
#[cfg_attr(docsrs, doc(cfg(feature = "huggingface")))]
pub mod huggingface;

#[cfg(feature = "azure")]
pub use azure::AzureOpenAIModel;

#[cfg(feature = "openrouter")]
pub use openrouter::OpenRouterModel;

#[cfg(feature = "huggingface")]
pub use huggingface::HuggingFaceModel;

/// Cohere models (Command-R, Command-R Plus).
#[cfg(feature = "cohere")]
#[cfg_attr(docsrs, doc(cfg(feature = "cohere")))]
pub mod cohere;

#[cfg(feature = "cohere")]
pub use cohere::CohereModel;

/// ChatGPT OAuth models (Codex API with OAuth tokens).
#[cfg(feature = "chatgpt-oauth")]
#[cfg_attr(docsrs, doc(cfg(feature = "chatgpt-oauth")))]
pub mod chatgpt_oauth;

#[cfg(feature = "chatgpt-oauth")]
pub use chatgpt_oauth::ChatGptOAuthModel;

/// Claude Code OAuth models (Anthropic API with OAuth tokens).
#[cfg(feature = "claude-code-oauth")]
#[cfg_attr(docsrs, doc(cfg(feature = "claude-code-oauth")))]
pub mod claude_code_oauth;

#[cfg(feature = "claude-code-oauth")]
pub use claude_code_oauth::ClaudeCodeOAuthModel;

// Mock for testing
pub mod mock;

// Re-exports
pub use error::{ModelError, ModelResult};
pub use fallback::{FallbackModel, RetryOn};
pub use mock::{FunctionModel, MockModel, TestModel};
pub use model::{
    BoxedModel, Model, ModelCapability, ModelRequestParameters, ModelWithMetadata,
    StreamedResponse, ToolChoice,
};
pub use profile::{
    anthropic_claude_profile, deepseek_profile, google_gemini_profile, mistral_profile,
    openai_gpt4o_profile, openai_o1_profile, qwen_profile, ModelProfile, OutputMode,
    DEFAULT_PROFILE, DEFAULT_PROMPTED_OUTPUT_TEMPLATE,
};
pub use schema_transformer::JsonSchemaTransformer;

// Re-export provider types for convenience
#[cfg(feature = "openai")]
pub use openai::{OpenAIChatModel, OpenAIResponsesModel, ReasoningEffort, ReasoningSummary};

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicModel;

#[cfg(feature = "gemini")]
pub use google::GoogleModel as GeminiModel;

#[cfg(feature = "google")]
pub use google::GoogleModel;

#[cfg(feature = "groq")]
pub use groq::GroqModel;

#[cfg(feature = "mistral")]
pub use mistral::MistralModel;

#[cfg(feature = "ollama")]
pub use ollama::OllamaModel;

#[cfg(feature = "bedrock")]
pub use bedrock::BedrockModel;

/// Prelude for common imports.
pub mod prelude {
    pub use crate::{
        BoxedModel, FallbackModel, FunctionModel, MockModel, Model, ModelCapability, ModelError,
        ModelProfile, ModelRequestParameters, ModelResult, RetryOn, StreamedResponse, TestModel,
        ToolChoice,
    };

    #[cfg(feature = "openai")]
    pub use crate::openai::{
        OpenAIChatModel, OpenAIResponsesModel, ReasoningEffort, ReasoningSummary,
    };

    #[cfg(feature = "anthropic")]
    pub use crate::anthropic::AnthropicModel;

    #[cfg(feature = "gemini")]
    pub use crate::google::GoogleModel as GeminiModel;

    #[cfg(feature = "google")]
    pub use crate::google::GoogleModel;

    #[cfg(feature = "groq")]
    pub use crate::groq::GroqModel;

    #[cfg(feature = "mistral")]
    pub use crate::mistral::MistralModel;

    #[cfg(feature = "ollama")]
    pub use crate::ollama::OllamaModel;

    #[cfg(feature = "bedrock")]
    pub use crate::bedrock::BedrockModel;

    #[cfg(feature = "openrouter")]
    pub use crate::openrouter::OpenRouterModel;

    #[cfg(feature = "huggingface")]
    pub use crate::huggingface::HuggingFaceModel;

    #[cfg(feature = "cohere")]
    pub use crate::cohere::CohereModel;
}

/// Infer a model from a string identifier.
///
/// Format: `provider:model_name` or just `model_name` (defaults to OpenAI).
///
/// # Examples
///
/// ```ignore
/// let model = infer_model("openai:gpt-4o")?;
/// let model = infer_model("anthropic:claude-3-opus")?;
/// let model = infer_model("groq:llama-3.1-70b-versatile")?;
/// let model = infer_model("ollama:llama3.1")?;
/// ```
/// Infer a model from a string identifier.
///
/// Format: `provider:model_name` or just `model_name` (defaults to OpenAI).
///
/// # Examples
///
/// ```ignore
/// let model = infer_model("openai:gpt-4o")?;
/// let model = infer_model("anthropic:claude-3-opus")?;
/// let model = infer_model("groq:llama-3.1-70b-versatile")?;
/// let model = infer_model("ollama:llama3.1")?;
/// ```
#[cfg(feature = "openai")]
pub fn infer_model(identifier: &str) -> ModelResult<std::sync::Arc<dyn Model>> {
    use std::sync::Arc;

    let (provider, model_name) = if identifier.contains(':') {
        let parts: Vec<&str> = identifier.splitn(2, ':').collect();
        (parts[0], parts[1])
    } else {
        ("openai", identifier)
    };

    match provider {
        "openai" | "gpt" => {
            #[cfg(feature = "openai")]
            {
                let model = OpenAIChatModel::from_env(model_name)?;
                Ok(Arc::new(model))
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(ModelError::Configuration(
                    "OpenAI support not enabled. Enable 'openai' feature.".to_string(),
                ))
            }
        }
        "anthropic" | "claude" => {
            #[cfg(feature = "anthropic")]
            {
                let model = AnthropicModel::from_env(model_name)?;
                Ok(Arc::new(model))
            }
            #[cfg(not(feature = "anthropic"))]
            {
                Err(ModelError::Configuration(
                    "Anthropic support not enabled. Enable 'anthropic' feature.".to_string(),
                ))
            }
        }
        #[cfg(feature = "groq")]
        "groq" => {
            let model = GroqModel::from_env(model_name)?;
            Ok(Arc::new(model))
        }
        #[cfg(feature = "mistral")]
        "mistral" => {
            let model = MistralModel::from_env(model_name)?;
            Ok(Arc::new(model))
        }
        #[cfg(feature = "ollama")]
        "ollama" => {
            let model = OllamaModel::from_env(model_name)?;
            Ok(Arc::new(model))
        }
        #[cfg(feature = "bedrock")]
        "bedrock" | "aws" => {
            let model = BedrockModel::new(model_name)?;
            Ok(Arc::new(model))
        }
        #[cfg(feature = "openrouter")]
        "openrouter" | "or" => {
            let model = OpenRouterModel::from_env(model_name)?;
            Ok(Arc::new(model))
        }
        #[cfg(feature = "huggingface")]
        "huggingface" | "hf" => {
            let model = HuggingFaceModel::from_env(model_name)?;
            Ok(Arc::new(model))
        }
        #[cfg(feature = "cohere")]
        "cohere" | "co" => {
            let model = CohereModel::from_env(model_name)?;
            Ok(Arc::new(model))
        }
        _ => Err(ModelError::Configuration(format!(
            "Unknown provider: {}. Supported: openai, anthropic, groq, mistral, ollama, bedrock, openrouter, huggingface, cohere",
            provider
        ))),
    }
}

/// Build a model with custom configuration.
///
/// This is a helper function for `serdes_ai_agent::ModelConfig` that creates
/// models with custom API keys, base URLs, and timeouts.
///
/// # Arguments
///
/// * `provider` - The provider name (e.g., "openai", "anthropic")
/// * `model_name` - The model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
/// * `api_key` - Optional API key (overrides environment variable)
/// * `base_url` - Optional base URL (for custom endpoints)
/// * `timeout` - Optional request timeout
pub fn build_model_with_config(
    provider: &str,
    model_name: &str,
    api_key: Option<&str>,
    base_url: Option<&str>,
    timeout: Option<std::time::Duration>,
) -> ModelResult<std::sync::Arc<dyn Model>> {
    use std::sync::Arc;

    match provider {
        "openai" | "gpt" => {
            #[cfg(feature = "openai")]
            {
                let model = if let Some(key) = api_key {
                    OpenAIChatModel::new(model_name, key)
                } else {
                    OpenAIChatModel::from_env(model_name)?
                };

                let model = if let Some(url) = base_url {
                    model.with_base_url(url)
                } else {
                    model
                };

                let model = if let Some(t) = timeout {
                    model.with_timeout(t)
                } else {
                    model
                };

                Ok(Arc::new(model))
            }
            #[cfg(not(feature = "openai"))]
            {
                let _ = (model_name, api_key, base_url, timeout);
                Err(ModelError::Configuration(
                    "OpenAI support not enabled. Enable 'openai' feature.".to_string(),
                ))
            }
        }
        "anthropic" | "claude" => {
            #[cfg(feature = "anthropic")]
            {
                let model = if let Some(key) = api_key {
                    AnthropicModel::new(model_name, key)
                } else {
                    AnthropicModel::from_env(model_name)?
                };

                let model = if let Some(url) = base_url {
                    model.with_base_url(url)
                } else {
                    model
                };

                let model = if let Some(t) = timeout {
                    model.with_timeout(t)
                } else {
                    model
                };

                Ok(Arc::new(model))
            }
            #[cfg(not(feature = "anthropic"))]
            {
                let _ = (model_name, api_key, base_url, timeout);
                Err(ModelError::Configuration(
                    "Anthropic support not enabled. Enable 'anthropic' feature.".to_string(),
                ))
            }
        }
        #[cfg(feature = "groq")]
        "groq" => {
            let model = if let Some(key) = api_key {
                GroqModel::new(model_name, key)
            } else {
                GroqModel::from_env(model_name)?
            };

            // GroqModel doesn't have with_base_url/with_timeout
            let _ = (base_url, timeout);

            Ok(Arc::new(model))
        }
        #[cfg(feature = "mistral")]
        "mistral" => {
            let model = if let Some(key) = api_key {
                MistralModel::new(model_name, key)
            } else {
                MistralModel::from_env(model_name)?
            };

            let model = if let Some(url) = base_url {
                model.with_base_url(url)
            } else {
                model
            };

            let model = if let Some(t) = timeout {
                model.with_timeout(t)
            } else {
                model
            };

            Ok(Arc::new(model))
        }
        #[cfg(feature = "ollama")]
        "ollama" => {
            let model = OllamaModel::from_env(model_name)?;

            let model = if let Some(url) = base_url {
                model.with_base_url(url)
            } else {
                model
            };

            // Ollama doesn't use API keys or timeouts in the same way
            let _ = (api_key, timeout);

            Ok(Arc::new(model))
        }
        #[cfg(feature = "google")]
        "google" | "gemini" => {
            // Google model requires an API key - no from_env helper
            let key: String = if let Some(k) = api_key {
                k.to_string()
            } else {
                std::env::var("GOOGLE_API_KEY").map_err(|_| {
                    ModelError::Configuration(
                        "Google/Gemini models require an API key. Use ModelConfig::with_api_key() \
                         or set GOOGLE_API_KEY environment variable.".to_string()
                    )
                })?
            };

            let model = GoogleModel::new(model_name, key);

            let model = if let Some(url) = base_url {
                model.with_base_url(url)
            } else {
                model
            };

            let _ = timeout; // Google model doesn't have with_timeout

            Ok(Arc::new(model))
        }
        _ => Err(ModelError::Configuration(format!(
            "Unknown or unsupported provider: '{}'. Supported providers depend on enabled features: \
             openai, anthropic, groq, mistral, ollama, google",
            provider
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_model() {
        let model = MockModel::new("test-model");
        assert_eq!(model.name(), "test-model");
        assert_eq!(model.system(), "mock");
    }

    #[test]
    fn test_build_model_unknown_provider() {
        let result = build_model_with_config("unknown", "model", None, None, None);
        assert!(result.is_err());
    }
}

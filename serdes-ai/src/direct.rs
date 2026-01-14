//! Direct model request functions.
//!
//! These functions allow making imperative requests to models with minimal abstraction.
//! The only abstraction is input/output schema translation for unified API access.
//!
//! Use these when you want simple, direct access to models without the full agent
//! infrastructure. Great for one-off queries, scripts, and simple integrations.
//!
//! # Examples
//!
//! ## Non-streaming request
//!
//! ```rust,ignore
//! use serdes_ai::direct::model_request;
//! use serdes_ai_core::ModelRequest;
//!
//! let response = model_request(
//!     "openai:gpt-4o",
//!     &[ModelRequest::user("What is the capital of France?")],
//!     None,
//!     None,
//! ).await?;
//!
//! println!("{}", response.text());
//! ```
//!
//! ## Streaming request
//!
//! ```rust,ignore
//! use serdes_ai::direct::model_request_stream;
//! use futures::StreamExt;
//!
//! let mut stream = model_request_stream(
//!     "anthropic:claude-3-5-sonnet",
//!     &[ModelRequest::user("Write a poem")],
//!     None,
//!     None,
//! ).await?;
//!
//! while let Some(event) = stream.next().await {
//!     // Handle streaming events
//! }
//! ```
//!
//! ## Using a pre-built model instance
//!
//! ```rust,ignore
//! use serdes_ai::direct::model_request;
//! use serdes_ai_models::openai::OpenAIChatModel;
//!
//! let model = OpenAIChatModel::from_env("gpt-4o")?;
//! let response = model_request(
//!     model,
//!     &[ModelRequest::user("Hello!")],
//!     None,
//!     None,
//! ).await?;
//! ```

use std::sync::Arc;

use futures::StreamExt;
use serdes_ai_core::{
    messages::ModelResponseStreamEvent, ModelRequest, ModelResponse, ModelSettings,
};
use serdes_ai_models::{BoxedModel, Model, ModelError, ModelRequestParameters, StreamedResponse};
use thiserror::Error;

// ============================================================================
// Error Type
// ============================================================================

/// Error type for direct requests.
#[derive(Debug, Error)]
pub enum DirectError {
    /// Invalid model name format.
    #[error("Invalid model name: {0}")]
    InvalidModelName(String),

    /// Model-level error (API, network, etc.).
    #[error("Model error: {0}")]
    ModelError(#[from] ModelError),

    /// Runtime error (e.g., sync functions called in async context).
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Provider not available (feature not enabled).
    #[error("Provider not available: {0}. Enable the corresponding feature.")]
    ProviderNotAvailable(String),
}

// ============================================================================
// Model Specification
// ============================================================================

/// Model specification - either a string like "openai:gpt-4o" or a Model instance.
///
/// This allows flexible model specification in the direct API functions.
///
/// # Examples
///
/// ```rust,ignore
/// // From string
/// let spec: ModelSpec = "openai:gpt-4o".into();
///
/// // From model instance
/// let model = OpenAIChatModel::from_env("gpt-4o")?;
/// let spec: ModelSpec = model.into();
/// ```
#[derive(Clone)]
pub enum ModelSpec {
    /// Model specified by name (e.g., "openai:gpt-4o").
    Name(String),
    /// Pre-built model instance.
    Instance(BoxedModel),
}

impl From<&str> for ModelSpec {
    fn from(s: &str) -> Self {
        ModelSpec::Name(s.to_string())
    }
}

impl From<String> for ModelSpec {
    fn from(s: String) -> Self {
        ModelSpec::Name(s)
    }
}

impl From<BoxedModel> for ModelSpec {
    fn from(model: BoxedModel) -> Self {
        ModelSpec::Instance(model)
    }
}

impl ModelSpec {
    /// Create a ModelSpec from any concrete Model type.
    ///
    /// This is a convenience method for wrapping concrete model types.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use serdes_ai::direct::ModelSpec;
    /// use serdes_ai_models::openai::OpenAIChatModel;
    ///
    /// let model = OpenAIChatModel::from_env("gpt-4o")?;
    /// let spec = ModelSpec::from_model(model);
    /// ```
    pub fn from_model<M: Model + 'static>(model: M) -> Self {
        ModelSpec::Instance(Arc::new(model))
    }
}

impl ModelSpec {
    /// Resolve the spec into a concrete model instance.
    fn resolve(self) -> Result<BoxedModel, DirectError> {
        match self {
            ModelSpec::Name(name) => parse_model_name(&name),
            ModelSpec::Instance(model) => Ok(model),
        }
    }
}

// ============================================================================
// Non-Streaming Requests
// ============================================================================

/// Make a non-streamed request to a model.
///
/// This is the simplest way to get a response from a model. It blocks until
/// the full response is available.
///
/// # Arguments
///
/// * `model` - Model specification (string like "openai:gpt-4o" or a Model instance)
/// * `messages` - Slice of request messages
/// * `model_settings` - Optional model settings (temperature, max_tokens, etc.)
/// * `model_request_parameters` - Optional request parameters (tools, output schema, etc.)
///
/// # Example
///
/// ```rust,ignore
/// use serdes_ai::direct::model_request;
/// use serdes_ai_core::ModelRequest;
///
/// let response = model_request(
///     "openai:gpt-4o",
///     &[ModelRequest::user("What is the capital of France?")],
///     None,
///     None,
/// ).await?;
///
/// println!("{}", response.text());
/// ```
pub async fn model_request(
    model: impl Into<ModelSpec>,
    messages: &[ModelRequest],
    model_settings: Option<ModelSettings>,
    model_request_parameters: Option<ModelRequestParameters>,
) -> Result<ModelResponse, DirectError> {
    let model = model.into().resolve()?;
    let settings = model_settings.unwrap_or_default();
    let params = model_request_parameters.unwrap_or_default();

    let response = model.request(messages, &settings, &params).await?;
    Ok(response)
}

/// Make a synchronous (blocking) non-streamed request.
///
/// This wraps `model_request` with a tokio runtime. It creates a new runtime
/// for each call, so it's not the most efficient for high-throughput scenarios.
///
/// # Warning
///
/// Cannot be used inside async code (will panic if called from an async context).
/// Use `model_request` instead in async contexts.
///
/// # Example
///
/// ```rust,ignore
/// use serdes_ai::direct::model_request_sync;
/// use serdes_ai_core::ModelRequest;
///
/// fn main() {
///     let response = model_request_sync(
///         "openai:gpt-4o",
///         &[ModelRequest::user("Hello!")],
///         None,
///         None,
///     ).unwrap();
///
///     println!("{}", response.text());
/// }
/// ```
pub fn model_request_sync(
    model: impl Into<ModelSpec>,
    messages: &[ModelRequest],
    model_settings: Option<ModelSettings>,
    model_request_parameters: Option<ModelRequestParameters>,
) -> Result<ModelResponse, DirectError> {
    // Check if we're already in an async context
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(DirectError::RuntimeError(
            "model_request_sync cannot be called from async context. Use model_request instead."
                .to_string(),
        ));
    }

    // Create a new runtime for the blocking call
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| DirectError::RuntimeError(format!("Failed to create runtime: {e}")))?;

    // Clone what we need since we can't move references
    let model_spec = model.into();
    let messages_owned: Vec<ModelRequest> = messages.to_vec();
    let settings = model_settings;
    let params = model_request_parameters;

    rt.block_on(async move { model_request(model_spec, &messages_owned, settings, params).await })
}

// ============================================================================
// Streaming Requests
// ============================================================================

/// Make a streaming request to a model.
///
/// Returns a stream of response events that can be processed as they arrive.
/// This is useful for real-time output and long responses.
///
/// # Arguments
///
/// * `model` - Model specification (string like "openai:gpt-4o" or a Model instance)
/// * `messages` - Slice of request messages
/// * `model_settings` - Optional model settings (temperature, max_tokens, etc.)
/// * `model_request_parameters` - Optional request parameters (tools, output schema, etc.)
///
/// # Example
///
/// ```rust,ignore
/// use serdes_ai::direct::model_request_stream;
/// use serdes_ai_core::messages::ModelResponseStreamEvent;
/// use futures::StreamExt;
///
/// let mut stream = model_request_stream(
///     "anthropic:claude-3-5-sonnet",
///     &[ModelRequest::user("Write a poem about Rust")],
///     None,
///     None,
/// ).await?;
///
/// while let Some(event) = stream.next().await {
///     match event? {
///         ModelResponseStreamEvent::PartDelta(delta) => {
///             if let Some(text) = delta.delta.content_delta() {
///                 print!("{}", text);
///             }
///         }
///         _ => {}
///     }
/// }
/// ```
pub async fn model_request_stream(
    model: impl Into<ModelSpec>,
    messages: &[ModelRequest],
    model_settings: Option<ModelSettings>,
    model_request_parameters: Option<ModelRequestParameters>,
) -> Result<StreamedResponse, DirectError> {
    let model = model.into().resolve()?;
    let settings = model_settings.unwrap_or_default();
    let params = model_request_parameters.unwrap_or_default();

    let stream = model.request_stream(messages, &settings, &params).await?;
    Ok(stream)
}

/// Synchronous streaming request wrapper.
///
/// This struct wraps a streaming response and provides a synchronous iterator
/// interface for consuming streaming events.
///
/// # Warning
///
/// Cannot be used inside async code (will panic if called from an async context).
pub struct StreamedResponseSync {
    /// The underlying async runtime.
    runtime: tokio::runtime::Runtime,
    /// The underlying async stream.
    stream: Option<StreamedResponse>,
}

impl StreamedResponseSync {
    /// Create a new sync wrapper around an async stream.
    fn new(stream: StreamedResponse) -> Result<Self, DirectError> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| DirectError::RuntimeError(format!("Failed to create runtime: {e}")))?;

        Ok(Self {
            runtime,
            stream: Some(stream),
        })
    }
}

impl Iterator for StreamedResponseSync {
    type Item = Result<ModelResponseStreamEvent, ModelError>;

    fn next(&mut self) -> Option<Self::Item> {
        let stream = self.stream.as_mut()?;
        self.runtime.block_on(stream.next())
    }
}

/// Synchronous streaming request.
///
/// This creates a streaming request and wraps it in a synchronous iterator.
///
/// # Warning
///
/// Cannot be used inside async code (will panic if called from an async context).
/// Use `model_request_stream` instead in async contexts.
///
/// # Example
///
/// ```rust,ignore
/// use serdes_ai::direct::model_request_stream_sync;
/// use serdes_ai_core::ModelRequest;
///
/// fn main() {
///     let stream = model_request_stream_sync(
///         "openai:gpt-4o",
///         &[ModelRequest::user("Tell me a story")],
///         None,
///         None,
///     ).unwrap();
///
///     for event in stream {
///         // Handle each event
///     }
/// }
/// ```
pub fn model_request_stream_sync(
    model: impl Into<ModelSpec>,
    messages: &[ModelRequest],
    model_settings: Option<ModelSettings>,
    model_request_parameters: Option<ModelRequestParameters>,
) -> Result<StreamedResponseSync, DirectError> {
    // Check if we're already in an async context
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(DirectError::RuntimeError(
            "model_request_stream_sync cannot be called from async context. Use model_request_stream instead."
                .to_string(),
        ));
    }

    // Create a runtime to set up the stream
    let setup_rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| DirectError::RuntimeError(format!("Failed to create runtime: {e}")))?;

    let model_spec = model.into();
    let messages_owned: Vec<ModelRequest> = messages.to_vec();
    let settings = model_settings;
    let params = model_request_parameters;

    let stream = setup_rt.block_on(async move {
        model_request_stream(model_spec, &messages_owned, settings, params).await
    })?;

    // Drop the setup runtime and create the iterator with its own runtime
    drop(setup_rt);

    StreamedResponseSync::new(stream)
}

// ============================================================================
// Model Parsing
// ============================================================================

/// Parse a model name like "openai:gpt-4o" into a model instance.
///
/// Supported formats:
/// - `provider:model_name` (e.g., "openai:gpt-4o", "anthropic:claude-3-5-sonnet")
/// - `model_name` (defaults to OpenAI)
///
/// Available providers (when their features are enabled):
/// - `openai` / `gpt`: OpenAI models
/// - `anthropic` / `claude`: Anthropic Claude models
/// - `groq`: Groq fast inference
/// - `mistral`: Mistral AI models
/// - `ollama`: Local Ollama models
/// - `bedrock` / `aws`: AWS Bedrock models
/// - `openrouter` / `or`: OpenRouter multi-provider
/// - `huggingface` / `hf`: HuggingFace Inference API
/// - `cohere` / `co`: Cohere models
fn parse_model_name(name: &str) -> Result<BoxedModel, DirectError> {
    // Use the infer_model function from serdes-ai-models
    #[cfg(feature = "openai")]
    {
        serdes_ai_models::infer_model(name).map_err(DirectError::ModelError)
    }

    #[cfg(not(feature = "openai"))]
    {
        // Without openai feature, we need manual parsing
        let (provider, model_name) = if name.contains(':') {
            let parts: Vec<&str> = name.splitn(2, ':').collect();
            (parts[0], parts[1])
        } else {
            return Err(DirectError::InvalidModelName(format!(
                "Model name '{}' requires a provider prefix (e.g., 'anthropic:{}') \
                 when the 'openai' feature is not enabled.",
                name, name
            )));
        };

        match provider {
            #[cfg(feature = "anthropic")]
            "anthropic" | "claude" => {
                let model = serdes_ai_models::AnthropicModel::from_env(model_name)
                    .map_err(DirectError::ModelError)?;
                Ok(Arc::new(model))
            }
            #[cfg(feature = "groq")]
            "groq" => {
                let model = serdes_ai_models::GroqModel::from_env(model_name)
                    .map_err(DirectError::ModelError)?;
                Ok(Arc::new(model))
            }
            #[cfg(feature = "mistral")]
            "mistral" => {
                let model = serdes_ai_models::MistralModel::from_env(model_name)
                    .map_err(DirectError::ModelError)?;
                Ok(Arc::new(model))
            }
            #[cfg(feature = "ollama")]
            "ollama" => {
                let model = serdes_ai_models::OllamaModel::from_env(model_name)
                    .map_err(DirectError::ModelError)?;
                Ok(Arc::new(model))
            }
            #[cfg(feature = "bedrock")]
            "bedrock" | "aws" => {
                let model = serdes_ai_models::BedrockModel::new(model_name)
                    .map_err(DirectError::ModelError)?;
                Ok(Arc::new(model))
            }
            _ => Err(DirectError::ProviderNotAvailable(provider.to_string())),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_spec_from_str() {
        let spec: ModelSpec = "openai:gpt-4o".into();
        assert!(matches!(spec, ModelSpec::Name(ref s) if s == "openai:gpt-4o"));
    }

    #[test]
    fn test_model_spec_from_string() {
        let spec: ModelSpec = String::from("anthropic:claude-3").into();
        assert!(matches!(spec, ModelSpec::Name(ref s) if s == "anthropic:claude-3"));
    }

    #[test]
    fn test_direct_error_display() {
        let err = DirectError::InvalidModelName("bad-model".to_string());
        assert!(err.to_string().contains("bad-model"));

        let err = DirectError::ProviderNotAvailable("unknown".to_string());
        assert!(err.to_string().contains("unknown"));

        let err = DirectError::RuntimeError("something went wrong".to_string());
        assert!(err.to_string().contains("something went wrong"));
    }

    #[test]
    fn test_sync_runtime_detection() {
        // In a normal sync context, this should not error due to runtime detection
        // (but might fail due to missing API keys)
        // We're just testing the runtime detection logic here

        // Can't easily test the async context detection without actually being in one
    }
}

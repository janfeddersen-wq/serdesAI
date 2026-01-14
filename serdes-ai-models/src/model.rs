//! Core model trait and types.
//!
//! This module defines the `Model` trait which is the primary interface
//! for interacting with language models.

use async_trait::async_trait;
use futures::Stream;
use serdes_ai_core::{
    messages::ModelResponseStreamEvent, ModelRequest, ModelResponse, ModelSettings,
};
use serdes_ai_output::OutputMode;
use serdes_ai_tools::{ObjectJsonSchema, ToolDefinition};
use std::pin::Pin;
use std::sync::Arc;

use crate::error::ModelError;
use crate::profile::ModelProfile;

/// Parameters for a model request.
#[derive(Debug, Clone, Default)]
pub struct ModelRequestParameters {
    /// Tool definitions to include (wrapped in Arc to avoid cloning on every step).
    pub tools: Arc<Vec<ToolDefinition>>,
    /// Output schema for structured output.
    pub output_schema: Option<ObjectJsonSchema>,
    /// Output mode (text, native, prompted, tool).
    pub output_mode: OutputMode,
    /// Whether to allow text response.
    pub allow_text_output: bool,
    /// Tool choice strategy.
    pub tool_choice: Option<ToolChoice>,
    /// Whether to include usage in streaming responses.
    pub stream_usage: bool,
}

impl ModelRequestParameters {
    /// Create new empty parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add tool definitions.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Arc::new(tools);
        self
    }

    /// Add tool definitions from an Arc (zero-copy for cached tools).
    #[must_use]
    pub fn with_tools_arc(mut self, tools: Arc<Vec<ToolDefinition>>) -> Self {
        self.tools = tools;
        self
    }

    /// Set output schema.
    #[must_use]
    pub fn with_output_schema(mut self, schema: ObjectJsonSchema) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Set output mode.
    #[must_use]
    pub fn with_output_mode(mut self, mode: OutputMode) -> Self {
        self.output_mode = mode;
        self
    }

    /// Set allow text output.
    #[must_use]
    pub fn with_allow_text(mut self, allow: bool) -> Self {
        self.allow_text_output = allow;
        self
    }

    /// Set tool choice.
    #[must_use]
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }
}

/// Tool choice strategy.
#[derive(Debug, Clone, Default)]
pub enum ToolChoice {
    /// Model decides whether to call tools.
    #[default]
    Auto,
    /// Model must call at least one tool.
    Required,
    /// Model should not call any tools.
    None,
    /// Model must call a specific tool.
    Specific(String),
}

/// Type alias for streaming response.
pub type StreamedResponse =
    Pin<Box<dyn Stream<Item = Result<ModelResponseStreamEvent, ModelError>> + Send>>;

/// Core model trait.
///
/// This trait defines the interface for all language model implementations.
/// It supports both synchronous requests and streaming.
#[async_trait]
pub trait Model: Send + Sync {
    /// Get the model name.
    fn name(&self) -> &str;

    /// Get the model system/provider (openai, anthropic, etc).
    fn system(&self) -> &str;

    /// Get the full model identifier.
    fn identifier(&self) -> String {
        format!("{}:{}", self.system(), self.name())
    }

    /// Make a request to the model.
    async fn request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError>;

    /// Make a streaming request to the model.
    ///
    /// Returns a stream of response events. Implementations may
    /// fall back to non-streaming if streaming is not supported.
    async fn request_stream(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<StreamedResponse, ModelError>;

    /// Get the model profile (capabilities, schema transforms).
    fn profile(&self) -> &ModelProfile;

    /// Count tokens for messages (if supported).
    ///
    /// Returns an error if token counting is not supported.
    async fn count_tokens(&self, _messages: &[ModelRequest]) -> Result<u64, ModelError> {
        Err(ModelError::not_supported("Token counting"))
    }

    /// Check if the model supports a specific capability.
    fn supports(&self, capability: ModelCapability) -> bool {
        let profile = self.profile();
        match capability {
            ModelCapability::Tools => profile.supports_tools,
            ModelCapability::ParallelTools => profile.supports_parallel_tools,
            ModelCapability::NativeStructuredOutput => profile.supports_native_structured_output,
            ModelCapability::StrictTools => profile.supports_strict_tools,
            ModelCapability::SystemMessages => profile.supports_system_messages,
            ModelCapability::Images => profile.supports_images,
            ModelCapability::Audio => profile.supports_audio,
            ModelCapability::Video => profile.supports_video,
            ModelCapability::Documents => profile.supports_documents,
            ModelCapability::Caching => profile.supports_caching,
            ModelCapability::Reasoning => profile.supports_reasoning,
            ModelCapability::Streaming => profile.supports_streaming,
        }
    }
}

/// Model capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelCapability {
    /// Tool/function calling.
    Tools,
    /// Parallel tool calls.
    ParallelTools,
    /// Native structured output.
    NativeStructuredOutput,
    /// Strict mode for tools.
    StrictTools,
    /// System messages.
    SystemMessages,
    /// Image input.
    Images,
    /// Audio input.
    Audio,
    /// Video input.
    Video,
    /// Document input.
    Documents,
    /// Prompt caching.
    Caching,
    /// Reasoning/thinking.
    Reasoning,
    /// Streaming responses.
    Streaming,
}

/// Boxed model for dynamic dispatch.
pub type BoxedModel = Arc<dyn Model>;

/// A model with additional metadata.
#[derive(Clone)]
pub struct ModelWithMetadata {
    /// The underlying model.
    pub model: BoxedModel,
    /// Display name for the model.
    pub display_name: Option<String>,
    /// Description of the model.
    pub description: Option<String>,
    /// Tags for categorization.
    pub tags: Vec<String>,
}

impl ModelWithMetadata {
    /// Create a new model with metadata.
    pub fn new(model: BoxedModel) -> Self {
        Self {
            model,
            display_name: None,
            description: None,
            tags: Vec::new(),
        }
    }

    /// Set display name.
    #[must_use]
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// Set description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a tag.
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_parameters_builder() {
        let params = ModelRequestParameters::new()
            .with_output_mode(OutputMode::Tool)
            .with_allow_text(true)
            .with_tool_choice(ToolChoice::Required);

        assert_eq!(params.output_mode, OutputMode::Tool);
        assert!(params.allow_text_output);
        assert!(matches!(params.tool_choice, Some(ToolChoice::Required)));
    }

    #[test]
    fn test_tool_choice_default() {
        let choice = ToolChoice::default();
        assert!(matches!(choice, ToolChoice::Auto));
    }
}

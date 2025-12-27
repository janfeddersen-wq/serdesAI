//! Model profiles and capabilities.
//!
//! This module defines model capabilities and configuration
//! for different AI model providers.

use crate::schema_transformer::JsonSchemaTransformer;

/// Structured output mode for models.
///
/// Different models support different ways of getting structured output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OutputMode {
    /// Use tool/function calling to get structured output.
    #[default]
    Tool,
    /// Use native structured output (e.g., OpenAI's JSON mode with schema).
    Native,
    /// Include schema in prompt and ask model to output JSON.
    Prompted,
    /// Plain text output (no structured output).
    Text,
}

impl std::fmt::Display for OutputMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputMode::Tool => write!(f, "tool"),
            OutputMode::Native => write!(f, "native"),
            OutputMode::Prompted => write!(f, "prompted"),
            OutputMode::Text => write!(f, "text"),
        }
    }
}

/// Model capabilities and behavior configuration.
///
/// Each model has a profile that describes its supported features
/// and how to transform schemas for compatibility.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    /// Model supports tool/function calling.
    pub supports_tools: bool,
    /// Model supports parallel tool calls.
    pub supports_parallel_tools: bool,
    /// Model supports native structured output.
    pub supports_native_structured_output: bool,
    /// Model supports strict mode for tools.
    pub supports_strict_tools: bool,
    /// Model supports system messages.
    pub supports_system_messages: bool,
    /// Model supports images in input.
    pub supports_images: bool,
    /// Model supports audio in input.
    pub supports_audio: bool,
    /// Model supports video in input.
    pub supports_video: bool,
    /// Model supports documents in input.
    pub supports_documents: bool,
    /// Model supports prompt caching.
    pub supports_caching: bool,
    /// Model supports reasoning/thinking.
    pub supports_reasoning: bool,
    /// JSON schema transformer for this model.
    pub json_schema_transformer: JsonSchemaTransformer,
    /// Maximum tokens for this model.
    pub max_tokens: Option<u64>,
    /// Context window size.
    pub context_window: Option<u64>,
    /// Supports streaming.
    pub supports_streaming: bool,
    /// Tags used to identify thinking/reasoning content in output.
    /// Default: ("<think>", "</think>")
    pub thinking_tags: (String, String),
    /// Whether to ignore leading whitespace in streamed responses.
    /// Workaround for models that emit empty text before tool calls.
    pub ignore_streamed_leading_whitespace: bool,
    /// Default structured output mode for this model.
    pub default_structured_output_mode: OutputMode,
    /// Template for prompted output instructions.
    /// Uses {schema} placeholder for JSON schema.
    pub prompted_output_template: String,
    /// Whether native output mode requires schema in instructions too.
    pub native_output_requires_schema_in_instructions: bool,
}
/// Default template for prompted structured output.
pub const DEFAULT_PROMPTED_OUTPUT_TEMPLATE: &str = r#"Output your response as JSON matching this schema:
```json
{schema}
```
Output only valid JSON, no additional text."#;
impl Default for ModelProfile {
    fn default() -> Self {
        Self {
            supports_tools: true,
            supports_parallel_tools: true,
            supports_native_structured_output: false,
            supports_strict_tools: false,
            supports_system_messages: true,
            supports_images: false,
            supports_audio: false,
            supports_video: false,
            supports_documents: false,
            supports_caching: false,
            supports_reasoning: false,
            json_schema_transformer: JsonSchemaTransformer::default(),
            max_tokens: None,
            context_window: None,
            supports_streaming: true,
            thinking_tags: ("<think>".to_string(), "</think>".to_string()),
            ignore_streamed_leading_whitespace: false,
            default_structured_output_mode: OutputMode::default(),
            prompted_output_template: DEFAULT_PROMPTED_OUTPUT_TEMPLATE.to_string(),
            native_output_requires_schema_in_instructions: false,
        }
    }
}
impl ModelProfile {
    /// Create a new model profile with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set tool support.
    #[must_use]
    pub fn with_tools(mut self, supported: bool) -> Self {
        self.supports_tools = supported;
        self
    }

    /// Set parallel tool calls support.
    #[must_use]
    pub fn with_parallel_tools(mut self, supported: bool) -> Self {
        self.supports_parallel_tools = supported;
        self
    }

    /// Set native structured output support.
    #[must_use]
    pub fn with_native_structured_output(mut self, supported: bool) -> Self {
        self.supports_native_structured_output = supported;
        self
    }

    /// Set strict tools support.
    #[must_use]
    pub fn with_strict_tools(mut self, supported: bool) -> Self {
        self.supports_strict_tools = supported;
        self
    }

    /// Set image support.
    #[must_use]
    pub fn with_images(mut self, supported: bool) -> Self {
        self.supports_images = supported;
        self
    }

    /// Set audio support.
    #[must_use]
    pub fn with_audio(mut self, supported: bool) -> Self {
        self.supports_audio = supported;
        self
    }

    /// Set document support.
    #[must_use]
    pub fn with_documents(mut self, supported: bool) -> Self {
        self.supports_documents = supported;
        self
    }

    /// Set caching support.
    #[must_use]
    pub fn with_caching(mut self, supported: bool) -> Self {
        self.supports_caching = supported;
        self
    }

    /// Set context window size.
    #[must_use]
    pub fn with_context_window(mut self, size: u64) -> Self {
        self.context_window = Some(size);
        self
    }

    /// Set max tokens.
    #[must_use]
    pub fn with_max_tokens(mut self, max: u64) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Set JSON schema transformer.
    #[must_use]
    pub fn with_schema_transformer(mut self, transformer: JsonSchemaTransformer) -> Self {
        self.json_schema_transformer = transformer;
        self
    }

    /// Set reasoning/thinking support.
    #[must_use]
    pub fn with_reasoning(mut self, supported: bool) -> Self {
        self.supports_reasoning = supported;
        self
    }

    /// Set thinking tags for reasoning content.
    #[must_use]
    pub fn with_thinking_tags(mut self, open: impl Into<String>, close: impl Into<String>) -> Self {
        self.thinking_tags = (open.into(), close.into());
        self
    }

    /// Set whether to ignore leading whitespace in streamed responses.
    #[must_use]
    pub fn with_ignore_streamed_leading_whitespace(mut self, ignore: bool) -> Self {
        self.ignore_streamed_leading_whitespace = ignore;
        self
    }

    /// Set default structured output mode.
    #[must_use]
    pub fn with_default_structured_output_mode(mut self, mode: OutputMode) -> Self {
        self.default_structured_output_mode = mode;
        self
    }

    /// Set prompted output template.
    /// Use {schema} as placeholder for the JSON schema.
    #[must_use]
    pub fn with_prompted_output_template(mut self, template: impl Into<String>) -> Self {
        self.prompted_output_template = template.into();
        self
    }

    /// Set whether native output mode requires schema in instructions.
    #[must_use]
    pub fn with_native_output_requires_schema_in_instructions(mut self, required: bool) -> Self {
        self.native_output_requires_schema_in_instructions = required;
        self
    }

    /// Get the opening thinking tag.
    #[must_use]
    pub fn thinking_open_tag(&self) -> &str {
        &self.thinking_tags.0
    }

    /// Get the closing thinking tag.
    #[must_use]
    pub fn thinking_close_tag(&self) -> &str {
        &self.thinking_tags.1
    }

    /// Format the prompted output template with a schema.
    #[must_use]
    pub fn format_prompted_output(&self, schema: &str) -> String {
        self.prompted_output_template.replace("{schema}", schema)
    }
}
/// Default model profile.
pub static DEFAULT_PROFILE: std::sync::LazyLock<ModelProfile> =
    std::sync::LazyLock::new(ModelProfile::default);

/// OpenAI GPT-4o profile.
pub fn openai_gpt4o_profile() -> ModelProfile {
    ModelProfile {
        supports_tools: true,
        supports_parallel_tools: true,
        supports_native_structured_output: true,
        supports_strict_tools: true,
        supports_system_messages: true,
        supports_images: true,
        supports_audio: true,
        supports_video: false,
        supports_documents: true,
        supports_caching: false,
        supports_reasoning: false,
        json_schema_transformer: JsonSchemaTransformer::openai(),
        max_tokens: Some(16384),
        context_window: Some(128000),
        supports_streaming: true,
        thinking_tags: ("<think>".to_string(), "</think>".to_string()),
        ignore_streamed_leading_whitespace: false,
        default_structured_output_mode: OutputMode::Native,
        prompted_output_template: DEFAULT_PROMPTED_OUTPUT_TEMPLATE.to_string(),
        native_output_requires_schema_in_instructions: false,
    }
}

/// OpenAI o1 profile (reasoning model).
pub fn openai_o1_profile() -> ModelProfile {
    ModelProfile {
        supports_tools: true,
        supports_parallel_tools: false,
        supports_native_structured_output: true,
        supports_strict_tools: true,
        supports_system_messages: false, // o1 uses developer messages
        supports_images: true,
        supports_audio: false,
        supports_video: false,
        supports_documents: true,
        supports_caching: false,
        supports_reasoning: true,
        json_schema_transformer: JsonSchemaTransformer::openai(),
        max_tokens: Some(100000),
        context_window: Some(200000),
        supports_streaming: true,
        // o1 models don't use visible thinking tags - reasoning is internal
        thinking_tags: ("<think>".to_string(), "</think>".to_string()),
        ignore_streamed_leading_whitespace: false,
        default_structured_output_mode: OutputMode::Native,
        prompted_output_template: DEFAULT_PROMPTED_OUTPUT_TEMPLATE.to_string(),
        native_output_requires_schema_in_instructions: false,
    }
}

/// Anthropic Claude profile.
pub fn anthropic_claude_profile() -> ModelProfile {
    ModelProfile {
        supports_tools: true,
        supports_parallel_tools: true,
        supports_native_structured_output: false,
        supports_strict_tools: false,
        supports_system_messages: true,
        supports_images: true,
        supports_audio: false,
        supports_video: false,
        supports_documents: true,
        supports_caching: true,
        supports_reasoning: false,
        json_schema_transformer: JsonSchemaTransformer::anthropic(),
        max_tokens: Some(8192),
        context_window: Some(200000),
        supports_streaming: true,
        // Claude uses <think> tags for extended thinking
        thinking_tags: ("<think>".to_string(), "</think>".to_string()),
        ignore_streamed_leading_whitespace: false,
        default_structured_output_mode: OutputMode::Tool,
        prompted_output_template: DEFAULT_PROMPTED_OUTPUT_TEMPLATE.to_string(),
        native_output_requires_schema_in_instructions: false,
    }
}

/// DeepSeek profile (reasoning model).
pub fn deepseek_profile() -> ModelProfile {
    ModelProfile {
        supports_tools: true,
        supports_parallel_tools: false,
        supports_native_structured_output: false,
        supports_strict_tools: false,
        supports_system_messages: true,
        supports_images: false,
        supports_audio: false,
        supports_video: false,
        supports_documents: false,
        supports_caching: false,
        supports_reasoning: true,
        json_schema_transformer: JsonSchemaTransformer::default(),
        max_tokens: Some(8192),
        context_window: Some(64000),
        supports_streaming: true,
        // DeepSeek uses <think> tags for chain-of-thought
        thinking_tags: ("<think>".to_string(), "</think>".to_string()),
        ignore_streamed_leading_whitespace: true,
        default_structured_output_mode: OutputMode::Prompted,
        prompted_output_template: DEFAULT_PROMPTED_OUTPUT_TEMPLATE.to_string(),
        native_output_requires_schema_in_instructions: false,
    }
}

/// Qwen profile (reasoning model).
pub fn qwen_profile() -> ModelProfile {
    ModelProfile {
        supports_tools: true,
        supports_parallel_tools: false,
        supports_native_structured_output: false,
        supports_strict_tools: false,
        supports_system_messages: true,
        supports_images: true,
        supports_audio: false,
        supports_video: false,
        supports_documents: false,
        supports_caching: false,
        supports_reasoning: true,
        json_schema_transformer: JsonSchemaTransformer::default(),
        max_tokens: Some(8192),
        context_window: Some(32000),
        supports_streaming: true,
        // Qwen uses <think> tags for reasoning
        thinking_tags: ("<think>".to_string(), "</think>".to_string()),
        ignore_streamed_leading_whitespace: true,
        default_structured_output_mode: OutputMode::Prompted,
        prompted_output_template: DEFAULT_PROMPTED_OUTPUT_TEMPLATE.to_string(),
        native_output_requires_schema_in_instructions: false,
    }
}

/// Google Gemini profile.
pub fn google_gemini_profile() -> ModelProfile {
    ModelProfile {
        supports_tools: true,
        supports_parallel_tools: true,
        supports_native_structured_output: true,
        supports_strict_tools: false,
        supports_system_messages: true,
        supports_images: true,
        supports_audio: true,
        supports_video: true,
        supports_documents: true,
        supports_caching: false,
        supports_reasoning: false,
        json_schema_transformer: JsonSchemaTransformer::default(),
        max_tokens: Some(8192),
        context_window: Some(1000000), // Gemini 1.5 Pro has 1M context
        supports_streaming: true,
        thinking_tags: ("<think>".to_string(), "</think>".to_string()),
        ignore_streamed_leading_whitespace: false,
        default_structured_output_mode: OutputMode::Native,
        prompted_output_template: DEFAULT_PROMPTED_OUTPUT_TEMPLATE.to_string(),
        native_output_requires_schema_in_instructions: true,
    }
}

/// Mistral profile.
pub fn mistral_profile() -> ModelProfile {
    ModelProfile {
        supports_tools: true,
        supports_parallel_tools: true,
        supports_native_structured_output: false,
        supports_strict_tools: false,
        supports_system_messages: true,
        supports_images: true,
        supports_audio: false,
        supports_video: false,
        supports_documents: false,
        supports_caching: false,
        supports_reasoning: false,
        json_schema_transformer: JsonSchemaTransformer::default(),
        max_tokens: Some(8192),
        context_window: Some(32000),
        supports_streaming: true,
        thinking_tags: ("<think>".to_string(), "</think>".to_string()),
        ignore_streamed_leading_whitespace: false,
        default_structured_output_mode: OutputMode::Tool,
        prompted_output_template: DEFAULT_PROMPTED_OUTPUT_TEMPLATE.to_string(),
        native_output_requires_schema_in_instructions: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_profile() {
        let profile = ModelProfile::default();
        assert!(profile.supports_tools);
        assert!(profile.supports_parallel_tools);
        assert!(profile.supports_system_messages);
        assert!(!profile.supports_native_structured_output);
        assert_eq!(profile.thinking_tags.0, "<think>");
        assert_eq!(profile.thinking_tags.1, "</think>");
        assert!(!profile.ignore_streamed_leading_whitespace);
        assert_eq!(profile.default_structured_output_mode, OutputMode::Tool);
        assert!(!profile.native_output_requires_schema_in_instructions);
    }

    #[test]
    fn test_output_mode_default() {
        assert_eq!(OutputMode::default(), OutputMode::Tool);
    }

    #[test]
    fn test_output_mode_display() {
        assert_eq!(OutputMode::Tool.to_string(), "tool");
        assert_eq!(OutputMode::Native.to_string(), "native");
        assert_eq!(OutputMode::Prompted.to_string(), "prompted");
        assert_eq!(OutputMode::Text.to_string(), "text");
    }

    #[test]
    fn test_openai_profile() {
        let profile = openai_gpt4o_profile();
        assert!(profile.supports_native_structured_output);
        assert!(profile.supports_strict_tools);
        assert!(profile.supports_images);
        assert!(profile.supports_audio);
        assert_eq!(profile.default_structured_output_mode, OutputMode::Native);
    }

    #[test]
    fn test_anthropic_profile() {
        let profile = anthropic_claude_profile();
        assert!(!profile.supports_native_structured_output);
        assert!(profile.supports_caching);
        assert!(profile.supports_images);
        assert_eq!(profile.thinking_tags.0, "<think>");
        assert_eq!(profile.thinking_tags.1, "</think>");
        assert_eq!(profile.default_structured_output_mode, OutputMode::Tool);
    }

    #[test]
    fn test_deepseek_profile() {
        let profile = deepseek_profile();
        assert!(profile.supports_reasoning);
        assert!(profile.ignore_streamed_leading_whitespace);
        assert_eq!(profile.thinking_tags.0, "<think>");
        assert_eq!(profile.thinking_tags.1, "</think>");
        assert_eq!(profile.default_structured_output_mode, OutputMode::Prompted);
    }

    #[test]
    fn test_qwen_profile() {
        let profile = qwen_profile();
        assert!(profile.supports_reasoning);
        assert!(profile.ignore_streamed_leading_whitespace);
        assert_eq!(profile.thinking_tags.0, "<think>");
        assert_eq!(profile.thinking_tags.1, "</think>");
        assert_eq!(profile.default_structured_output_mode, OutputMode::Prompted);
    }

    #[test]
    fn test_gemini_profile() {
        let profile = google_gemini_profile();
        assert!(profile.supports_native_structured_output);
        assert!(profile.supports_video);
        assert!(profile.native_output_requires_schema_in_instructions);
        assert_eq!(profile.default_structured_output_mode, OutputMode::Native);
    }

    #[test]
    fn test_mistral_profile() {
        let profile = mistral_profile();
        assert!(profile.supports_tools);
        assert!(profile.supports_images);
        assert_eq!(profile.default_structured_output_mode, OutputMode::Tool);
    }

    #[test]
    fn test_profile_builder() {
        let profile = ModelProfile::new()
            .with_tools(true)
            .with_images(true)
            .with_context_window(100000)
            .with_max_tokens(4096);

        assert!(profile.supports_tools);
        assert!(profile.supports_images);
        assert_eq!(profile.context_window, Some(100000));
        assert_eq!(profile.max_tokens, Some(4096));
    }

    #[test]
    fn test_profile_builder_new_fields() {
        let profile = ModelProfile::new()
            .with_thinking_tags("<reasoning>", "</reasoning>")
            .with_ignore_streamed_leading_whitespace(true)
            .with_default_structured_output_mode(OutputMode::Prompted)
            .with_prompted_output_template("Custom: {schema}")
            .with_native_output_requires_schema_in_instructions(true)
            .with_reasoning(true);

        assert_eq!(profile.thinking_tags.0, "<reasoning>");
        assert_eq!(profile.thinking_tags.1, "</reasoning>");
        assert!(profile.ignore_streamed_leading_whitespace);
        assert_eq!(profile.default_structured_output_mode, OutputMode::Prompted);
        assert_eq!(profile.prompted_output_template, "Custom: {schema}");
        assert!(profile.native_output_requires_schema_in_instructions);
        assert!(profile.supports_reasoning);
    }

    #[test]
    fn test_thinking_tag_accessors() {
        let profile = ModelProfile::new()
            .with_thinking_tags("<thought>", "</thought>");

        assert_eq!(profile.thinking_open_tag(), "<thought>");
        assert_eq!(profile.thinking_close_tag(), "</thought>");
    }

    #[test]
    fn test_format_prompted_output() {
        let profile = ModelProfile::new()
            .with_prompted_output_template("Please output JSON: {schema}");

        let formatted = profile.format_prompted_output(r#"{"type": "object"}"#);
        assert_eq!(formatted, r#"Please output JSON: {"type": "object"}"#);
    }

    #[test]
    fn test_default_prompted_output_template() {
        let profile = ModelProfile::default();
        let formatted = profile.format_prompted_output("{}");
        assert!(formatted.contains("{}"));
        assert!(formatted.contains("JSON"));
    }
}

//! Anthropic API types.
//!
//! This module contains all the request/response types for the Anthropic Messages API.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

// ============================================================================
// Request Types
// ============================================================================

/// Messages API request.
#[derive(Debug, Clone, Serialize)]
pub struct MessagesRequest {
    /// Model to use.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<AnthropicMessage>,
    /// Maximum tokens to generate.
    pub max_tokens: u64,
    /// System prompt (separate from messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemContent>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top-k sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Tool definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    /// Tool choice strategy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
    /// Request metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<RequestMetadata>,
    /// Whether to stream the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Extended thinking configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

impl MessagesRequest {
    /// Create a new request.
    pub fn new(model: impl Into<String>, messages: Vec<AnthropicMessage>, max_tokens: u64) -> Self {
        Self {
            model: model.into(),
            messages,
            max_tokens,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            stream: None,
            thinking: None,
        }
    }
}

/// System content - can be simple text or blocks with cache control.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemContent {
    /// Simple text system prompt.
    Text(String),
    /// Blocks with optional cache control.
    Blocks(Vec<SystemBlock>),
}

impl SystemContent {
    /// Create text system content.
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    /// Create system content with cache control.
    pub fn cached(s: impl Into<String>) -> Self {
        Self::Blocks(vec![SystemBlock::Text {
            text: s.into(),
            cache_control: Some(CacheControl::ephemeral()),
        }])
    }
}

/// System block for structured system prompts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SystemBlock {
    /// Text block.
    Text {
        /// The text content.
        text: String,
        /// Cache control settings.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

/// Cache control settings for prompt caching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    /// Cache type.
    #[serde(rename = "type")]
    pub cache_type: String,
}

impl CacheControl {
    /// Create ephemeral cache control.
    pub fn ephemeral() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
        }
    }
}

/// A message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    /// Role: "user" or "assistant".
    pub role: String,
    /// Message content.
    pub content: AnthropicContent,
}

impl AnthropicMessage {
    /// Create a user message with text.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: AnthropicContent::Text(content.into()),
        }
    }

    /// Create a user message with blocks.
    pub fn user_blocks(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: "user".to_string(),
            content: AnthropicContent::Blocks(blocks),
        }
    }

    /// Create an assistant message with text.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: AnthropicContent::Text(content.into()),
        }
    }

    /// Create an assistant message with blocks.
    pub fn assistant_blocks(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: AnthropicContent::Blocks(blocks),
        }
    }
}

/// Message content - can be simple text or content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    /// Simple text content.
    Text(String),
    /// Content blocks.
    Blocks(Vec<ContentBlock>),
}

impl AnthropicContent {
    /// Create text content.
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    /// Create content from blocks.
    pub fn blocks(blocks: Vec<ContentBlock>) -> Self {
        Self::Blocks(blocks)
    }
}

/// Content block types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content.
    Text {
        /// The text.
        text: String,
        /// Cache control.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// Image content.
    Image {
        /// Image source.
        source: ImageSource,
        /// Cache control.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// Document content (PDF, etc.).
    Document {
        /// Document source.
        source: DocumentSource,
        /// Cache control.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// Tool use (in assistant messages).
    ToolUse {
        /// Tool call ID.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input.
        input: JsonValue,
    },

    /// Tool result (in user messages).
    ToolResult {
        /// Tool use ID being responded to.
        tool_use_id: String,
        /// Result content.
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<ToolResultContent>,
        /// Whether this is an error.
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },

    /// Thinking content (extended thinking).
    Thinking {
        /// The thinking content.
        thinking: String,
        /// Signature for verification.
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },

    /// Redacted thinking (when thinking is hidden).
    RedactedThinking {
        /// Redacted data.
        data: String,
    },
}

impl ContentBlock {
    /// Create a text block.
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text {
            text: s.into(),
            cache_control: None,
        }
    }

    /// Create a text block with cache control.
    pub fn text_cached(s: impl Into<String>) -> Self {
        Self::Text {
            text: s.into(),
            cache_control: Some(CacheControl::ephemeral()),
        }
    }

    /// Create an image block from base64 data.
    pub fn image_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::Image {
            source: ImageSource::base64(media_type, data),
            cache_control: None,
        }
    }

    /// Create an image block from URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            source: ImageSource::url(url),
            cache_control: None,
        }
    }

    /// Create a tool use block.
    pub fn tool_use(id: impl Into<String>, name: impl Into<String>, input: JsonValue) -> Self {
        Self::ToolUse {
            id: id.into(),
            name: name.into(),
            input,
        }
    }

    /// Create a tool result block.
    pub fn tool_result(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: Some(ToolResultContent::Text(content.into())),
            is_error: None,
        }
    }

    /// Create a tool error result block.
    pub fn tool_error(tool_use_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: Some(ToolResultContent::Text(error.into())),
            is_error: Some(true),
        }
    }
}

/// Image source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    /// Source type: "base64" or "url".
    #[serde(rename = "type")]
    pub source_type: String,
    /// Media type (e.g., "image/jpeg").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// Base64-encoded data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    /// URL for url-type sources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

impl ImageSource {
    /// Create a base64 image source.
    pub fn base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            source_type: "base64".to_string(),
            media_type: Some(media_type.into()),
            data: Some(data.into()),
            url: None,
        }
    }

    /// Create a URL image source.
    pub fn url(url: impl Into<String>) -> Self {
        Self {
            source_type: "url".to_string(),
            media_type: None,
            data: None,
            url: Some(url.into()),
        }
    }
}

/// Document source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSource {
    /// Source type: "base64".
    #[serde(rename = "type")]
    pub source_type: String,
    /// Media type (e.g., "application/pdf").
    pub media_type: String,
    /// Base64-encoded data.
    pub data: String,
}

impl DocumentSource {
    /// Create a base64 document source.
    pub fn base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            source_type: "base64".to_string(),
            media_type: media_type.into(),
            data: data.into(),
        }
    }
}

/// Tool result content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    /// Simple text result.
    Text(String),
    /// Multiple content blocks.
    Blocks(Vec<ToolResultBlock>),
}

impl ToolResultContent {
    /// Create text result.
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    /// Create result from blocks.
    pub fn blocks(blocks: Vec<ToolResultBlock>) -> Self {
        Self::Blocks(blocks)
    }
}

/// Tool result block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultBlock {
    /// Text result.
    Text {
        /// The text.
        text: String,
    },
    /// Image result.
    Image {
        /// Image source.
        source: ImageSource,
    },
}

/// Tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// Input schema (JSON Schema).
    pub input_schema: JsonValue,
    /// Cache control.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl AnthropicTool {
    /// Create a new tool.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: JsonValue,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            cache_control: None,
        }
    }

    /// Add cache control.
    pub fn with_cache(mut self) -> Self {
        self.cache_control = Some(CacheControl::ephemeral());
        self
    }
}

/// Tool choice strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolChoice {
    /// Model decides whether to use tools.
    Auto,
    /// Model must use at least one tool.
    Any,
    /// Model must use a specific tool.
    Tool {
        /// Tool name.
        name: String,
    },
}

impl AnthropicToolChoice {
    /// Create auto choice.
    pub fn auto() -> Self {
        Self::Auto
    }

    /// Create any choice.
    pub fn any() -> Self {
        Self::Any
    }

    /// Create specific tool choice.
    pub fn tool(name: impl Into<String>) -> Self {
        Self::Tool { name: name.into() }
    }
}

/// Extended thinking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Thinking type: "enabled".
    #[serde(rename = "type")]
    pub thinking_type: String,
    /// Budget tokens for thinking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u64>,
}

impl ThinkingConfig {
    /// Create enabled thinking config.
    pub fn enabled() -> Self {
        Self {
            thinking_type: "enabled".to_string(),
            budget_tokens: None,
        }
    }

    /// Create thinking config with budget.
    pub fn with_budget(budget: u64) -> Self {
        Self {
            thinking_type: "enabled".to_string(),
            budget_tokens: Some(budget),
        }
    }
}

/// Request metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetadata {
    /// User ID for tracking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

// ============================================================================
// Response Types
// ============================================================================

/// Messages API response.
#[derive(Debug, Clone, Deserialize)]
pub struct MessagesResponse {
    /// Response ID.
    pub id: String,
    /// Object type.
    #[serde(rename = "type")]
    pub response_type: String,
    /// Role (always "assistant").
    pub role: String,
    /// Content blocks.
    pub content: Vec<ResponseContentBlock>,
    /// Model used.
    pub model: String,
    /// Reason for stopping.
    pub stop_reason: Option<String>,
    /// Stop sequence if hit.
    pub stop_sequence: Option<String>,
    /// Token usage.
    pub usage: AnthropicUsage,
}

/// Response content block.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentBlock {
    /// Text content.
    Text {
        /// The text.
        text: String,
    },
    /// Tool use.
    ToolUse {
        /// Tool call ID.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input.
        input: JsonValue,
    },
    /// Thinking content.
    Thinking {
        /// The thinking.
        thinking: String,
        /// Signature.
        #[serde(default)]
        signature: Option<String>,
    },
    /// Redacted thinking.
    RedactedThinking {
        /// Redacted data.
        data: String,
    },
}

/// Token usage.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct AnthropicUsage {
    /// Input tokens.
    #[serde(default)]
    pub input_tokens: u64,
    /// Output tokens.
    #[serde(default)]
    pub output_tokens: u64,
    /// Tokens used to create cache.
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
    /// Tokens read from cache.
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
}

// ============================================================================
// Streaming Types
// ============================================================================

/// SSE stream event.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Message start.
    MessageStart {
        /// Partial message.
        message: PartialMessage,
    },
    /// Content block start.
    ContentBlockStart {
        /// Block index.
        index: usize,
        /// Content block.
        content_block: ContentBlockStart,
    },
    /// Content block delta.
    ContentBlockDelta {
        /// Block index.
        index: usize,
        /// Delta content.
        delta: ContentBlockDelta,
    },
    /// Content block stop.
    ContentBlockStop {
        /// Block index.
        index: usize,
    },
    /// Message delta.
    MessageDelta {
        /// Delta.
        delta: MessageDelta,
        /// Usage.
        #[serde(default)]
        usage: Option<DeltaUsage>,
    },
    /// Message stop.
    MessageStop,
    /// Ping (keep-alive).
    Ping,
    /// Error.
    Error {
        /// Error details.
        error: StreamError,
    },
}

/// Partial message at stream start.
#[derive(Debug, Clone, Deserialize)]
pub struct PartialMessage {
    /// Message ID.
    pub id: String,
    /// Message type.
    #[serde(rename = "type")]
    pub message_type: String,
    /// Role.
    pub role: String,
    /// Model.
    pub model: String,
    /// Initial usage.
    #[serde(default)]
    pub usage: AnthropicUsage,
}

/// Content block start.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockStart {
    /// Text block start.
    Text {
        /// Initial text (usually empty).
        text: String,
    },
    /// Tool use start.
    ToolUse {
        /// Tool call ID.
        id: String,
        /// Tool name.
        name: String,
        /// Initial input (usually empty object).
        input: JsonValue,
    },
    /// Thinking start.
    Thinking {
        /// Initial thinking (usually empty).
        thinking: String,
    },
    /// Redacted thinking start.
    RedactedThinking {
        /// Redacted data (signature).
        data: String,
    },
}

/// Content block delta.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
    /// Text delta.
    TextDelta {
        /// Text content.
        text: String,
    },
    /// Tool input JSON delta.
    InputJsonDelta {
        /// Partial JSON.
        partial_json: String,
    },
    /// Thinking delta.
    ThinkingDelta {
        /// Thinking content.
        thinking: String,
    },
    /// Signature delta.
    SignatureDelta {
        /// Signature content.
        signature: String,
    },
}

/// Message delta at stream end.
#[derive(Debug, Clone, Deserialize)]
pub struct MessageDelta {
    /// Stop reason.
    pub stop_reason: Option<String>,
    /// Stop sequence.
    pub stop_sequence: Option<String>,
}

/// Usage delta.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct DeltaUsage {
    /// Output tokens so far.
    #[serde(default)]
    pub output_tokens: u64,
}

/// Stream error.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamError {
    /// Error type.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Error message.
    pub message: String,
}

// ============================================================================
// Error Types
// ============================================================================

/// Anthropic API error response.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicError {
    /// Error type.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Error details.
    pub error: AnthropicErrorBody,
}

/// Anthropic error body.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicErrorBody {
    /// Error type.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Error message.
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_message() {
        let msg = AnthropicMessage::user("Hello!");
        assert_eq!(msg.role, "user");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn test_assistant_message_with_tool() {
        let msg = AnthropicMessage::assistant_blocks(vec![
            ContentBlock::text("Let me search for that."),
            ContentBlock::tool_use("tool_1", "search", serde_json::json!({"query": "rust"})),
        ]);
        assert_eq!(msg.role, "assistant");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("tool_use"));
        assert!(json.contains("search"));
    }

    #[test]
    fn test_tool_result() {
        let block = ContentBlock::tool_result("tool_1", "Search results: ...");
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("tool_result"));
        assert!(json.contains("tool_1"));
    }

    #[test]
    fn test_tool_error() {
        let block = ContentBlock::tool_error("tool_1", "Search failed");
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("is_error"));
        assert!(json.contains("true"));
    }

    #[test]
    fn test_system_content() {
        let system = SystemContent::text("You are a helpful assistant.");
        let json = serde_json::to_string(&system).unwrap();
        assert_eq!(json, "\"You are a helpful assistant.\"");
    }

    #[test]
    fn test_system_cached() {
        let system = SystemContent::cached("You are a helpful assistant.");
        let json = serde_json::to_string(&system).unwrap();
        assert!(json.contains("cache_control"));
        assert!(json.contains("ephemeral"));
    }

    #[test]
    fn test_image_source_base64() {
        let source = ImageSource::base64("image/png", "abc123");
        let json = serde_json::to_string(&source).unwrap();
        assert!(json.contains("base64"));
        assert!(json.contains("image/png"));
    }

    #[test]
    fn test_tool_choice() {
        let auto = AnthropicToolChoice::auto();
        let json = serde_json::to_string(&auto).unwrap();
        assert!(json.contains("auto"));

        let specific = AnthropicToolChoice::tool("search");
        let json = serde_json::to_string(&specific).unwrap();
        assert!(json.contains("search"));
    }

    #[test]
    fn test_thinking_config() {
        let config = ThinkingConfig::with_budget(10000);
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("enabled"));
        assert!(json.contains("10000"));
    }

    #[test]
    fn test_deserialize_response() {
        let json = r#"{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;

        let resp: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "msg_123");
        assert_eq!(resp.content.len(), 1);
        assert_eq!(resp.usage.input_tokens, 10);
    }

    #[test]
    fn test_deserialize_tool_use_response() {
        let json = r#"{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me search."},
                {"type": "tool_use", "id": "tool_1", "name": "search", "input": {"q": "rust"}}
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        }"#;

        let resp: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.stop_reason, Some("tool_use".to_string()));
        assert_eq!(resp.content.len(), 2);
    }

    #[test]
    fn test_deserialize_stream_events() {
        // Message start
        let json = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, StreamEvent::MessageStart { .. }));

        // Content block start
        let json =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, StreamEvent::ContentBlockStart { .. }));

        // Text delta
        let json = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        if let StreamEvent::ContentBlockDelta { delta, .. } = event {
            assert!(matches!(delta, ContentBlockDelta::TextDelta { text } if text == "Hello"));
        } else {
            panic!("Expected ContentBlockDelta");
        }

        // Message stop
        let json = r#"{"type":"message_stop"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, StreamEvent::MessageStop));
    }
}

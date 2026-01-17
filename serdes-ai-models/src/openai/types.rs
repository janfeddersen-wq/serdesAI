//! OpenAI API types.
//!
//! This module contains all the request/response types for the OpenAI API.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

// ============================================================================
// Request Types
// ============================================================================

/// Chat completion request.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionRequest {
    /// Model to use.
    pub model: String,
    /// Messages in the conversation.
    pub messages: Vec<ChatMessage>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Maximum tokens to generate (deprecated, use max_completion_tokens).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Maximum completion tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Presence penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Frequency penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Random seed for reproducibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Tool definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatTool>>,
    /// Tool choice strategy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoiceValue>,
    /// Whether to allow parallel tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Response format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// User identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Whether to stream the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Stream options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// Log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Top log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
}

impl ChatCompletionRequest {
    /// Create a new request.
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_completion_tokens: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            response_format: None,
            user: None,
            stream: None,
            stream_options: None,
            logprobs: None,
            top_logprobs: None,
        }
    }
}

/// Chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message author.
    pub role: String,
    /// Message content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    /// Name of the author.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool calls made by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// ID of the tool call being responded to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(MessageContent::Text(content.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(MessageContent::Text(content.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(MessageContent::Text(content.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a tool response message.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(MessageContent::Text(content.into())),
            name: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// Message content (can be text or multipart).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content.
    Text(String),
    /// Multipart content.
    Parts(Vec<ContentPart>),
}

/// Content part for multipart messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content.
    #[serde(rename = "text")]
    Text {
        /// The text content.
        text: String,
    },
    /// Image URL content.
    #[serde(rename = "image_url")]
    ImageUrl {
        /// Image URL details.
        image_url: ImageUrlContent,
    },
    /// Audio content.
    #[serde(rename = "input_audio")]
    Audio {
        /// Audio data.
        input_audio: AudioContent,
    },
}

impl ContentPart {
    /// Create a text part.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create an image URL part.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrlContent {
                url: url.into(),
                detail: None,
            },
        }
    }

    /// Create an image URL part with detail level.
    pub fn image_url_with_detail(url: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrlContent {
                url: url.into(),
                detail: Some(detail.into()),
            },
        }
    }
}

/// Image URL content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrlContent {
    /// The image URL.
    pub url: String,
    /// Detail level (auto, low, high).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Audio content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioContent {
    /// Base64-encoded audio data.
    pub data: String,
    /// Audio format (wav, mp3).
    pub format: String,
}

/// Tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTool {
    /// Tool type (always "function").
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function definition.
    pub function: FunctionDefinition,
}

impl ChatTool {
    /// Create a function tool.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: JsonValue,
    ) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
                strict: None,
            },
        }
    }

    /// Create a function tool with strict mode.
    pub fn function_strict(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: JsonValue,
    ) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
                strict: Some(true),
            },
        }
    }
}

/// Function definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Function name.
    pub name: String,
    /// Function description.
    pub description: String,
    /// Parameter schema.
    pub parameters: JsonValue,
    /// Whether to use strict mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Tool call in a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Tool call ID.
    pub id: String,
    /// Tool type.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function call details.
    pub function: FunctionCall,
}

/// Function call details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Function name.
    pub name: String,
    /// Arguments as JSON string.
    pub arguments: String,
}

/// Tool choice value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoiceValue {
    /// String choice (auto, none, required).
    String(String),
    /// Specific tool choice.
    Specific {
        /// Tool type.
        #[serde(rename = "type")]
        tool_type: String,
        /// Function to call.
        function: FunctionName,
    },
}

impl ToolChoiceValue {
    /// Auto mode.
    pub fn auto() -> Self {
        Self::String("auto".to_string())
    }

    /// None mode.
    pub fn none() -> Self {
        Self::String("none".to_string())
    }

    /// Required mode.
    pub fn required() -> Self {
        Self::String("required".to_string())
    }

    /// Specific function.
    pub fn function(name: impl Into<String>) -> Self {
        Self::Specific {
            tool_type: "function".to_string(),
            function: FunctionName { name: name.into() },
        }
    }
}

/// Function name for tool choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionName {
    /// The function name.
    pub name: String,
}

/// Response format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    /// Format type (text, json_object, json_schema).
    #[serde(rename = "type")]
    pub format_type: String,
    /// JSON schema for structured output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<JsonSchemaFormat>,
}

impl ResponseFormat {
    /// Text format.
    pub fn text() -> Self {
        Self {
            format_type: "text".to_string(),
            json_schema: None,
        }
    }

    /// JSON object format.
    pub fn json_object() -> Self {
        Self {
            format_type: "json_object".to_string(),
            json_schema: None,
        }
    }

    /// JSON schema format.
    pub fn json_schema(name: impl Into<String>, schema: JsonValue, strict: bool) -> Self {
        Self {
            format_type: "json_schema".to_string(),
            json_schema: Some(JsonSchemaFormat {
                name: name.into(),
                schema,
                strict: Some(strict),
            }),
        }
    }
}

/// JSON schema format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchemaFormat {
    /// Schema name.
    pub name: String,
    /// The JSON schema.
    pub schema: JsonValue,
    /// Whether to use strict mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Stream options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamOptions {
    /// Include usage in stream.
    pub include_usage: bool,
}

// ============================================================================
// Response Types
// ============================================================================

/// Chat completion response.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    /// Response ID.
    pub id: String,
    /// Object type.
    pub object: String,
    /// Creation timestamp.
    pub created: u64,
    /// Model used.
    pub model: String,
    /// Response choices.
    pub choices: Vec<ChatChoice>,
    /// Token usage.
    pub usage: Option<Usage>,
    /// System fingerprint.
    pub system_fingerprint: Option<String>,
}

/// Chat choice.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatChoice {
    /// Choice index.
    pub index: u32,
    /// The message.
    pub message: ResponseMessage,
    /// Reason for stopping.
    pub finish_reason: Option<String>,
    /// Log probabilities.
    pub logprobs: Option<JsonValue>,
}

/// Response message.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseMessage {
    /// Role.
    pub role: String,
    /// Text content.
    pub content: Option<String>,
    /// Tool calls.
    pub tool_calls: Option<Vec<ResponseToolCall>>,
    /// Refusal (for content filter).
    pub refusal: Option<String>,
    /// Reasoning/thinking content (for models like GLM-4 that support chain-of-thought).
    /// This is returned by some OpenAI-compatible providers when the model does reasoning.
    pub reasoning_content: Option<String>,
}

/// Response tool call.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseToolCall {
    /// Tool call ID.
    pub id: String,
    /// Tool type.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function call.
    pub function: FunctionCall,
}

/// Token usage.
#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    /// Prompt tokens.
    pub prompt_tokens: u64,
    /// Completion tokens.
    pub completion_tokens: u64,
    /// Total tokens.
    pub total_tokens: u64,
    /// Prompt token details.
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Completion token details.
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// Prompt token details.
#[derive(Debug, Clone, Deserialize)]
pub struct PromptTokensDetails {
    /// Cached tokens.
    pub cached_tokens: Option<u64>,
    /// Audio tokens.
    pub audio_tokens: Option<u64>,
}

/// Completion token details.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionTokensDetails {
    /// Reasoning tokens.
    pub reasoning_tokens: Option<u64>,
    /// Audio tokens.
    pub audio_tokens: Option<u64>,
}

// ============================================================================
// Streaming Types
// ============================================================================

/// Chat completion chunk (streaming).
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionChunk {
    /// Response ID.
    pub id: String,
    /// Object type.
    pub object: String,
    /// Creation timestamp.
    pub created: u64,
    /// Model used.
    pub model: String,
    /// Response choices.
    pub choices: Vec<ChunkChoice>,
    /// Token usage (if stream_options.include_usage is true).
    pub usage: Option<Usage>,
    /// System fingerprint.
    pub system_fingerprint: Option<String>,
}

/// Chunk choice.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkChoice {
    /// Choice index.
    pub index: u32,
    /// Delta content.
    pub delta: ChunkDelta,
    /// Finish reason.
    pub finish_reason: Option<String>,
    /// Log probabilities.
    pub logprobs: Option<JsonValue>,
}

/// Chunk delta.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ChunkDelta {
    /// Role (usually only in first chunk).
    pub role: Option<String>,
    /// Text content delta.
    pub content: Option<String>,
    /// Tool calls delta.
    pub tool_calls: Option<Vec<ChunkToolCall>>,
    /// Refusal.
    pub refusal: Option<String>,
    /// Reasoning/thinking content delta (for models like GLM-4).
    pub reasoning_content: Option<String>,
}

/// Chunk tool call.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkToolCall {
    /// Index of this tool call.
    pub index: u32,
    /// Tool call ID (only in first chunk for this tool).
    pub id: Option<String>,
    /// Tool type.
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    /// Function call delta.
    pub function: Option<ChunkFunction>,
}

/// Chunk function.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkFunction {
    /// Function name (only in first chunk).
    pub name: Option<String>,
    /// Arguments delta.
    pub arguments: Option<String>,
}

// ============================================================================
// Error Types
// ============================================================================

/// OpenAI API error response.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIError {
    /// Error details.
    pub error: OpenAIErrorBody,
}

/// OpenAI error body.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIErrorBody {
    /// Error message.
    pub message: String,
    /// Error type.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Parameter that caused the error.
    pub param: Option<String>,
    /// Error code.
    pub code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_system() {
        let msg = ChatMessage::system("You are a helpful assistant.");
        assert_eq!(msg.role, "system");
        assert!(matches!(msg.content, Some(MessageContent::Text(_))));
    }

    #[test]
    fn test_chat_message_tool() {
        let msg = ChatMessage::tool("call_123", "Result: 42");
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_content_part_image() {
        let part = ContentPart::image_url("https://example.com/image.png");
        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("image_url"));
    }

    #[test]
    fn test_tool_choice_values() {
        let auto = ToolChoiceValue::auto();
        let json = serde_json::to_string(&auto).unwrap();
        assert_eq!(json, "\"auto\"");

        let specific = ToolChoiceValue::function("my_func");
        let json = serde_json::to_string(&specific).unwrap();
        assert!(json.contains("my_func"));
    }

    #[test]
    fn test_response_format() {
        let format = ResponseFormat::json_object();
        let json = serde_json::to_string(&format).unwrap();
        assert!(json.contains("json_object"));
    }

    #[test]
    fn test_deserialize_response() {
        let json = r#"{
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let response: ChatCompletionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "chatcmpl-123");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(
            response.choices[0].message.content,
            Some("Hello!".to_string())
        );
    }

    #[test]
    fn test_deserialize_chunk() {
        let json = r#"{
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "Hello"
                },
                "finish_reason": null
            }]
        }"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    }
}

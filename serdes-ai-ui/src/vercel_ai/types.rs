//! Vercel AI SDK SSE chunk types.
//!
//! Based on: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
//!
//! This module implements the complete Vercel AI Data Stream Protocol,
//! enabling serdesAI agents to stream responses to Vercel AI SDK clients.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Provider metadata type.
pub type ProviderMetadata = HashMap<String, HashMap<String, Value>>;

/// Finish reason for model completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum FinishReason {
    /// Normal stop (end of response).
    Stop,
    /// Maximum token length reached.
    Length,
    /// Content filter triggered.
    ContentFilter,
    /// Tool calls need to be executed.
    ToolCalls,
    /// Error occurred.
    Error,
    /// Other/custom reason.
    Other,
    /// Unknown reason.
    #[default]
    Unknown,
}

/// Base trait for all Vercel AI chunk types.
pub trait Chunk: erased_serde::Serialize + Send + Sync {
    /// Get the chunk type identifier.
    fn chunk_type(&self) -> &'static str;

    /// Encode the chunk as JSON.
    fn encode(&self) -> String
    where
        Self: Sized + serde::Serialize,
    {
        serde_json::to_string(self).unwrap_or_default()
    }
}

// Allow Box<dyn Chunk> to be serialized
erased_serde::serialize_trait_object!(Chunk);

/// Encode any chunk (including dyn Chunk) to JSON string.
pub fn encode_chunk<C: ?Sized + erased_serde::Serialize>(chunk: &C) -> String {
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::new(&mut buf);
    if erased_serde::serialize(chunk, &mut serializer).is_ok() {
        String::from_utf8(buf).unwrap_or_default()
    } else {
        String::new()
    }
}

// ============================================================================
// Message Lifecycle Chunks
// ============================================================================

/// Start of a new message stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Unique message identifier.
    pub message_id: String,
}

impl StartChunk {
    /// Create a new start chunk.
    pub fn new(message_id: impl Into<String>) -> Self {
        Self {
            chunk_type: "start".to_string(),
            message_id: message_id.into(),
        }
    }
}

impl Chunk for StartChunk {
    fn chunk_type(&self) -> &'static str {
        "start"
    }
}

/// Start of a new step within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartStepChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Unique message identifier.
    pub message_id: String,
    /// Step number (0-indexed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<u32>,
}

impl StartStepChunk {
    /// Create a new start step chunk.
    pub fn new(message_id: impl Into<String>) -> Self {
        Self {
            chunk_type: "start-step".to_string(),
            message_id: message_id.into(),
            step: None,
        }
    }

    /// Set the step number.
    pub fn with_step(mut self, step: u32) -> Self {
        self.step = Some(step);
        self
    }
}

impl Chunk for StartStepChunk {
    fn chunk_type(&self) -> &'static str {
        "start-step"
    }
}

/// End of a step within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FinishStepChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Unique message identifier.
    pub message_id: String,
    /// Why the step finished.
    pub finish_reason: FinishReason,
    /// Token usage for this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageInfo>,
    /// Whether tool calls are pending.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_continued: Option<bool>,
}

impl FinishStepChunk {
    /// Create a new finish step chunk.
    pub fn new(message_id: impl Into<String>, finish_reason: FinishReason) -> Self {
        Self {
            chunk_type: "finish-step".to_string(),
            message_id: message_id.into(),
            finish_reason,
            usage: None,
            is_continued: None,
        }
    }

    /// Set usage information.
    pub fn with_usage(mut self, usage: UsageInfo) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Mark as continued (tool calls pending).
    pub fn with_continued(mut self, continued: bool) -> Self {
        self.is_continued = Some(continued);
        self
    }
}

impl Chunk for FinishStepChunk {
    fn chunk_type(&self) -> &'static str {
        "finish-step"
    }
}

/// End of a message stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FinishChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Unique message identifier.
    pub message_id: String,
    /// Why the message finished.
    pub finish_reason: FinishReason,
    /// Total token usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageInfo>,
}

impl FinishChunk {
    /// Create a new finish chunk.
    pub fn new(message_id: impl Into<String>, finish_reason: FinishReason) -> Self {
        Self {
            chunk_type: "finish".to_string(),
            message_id: message_id.into(),
            finish_reason,
            usage: None,
        }
    }

    /// Set usage information.
    pub fn with_usage(mut self, usage: UsageInfo) -> Self {
        self.usage = Some(usage);
        self
    }
}

impl Chunk for FinishChunk {
    fn chunk_type(&self) -> &'static str {
        "finish"
    }
}

/// Final done signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DoneChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
}

impl DoneChunk {
    /// Create a new done chunk.
    pub fn new() -> Self {
        Self {
            chunk_type: "done".to_string(),
        }
    }
}

impl Default for DoneChunk {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunk for DoneChunk {
    fn chunk_type(&self) -> &'static str {
        "done"
    }
}

/// Stream was aborted.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AbortChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
}

impl AbortChunk {
    /// Create a new abort chunk.
    pub fn new() -> Self {
        Self {
            chunk_type: "abort".to_string(),
        }
    }
}

impl Default for AbortChunk {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunk for AbortChunk {
    fn chunk_type(&self) -> &'static str {
        "abort"
    }
}

// ============================================================================
// Text Streaming Chunks
// ============================================================================

/// Start of text content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextStartChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
}

impl TextStartChunk {
    /// Create a new text start chunk.
    pub fn new() -> Self {
        Self {
            chunk_type: "text-start".to_string(),
        }
    }
}

impl Default for TextStartChunk {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunk for TextStartChunk {
    fn chunk_type(&self) -> &'static str {
        "text-start"
    }
}

/// Text delta/fragment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextDeltaChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// The text content.
    pub text_delta: String,
}

impl TextDeltaChunk {
    /// Create a new text delta chunk.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            chunk_type: "text-delta".to_string(),
            text_delta: text.into(),
        }
    }
}

impl Chunk for TextDeltaChunk {
    fn chunk_type(&self) -> &'static str {
        "text-delta"
    }
}

/// End of text content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextEndChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
}

impl TextEndChunk {
    /// Create a new text end chunk.
    pub fn new() -> Self {
        Self {
            chunk_type: "text-end".to_string(),
        }
    }
}

impl Default for TextEndChunk {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunk for TextEndChunk {
    fn chunk_type(&self) -> &'static str {
        "text-end"
    }
}

// ============================================================================
// Reasoning/Thinking Chunks (for Claude extended thinking)
// ============================================================================

/// Start of reasoning/thinking content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningStartChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
}

impl ReasoningStartChunk {
    /// Create a new reasoning start chunk.
    pub fn new() -> Self {
        Self {
            chunk_type: "reasoning-start".to_string(),
        }
    }
}

impl Default for ReasoningStartChunk {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunk for ReasoningStartChunk {
    fn chunk_type(&self) -> &'static str {
        "reasoning-start"
    }
}

/// Reasoning/thinking delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningDeltaChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// The reasoning/thinking content.
    pub reasoning_delta: String,
}

impl ReasoningDeltaChunk {
    /// Create a new reasoning delta chunk.
    pub fn new(reasoning: impl Into<String>) -> Self {
        Self {
            chunk_type: "reasoning-delta".to_string(),
            reasoning_delta: reasoning.into(),
        }
    }
}

impl Chunk for ReasoningDeltaChunk {
    fn chunk_type(&self) -> &'static str {
        "reasoning-delta"
    }
}

/// End of reasoning/thinking content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningEndChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
}

impl ReasoningEndChunk {
    /// Create a new reasoning end chunk.
    pub fn new() -> Self {
        Self {
            chunk_type: "reasoning-end".to_string(),
        }
    }
}

impl Default for ReasoningEndChunk {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunk for ReasoningEndChunk {
    fn chunk_type(&self) -> &'static str {
        "reasoning-end"
    }
}

// ============================================================================
// Tool Input Chunks
// ============================================================================

/// Start of tool call input.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolInputStartChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Unique identifier for this tool call.
    pub tool_call_id: String,
    /// Name of the tool being called.
    pub tool_name: String,
}

impl ToolInputStartChunk {
    /// Create a new tool input start chunk.
    pub fn new(tool_call_id: impl Into<String>, tool_name: impl Into<String>) -> Self {
        Self {
            chunk_type: "tool-input-start".to_string(),
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
        }
    }
}

impl Chunk for ToolInputStartChunk {
    fn chunk_type(&self) -> &'static str {
        "tool-input-start"
    }
}

/// Tool call arguments delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolInputDeltaChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Unique identifier for this tool call.
    pub tool_call_id: String,
    /// Arguments delta (JSON fragment).
    pub args_text_delta: String,
}

impl ToolInputDeltaChunk {
    /// Create a new tool input delta chunk.
    pub fn new(tool_call_id: impl Into<String>, args_delta: impl Into<String>) -> Self {
        Self {
            chunk_type: "tool-input-delta".to_string(),
            tool_call_id: tool_call_id.into(),
            args_text_delta: args_delta.into(),
        }
    }
}

impl Chunk for ToolInputDeltaChunk {
    fn chunk_type(&self) -> &'static str {
        "tool-input-delta"
    }
}

/// Tool call input complete and available.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolInputAvailableChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Unique identifier for this tool call.
    pub tool_call_id: String,
    /// Name of the tool being called.
    pub tool_name: String,
    /// Complete arguments.
    pub args: Value,
}

impl ToolInputAvailableChunk {
    /// Create a new tool input available chunk.
    pub fn new(tool_call_id: impl Into<String>, tool_name: impl Into<String>, args: Value) -> Self {
        Self {
            chunk_type: "tool-input-available".to_string(),
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            args,
        }
    }
}

impl Chunk for ToolInputAvailableChunk {
    fn chunk_type(&self) -> &'static str {
        "tool-input-available"
    }
}

/// Tool input parsing/validation error.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolInputErrorChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Unique identifier for this tool call.
    pub tool_call_id: String,
    /// Error message.
    pub error: String,
}

impl ToolInputErrorChunk {
    /// Create a new tool input error chunk.
    pub fn new(tool_call_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            chunk_type: "tool-input-error".to_string(),
            tool_call_id: tool_call_id.into(),
            error: error.into(),
        }
    }
}

impl Chunk for ToolInputErrorChunk {
    fn chunk_type(&self) -> &'static str {
        "tool-input-error"
    }
}

// ============================================================================
// Tool Output Chunks
// ============================================================================

/// Tool execution result available.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolOutputAvailableChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Unique identifier for this tool call.
    pub tool_call_id: String,
    /// Tool execution result.
    pub output: Value,
}

impl ToolOutputAvailableChunk {
    /// Create a new tool output available chunk.
    pub fn new(tool_call_id: impl Into<String>, output: Value) -> Self {
        Self {
            chunk_type: "tool-output-available".to_string(),
            tool_call_id: tool_call_id.into(),
            output,
        }
    }
}

impl Chunk for ToolOutputAvailableChunk {
    fn chunk_type(&self) -> &'static str {
        "tool-output-available"
    }
}

/// Tool execution error.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolOutputErrorChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Unique identifier for this tool call.
    pub tool_call_id: String,
    /// Error message.
    pub error: String,
}

impl ToolOutputErrorChunk {
    /// Create a new tool output error chunk.
    pub fn new(tool_call_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            chunk_type: "tool-output-error".to_string(),
            tool_call_id: tool_call_id.into(),
            error: error.into(),
        }
    }
}

impl Chunk for ToolOutputErrorChunk {
    fn chunk_type(&self) -> &'static str {
        "tool-output-error"
    }
}

/// Tool execution was denied (user rejected).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolOutputDeniedChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Unique identifier for this tool call.
    pub tool_call_id: String,
    /// Reason for denial (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl ToolOutputDeniedChunk {
    /// Create a new tool output denied chunk.
    pub fn new(tool_call_id: impl Into<String>) -> Self {
        Self {
            chunk_type: "tool-output-denied".to_string(),
            tool_call_id: tool_call_id.into(),
            reason: None,
        }
    }

    /// Set the denial reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

impl Chunk for ToolOutputDeniedChunk {
    fn chunk_type(&self) -> &'static str {
        "tool-output-denied"
    }
}

/// Request user approval for tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolApprovalRequestChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Unique identifier for this tool call.
    pub tool_call_id: String,
    /// Name of the tool requesting approval.
    pub tool_name: String,
    /// Arguments for the tool call.
    pub args: Value,
    /// Human-readable description of what the tool will do.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl ToolApprovalRequestChunk {
    /// Create a new tool approval request chunk.
    pub fn new(tool_call_id: impl Into<String>, tool_name: impl Into<String>, args: Value) -> Self {
        Self {
            chunk_type: "tool-approval-request".to_string(),
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            args,
            description: None,
        }
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

impl Chunk for ToolApprovalRequestChunk {
    fn chunk_type(&self) -> &'static str {
        "tool-approval-request"
    }
}

// ============================================================================
// Source/Citation Chunks
// ============================================================================

/// URL source citation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceUrlChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Unique identifier for this source.
    pub source_id: String,
    /// Source URL.
    pub url: String,
    /// Source title (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl SourceUrlChunk {
    /// Create a new source URL chunk.
    pub fn new(source_id: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            chunk_type: "source-url".to_string(),
            source_id: source_id.into(),
            url: url.into(),
            title: None,
        }
    }

    /// Set the title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
}

impl Chunk for SourceUrlChunk {
    fn chunk_type(&self) -> &'static str {
        "source-url"
    }
}

/// Document source citation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceDocumentChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Unique identifier for this source.
    pub source_id: String,
    /// Document identifier.
    pub document_id: String,
    /// Document title (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Document content snippet (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

impl SourceDocumentChunk {
    /// Create a new source document chunk.
    pub fn new(source_id: impl Into<String>, document_id: impl Into<String>) -> Self {
        Self {
            chunk_type: "source-document".to_string(),
            source_id: source_id.into(),
            document_id: document_id.into(),
            title: None,
            snippet: None,
        }
    }

    /// Set the title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the snippet.
    pub fn with_snippet(mut self, snippet: impl Into<String>) -> Self {
        self.snippet = Some(snippet.into());
        self
    }
}

impl Chunk for SourceDocumentChunk {
    fn chunk_type(&self) -> &'static str {
        "source-document"
    }
}

// ============================================================================
// File and Data Chunks
// ============================================================================

/// File attachment chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Unique identifier for this file.
    pub file_id: String,
    /// File name.
    pub name: String,
    /// MIME type.
    pub mime_type: String,
    /// File data (base64 encoded or URL).
    pub data: String,
}

impl FileChunk {
    /// Create a new file chunk.
    pub fn new(
        file_id: impl Into<String>,
        name: impl Into<String>,
        mime_type: impl Into<String>,
        data: impl Into<String>,
    ) -> Self {
        Self {
            chunk_type: "file".to_string(),
            file_id: file_id.into(),
            name: name.into(),
            mime_type: mime_type.into(),
            data: data.into(),
        }
    }
}

impl Chunk for FileChunk {
    fn chunk_type(&self) -> &'static str {
        "file"
    }
}

/// Generic data chunk for custom data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Data key/type identifier.
    pub key: String,
    /// Data value.
    pub value: Value,
}

impl DataChunk {
    /// Create a new data chunk.
    pub fn new(key: impl Into<String>, value: Value) -> Self {
        Self {
            chunk_type: "data".to_string(),
            key: key.into(),
            value,
        }
    }
}

impl Chunk for DataChunk {
    fn chunk_type(&self) -> &'static str {
        "data"
    }
}

// ============================================================================
// Error and Metadata Chunks
// ============================================================================

/// Error chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErrorChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Error message.
    pub error: String,
    /// Error code (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl ErrorChunk {
    /// Create a new error chunk.
    pub fn new(error: impl Into<String>) -> Self {
        Self {
            chunk_type: "error".to_string(),
            error: error.into(),
            code: None,
        }
    }

    /// Set the error code.
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }
}

impl Chunk for ErrorChunk {
    fn chunk_type(&self) -> &'static str {
        "error"
    }
}

/// Message metadata chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageMetadataChunk {
    /// Type discriminator.
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Message identifier.
    pub message_id: String,
    /// Metadata key-value pairs.
    #[serde(flatten)]
    pub metadata: HashMap<String, Value>,
}

impl MessageMetadataChunk {
    /// Create a new message metadata chunk.
    pub fn new(message_id: impl Into<String>) -> Self {
        Self {
            chunk_type: "message-metadata".to_string(),
            message_id: message_id.into(),
            metadata: HashMap::new(),
        }
    }

    /// Add a metadata entry.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

impl Chunk for MessageMetadataChunk {
    fn chunk_type(&self) -> &'static str {
        "message-metadata"
    }
}

// ============================================================================
// Usage Info
// ============================================================================

/// Token usage information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageInfo {
    /// Number of prompt/input tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    /// Number of completion/output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    /// Total tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
}

impl UsageInfo {
    /// Create new usage info.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set prompt tokens.
    pub fn with_prompt_tokens(mut self, tokens: u32) -> Self {
        self.prompt_tokens = Some(tokens);
        self
    }

    /// Set completion tokens.
    pub fn with_completion_tokens(mut self, tokens: u32) -> Self {
        self.completion_tokens = Some(tokens);
        self
    }

    /// Set total tokens.
    pub fn with_total_tokens(mut self, tokens: u32) -> Self {
        self.total_tokens = Some(tokens);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_start_chunk() {
        let chunk = StartChunk::new("msg-123");
        let json = chunk.encode();
        assert!(json.contains(r#""type":"start"#));
        assert!(json.contains(r#""messageId":"msg-123"#));
    }

    #[test]
    fn test_text_delta_chunk() {
        let chunk = TextDeltaChunk::new("Hello, world!");
        let json = chunk.encode();
        assert!(json.contains(r#""type":"text-delta"#));
        assert!(json.contains(r#""textDelta":"Hello, world!"#));
    }

    #[test]
    fn test_tool_input_start_chunk() {
        let chunk = ToolInputStartChunk::new("call-123", "get_weather");
        let json = chunk.encode();
        assert!(json.contains(r#""type":"tool-input-start"#));
        assert!(json.contains(r#""toolCallId":"call-123"#));
        assert!(json.contains(r#""toolName":"get_weather"#));
    }

    #[test]
    fn test_finish_reason_serialization() {
        let chunk = FinishChunk::new("msg-123", FinishReason::ToolCalls);
        let json = chunk.encode();
        assert!(json.contains(r#""finishReason":"tool-calls"#));
    }

    #[test]
    fn test_usage_info() {
        let usage = UsageInfo::new()
            .with_prompt_tokens(100)
            .with_completion_tokens(50)
            .with_total_tokens(150);
        let json = serde_json::to_string(&usage).unwrap();
        assert!(json.contains(r#""promptTokens":100"#));
        assert!(json.contains(r#""completionTokens":50"#));
    }

    #[test]
    fn test_done_chunk() {
        let chunk = DoneChunk::new();
        assert_eq!(chunk.chunk_type(), "done");
    }

    #[test]
    fn test_reasoning_delta() {
        let chunk = ReasoningDeltaChunk::new("Let me think about this...");
        let json = chunk.encode();
        assert!(json.contains(r#""type":"reasoning-delta"#));
        assert!(json.contains(r#""reasoningDelta"#));
    }
}

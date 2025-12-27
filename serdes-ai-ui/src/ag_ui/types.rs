//! AG-UI protocol event types.
//!
//! Agent-User Interaction protocol for real-time agent streaming.
//! This implements the AG-UI specification for bidirectional agent communication.
//!
//! # Event Structure
//!
//! All events share common fields:
//! - `type`: Event type discriminator
//! - `timestamp`: Optional Unix timestamp in milliseconds
//!
//! Events are organized into categories:
//! - **Run lifecycle**: `RUN_STARTED`, `RUN_FINISHED`, `RUN_ERROR`
//! - **Text messages**: `TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_END`
//! - **Thinking**: `THINKING_START`, `THINKING_END` with nested text messages
//! - **Tool calls**: `TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_END`, `TOOL_CALL_RESULT`
//! - **State**: `STATE_SNAPSHOT`, `STATE_DELTA`, `MESSAGES_SNAPSHOT`

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Event type discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EventType {
    /// Run has started.
    RunStarted,
    /// Run has finished successfully.
    RunFinished,
    /// Run encountered an error.
    RunError,
    /// Text message started.
    TextMessageStart,
    /// Text message content delta.
    TextMessageContent,
    /// Text message ended.
    TextMessageEnd,
    /// Thinking/reasoning started.
    ThinkingStart,
    /// Thinking/reasoning ended.
    ThinkingEnd,
    /// Thinking text message started (nested in thinking).
    ThinkingTextMessageStart,
    /// Thinking text message content delta.
    ThinkingTextMessageContent,
    /// Thinking text message ended.
    ThinkingTextMessageEnd,
    /// Tool call started.
    ToolCallStart,
    /// Tool call arguments delta.
    ToolCallArgs,
    /// Tool call ended (arguments complete).
    ToolCallEnd,
    /// Tool call result received.
    ToolCallResult,
    /// State snapshot (full state).
    StateSnapshot,
    /// State delta (partial update).
    StateDelta,
    /// Messages snapshot.
    MessagesSnapshot,
    /// Custom event.
    Custom,
    /// Raw event (passthrough).
    Raw,
}

/// Base event trait for all AG-UI events.
pub trait Event: erased_serde::Serialize + Send + Sync {
    /// Get the event type.
    fn event_type(&self) -> EventType;

    /// Get the timestamp (milliseconds since epoch).
    fn timestamp(&self) -> Option<i64>;

    /// Encode the event as JSON.
    fn encode(&self) -> String
    where
        Self: Sized + Serialize,
    {
        serde_json::to_string(self).unwrap_or_default()
    }
}

// Allow Box<dyn Event> to be serialized
erased_serde::serialize_trait_object!(Event);

/// Encode any event (including dyn Event) to JSON string.
pub fn encode_event<E: ?Sized + erased_serde::Serialize>(event: &E) -> String {
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::new(&mut buf);
    if erased_serde::serialize(event, &mut serializer).is_ok() {
        String::from_utf8(buf).unwrap_or_default()
    } else {
        String::new()
    }
}

// ============================================================================
// Run Lifecycle Events
// ============================================================================

/// Run started event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunStartedEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Thread identifier.
    pub thread_id: String,
    /// Run identifier.
    pub run_id: String,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl RunStartedEvent {
    /// Create a new run started event.
    pub fn new(thread_id: impl Into<String>, run_id: impl Into<String>) -> Self {
        Self {
            event_type: EventType::RunStarted,
            thread_id: thread_id.into(),
            run_id: run_id.into(),
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

impl Event for RunStartedEvent {
    fn event_type(&self) -> EventType {
        EventType::RunStarted
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

/// Run finished event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunFinishedEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Thread identifier.
    pub thread_id: String,
    /// Run identifier.
    pub run_id: String,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl RunFinishedEvent {
    /// Create a new run finished event.
    pub fn new(thread_id: impl Into<String>, run_id: impl Into<String>) -> Self {
        Self {
            event_type: EventType::RunFinished,
            thread_id: thread_id.into(),
            run_id: run_id.into(),
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

impl Event for RunFinishedEvent {
    fn event_type(&self) -> EventType {
        EventType::RunFinished
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

/// Run error event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunErrorEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Error message.
    pub message: String,
    /// Error code (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl RunErrorEvent {
    /// Create a new run error event.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            event_type: EventType::RunError,
            message: message.into(),
            code: None,
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }

    /// Set the error code.
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }
}

impl Event for RunErrorEvent {
    fn event_type(&self) -> EventType {
        EventType::RunError
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

// ============================================================================
// Text Message Events
// ============================================================================

/// Text message start event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextMessageStartEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Message identifier.
    pub message_id: String,
    /// Role of the message sender.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl TextMessageStartEvent {
    /// Create a new text message start event.
    pub fn new(message_id: impl Into<String>) -> Self {
        Self {
            event_type: EventType::TextMessageStart,
            message_id: message_id.into(),
            role: Some("assistant".to_string()),
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }

    /// Set the role.
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.role = Some(role.into());
        self
    }
}

impl Event for TextMessageStartEvent {
    fn event_type(&self) -> EventType {
        EventType::TextMessageStart
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

/// Text message content event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextMessageContentEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Message identifier.
    pub message_id: String,
    /// Content delta.
    pub delta: String,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl TextMessageContentEvent {
    /// Create a new text message content event.
    pub fn new(message_id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self {
            event_type: EventType::TextMessageContent,
            message_id: message_id.into(),
            delta: delta.into(),
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

impl Event for TextMessageContentEvent {
    fn event_type(&self) -> EventType {
        EventType::TextMessageContent
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

/// Text message end event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextMessageEndEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Message identifier.
    pub message_id: String,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl TextMessageEndEvent {
    /// Create a new text message end event.
    pub fn new(message_id: impl Into<String>) -> Self {
        Self {
            event_type: EventType::TextMessageEnd,
            message_id: message_id.into(),
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

impl Event for TextMessageEndEvent {
    fn event_type(&self) -> EventType {
        EventType::TextMessageEnd
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

// ============================================================================
// Thinking Events
// ============================================================================

/// Thinking start event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingStartEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl ThinkingStartEvent {
    /// Create a new thinking start event.
    pub fn new() -> Self {
        Self {
            event_type: EventType::ThinkingStart,
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

impl Default for ThinkingStartEvent {
    fn default() -> Self {
        Self::new()
    }
}

impl Event for ThinkingStartEvent {
    fn event_type(&self) -> EventType {
        EventType::ThinkingStart
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

/// Thinking end event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingEndEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl ThinkingEndEvent {
    /// Create a new thinking end event.
    pub fn new() -> Self {
        Self {
            event_type: EventType::ThinkingEnd,
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

impl Default for ThinkingEndEvent {
    fn default() -> Self {
        Self::new()
    }
}

impl Event for ThinkingEndEvent {
    fn event_type(&self) -> EventType {
        EventType::ThinkingEnd
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

/// Thinking text message start event (nested within thinking).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingTextMessageStartEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Message identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl ThinkingTextMessageStartEvent {
    /// Create a new thinking text message start event.
    pub fn new() -> Self {
        Self {
            event_type: EventType::ThinkingTextMessageStart,
            message_id: None,
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }

    /// Set the message ID.
    pub fn with_message_id(mut self, id: impl Into<String>) -> Self {
        self.message_id = Some(id.into());
        self
    }
}

impl Default for ThinkingTextMessageStartEvent {
    fn default() -> Self {
        Self::new()
    }
}

impl Event for ThinkingTextMessageStartEvent {
    fn event_type(&self) -> EventType {
        EventType::ThinkingTextMessageStart
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

/// Thinking text message content event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingTextMessageContentEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Content delta.
    pub delta: String,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl ThinkingTextMessageContentEvent {
    /// Create a new thinking text message content event.
    pub fn new(delta: impl Into<String>) -> Self {
        Self {
            event_type: EventType::ThinkingTextMessageContent,
            delta: delta.into(),
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

impl Event for ThinkingTextMessageContentEvent {
    fn event_type(&self) -> EventType {
        EventType::ThinkingTextMessageContent
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

/// Thinking text message end event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingTextMessageEndEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl ThinkingTextMessageEndEvent {
    /// Create a new thinking text message end event.
    pub fn new() -> Self {
        Self {
            event_type: EventType::ThinkingTextMessageEnd,
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

impl Default for ThinkingTextMessageEndEvent {
    fn default() -> Self {
        Self::new()
    }
}

impl Event for ThinkingTextMessageEndEvent {
    fn event_type(&self) -> EventType {
        EventType::ThinkingTextMessageEnd
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

// ============================================================================
// Tool Call Events
// ============================================================================

/// Tool call start event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallStartEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Tool call identifier.
    pub tool_call_id: String,
    /// Tool name.
    pub tool_call_name: String,
    /// Parent message ID (the assistant message containing this tool call).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_message_id: Option<String>,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl ToolCallStartEvent {
    /// Create a new tool call start event.
    pub fn new(tool_call_id: impl Into<String>, tool_call_name: impl Into<String>) -> Self {
        Self {
            event_type: EventType::ToolCallStart,
            tool_call_id: tool_call_id.into(),
            tool_call_name: tool_call_name.into(),
            parent_message_id: None,
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }

    /// Set the parent message ID.
    pub fn with_parent_message_id(mut self, id: impl Into<String>) -> Self {
        self.parent_message_id = Some(id.into());
        self
    }
}

impl Event for ToolCallStartEvent {
    fn event_type(&self) -> EventType {
        EventType::ToolCallStart
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

/// Tool call args event (streaming arguments).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallArgsEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Tool call identifier.
    pub tool_call_id: String,
    /// Arguments delta (JSON fragment).
    pub delta: String,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl ToolCallArgsEvent {
    /// Create a new tool call args event.
    pub fn new(tool_call_id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self {
            event_type: EventType::ToolCallArgs,
            tool_call_id: tool_call_id.into(),
            delta: delta.into(),
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

impl Event for ToolCallArgsEvent {
    fn event_type(&self) -> EventType {
        EventType::ToolCallArgs
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

/// Tool call end event (arguments complete).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallEndEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Tool call identifier.
    pub tool_call_id: String,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl ToolCallEndEvent {
    /// Create a new tool call end event.
    pub fn new(tool_call_id: impl Into<String>) -> Self {
        Self {
            event_type: EventType::ToolCallEnd,
            tool_call_id: tool_call_id.into(),
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

impl Event for ToolCallEndEvent {
    fn event_type(&self) -> EventType {
        EventType::ToolCallEnd
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

/// Tool call result event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallResultEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Tool call identifier.
    pub tool_call_id: String,
    /// Result value (if successful).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error message (if failed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl ToolCallResultEvent {
    /// Create a new successful tool call result event.
    pub fn success(tool_call_id: impl Into<String>, result: Value) -> Self {
        Self {
            event_type: EventType::ToolCallResult,
            tool_call_id: tool_call_id.into(),
            result: Some(result),
            error_message: None,
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }

    /// Create a new failed tool call result event.
    pub fn error(tool_call_id: impl Into<String>, error_message: impl Into<String>) -> Self {
        Self {
            event_type: EventType::ToolCallResult,
            tool_call_id: tool_call_id.into(),
            result: None,
            error_message: Some(error_message.into()),
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

impl Event for ToolCallResultEvent {
    fn event_type(&self) -> EventType {
        EventType::ToolCallResult
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

// ============================================================================
// State Events
// ============================================================================

/// State snapshot event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateSnapshotEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// State snapshot data.
    pub snapshot: Value,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl StateSnapshotEvent {
    /// Create a new state snapshot event.
    pub fn new(snapshot: Value) -> Self {
        Self {
            event_type: EventType::StateSnapshot,
            snapshot,
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

impl Event for StateSnapshotEvent {
    fn event_type(&self) -> EventType {
        EventType::StateSnapshot
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

/// State delta event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateDeltaEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// State delta data.
    pub delta: Value,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl StateDeltaEvent {
    /// Create a new state delta event.
    pub fn new(delta: Value) -> Self {
        Self {
            event_type: EventType::StateDelta,
            delta,
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

impl Event for StateDeltaEvent {
    fn event_type(&self) -> EventType {
        EventType::StateDelta
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

/// Messages snapshot event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessagesSnapshotEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Messages array.
    pub messages: Vec<Value>,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl MessagesSnapshotEvent {
    /// Create a new messages snapshot event.
    pub fn new(messages: Vec<Value>) -> Self {
        Self {
            event_type: EventType::MessagesSnapshot,
            messages,
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

impl Event for MessagesSnapshotEvent {
    fn event_type(&self) -> EventType {
        EventType::MessagesSnapshot
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

// ============================================================================
// Custom/Raw Events
// ============================================================================

/// Custom event for application-specific data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Event name/subtype.
    pub name: String,
    /// Event data.
    pub data: Value,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl CustomEvent {
    /// Create a new custom event.
    pub fn new(name: impl Into<String>, data: Value) -> Self {
        Self {
            event_type: EventType::Custom,
            name: name.into(),
            data,
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

impl Event for CustomEvent {
    fn event_type(&self) -> EventType {
        EventType::Custom
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

/// Raw event for passthrough data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RawEvent {
    /// Event type discriminator.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Raw event data.
    pub data: Value,
    /// Timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl RawEvent {
    /// Create a new raw event.
    pub fn new(data: Value) -> Self {
        Self {
            event_type: EventType::Raw,
            data,
            timestamp: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

impl Event for RawEvent {
    fn event_type(&self) -> EventType {
        EventType::Raw
    }

    fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_started_event() {
        let event = RunStartedEvent::new("thread-1", "run-1");
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"RUN_STARTED"#));
        assert!(json.contains(r#""threadId":"thread-1"#));
        assert!(json.contains(r#""runId":"run-1"#));
    }

    #[test]
    fn test_text_message_content_event() {
        let event = TextMessageContentEvent::new("msg-1", "Hello, world!");
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"TEXT_MESSAGE_CONTENT"#));
        assert!(json.contains(r#""messageId":"msg-1"#));
        assert!(json.contains(r#""delta":"Hello, world!"#));
    }

    #[test]
    fn test_tool_call_start_event() {
        let event = ToolCallStartEvent::new("call-1", "get_weather")
            .with_parent_message_id("msg-1");
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"TOOL_CALL_START"#));
        assert!(json.contains(r#""toolCallId":"call-1"#));
        assert!(json.contains(r#""toolCallName":"get_weather"#));
        assert!(json.contains(r#""parentMessageId":"msg-1"#));
    }

    #[test]
    fn test_tool_call_result_success() {
        let event = ToolCallResultEvent::success("call-1", serde_json::json!({"temp": 20}));
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"TOOL_CALL_RESULT"#));
        assert!(json.contains(r#""result":{"temp":20}"#));
        assert!(!json.contains("errorMessage"));
    }

    #[test]
    fn test_tool_call_result_error() {
        let event = ToolCallResultEvent::error("call-1", "City not found");
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""errorMessage":"City not found"#));
        assert!(!json.contains(r#""result"#) || json.contains(r#""result":null"#));
    }

    #[test]
    fn test_thinking_events() {
        let start = ThinkingStartEvent::new();
        let content = ThinkingTextMessageContentEvent::new("Let me think...");
        let end = ThinkingEndEvent::new();

        assert_eq!(start.event_type(), EventType::ThinkingStart);
        assert_eq!(content.event_type(), EventType::ThinkingTextMessageContent);
        assert_eq!(end.event_type(), EventType::ThinkingEnd);
    }

    #[test]
    fn test_encode_event() {
        let event = TextMessageContentEvent::new("msg-1", "Hi");
        let encoded = encode_event(&event);
        assert!(encoded.contains("TEXT_MESSAGE_CONTENT"));
        assert!(encoded.contains("Hi"));
    }

    #[test]
    fn test_event_trait_timestamp() {
        let event = RunStartedEvent::new("t", "r");
        assert!(event.timestamp().is_some());
        assert!(event.timestamp().unwrap() > 0);
    }
}

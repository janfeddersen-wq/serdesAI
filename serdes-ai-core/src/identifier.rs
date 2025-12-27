//! ID generation utilities.
//!
//! This module provides functions for generating unique identifiers
//! for tool calls, runs, and other entities.

use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Generate a unique tool call ID.
///
/// Returns a UUID v4 string in the format used by most LLM providers.
///
/// # Example
///
/// ```rust
/// use serdes_ai_core::identifier::generate_tool_call_id;
///
/// let id = generate_tool_call_id();
/// assert!(id.starts_with("call_"));
/// assert_eq!(id.len(), 37); // "call_" + 32 hex chars
/// ```
#[must_use]
pub fn generate_tool_call_id() -> String {
    format!("call_{}", Uuid::new_v4().simple())
}

/// Generate a unique run ID.
///
/// Returns a UUID v4 string prefixed with "run_".
///
/// # Example
///
/// ```rust
/// use serdes_ai_core::identifier::generate_run_id;
///
/// let id = generate_run_id();
/// assert!(id.starts_with("run_"));
/// ```
#[must_use]
pub fn generate_run_id() -> String {
    format!("run_{}", Uuid::new_v4().simple())
}

/// Generate a unique message ID.
///
/// Returns a UUID v4 string prefixed with "msg_".
#[must_use]
pub fn generate_message_id() -> String {
    format!("msg_{}", Uuid::new_v4().simple())
}

/// Generate a unique conversation ID.
///
/// Returns a UUID v4 string prefixed with "conv_".
#[must_use]
pub fn generate_conversation_id() -> String {
    format!("conv_{}", Uuid::new_v4().simple())
}

/// Generate a raw UUID v4 string (no prefix).
#[must_use]
pub fn generate_uuid() -> String {
    Uuid::new_v4().to_string()
}

/// Generate a short ID suitable for display.
///
/// Returns the first 8 characters of a UUID.
#[must_use]
pub fn generate_short_id() -> String {
    Uuid::new_v4().simple().to_string()[..8].to_string()
}

/// Get the current UTC timestamp.
///
/// # Example
///
/// ```rust
/// use serdes_ai_core::identifier::now_utc;
///
/// let timestamp = now_utc();
/// println!("Current time: {}", timestamp);
/// ```
#[must_use]
pub fn now_utc() -> DateTime<Utc> {
    Utc::now()
}

/// Parse a timestamp from an ISO 8601 string.
///
/// # Errors
///
/// Returns an error if the string is not a valid ISO 8601 timestamp.
pub fn parse_timestamp(s: &str) -> Result<DateTime<Utc>, chrono::ParseError> {
    s.parse()
}

/// Format a timestamp as ISO 8601.
#[must_use]
pub fn format_timestamp(dt: &DateTime<Utc>) -> String {
    dt.to_rfc3339()
}

/// Type-safe wrapper for a tool call ID.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct ToolCallId(String);

impl ToolCallId {
    /// Create a new tool call ID.
    #[must_use]
    pub fn new() -> Self {
        Self(generate_tool_call_id())
    }

    /// Create from an existing string.
    #[must_use]
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get the ID as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for ToolCallId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ToolCallId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for ToolCallId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for ToolCallId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl AsRef<str> for ToolCallId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Type-safe wrapper for a run ID.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct RunId(String);

impl RunId {
    /// Create a new run ID.
    #[must_use]
    pub fn new() -> Self {
        Self(generate_run_id())
    }

    /// Create from an existing string.
    #[must_use]
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get the ID as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for RunId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for RunId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl AsRef<str> for RunId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Type-safe wrapper for a conversation ID.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct ConversationId(String);

impl ConversationId {
    /// Create a new conversation ID.
    #[must_use]
    pub fn new() -> Self {
        Self(generate_conversation_id())
    }

    /// Create from an existing string.
    #[must_use]
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get the ID as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for ConversationId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ConversationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for ConversationId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for ConversationId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl AsRef<str> for ConversationId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_tool_call_id() {
        let id = generate_tool_call_id();
        assert!(id.starts_with("call_"));
        assert_eq!(id.len(), 37);
    }

    #[test]
    fn test_generate_run_id() {
        let id = generate_run_id();
        assert!(id.starts_with("run_"));
    }

    #[test]
    fn test_generate_unique_ids() {
        let id1 = generate_tool_call_id();
        let id2 = generate_tool_call_id();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_tool_call_id_type() {
        let id = ToolCallId::new();
        assert!(id.as_str().starts_with("call_"));

        let from_str = ToolCallId::from_string("call_custom");
        assert_eq!(from_str.as_str(), "call_custom");
    }

    #[test]
    fn test_now_utc() {
        let now = now_utc();
        let formatted = format_timestamp(&now);
        let parsed = parse_timestamp(&formatted).unwrap();
        // Times should be very close (within a second)
        assert!((now - parsed).num_seconds().abs() <= 1);
    }

    #[test]
    fn test_serde_roundtrip() {
        let id = ToolCallId::new();
        let json = serde_json::to_string(&id).unwrap();
        let parsed: ToolCallId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, parsed);
    }
}

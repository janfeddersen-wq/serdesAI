//! Message part types for model responses.
//!
//! This module defines the individual parts that can appear in model responses,
//! including text, tool calls, and thinking/reasoning content.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Text content part.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TextPart {
    /// The text content.
    pub content: String,
    /// Optional unique identifier for this part.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Provider-specific details/metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_details: Option<serde_json::Map<String, serde_json::Value>>,
}

impl TextPart {
    /// Part kind identifier.
    pub const PART_KIND: &'static str = "text";

    /// Create a new text part.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            id: None,
            provider_details: None,
        }
    }

    /// Set the part ID.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set provider-specific details.
    #[must_use]
    pub fn with_provider_details(mut self, details: serde_json::Map<String, serde_json::Value>) -> Self {
        self.provider_details = Some(details);
        self
    }

    /// Get the part kind.
    #[must_use]
    pub fn part_kind(&self) -> &'static str {
        Self::PART_KIND
    }

    /// Check if the content is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Get the content length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.content.len()
    }
}

// Note: Eq is not derived because serde_json::Map doesn't implement Eq
// (due to potential NaN values in JSON numbers). Use PartialEq for comparisons.

impl From<String> for TextPart {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for TextPart {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

/// Tool call arguments - can be either parsed JSON or raw string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolCallArgs {
    /// Parsed JSON arguments.
    Json(serde_json::Value),
    /// Raw string arguments (for streaming or parse failures).
    String(String),
}

/// Attempt to repair common JSON malformations from LLM outputs.
///
/// This function tries various repairs on malformed JSON strings:
/// - Removes trailing commas before `}` or `]`
/// - Fixes unquoted keys: `{foo: "bar"}` -> `{"foo": "bar"}`
/// - Replaces single quotes with double quotes
/// - Closes unclosed braces/brackets
///
/// Returns `Some(Value)` if parsing succeeds (before or after repair),
/// or `None` if the string cannot be salvaged.
fn repair_json(s: &str) -> Option<serde_json::Value> {
    let s = s.trim();

    // Already valid JSON?
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
        return Some(v);
    }

    let mut repaired = s.to_string();

    // 1. Remove trailing commas before } or ]
    repaired = remove_trailing_commas(&repaired);

    // 2. Try to fix unquoted keys: {foo: "bar"} -> {"foo": "bar"}
    repaired = quote_unquoted_keys(&repaired);

    // 3. Replace single quotes with double quotes (only if no double quotes present)
    if repaired.contains('\'') && !repaired.contains('"') {
        repaired = repaired.replace('\'', "\"");
    }

    // 4. Try to close unclosed braces/brackets
    let open_braces = repaired.matches('{').count();
    let close_braces = repaired.matches('}').count();
    if open_braces > close_braces {
        repaired.push_str(&"}".repeat(open_braces - close_braces));
    }

    let open_brackets = repaired.matches('[').count();
    let close_brackets = repaired.matches(']').count();
    if open_brackets > close_brackets {
        repaired.push_str(&"]".repeat(open_brackets - close_brackets));
    }

    // Try parsing the repaired string
    serde_json::from_str(&repaired).ok()
}

/// Remove trailing commas before `}` or `]`.
/// E.g., `{"a": 1,}` -> `{"a": 1}`
fn remove_trailing_commas(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();

    let mut i = 0;
    while i < len {
        let c = chars[i];
        if c == ',' {
            // Look ahead for whitespace followed by } or ]
            let mut j = i + 1;
            while j < len && chars[j].is_whitespace() {
                j += 1;
            }
            if j < len && (chars[j] == '}' || chars[j] == ']') {
                // Skip this comma
                i += 1;
                continue;
            }
        }
        result.push(c);
        i += 1;
    }
    result
}

/// Attempt to quote unquoted keys in JSON-like strings.
/// E.g., `{foo: "bar"}` -> `{"foo": "bar"}`
fn quote_unquoted_keys(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 32);
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();

    let mut i = 0;
    while i < len {
        let c = chars[i];

        // After { or , we might have an unquoted key
        if c == '{' || c == ',' {
            result.push(c);
            i += 1;

            // Skip whitespace
            while i < len && chars[i].is_whitespace() {
                result.push(chars[i]);
                i += 1;
            }

            // Check if we have an unquoted identifier followed by :
            if i < len && is_ident_start(chars[i]) {
                let key_start = i;
                while i < len && is_ident_char(chars[i]) {
                    i += 1;
                }
                let key = &s[key_start..i];

                // Skip whitespace after key
                while i < len && chars[i].is_whitespace() {
                    i += 1;
                }

                // If followed by :, this was an unquoted key
                if i < len && chars[i] == ':' {
                    result.push('"');
                    result.push_str(key);
                    result.push('"');
                } else {
                    // Not a key, just push what we read
                    result.push_str(key);
                }
            }
        } else {
            result.push(c);
            i += 1;
        }
    }
    result
}

/// Check if a character can start an identifier.
#[inline]
fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

/// Check if a character can be part of an identifier.
#[inline]
fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

impl ToolCallArgs {
    /// Create from JSON value.
    #[must_use]
    pub fn json(value: serde_json::Value) -> Self {
        Self::Json(value)
    }

    /// Create from string.
    #[must_use]
    pub fn string(s: impl Into<String>) -> Self {
        Self::String(s.into())
    }

    /// Try to get as JSON object.
    #[must_use]
    pub fn as_object(&self) -> Option<serde_json::Map<String, serde_json::Value>> {
        match self {
            Self::Json(serde_json::Value::Object(obj)) => Some(obj.clone()),
            Self::String(s) => serde_json::from_str::<serde_json::Value>(s)
                .ok()
                .and_then(|value| match value {
                    serde_json::Value::Object(map) => Some(map),
                    _ => None,
                }),
            _ => None,
        }
    }

    /// Convert to JSON value, guaranteeing a valid JSON object.
    ///
    /// This method ensures the result is always a JSON object (dictionary),
    /// which is required by APIs like Anthropic's `tool_use.input` field.
    ///
    /// # Behavior
    ///
    /// - If already a JSON object, returns it as-is
    /// - If a JSON array or primitive, wraps in `{"_value": ...}`
    /// - If a string, attempts to parse and repair malformed JSON
    /// - If all parsing fails, returns `{"_raw": "<original>", "_error": "parse_failed"}`
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Json(v) => {
                if v.is_object() {
                    v.clone()
                } else {
                    // Wrap non-objects
                    serde_json::json!({ "_value": v })
                }
            }
            Self::String(s) => {
                // Try direct parse first
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
                    if v.is_object() {
                        return v;
                    }
                    return serde_json::json!({ "_value": v });
                }

                // Try to repair malformed JSON
                if let Some(v) = repair_json(s) {
                    if v.is_object() {
                        return v;
                    }
                    return serde_json::json!({ "_value": v });
                }

                // All parsing failed - wrap the raw string
                serde_json::json!({
                    "_raw": s,
                    "_error": "parse_failed"
                })
            }
        }
    }

    /// Convert to a JSON object map, guaranteed.
    ///
    /// Similar to [`to_json()`](Self::to_json) but returns the inner `Map` directly.
    /// This is useful when you need to work with the map directly.
    #[must_use]
    pub fn to_json_object(&self) -> serde_json::Map<String, serde_json::Value> {
        match self.to_json() {
            serde_json::Value::Object(map) => map,
            // SAFETY: to_json() guarantees an object, so this is unreachable
            _ => unreachable!("to_json() guarantees an object"),
        }
    }

    /// Convert to JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        match self {
            Self::Json(v) => serde_json::to_string(v),
            Self::String(s) => Ok(s.clone()),
        }
    }

    /// Check if this is valid JSON.
    #[must_use]
    pub fn is_valid_json(&self) -> bool {
        match self {
            Self::Json(_) => true,
            Self::String(s) => serde_json::from_str::<serde_json::Value>(s).is_ok(),
        }
    }
}

impl Default for ToolCallArgs {
    fn default() -> Self {
        Self::Json(serde_json::Value::Object(serde_json::Map::new()))
    }
}

impl From<serde_json::Value> for ToolCallArgs {
    fn from(v: serde_json::Value) -> Self {
        Self::Json(v)
    }
}

impl From<String> for ToolCallArgs {
    fn from(s: String) -> Self {
        // Try to parse as JSON first
        match serde_json::from_str(&s) {
            Ok(v) => Self::Json(v),
            Err(_) => Self::String(s),
        }
    }
}

impl From<&str> for ToolCallArgs {
    fn from(s: &str) -> Self {
        Self::from(s.to_string())
    }
}

/// Tool call part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallPart {
    /// Name of the tool being called.
    pub tool_name: String,
    /// Arguments for the tool call.
    pub args: ToolCallArgs,
    /// Unique identifier for this tool call (provider-assigned).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional unique identifier for this message part.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Provider-specific details/metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_details: Option<serde_json::Map<String, serde_json::Value>>,
}

impl ToolCallPart {
    /// Part kind identifier.
    pub const PART_KIND: &'static str = "tool-call";

    /// Create a new tool call part.
    #[must_use]
    pub fn new(tool_name: impl Into<String>, args: impl Into<ToolCallArgs>) -> Self {
        Self {
            tool_name: tool_name.into(),
            args: args.into(),
            tool_call_id: None,
            id: None,
            provider_details: None,
        }
    }

    /// Get the part kind.
    #[must_use]
    pub fn part_kind(&self) -> &'static str {
        Self::PART_KIND
    }

    /// Set the tool call ID (provider-assigned identifier for the tool call).
    #[must_use]
    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    /// Alias for `with_tool_call_id` - kept for backward compatibility.
    #[must_use]
    #[deprecated(since = "0.2.0", note = "Use with_tool_call_id() instead for clarity")]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    /// Set the part ID (unique identifier for this message part).
    #[must_use]
    pub fn with_part_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set provider-specific details.
    #[must_use]
    pub fn with_provider_details(mut self, details: serde_json::Map<String, serde_json::Value>) -> Self {
        self.provider_details = Some(details);
        self
    }

    /// Get arguments as a dictionary/object.
    #[must_use]
    pub fn args_as_dict(&self) -> serde_json::Value {
        self.args.to_json()
    }

    /// Get arguments as JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn args_as_json_str(&self) -> Result<String, serde_json::Error> {
        self.args.to_json_string()
    }

    /// Try to deserialize arguments into a typed struct.
    pub fn parse_args<T: for<'de> Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        let json = self.args.to_json();
        serde_json::from_value(json)
    }
}

/// Thinking/reasoning content part.
///
/// Used for models that support "thinking" or chain-of-thought reasoning,
/// like Claude's extended thinking feature.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ThinkingPart {
    /// The thinking content.
    pub content: String,
    /// Optional unique identifier for this part.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Optional signature for verification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Provider name that generated this thinking content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_name: Option<String>,
    /// Provider-specific details/metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_details: Option<serde_json::Map<String, serde_json::Value>>,
}

impl ThinkingPart {
    /// Part kind identifier.
    pub const PART_KIND: &'static str = "thinking";

    /// ID used for Anthropic redacted thinking blocks.
    pub const REDACTED_THINKING_ID: &'static str = "redacted_thinking";

    /// ID used for Bedrock redacted content blocks.
    pub const REDACTED_CONTENT_ID: &'static str = "redacted_content";

    /// Create a new thinking part.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            id: None,
            signature: None,
            provider_name: None,
            provider_details: None,
        }
    }

    /// Create a redacted thinking part.
    ///
    /// Redacted thinking blocks contain encrypted/hidden content. The signature
    /// must be preserved and sent back to the API in subsequent requests.
    ///
    /// # Arguments
    /// * `signature` - The encrypted/encoded data from the provider
    /// * `provider_name` - The provider that generated this (e.g., "anthropic", "bedrock")
    #[must_use]
    pub fn redacted(signature: impl Into<String>, provider_name: impl Into<String>) -> Self {
        let provider = provider_name.into();
        let id = if provider.contains("bedrock") {
            Self::REDACTED_CONTENT_ID
        } else {
            Self::REDACTED_THINKING_ID
        };
        Self {
            content: String::new(),
            id: Some(id.to_string()),
            signature: Some(signature.into()),
            provider_name: Some(provider),
            provider_details: None,
        }
    }

    /// Create a redacted thinking part with a specific ID.
    #[must_use]
    pub fn redacted_with_id(
        id: impl Into<String>,
        signature: impl Into<String>,
        provider_name: impl Into<String>,
    ) -> Self {
        Self {
            content: String::new(),
            id: Some(id.into()),
            signature: Some(signature.into()),
            provider_name: Some(provider_name.into()),
            provider_details: None,
        }
    }

    /// Get the part kind.
    #[must_use]
    pub fn part_kind(&self) -> &'static str {
        Self::PART_KIND
    }

    /// Set the part ID.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the signature.
    #[must_use]
    pub fn with_signature(mut self, signature: impl Into<String>) -> Self {
        self.signature = Some(signature.into());
        self
    }

    /// Set the provider name.
    #[must_use]
    pub fn with_provider_name(mut self, name: impl Into<String>) -> Self {
        self.provider_name = Some(name.into());
        self
    }

    /// Set provider-specific details.
    #[must_use]
    pub fn with_provider_details(mut self, details: serde_json::Map<String, serde_json::Value>) -> Self {
        self.provider_details = Some(details);
        self
    }

    /// Check if the content is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Check if this is a redacted thinking block.
    ///
    /// Redacted blocks have their content hidden/encrypted by the provider.
    /// The `signature` field contains the encrypted data that must be
    /// round-tripped back to the API.
    #[must_use]
    pub fn is_redacted(&self) -> bool {
        self.id
            .as_ref()
            .map(|id| id.starts_with("redacted"))
            .unwrap_or(false)
    }

    /// Get the signature if this is a redacted block.
    #[must_use]
    pub fn redacted_signature(&self) -> Option<&str> {
        if self.is_redacted() {
            self.signature.as_deref()
        } else {
            None
        }
    }
}

/// Binary content container for file data.
///
/// Encapsulates raw binary data along with its MIME type for type-safe
/// handling of file content like images, audio, or documents.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BinaryContent {
    /// The raw binary data.
    #[serde(with = "base64_serde")]
    pub data: Vec<u8>,
    /// The MIME type of the content (e.g., "image/png", "audio/wav").
    pub media_type: String,
}

impl BinaryContent {
    /// Create new binary content.
    #[must_use]
    pub fn new(data: Vec<u8>, media_type: impl Into<String>) -> Self {
        Self {
            data,
            media_type: media_type.into(),
        }
    }

    /// Check if the content is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the length of the binary data.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

/// Custom serde module for base64 encoding/decoding of binary data.
mod base64_serde {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(data: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&STANDARD.encode(data))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        STANDARD.decode(&s).map_err(serde::de::Error::custom)
    }
}

/// A file response from a model.
///
/// Represents file content generated by models, such as images from
/// image generation APIs. This is the Rust equivalent of pydantic-ai's
/// `FilePart` type.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FilePart {
    /// The binary content of the file.
    pub content: BinaryContent,
    /// Optional unique identifier for this part.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Provider name that generated this file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_name: Option<String>,
    /// Provider-specific details/metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_details: Option<serde_json::Map<String, serde_json::Value>>,
}

impl FilePart {
    /// Part kind identifier.
    pub const PART_KIND: &'static str = "file";

    /// Create a new file part.
    #[must_use]
    pub fn new(content: BinaryContent) -> Self {
        Self {
            content,
            id: None,
            provider_name: None,
            provider_details: None,
        }
    }

    /// Create a new file part from raw data and media type.
    #[must_use]
    pub fn from_bytes(data: Vec<u8>, media_type: impl Into<String>) -> Self {
        Self::new(BinaryContent::new(data, media_type))
    }

    /// Get the part kind.
    #[must_use]
    pub fn part_kind(&self) -> &'static str {
        Self::PART_KIND
    }

    /// Set the part ID.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the provider name.
    #[must_use]
    pub fn with_provider_name(mut self, name: impl Into<String>) -> Self {
        self.provider_name = Some(name.into());
        self
    }

    /// Set provider-specific details.
    #[must_use]
    pub fn with_provider_details(mut self, details: serde_json::Map<String, serde_json::Value>) -> Self {
        self.provider_details = Some(details);
        self
    }

    /// Check if the file has content.
    #[must_use]
    pub fn has_content(&self) -> bool {
        !self.content.is_empty()
    }

    /// Get the media type of the file content.
    #[must_use]
    pub fn media_type(&self) -> &str {
        &self.content.media_type
    }

    /// Get the raw binary data.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.content.data
    }

    /// Get the size of the file in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.content.len()
    }
}

/// A builtin tool call from a model (web search, code execution, etc.).
///
/// Builtin tools are provider-native capabilities like web search, code execution,
/// and file search. They are distinct from user-defined function tools and require
/// special handling for UI rendering and history management.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuiltinToolCallPart {
    /// Name of the builtin tool being called (e.g., "web_search", "code_execution").
    pub tool_name: String,
    /// Arguments for the tool call.
    pub args: ToolCallArgs,
    /// Unique identifier for this tool call (provider-assigned).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional unique identifier for this message part.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Provider-specific details/metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_details: Option<serde_json::Map<String, serde_json::Value>>,
}

impl BuiltinToolCallPart {
    /// Part kind identifier.
    pub const PART_KIND: &'static str = "builtin-tool-call";

    /// Create a new builtin tool call part.
    #[must_use]
    pub fn new(tool_name: impl Into<String>, args: impl Into<ToolCallArgs>) -> Self {
        Self {
            tool_name: tool_name.into(),
            args: args.into(),
            tool_call_id: None,
            id: None,
            provider_details: None,
        }
    }

    /// Get the part kind.
    #[must_use]
    pub fn part_kind(&self) -> &'static str {
        Self::PART_KIND
    }

    /// Set the tool call ID (provider-assigned identifier for the tool call).
    #[must_use]
    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    /// Set the part ID (unique identifier for this message part).
    #[must_use]
    pub fn with_part_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set provider-specific details.
    #[must_use]
    pub fn with_provider_details(
        mut self,
        details: serde_json::Map<String, serde_json::Value>,
    ) -> Self {
        self.provider_details = Some(details);
        self
    }

    /// Get arguments as a dictionary/object.
    #[must_use]
    pub fn args_as_dict(&self) -> serde_json::Value {
        self.args.to_json()
    }

    /// Get arguments as JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn args_as_json_str(&self) -> Result<String, serde_json::Error> {
        self.args.to_json_string()
    }

    /// Try to deserialize arguments into a typed struct.
    pub fn parse_args<T: for<'de> Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        let json = self.args.to_json();
        serde_json::from_value(json)
    }
}

/// A single web search result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebSearchResult {
    /// The title of the search result.
    pub title: String,
    /// The URL of the search result.
    pub url: String,
    /// A snippet/summary of the content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    /// The full content if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl WebSearchResult {
    /// Create a new web search result.
    #[must_use]
    pub fn new(title: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            url: url.into(),
            snippet: None,
            content: None,
        }
    }

    /// Set the snippet.
    #[must_use]
    pub fn with_snippet(mut self, snippet: impl Into<String>) -> Self {
        self.snippet = Some(snippet.into());
        self
    }

    /// Set the full content.
    #[must_use]
    pub fn with_content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }
}

/// Web search results from a builtin search tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebSearchResults {
    /// The search query that was executed.
    pub query: String,
    /// The list of search results.
    pub results: Vec<WebSearchResult>,
    /// Total number of results (may be more than returned).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_results: Option<u64>,
}

impl WebSearchResults {
    /// Create new web search results.
    #[must_use]
    pub fn new(query: impl Into<String>, results: Vec<WebSearchResult>) -> Self {
        Self {
            query: query.into(),
            results,
            total_results: None,
        }
    }

    /// Set the total results count.
    #[must_use]
    pub fn with_total_results(mut self, total: u64) -> Self {
        self.total_results = Some(total);
        self
    }

    /// Check if there are any results.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get the number of results.
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.len()
    }
}

/// Result from code execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeExecutionResult {
    /// The code that was executed.
    pub code: String,
    /// Standard output from the execution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout: Option<String>,
    /// Standard error from the execution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr: Option<String>,
    /// Exit code from the execution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// Any images/files generated by the code.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub output_files: Vec<BinaryContent>,
    /// Error message if execution failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl CodeExecutionResult {
    /// Create a new code execution result.
    #[must_use]
    pub fn new(code: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            stdout: None,
            stderr: None,
            exit_code: None,
            output_files: Vec::new(),
            error: None,
        }
    }

    /// Set the stdout.
    #[must_use]
    pub fn with_stdout(mut self, stdout: impl Into<String>) -> Self {
        self.stdout = Some(stdout.into());
        self
    }

    /// Set the stderr.
    #[must_use]
    pub fn with_stderr(mut self, stderr: impl Into<String>) -> Self {
        self.stderr = Some(stderr.into());
        self
    }

    /// Set the exit code.
    #[must_use]
    pub fn with_exit_code(mut self, code: i32) -> Self {
        self.exit_code = Some(code);
        self
    }

    /// Add an output file.
    #[must_use]
    pub fn with_output_file(mut self, file: BinaryContent) -> Self {
        self.output_files.push(file);
        self
    }

    /// Set the error message.
    #[must_use]
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self
    }

    /// Check if execution succeeded (exit_code == 0 and no error).
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.error.is_none() && self.exit_code.map_or(true, |c| c == 0)
    }
}

/// A single file search result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileSearchResult {
    /// The file name or path.
    pub file_name: String,
    /// The matched content or snippet.
    pub content: String,
    /// Relevance score if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// File metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

impl FileSearchResult {
    /// Create a new file search result.
    #[must_use]
    pub fn new(file_name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            file_name: file_name.into(),
            content: content.into(),
            score: None,
            metadata: None,
        }
    }

    /// Set the relevance score.
    #[must_use]
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = Some(score);
        self
    }

    /// Set file metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: serde_json::Map<String, serde_json::Value>) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// File search results from a builtin file search tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileSearchResults {
    /// The search query that was executed.
    pub query: String,
    /// The list of file search results.
    pub results: Vec<FileSearchResult>,
}

impl FileSearchResults {
    /// Create new file search results.
    #[must_use]
    pub fn new(query: impl Into<String>, results: Vec<FileSearchResult>) -> Self {
        Self {
            query: query.into(),
            results,
        }
    }

    /// Check if there are any results.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get the number of results.
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.len()
    }
}

/// Content returned from builtin tools.
///
/// This enum represents the different types of structured content that can be
/// returned from builtin tools like web search, code execution, and file search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BuiltinToolReturnContent {
    /// Web search results.
    WebSearch(WebSearchResults),
    /// Code execution result.
    CodeExecution(CodeExecutionResult),
    /// File search results.
    FileSearch(FileSearchResults),
    /// Generic/other structured content for provider-specific results.
    Other {
        /// The type identifier for this content.
        kind: String,
        /// The content data.
        data: serde_json::Value,
    },
}

impl BuiltinToolReturnContent {
    /// Create web search content.
    #[must_use]
    pub fn web_search(results: WebSearchResults) -> Self {
        Self::WebSearch(results)
    }

    /// Create code execution content.
    #[must_use]
    pub fn code_execution(result: CodeExecutionResult) -> Self {
        Self::CodeExecution(result)
    }

    /// Create file search content.
    #[must_use]
    pub fn file_search(results: FileSearchResults) -> Self {
        Self::FileSearch(results)
    }

    /// Create other/generic content.
    #[must_use]
    pub fn other(kind: impl Into<String>, data: serde_json::Value) -> Self {
        Self::Other {
            kind: kind.into(),
            data,
        }
    }

    /// Get the content type name.
    #[must_use]
    pub fn content_type(&self) -> &str {
        match self {
            Self::WebSearch(_) => "web_search",
            Self::CodeExecution(_) => "code_execution",
            Self::FileSearch(_) => "file_search",
            Self::Other { kind, .. } => kind,
        }
    }
}

/// Return from a builtin tool with structured content.
///
/// This part represents the result of a builtin tool execution, containing
/// structured content appropriate for the tool type (search results, code output, etc.).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuiltinToolReturnPart {
    /// Name of the builtin tool that was called.
    pub tool_name: String,
    /// The structured content returned by the tool.
    pub content: BuiltinToolReturnContent,
    /// ID of the tool call this is responding to.
    pub tool_call_id: String,
    /// When this return was generated.
    pub timestamp: DateTime<Utc>,
    /// Optional unique identifier for this part.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Provider-specific details/metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_details: Option<serde_json::Map<String, serde_json::Value>>,
}

impl BuiltinToolReturnPart {
    /// Part kind identifier.
    pub const PART_KIND: &'static str = "builtin-tool-return";

    /// Create a new builtin tool return part.
    #[must_use]
    pub fn new(
        tool_name: impl Into<String>,
        content: BuiltinToolReturnContent,
        tool_call_id: impl Into<String>,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            content,
            tool_call_id: tool_call_id.into(),
            timestamp: Utc::now(),
            id: None,
            provider_details: None,
        }
    }

    /// Get the part kind.
    #[must_use]
    pub fn part_kind(&self) -> &'static str {
        Self::PART_KIND
    }

    /// Set the timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Set the part ID.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set provider-specific details.
    #[must_use]
    pub fn with_provider_details(
        mut self,
        details: serde_json::Map<String, serde_json::Value>,
    ) -> Self {
        self.provider_details = Some(details);
        self
    }

    /// Get the content type of the return.
    #[must_use]
    pub fn content_type(&self) -> &str {
        self.content.content_type()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_part() {
        let part = TextPart::new("Hello, world!");
        assert_eq!(part.content, "Hello, world!");
        assert_eq!(part.part_kind(), "text");
        assert!(!part.is_empty());
        assert_eq!(part.len(), 13);
        assert!(part.id.is_none());
        assert!(part.provider_details.is_none());
    }

    #[test]
    fn test_text_part_with_builders() {
        let mut details = serde_json::Map::new();
        details.insert("model".to_string(), serde_json::json!("gpt-4"));
        
        let part = TextPart::new("Hello!")
            .with_id("part-123")
            .with_provider_details(details.clone());
        
        assert_eq!(part.id, Some("part-123".to_string()));
        assert_eq!(part.provider_details, Some(details));
    }

    #[test]
    fn test_tool_call_args_from_json() {
        let args = ToolCallArgs::json(serde_json::json!({"location": "NYC"}));
        assert!(args.is_valid_json());
        assert_eq!(args.to_json_string().unwrap(), r#"{"location":"NYC"}"#);
    }

    #[test]
    fn test_tool_call_args_from_string() {
        let args: ToolCallArgs = r#"{"x": 1}"#.into();
        assert!(args.is_valid_json());
        // Should parse into Json variant
        if let ToolCallArgs::Json(v) = &args {
            assert_eq!(v["x"], 1);
        } else {
            panic!("Expected Json variant");
        }
    }

    // ==================== JSON Repair Tests ====================

    #[test]
    fn test_to_json_always_returns_object() {
        // Valid JSON object should pass through
        let args = ToolCallArgs::json(serde_json::json!({"foo": "bar"}));
        let result = args.to_json();
        assert!(result.is_object());
        assert_eq!(result["foo"], "bar");
    }

    #[test]
    fn test_to_json_wraps_array() {
        // Arrays should be wrapped in {"_value": ...}
        let args = ToolCallArgs::json(serde_json::json!([1, 2, 3]));
        let result = args.to_json();
        assert!(result.is_object());
        assert_eq!(result["_value"], serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn test_to_json_wraps_primitive() {
        // Primitives should be wrapped in {"_value": ...}
        let args = ToolCallArgs::json(serde_json::json!(42));
        let result = args.to_json();
        assert!(result.is_object());
        assert_eq!(result["_value"], 42);

        let args = ToolCallArgs::json(serde_json::json!("hello"));
        let result = args.to_json();
        assert!(result.is_object());
        assert_eq!(result["_value"], "hello");
    }

    #[test]
    fn test_to_json_repairs_trailing_comma() {
        // Trailing commas should be repaired
        let args = ToolCallArgs::string(r#"{"a": 1,}"#.to_string());
        let result = args.to_json();
        assert!(result.is_object());
        assert_eq!(result["a"], 1);
    }

    #[test]
    fn test_to_json_repairs_unquoted_keys() {
        // Unquoted keys should be repaired
        let args = ToolCallArgs::string(r#"{foo: "bar"}"#.to_string());
        let result = args.to_json();
        assert!(result.is_object());
        assert_eq!(result["foo"], "bar");
    }

    #[test]
    fn test_to_json_repairs_single_quotes() {
        // Single quotes should be converted to double quotes
        let args = ToolCallArgs::string("{'a': 'b'}".to_string());
        let result = args.to_json();
        assert!(result.is_object());
        assert_eq!(result["a"], "b");
    }

    #[test]
    fn test_to_json_repairs_unclosed_braces() {
        // Unclosed braces should be closed
        let args = ToolCallArgs::string(r#"{"x": 1"#.to_string());
        let result = args.to_json();
        assert!(result.is_object());
        assert_eq!(result["x"], 1);
    }

    #[test]
    fn test_to_json_handles_completely_invalid() {
        // Completely invalid JSON should return error object
        let args = ToolCallArgs::string("this is not json at all".to_string());
        let result = args.to_json();
        assert!(result.is_object());
        assert_eq!(result["_error"], "parse_failed");
        assert_eq!(result["_raw"], "this is not json at all");
    }

    #[test]
    fn test_to_json_object_returns_map() {
        let args = ToolCallArgs::json(serde_json::json!({"key": "value"}));
        let map = args.to_json_object();
        assert_eq!(map.get("key"), Some(&serde_json::json!("value")));
    }

    #[test]
    fn test_repair_json_valid_passthrough() {
        // Already valid JSON should pass through
        let result = repair_json(r#"{"valid": true}"#);
        assert!(result.is_some());
        assert_eq!(result.unwrap()["valid"], true);
    }

    #[test]
    fn test_repair_json_nested_trailing_comma() {
        let result = repair_json(r#"{"outer": {"inner": 1,},}"#);
        assert!(result.is_some());
        let v = result.unwrap();
        assert_eq!(v["outer"]["inner"], 1);
    }

    #[test]
    fn test_repair_json_array_trailing_comma() {
        let result = repair_json(r#"[1, 2, 3,]"#);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn test_repair_json_multiple_unquoted_keys() {
        let result = repair_json(r#"{foo: 1, bar: 2}"#);
        assert!(result.is_some());
        let v = result.unwrap();
        assert_eq!(v["foo"], 1);
        assert_eq!(v["bar"], 2);
    }

    #[test]
    fn test_remove_trailing_commas_helper() {
        assert_eq!(remove_trailing_commas("{\"a\": 1,}"), "{\"a\": 1}");
        assert_eq!(remove_trailing_commas("[1, 2,]"), "[1, 2]");
        assert_eq!(remove_trailing_commas("{\"a\": 1,  }"), "{\"a\": 1  }");
    }

    #[test]
    fn test_quote_unquoted_keys_helper() {
        assert_eq!(quote_unquoted_keys("{foo: 1}"), "{\"foo\": 1}");
        assert_eq!(quote_unquoted_keys("{foo: 1, bar: 2}"), "{\"foo\": 1, \"bar\": 2}");
    }

    // ==================== End JSON Repair Tests ====================

    #[test]
    fn test_tool_call_part() {
        let part = ToolCallPart::new("get_weather", serde_json::json!({"city": "NYC"}))
            .with_tool_call_id("call_123");
        assert_eq!(part.tool_name, "get_weather");
        assert_eq!(part.tool_call_id, Some("call_123".to_string()));
        assert_eq!(part.part_kind(), "tool-call");
        assert!(part.id.is_none());
        assert!(part.provider_details.is_none());
    }

    #[test]
    fn test_tool_call_part_with_all_fields() {
        let mut details = serde_json::Map::new();
        details.insert("temperature".to_string(), serde_json::json!(0.7));
        
        let part = ToolCallPart::new("search", serde_json::json!({"query": "rust"}))
            .with_tool_call_id("call_456")
            .with_part_id("part-789")
            .with_provider_details(details.clone());
        
        assert_eq!(part.tool_call_id, Some("call_456".to_string()));
        assert_eq!(part.id, Some("part-789".to_string()));
        assert_eq!(part.provider_details, Some(details));
    }

    #[test]
    #[allow(deprecated)]
    fn test_tool_call_part_deprecated_with_id() {
        // Test backward compatibility with deprecated with_id()
        let part = ToolCallPart::new("test", serde_json::json!({}))
            .with_id("call_compat");
        assert_eq!(part.tool_call_id, Some("call_compat".to_string()));
    }

    #[test]
    fn test_tool_call_parse_args() {
        #[derive(Deserialize, PartialEq, Debug)]
        struct WeatherArgs {
            city: String,
        }

        let part = ToolCallPart::new("get_weather", serde_json::json!({"city": "NYC"}));
        let args: WeatherArgs = part.parse_args().unwrap();
        assert_eq!(args.city, "NYC");
    }

    #[test]
    fn test_thinking_part() {
        let part = ThinkingPart::new("Let me think about this...")
            .with_signature("sig123");
        assert_eq!(part.content, "Let me think about this...");
        assert_eq!(part.signature, Some("sig123".to_string()));
        assert!(part.id.is_none());
        assert!(part.provider_name.is_none());
        assert!(part.provider_details.is_none());
    }

    #[test]
    fn test_thinking_part_with_all_fields() {
        let mut details = serde_json::Map::new();
        details.insert("thinking_tokens".to_string(), serde_json::json!(1500));
        
        let part = ThinkingPart::new("Deep thoughts...")
            .with_id("think-001")
            .with_signature("sig456")
            .with_provider_name("anthropic")
            .with_provider_details(details.clone());
        
        assert_eq!(part.id, Some("think-001".to_string()));
        assert_eq!(part.signature, Some("sig456".to_string()));
        assert_eq!(part.provider_name, Some("anthropic".to_string()));
        assert_eq!(part.provider_details, Some(details));
    }

    #[test]
    fn test_thinking_part_redacted_anthropic() {
        let part = ThinkingPart::redacted("encrypted_signature_data", "anthropic");
        
        assert!(part.is_redacted());
        assert!(part.content.is_empty());
        assert_eq!(part.id, Some("redacted_thinking".to_string()));
        assert_eq!(part.signature, Some("encrypted_signature_data".to_string()));
        assert_eq!(part.provider_name, Some("anthropic".to_string()));
        assert_eq!(part.redacted_signature(), Some("encrypted_signature_data"));
    }

    #[test]
    fn test_thinking_part_redacted_bedrock() {
        let part = ThinkingPart::redacted("base64_encoded_content", "aws-bedrock");
        
        assert!(part.is_redacted());
        assert!(part.content.is_empty());
        assert_eq!(part.id, Some("redacted_content".to_string()));
        assert_eq!(part.signature, Some("base64_encoded_content".to_string()));
        assert_eq!(part.provider_name, Some("aws-bedrock".to_string()));
    }

    #[test]
    fn test_thinking_part_redacted_with_custom_id() {
        let part = ThinkingPart::redacted_with_id(
            "redacted_custom_type",
            "my_signature",
            "custom-provider"
        );
        
        assert!(part.is_redacted());
        assert_eq!(part.id, Some("redacted_custom_type".to_string()));
        assert_eq!(part.signature, Some("my_signature".to_string()));
    }

    #[test]
    fn test_thinking_part_not_redacted() {
        let part = ThinkingPart::new("Regular thinking content")
            .with_id("think-123");
        
        assert!(!part.is_redacted());
        assert_eq!(part.redacted_signature(), None);
    }

    #[test]
    fn test_thinking_part_redacted_constants() {
        assert_eq!(ThinkingPart::REDACTED_THINKING_ID, "redacted_thinking");
        assert_eq!(ThinkingPart::REDACTED_CONTENT_ID, "redacted_content");
    }

    #[test]
    fn test_serde_roundtrip_redacted_thinking() {
        let part = ThinkingPart::redacted("encrypted_data_here", "anthropic");
        
        let json = serde_json::to_string(&part).unwrap();
        let parsed: ThinkingPart = serde_json::from_str(&json).unwrap();
        
        assert_eq!(part, parsed);
        assert!(parsed.is_redacted());
        assert_eq!(parsed.redacted_signature(), Some("encrypted_data_here"));
    }

    #[test]
    fn test_serde_roundtrip_tool_call() {
        let mut details = serde_json::Map::new();
        details.insert("key".to_string(), serde_json::json!("value"));
        
        let part = ToolCallPart::new("test", serde_json::json!({"a": 1}))
            .with_tool_call_id("call_1")
            .with_part_id("part_1")
            .with_provider_details(details);
        
        let json = serde_json::to_string(&part).unwrap();
        let parsed: ToolCallPart = serde_json::from_str(&json).unwrap();
        assert_eq!(part, parsed);
    }

    #[test]
    fn test_serde_roundtrip_text() {
        let mut details = serde_json::Map::new();
        details.insert("tokens".to_string(), serde_json::json!(42));
        
        let part = TextPart::new("Hello")
            .with_id("text-1")
            .with_provider_details(details);
        
        let json = serde_json::to_string(&part).unwrap();
        let parsed: TextPart = serde_json::from_str(&json).unwrap();
        assert_eq!(part, parsed);
    }

    #[test]
    fn test_serde_roundtrip_thinking() {
        let mut details = serde_json::Map::new();
        details.insert("budget".to_string(), serde_json::json!(10000));
        
        let part = ThinkingPart::new("Thinking...")
            .with_id("think-1")
            .with_signature("sig")
            .with_provider_name("anthropic")
            .with_provider_details(details);
        
        let json = serde_json::to_string(&part).unwrap();
        let parsed: ThinkingPart = serde_json::from_str(&json).unwrap();
        assert_eq!(part, parsed);
    }

    #[test]
    fn test_serde_skip_none_fields() {
        // Verify that None fields are not serialized
        let part = TextPart::new("Hello");
        let json = serde_json::to_string(&part).unwrap();
        
        // Should only have "content" field, not id or provider_details
        assert!(json.contains("content"));
        assert!(!json.contains("id"));
        assert!(!json.contains("provider_details"));
    }

    #[test]
    fn test_backward_compat_deserialization() {
        // Verify we can deserialize old JSON without the new fields
        let old_json = r#"{"content":"Hello, world!"}"#;
        let part: TextPart = serde_json::from_str(old_json).unwrap();
        assert_eq!(part.content, "Hello, world!");
        assert!(part.id.is_none());
        assert!(part.provider_details.is_none());
    }

    #[test]
    fn test_binary_content() {
        let data = vec![0x89, 0x50, 0x4E, 0x47]; // PNG magic bytes
        let content = BinaryContent::new(data.clone(), "image/png");
        
        assert_eq!(content.data, data);
        assert_eq!(content.media_type, "image/png");
        assert!(!content.is_empty());
        assert_eq!(content.len(), 4);
    }

    #[test]
    fn test_binary_content_empty() {
        let content = BinaryContent::default();
        assert!(content.is_empty());
        assert_eq!(content.len(), 0);
        assert!(content.media_type.is_empty());
    }

    #[test]
    fn test_file_part() {
        let data = vec![0xFF, 0xD8, 0xFF]; // JPEG magic bytes
        let part = FilePart::from_bytes(data.clone(), "image/jpeg");
        
        assert_eq!(part.content.data, data);
        assert_eq!(part.media_type(), "image/jpeg");
        assert_eq!(part.part_kind(), "file");
        assert!(part.has_content());
        assert_eq!(part.size(), 3);
        assert!(part.id.is_none());
        assert!(part.provider_name.is_none());
        assert!(part.provider_details.is_none());
    }

    #[test]
    fn test_file_part_with_builders() {
        let mut details = serde_json::Map::new();
        details.insert("model".to_string(), serde_json::json!("dall-e-3"));
        details.insert("revised_prompt".to_string(), serde_json::json!("A cute puppy"));
        
        let data = vec![0x89, 0x50, 0x4E, 0x47];
        let part = FilePart::from_bytes(data, "image/png")
            .with_id("file-123")
            .with_provider_name("openai")
            .with_provider_details(details.clone());
        
        assert_eq!(part.id, Some("file-123".to_string()));
        assert_eq!(part.provider_name, Some("openai".to_string()));
        assert_eq!(part.provider_details, Some(details));
    }

    #[test]
    fn test_file_part_empty_content() {
        let part = FilePart::from_bytes(vec![], "application/octet-stream");
        assert!(!part.has_content());
        assert_eq!(part.size(), 0);
    }

    #[test]
    fn test_serde_roundtrip_binary_content() {
        let data = vec![0x00, 0x01, 0x02, 0xFF, 0xFE, 0xFD];
        let content = BinaryContent::new(data.clone(), "application/octet-stream");
        
        let json = serde_json::to_string(&content).unwrap();
        let parsed: BinaryContent = serde_json::from_str(&json).unwrap();
        
        assert_eq!(content, parsed);
        assert_eq!(parsed.data, data);
    }

    #[test]
    fn test_serde_roundtrip_file_part() {
        let mut details = serde_json::Map::new();
        details.insert("quality".to_string(), serde_json::json!("hd"));
        
        let data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let part = FilePart::from_bytes(data.clone(), "image/png")
            .with_id("img-001")
            .with_provider_name("openai")
            .with_provider_details(details);
        
        let json = serde_json::to_string(&part).unwrap();
        let parsed: FilePart = serde_json::from_str(&json).unwrap();
        
        assert_eq!(part, parsed);
        assert_eq!(parsed.data(), &data);
    }

    #[test]
    fn test_file_part_serde_skip_none() {
        // Verify that None fields are not serialized
        let part = FilePart::from_bytes(vec![0x00], "application/octet-stream");
        let json = serde_json::to_string(&part).unwrap();
        
        assert!(json.contains("content"));
        assert!(!json.contains("id"));
        assert!(!json.contains("provider_name"));
        assert!(!json.contains("provider_details"));
    }

    #[test]
    fn test_builtin_tool_call_part() {
        let part = BuiltinToolCallPart::new("web_search", serde_json::json!({"query": "rust"}))
            .with_tool_call_id("call_123");
        
        assert_eq!(part.tool_name, "web_search");
        assert_eq!(part.tool_call_id, Some("call_123".to_string()));
        assert_eq!(part.part_kind(), "builtin-tool-call");
    }

    #[test]
    fn test_builtin_tool_call_part_with_all_fields() {
        let mut details = serde_json::Map::new();
        details.insert("provider".to_string(), serde_json::json!("google"));
        
        let part = BuiltinToolCallPart::new("code_execution", serde_json::json!({"code": "print(1)"}))
            .with_tool_call_id("call_456")
            .with_part_id("part-789")
            .with_provider_details(details.clone());
        
        assert_eq!(part.tool_call_id, Some("call_456".to_string()));
        assert_eq!(part.id, Some("part-789".to_string()));
        assert_eq!(part.provider_details, Some(details));
    }

    #[test]
    fn test_web_search_result() {
        let result = WebSearchResult::new("Rust Programming", "https://rust-lang.org")
            .with_snippet("A systems programming language")
            .with_content("Full article content...");
        
        assert_eq!(result.title, "Rust Programming");
        assert_eq!(result.url, "https://rust-lang.org");
        assert_eq!(result.snippet, Some("A systems programming language".to_string()));
        assert!(result.content.is_some());
    }

    #[test]
    fn test_web_search_results() {
        let results = WebSearchResults::new(
            "rust programming",
            vec![
                WebSearchResult::new("Rust", "https://rust-lang.org"),
                WebSearchResult::new("Crates.io", "https://crates.io"),
            ],
        ).with_total_results(1000);
        
        assert_eq!(results.query, "rust programming");
        assert_eq!(results.len(), 2);
        assert!(!results.is_empty());
        assert_eq!(results.total_results, Some(1000));
    }

    #[test]
    fn test_code_execution_result() {
        let result = CodeExecutionResult::new("print('hello')")
            .with_stdout("hello\n")
            .with_exit_code(0);
        
        assert_eq!(result.code, "print('hello')");
        assert_eq!(result.stdout, Some("hello\n".to_string()));
        assert!(result.is_success());
    }

    #[test]
    fn test_code_execution_result_with_error() {
        let result = CodeExecutionResult::new("invalid code")
            .with_stderr("SyntaxError")
            .with_exit_code(1)
            .with_error("Compilation failed");
        
        assert!(!result.is_success());
        assert_eq!(result.error, Some("Compilation failed".to_string()));
    }

    #[test]
    fn test_code_execution_result_with_output_file() {
        let image = BinaryContent::new(vec![0x89, 0x50, 0x4E, 0x47], "image/png");
        let result = CodeExecutionResult::new("plot()")
            .with_stdout("Plot saved")
            .with_output_file(image);
        
        assert_eq!(result.output_files.len(), 1);
        assert_eq!(result.output_files[0].media_type, "image/png");
    }

    #[test]
    fn test_file_search_result() {
        let mut metadata = serde_json::Map::new();
        metadata.insert("size".to_string(), serde_json::json!(1024));
        
        let result = FileSearchResult::new("main.rs", "fn main() {}")
            .with_score(0.95)
            .with_metadata(metadata);
        
        assert_eq!(result.file_name, "main.rs");
        assert_eq!(result.score, Some(0.95));
        assert!(result.metadata.is_some());
    }

    #[test]
    fn test_file_search_results() {
        let results = FileSearchResults::new(
            "main function",
            vec![FileSearchResult::new("main.rs", "fn main() {}")],
        );
        
        assert_eq!(results.query, "main function");
        assert_eq!(results.len(), 1);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_builtin_tool_return_content() {
        let web_content = BuiltinToolReturnContent::web_search(
            WebSearchResults::new("test", vec![])
        );
        assert_eq!(web_content.content_type(), "web_search");
        
        let code_content = BuiltinToolReturnContent::code_execution(
            CodeExecutionResult::new("x = 1")
        );
        assert_eq!(code_content.content_type(), "code_execution");
        
        let file_content = BuiltinToolReturnContent::file_search(
            FileSearchResults::new("query", vec![])
        );
        assert_eq!(file_content.content_type(), "file_search");
        
        let other_content = BuiltinToolReturnContent::other(
            "custom_tool",
            serde_json::json!({"result": "data"})
        );
        assert_eq!(other_content.content_type(), "custom_tool");
    }

    #[test]
    fn test_builtin_tool_return_part() {
        let content = BuiltinToolReturnContent::web_search(
            WebSearchResults::new("rust", vec![
                WebSearchResult::new("Rust", "https://rust-lang.org")
            ])
        );
        
        let part = BuiltinToolReturnPart::new("web_search", content, "call_123")
            .with_id("return-001");
        
        assert_eq!(part.tool_name, "web_search");
        assert_eq!(part.tool_call_id, "call_123");
        assert_eq!(part.part_kind(), "builtin-tool-return");
        assert_eq!(part.content_type(), "web_search");
        assert_eq!(part.id, Some("return-001".to_string()));
    }

    #[test]
    fn test_serde_roundtrip_builtin_tool_call() {
        let part = BuiltinToolCallPart::new("web_search", serde_json::json!({"q": "test"}))
            .with_tool_call_id("call_1");
        
        let json = serde_json::to_string(&part).unwrap();
        let parsed: BuiltinToolCallPart = serde_json::from_str(&json).unwrap();
        assert_eq!(part, parsed);
    }

    #[test]
    fn test_serde_roundtrip_web_search_results() {
        let results = WebSearchResults::new(
            "rust",
            vec![
                WebSearchResult::new("Rust", "https://rust-lang.org")
                    .with_snippet("Systems programming")
            ]
        ).with_total_results(100);
        
        let json = serde_json::to_string(&results).unwrap();
        let parsed: WebSearchResults = serde_json::from_str(&json).unwrap();
        assert_eq!(results, parsed);
    }

    #[test]
    fn test_serde_roundtrip_builtin_tool_return() {
        let content = BuiltinToolReturnContent::code_execution(
            CodeExecutionResult::new("print(1)")
                .with_stdout("1\n")
                .with_exit_code(0)
        );
        
        let part = BuiltinToolReturnPart::new("code_execution", content, "call_xyz")
            .with_id("ret-1");
        
        let json = serde_json::to_string(&part).unwrap();
        let parsed: BuiltinToolReturnPart = serde_json::from_str(&json).unwrap();
        
        assert_eq!(part.tool_name, parsed.tool_name);
        assert_eq!(part.tool_call_id, parsed.tool_call_id);
        assert_eq!(part.id, parsed.id);
    }
}

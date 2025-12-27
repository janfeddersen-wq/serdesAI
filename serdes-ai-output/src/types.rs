//! Output type wrappers and markers.
//!
//! This module provides marker types and wrappers for different output modes.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as JsonValue};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use crate::structured::{DEFAULT_OUTPUT_TOOL_DESCRIPTION, DEFAULT_OUTPUT_TOOL_NAME};

/// Marker for native structured output mode.
///
/// Use this to indicate that output should use the model's native
/// structured output feature (like OpenAI's response_format).
#[derive(Debug, Clone)]
pub struct NativeOutput<T> {
    _phantom: PhantomData<T>,
}

impl<T> NativeOutput<T> {
    /// Create a new native output marker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T> Default for NativeOutput<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Marker for prompted output mode (JSON in text).
///
/// Use this to indicate that output should be prompted as JSON
/// but without native structured output enforcement.
#[derive(Debug, Clone)]
pub struct PromptedOutput<T> {
    _phantom: PhantomData<T>,
}

impl<T> PromptedOutput<T> {
    /// Create a new prompted output marker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T> Default for PromptedOutput<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Marker for tool-based output mode.
///
/// Use this to indicate that output should be captured via a tool call.
/// This is the most reliable method for structured output.
#[derive(Debug, Clone)]
pub struct ToolOutput<T> {
    /// The tool name.
    pub tool_name: String,
    /// The tool description.
    pub tool_description: String,
    _phantom: PhantomData<T>,
}

impl<T> ToolOutput<T> {
    /// Create a new tool output marker with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tool_name: DEFAULT_OUTPUT_TOOL_NAME.to_string(),
            tool_description: DEFAULT_OUTPUT_TOOL_DESCRIPTION.to_string(),
            _phantom: PhantomData,
        }
    }

    /// Set the tool name.
    #[must_use]
    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = name.into();
        self
    }

    /// Set the tool description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.tool_description = desc.into();
        self
    }
}

impl<T> Default for ToolOutput<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Plain text output (no structure).
///
/// Use this when you just want the model's text response without
/// any structured parsing.
#[derive(Debug, Clone, Default)]
pub struct TextOutput;

impl TextOutput {
    /// Create a new text output marker.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Structured dictionary output (like TypedDict in Python).
///
/// This provides a flexible key-value structure for when you
/// don't want to define a specific struct type.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct StructuredDict(pub Map<String, JsonValue>);

impl StructuredDict {
    /// Create a new empty structured dict.
    #[must_use]
    pub fn new() -> Self {
        Self(Map::new())
    }

    /// Create from a JSON map.
    #[must_use]
    pub fn from_map(map: Map<String, JsonValue>) -> Self {
        Self(map)
    }

    /// Get a value by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.0.get(key)
    }

    /// Get a mutable value by key.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut JsonValue> {
        self.0.get_mut(key)
    }

    /// Insert a value.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<JsonValue>) {
        self.0.insert(key.into(), value.into());
    }

    /// Remove a value.
    pub fn remove(&mut self, key: &str) -> Option<JsonValue> {
        self.0.remove(key)
    }

    /// Check if the dict contains a key.
    #[must_use]
    pub fn contains_key(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    /// Get the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get an iterator over keys.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }

    /// Get an iterator over values.
    pub fn values(&self) -> impl Iterator<Item = &JsonValue> {
        self.0.values()
    }

    /// Get an iterator over entries.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &JsonValue)> {
        self.0.iter()
    }

    /// Convert to a JSON value.
    #[must_use]
    pub fn to_json(&self) -> JsonValue {
        JsonValue::Object(self.0.clone())
    }

    /// Try to get a string value.
    #[must_use]
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.as_str())
    }

    /// Try to get an i64 value.
    #[must_use]
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.0.get(key).and_then(|v| v.as_i64())
    }

    /// Try to get a f64 value.
    #[must_use]
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.0.get(key).and_then(|v| v.as_f64())
    }

    /// Try to get a bool value.
    #[must_use]
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.0.get(key).and_then(|v| v.as_bool())
    }

    /// Try to get an array value.
    #[must_use]
    pub fn get_array(&self, key: &str) -> Option<&Vec<JsonValue>> {
        self.0.get(key).and_then(|v| v.as_array())
    }

    /// Try to get an object value.
    #[must_use]
    pub fn get_object(&self, key: &str) -> Option<&Map<String, JsonValue>> {
        self.0.get(key).and_then(|v| v.as_object())
    }
}

impl Deref for StructuredDict {
    type Target = Map<String, JsonValue>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for StructuredDict {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Map<String, JsonValue>> for StructuredDict {
    fn from(map: Map<String, JsonValue>) -> Self {
        Self(map)
    }
}

impl From<StructuredDict> for JsonValue {
    fn from(dict: StructuredDict) -> Self {
        JsonValue::Object(dict.0)
    }
}

impl TryFrom<JsonValue> for StructuredDict {
    type Error = &'static str;

    fn try_from(value: JsonValue) -> Result<Self, Self::Error> {
        match value {
            JsonValue::Object(map) => Ok(Self(map)),
            _ => Err("Expected JSON object"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_output() {
        let _output: NativeOutput<String> = NativeOutput::new();
    }

    #[test]
    fn test_prompted_output() {
        let _output: PromptedOutput<String> = PromptedOutput::new();
    }

    #[test]
    fn test_tool_output() {
        let output: ToolOutput<String> = ToolOutput::new()
            .with_tool_name("result")
            .with_description("The result");

        assert_eq!(output.tool_name, "result");
        assert_eq!(output.tool_description, "The result");
    }

    #[test]
    fn test_text_output() {
        let _output = TextOutput::new();
    }

    #[test]
    fn test_structured_dict_new() {
        let dict = StructuredDict::new();
        assert!(dict.is_empty());
    }

    #[test]
    fn test_structured_dict_insert_get() {
        let mut dict = StructuredDict::new();
        dict.insert("name", "Alice");
        dict.insert("age", 30);

        assert_eq!(dict.get_str("name"), Some("Alice"));
        assert_eq!(dict.get_i64("age"), Some(30));
    }

    #[test]
    fn test_structured_dict_remove() {
        let mut dict = StructuredDict::new();
        dict.insert("key", "value");
        assert!(dict.contains_key("key"));

        dict.remove("key");
        assert!(!dict.contains_key("key"));
    }

    #[test]
    fn test_structured_dict_serde() {
        let mut dict = StructuredDict::new();
        dict.insert("name", "Bob");
        dict.insert("score", 95.5);

        let json = serde_json::to_string(&dict).unwrap();
        let parsed: StructuredDict = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.get_str("name"), Some("Bob"));
        assert_eq!(parsed.get_f64("score"), Some(95.5));
    }

    #[test]
    fn test_structured_dict_from_json() {
        let json = serde_json::json!({"a": 1, "b": "two"});
        let dict: StructuredDict = json.try_into().unwrap();

        assert_eq!(dict.get_i64("a"), Some(1));
        assert_eq!(dict.get_str("b"), Some("two"));
    }

    #[test]
    fn test_structured_dict_to_json() {
        let mut dict = StructuredDict::new();
        dict.insert("key", "value");

        let json = dict.to_json();
        assert_eq!(json["key"], "value");
    }

    #[test]
    fn test_structured_dict_iter() {
        let mut dict = StructuredDict::new();
        dict.insert("a", 1);
        dict.insert("b", 2);

        let keys: Vec<_> = dict.keys().collect();
        assert_eq!(keys.len(), 2);
    }
}

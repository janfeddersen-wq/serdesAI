//! Output schema trait and core types.
//!
//! This module provides the `OutputSchema` trait which defines how to
//! parse and validate model responses into typed output.

use async_trait::async_trait;
use serde_json::Value as JsonValue;
use serdes_ai_tools::{ObjectJsonSchema, ToolDefinition};

use crate::error::OutputParseError;
use crate::mode::OutputMode;

/// Trait for output schemas that can validate model responses.
///
/// This trait defines how to parse model output from various formats
/// (text, tool calls, native JSON) and optionally validate it.
///
/// # Type Parameters
///
/// - `T`: The output type to parse into.
#[async_trait]
pub trait OutputSchema<T: Send>: Send + Sync {
    /// The preferred output mode for this schema.
    fn mode(&self) -> OutputMode;

    /// Get tool definitions if using tool mode.
    ///
    /// Returns an empty vector if not using tool mode.
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![]
    }

    /// Get JSON schema for native/prompted mode.
    ///
    /// Returns `None` if no schema is available.
    fn json_schema(&self) -> Option<ObjectJsonSchema> {
        None
    }

    /// Whether this schema supports a given output mode.
    fn supports_mode(&self, mode: OutputMode) -> bool {
        match mode {
            OutputMode::Text => true, // Text is always supported
            OutputMode::Tool => !self.tool_definitions().is_empty(),
            OutputMode::Native | OutputMode::Prompted => self.json_schema().is_some(),
        }
    }

    /// Parse output from text.
    fn parse_text(&self, text: &str) -> Result<T, OutputParseError>;

    /// Parse output from a tool call.
    fn parse_tool_call(&self, name: &str, args: &JsonValue) -> Result<T, OutputParseError>;

    /// Parse output from native structured response.
    fn parse_native(&self, value: &JsonValue) -> Result<T, OutputParseError>;

    /// Parse output based on the mode.
    fn parse(
        &self,
        mode: OutputMode,
        text: Option<&str>,
        tool_name: Option<&str>,
        args: Option<&JsonValue>,
    ) -> Result<T, OutputParseError> {
        match mode {
            OutputMode::Text => {
                let text = text.ok_or_else(|| OutputParseError::custom("No text output"))?;
                self.parse_text(text)
            }
            OutputMode::Tool => {
                let name = tool_name.ok_or_else(|| OutputParseError::custom("No tool call"))?;
                let args = args.ok_or_else(|| OutputParseError::custom("No tool arguments"))?;
                self.parse_tool_call(name, args)
            }
            OutputMode::Native | OutputMode::Prompted => {
                // Try tool call first, then native JSON
                if let (Some(name), Some(args)) = (tool_name, args) {
                    return self.parse_tool_call(name, args);
                }
                if let Some(args) = args {
                    return self.parse_native(args);
                }
                if let Some(text) = text {
                    return self.parse_text(text);
                }
                Err(OutputParseError::custom("No output to parse"))
            }
        }
    }
}

/// Boxed output schema for dynamic dispatch.
pub type BoxedOutputSchema<T> = Box<dyn OutputSchema<T>>;

/// A simple wrapper for output schemas that implements Send + Sync.
#[derive(Debug)]
pub struct OutputSchemaWrapper<S, T> {
    inner: S,
    _phantom: std::marker::PhantomData<T>,
}

impl<S, T> OutputSchemaWrapper<S, T> {
    /// Create a new wrapper.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the inner schema.
    pub fn inner(&self) -> &S {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockSchema;

    #[async_trait]
    impl OutputSchema<String> for MockSchema {
        fn mode(&self) -> OutputMode {
            OutputMode::Text
        }

        fn parse_text(&self, text: &str) -> Result<String, OutputParseError> {
            Ok(text.to_string())
        }

        fn parse_tool_call(
            &self,
            _name: &str,
            args: &JsonValue,
        ) -> Result<String, OutputParseError> {
            args.as_str()
                .map(String::from)
                .ok_or(OutputParseError::NotJson)
        }

        fn parse_native(&self, value: &JsonValue) -> Result<String, OutputParseError> {
            value
                .as_str()
                .map(String::from)
                .ok_or(OutputParseError::NotJson)
        }
    }

    #[test]
    fn test_mock_schema_parse_text() {
        let schema = MockSchema;
        let result = schema.parse_text("hello").unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_mock_schema_supports_mode() {
        let schema = MockSchema;
        assert!(schema.supports_mode(OutputMode::Text));
        assert!(!schema.supports_mode(OutputMode::Tool));
        assert!(!schema.supports_mode(OutputMode::Native));
    }

    #[test]
    fn test_parse_dispatch() {
        let schema = MockSchema;

        // Text mode
        let result = schema
            .parse(OutputMode::Text, Some("hello"), None, None)
            .unwrap();
        assert_eq!(result, "hello");

        // Missing text
        let result = schema.parse(OutputMode::Text, None, None, None);
        assert!(result.is_err());
    }
}

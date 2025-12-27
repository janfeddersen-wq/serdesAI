//! Text output schema implementation.
//!
//! This module provides `TextOutputSchema` for handling plain text output
//! with optional validation constraints like patterns, length limits, etc.

use async_trait::async_trait;
use regex::Regex;
use serde_json::Value as JsonValue;
use crate::error::OutputParseError;
use crate::mode::OutputMode;
use crate::schema::OutputSchema;

/// Schema for plain text output.
///
/// This schema validates text output with optional constraints:
/// - Pattern matching via regex
/// - Minimum/maximum length
/// - Trim whitespace
///
/// # Example
///
/// ```rust
/// use serdes_ai_output::TextOutputSchema;
///
/// let schema = TextOutputSchema::new()
///     .with_min_length(10)
///     .with_max_length(1000)
///     .trim();
/// ```
#[derive(Debug, Clone, Default)]
pub struct TextOutputSchema {
    /// Optional regex pattern to match.
    pattern: Option<Regex>,
    /// Pattern string (for error messages).
    pattern_str: Option<String>,
    /// Minimum length.
    min_length: Option<usize>,
    /// Maximum length.
    max_length: Option<usize>,
    /// Whether to trim whitespace.
    trim_whitespace: bool,
}

impl TextOutputSchema {
    /// Create a new text output schema with no constraints.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a regex pattern the output must match.
    ///
    /// # Errors
    ///
    /// Returns an error if the pattern is invalid.
    pub fn with_pattern(mut self, pattern: &str) -> Result<Self, regex::Error> {
        self.pattern = Some(Regex::new(pattern)?);
        self.pattern_str = Some(pattern.to_string());
        Ok(self)
    }

    /// Set the minimum length constraint.
    #[must_use]
    pub fn with_min_length(mut self, len: usize) -> Self {
        self.min_length = Some(len);
        self
    }

    /// Set the maximum length constraint.
    #[must_use]
    pub fn with_max_length(mut self, len: usize) -> Self {
        self.max_length = Some(len);
        self
    }

    /// Enable whitespace trimming.
    #[must_use]
    pub fn trim(mut self) -> Self {
        self.trim_whitespace = true;
        self
    }

    /// Validate the text against configured constraints.
    fn validate_text(&self, text: &str) -> Result<String, OutputParseError> {
        let text = if self.trim_whitespace {
            text.trim().to_string()
        } else {
            text.to_string()
        };

        // Check minimum length
        if let Some(min) = self.min_length {
            if text.len() < min {
                return Err(OutputParseError::too_short(text.len(), min));
            }
        }

        // Check maximum length
        if let Some(max) = self.max_length {
            if text.len() > max {
                return Err(OutputParseError::too_long(text.len(), max));
            }
        }

        // Check pattern
        if let Some(ref pattern) = self.pattern {
            if !pattern.is_match(&text) {
                return Err(OutputParseError::PatternMismatch {
                    pattern: self.pattern_str.clone().unwrap_or_default(),
                });
            }
        }

        Ok(text)
    }
}

#[async_trait]
impl OutputSchema<String> for TextOutputSchema {
    fn mode(&self) -> OutputMode {
        OutputMode::Text
    }

    fn parse_text(&self, text: &str) -> Result<String, OutputParseError> {
        self.validate_text(text)
    }

    fn parse_tool_call(&self, _name: &str, args: &JsonValue) -> Result<String, OutputParseError> {
        // Try to extract text from tool arguments
        if let Some(text) = args.as_str() {
            return self.validate_text(text);
        }

        // Try common field names
        for field in ["text", "content", "message", "result", "output"] {
            if let Some(text) = args.get(field).and_then(|v| v.as_str()) {
                return self.validate_text(text);
            }
        }

        // Fall back to JSON string representation
        let text = serde_json::to_string(args).map_err(OutputParseError::JsonParse)?;
        self.validate_text(&text)
    }

    fn parse_native(&self, value: &JsonValue) -> Result<String, OutputParseError> {
        if let Some(text) = value.as_str() {
            return self.validate_text(text);
        }

        // Try common field names for objects
        if let Some(obj) = value.as_object() {
            for field in ["text", "content", "message", "result", "output"] {
                if let Some(text) = obj.get(field).and_then(|v| v.as_str()) {
                    return self.validate_text(text);
                }
            }
        }

        Err(OutputParseError::NotJson)
    }
}

/// Builder for text output schema with more options.
#[derive(Debug, Default)]
pub struct TextOutputSchemaBuilder {
    schema: TextOutputSchema,
}

impl TextOutputSchemaBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a regex pattern.
    pub fn pattern(mut self, pattern: &str) -> Result<Self, regex::Error> {
        self.schema = self.schema.with_pattern(pattern)?;
        Ok(self)
    }

    /// Set minimum length.
    #[must_use]
    pub fn min_length(mut self, len: usize) -> Self {
        self.schema = self.schema.with_min_length(len);
        self
    }

    /// Set maximum length.
    #[must_use]
    pub fn max_length(mut self, len: usize) -> Self {
        self.schema = self.schema.with_max_length(len);
        self
    }

    /// Enable trimming.
    #[must_use]
    pub fn trim(mut self) -> Self {
        self.schema = self.schema.trim();
        self
    }

    /// Build the schema.
    #[must_use]
    pub fn build(self) -> TextOutputSchema {
        self.schema
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_schema_default() {
        let schema = TextOutputSchema::new();
        assert_eq!(schema.mode(), OutputMode::Text);
    }

    #[test]
    fn test_text_schema_parse_text() {
        let schema = TextOutputSchema::new();
        let result = schema.parse_text("hello world").unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_text_schema_trim() {
        let schema = TextOutputSchema::new().trim();
        let result = schema.parse_text("  hello world  ").unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_text_schema_min_length() {
        let schema = TextOutputSchema::new().with_min_length(10);

        // Too short
        let result = schema.parse_text("short");
        assert!(result.is_err());

        // Long enough
        let result = schema.parse_text("this is long enough");
        assert!(result.is_ok());
    }

    #[test]
    fn test_text_schema_max_length() {
        let schema = TextOutputSchema::new().with_max_length(10);

        // Too long
        let result = schema.parse_text("this is too long");
        assert!(result.is_err());

        // Short enough
        let result = schema.parse_text("short");
        assert!(result.is_ok());
    }

    #[test]
    fn test_text_schema_pattern() {
        let schema = TextOutputSchema::new()
            .with_pattern(r"^\d{3}-\d{4}$")
            .unwrap();

        // Matches
        let result = schema.parse_text("123-4567");
        assert!(result.is_ok());

        // Doesn't match
        let result = schema.parse_text("abc-defg");
        assert!(result.is_err());
    }

    #[test]
    fn test_text_schema_combined_constraints() {
        let schema = TextOutputSchema::new()
            .with_min_length(5)
            .with_max_length(20)
            .trim();

        // Valid
        let result = schema.parse_text("  hello world  ");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "hello world");

        // Too short after trim
        let result = schema.parse_text("  hi  ");
        assert!(result.is_err());
    }

    #[test]
    fn test_text_schema_parse_tool_call_string() {
        let schema = TextOutputSchema::new();
        let args = serde_json::json!("hello");
        let result = schema.parse_tool_call("result", &args).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_text_schema_parse_tool_call_object() {
        let schema = TextOutputSchema::new();
        let args = serde_json::json!({"text": "hello from tool"});
        let result = schema.parse_tool_call("result", &args).unwrap();
        assert_eq!(result, "hello from tool");
    }

    #[test]
    fn test_text_schema_parse_native() {
        let schema = TextOutputSchema::new();

        // Direct string
        let value = serde_json::json!("native output");
        let result = schema.parse_native(&value).unwrap();
        assert_eq!(result, "native output");

        // Object with content field
        let value = serde_json::json!({"content": "from content"});
        let result = schema.parse_native(&value).unwrap();
        assert_eq!(result, "from content");
    }

    #[test]
    fn test_builder() {
        let schema = TextOutputSchemaBuilder::new()
            .min_length(5)
            .max_length(100)
            .trim()
            .build();

        let result = schema.parse_text("  hello  ").unwrap();
        assert_eq!(result, "hello");
    }
}

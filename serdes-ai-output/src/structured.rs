//! Structured output schema implementation.
//!
//! This module provides `StructuredOutputSchema` for handling typed
//! structured output using serde deserialization.

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value as JsonValue;
use serdes_ai_tools::{ObjectJsonSchema, ToolDefinition};
use std::marker::PhantomData;

use crate::error::OutputParseError;
use crate::mode::OutputMode;
use crate::schema::OutputSchema;

/// Default tool name for structured output.
pub const DEFAULT_OUTPUT_TOOL_NAME: &str = "final_result";

/// Default tool description for structured output.
pub const DEFAULT_OUTPUT_TOOL_DESCRIPTION: &str = "The final response which ends this conversation";

/// Schema for structured output using serde.
///
/// This schema parses model output into a typed Rust struct using serde.
/// It supports multiple output modes (tool, native, prompted, text).
///
/// # Example
///
/// ```rust
/// use serdes_ai_output::StructuredOutputSchema;
/// use serdes_ai_tools::ObjectJsonSchema;
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// let schema = ObjectJsonSchema::new()
///     .with_property("name", serdes_ai_tools::PropertySchema::string("Name").build(), true)
///     .with_property("age", serdes_ai_tools::PropertySchema::integer("Age").build(), true);
///
/// let output_schema: StructuredOutputSchema<Person> = StructuredOutputSchema::new(schema);
/// ```
#[derive(Debug, Clone)]
pub struct StructuredOutputSchema<T> {
    /// Tool name when using tool mode.
    pub tool_name: String,
    /// Tool description.
    pub tool_description: String,
    /// JSON schema for the output.
    pub schema: ObjectJsonSchema,
    /// Whether to use strict mode (for OpenAI).
    pub strict: Option<bool>,
    /// Output mode preference.
    mode: OutputMode,
    _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned + Send + Sync> StructuredOutputSchema<T> {
    /// Create a new structured output schema.
    #[must_use]
    pub fn new(schema: ObjectJsonSchema) -> Self {
        Self {
            tool_name: DEFAULT_OUTPUT_TOOL_NAME.to_string(),
            tool_description: DEFAULT_OUTPUT_TOOL_DESCRIPTION.to_string(),
            schema,
            strict: None,
            mode: OutputMode::Tool,
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

    /// Set strict mode (for OpenAI structured outputs).
    #[must_use]
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    /// Set the preferred output mode.
    #[must_use]
    pub fn with_mode(mut self, mode: OutputMode) -> Self {
        self.mode = mode;
        self
    }
}

#[async_trait]
impl<T: DeserializeOwned + Send + Sync> OutputSchema<T> for StructuredOutputSchema<T> {
    fn mode(&self) -> OutputMode {
        self.mode
    }

    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(&self.tool_name, &self.tool_description)
            .with_parameters(self.schema.clone())
            .with_strict(self.strict.unwrap_or(false))]
    }

    fn json_schema(&self) -> Option<ObjectJsonSchema> {
        Some(self.schema.clone())
    }

    fn parse_text(&self, text: &str) -> Result<T, OutputParseError> {
        // Try to extract JSON from text (may be wrapped in markdown)
        let json_str = extract_json(text)?;
        serde_json::from_str(&json_str).map_err(OutputParseError::JsonParse)
    }

    fn parse_tool_call(&self, name: &str, args: &JsonValue) -> Result<T, OutputParseError> {
        if name != self.tool_name {
            return Err(OutputParseError::unexpected_tool(&self.tool_name, name));
        }
        serde_json::from_value(args.clone()).map_err(OutputParseError::JsonParse)
    }

    fn parse_native(&self, value: &JsonValue) -> Result<T, OutputParseError> {
        serde_json::from_value(value.clone()).map_err(OutputParseError::JsonParse)
    }
}

/// Extract JSON from text that might be wrapped in markdown code blocks.
///
/// This function handles common patterns:
/// - ` ```json ... ``` ` blocks
/// - ` ``` ... ``` ` blocks (no language)
/// - Raw JSON objects `{ ... }`
/// - Raw JSON arrays `[ ... ]`
pub fn extract_json(text: &str) -> Result<String, OutputParseError> {
    let text = text.trim();

    // Check for markdown code blocks with json language
    if let Some(rest) = text.strip_prefix("```json") {
        if let Some(end) = rest.find("```") {
            return Ok(rest[..end].trim().to_string());
        }
    }

    // Check for markdown code blocks without language
    if let Some(rest) = text.strip_prefix("```") {
        // Skip any language identifier on the first line
        let rest = if let Some(newline) = rest.find('\n') {
            &rest[newline + 1..]
        } else {
            rest
        };
        if let Some(end) = rest.find("```") {
            return Ok(rest[..end].trim().to_string());
        }
    }

    // Look for JSON object
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            if end > start {
                let candidate = &text[start..=end];
                // Validate it's actually JSON
                if serde_json::from_str::<JsonValue>(candidate).is_ok() {
                    return Ok(candidate.to_string());
                }
            }
        }
    }

    // Look for JSON array
    if let Some(start) = text.find('[') {
        if let Some(end) = text.rfind(']') {
            if end > start {
                let candidate = &text[start..=end];
                // Validate it's actually JSON
                if serde_json::from_str::<JsonValue>(candidate).is_ok() {
                    return Ok(candidate.to_string());
                }
            }
        }
    }

    // Try parsing the whole thing as JSON
    if serde_json::from_str::<JsonValue>(text).is_ok() {
        return Ok(text.to_string());
    }

    Err(OutputParseError::NoJsonFound)
}

/// Schema that accepts any valid JSON value.
#[derive(Debug, Clone, Default)]
pub struct AnyJsonSchema {
    tool_name: String,
    tool_description: String,
}

impl AnyJsonSchema {
    /// Create a new any JSON schema.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tool_name: DEFAULT_OUTPUT_TOOL_NAME.to_string(),
            tool_description: DEFAULT_OUTPUT_TOOL_DESCRIPTION.to_string(),
        }
    }

    /// Set the tool name.
    #[must_use]
    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = name.into();
        self
    }
}

#[async_trait]
impl OutputSchema<JsonValue> for AnyJsonSchema {
    fn mode(&self) -> OutputMode {
        OutputMode::Tool
    }

    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(&self.tool_name, &self.tool_description)]
    }

    fn parse_text(&self, text: &str) -> Result<JsonValue, OutputParseError> {
        let json_str = extract_json(text)?;
        serde_json::from_str(&json_str).map_err(OutputParseError::JsonParse)
    }

    fn parse_tool_call(
        &self,
        _name: &str,
        args: &JsonValue,
    ) -> Result<JsonValue, OutputParseError> {
        Ok(args.clone())
    }

    fn parse_native(&self, value: &JsonValue) -> Result<JsonValue, OutputParseError> {
        Ok(value.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serdes_ai_tools::PropertySchema;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Person {
        name: String,
        age: u32,
    }

    fn person_schema() -> ObjectJsonSchema {
        ObjectJsonSchema::new()
            .with_property("name", PropertySchema::string("Name").build(), true)
            .with_property("age", PropertySchema::integer("Age").build(), true)
    }

    #[test]
    fn test_structured_schema_new() {
        let schema: StructuredOutputSchema<Person> = StructuredOutputSchema::new(person_schema());
        assert_eq!(schema.tool_name, DEFAULT_OUTPUT_TOOL_NAME);
        assert_eq!(schema.mode(), OutputMode::Tool);
    }

    #[test]
    fn test_structured_schema_with_tool_name() {
        let schema: StructuredOutputSchema<Person> =
            StructuredOutputSchema::new(person_schema()).with_tool_name("submit_person");
        assert_eq!(schema.tool_name, "submit_person");
    }

    #[test]
    fn test_structured_schema_tool_definitions() {
        let schema: StructuredOutputSchema<Person> = StructuredOutputSchema::new(person_schema())
            .with_tool_name("result")
            .with_description("Submit the person");

        let defs = schema.tool_definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "result");
        assert_eq!(defs[0].description, "Submit the person");
    }

    #[test]
    fn test_structured_schema_parse_tool_call() {
        let schema: StructuredOutputSchema<Person> = StructuredOutputSchema::new(person_schema());

        let args = serde_json::json!({"name": "Alice", "age": 30});
        let result = schema.parse_tool_call("final_result", &args).unwrap();
        assert_eq!(result.name, "Alice");
        assert_eq!(result.age, 30);
    }

    #[test]
    fn test_structured_schema_parse_tool_call_wrong_name() {
        let schema: StructuredOutputSchema<Person> = StructuredOutputSchema::new(person_schema());

        let args = serde_json::json!({"name": "Alice", "age": 30});
        let result = schema.parse_tool_call("wrong_tool", &args);
        assert!(result.is_err());
    }

    #[test]
    fn test_structured_schema_parse_native() {
        let schema: StructuredOutputSchema<Person> = StructuredOutputSchema::new(person_schema());

        let value = serde_json::json!({"name": "Bob", "age": 25});
        let result = schema.parse_native(&value).unwrap();
        assert_eq!(result.name, "Bob");
        assert_eq!(result.age, 25);
    }

    #[test]
    fn test_structured_schema_parse_text_raw_json() {
        let schema: StructuredOutputSchema<Person> = StructuredOutputSchema::new(person_schema());

        let text = r#"{"name": "Charlie", "age": 35}"#;
        let result = schema.parse_text(text).unwrap();
        assert_eq!(result.name, "Charlie");
        assert_eq!(result.age, 35);
    }

    #[test]
    fn test_structured_schema_parse_text_markdown() {
        let schema: StructuredOutputSchema<Person> = StructuredOutputSchema::new(person_schema());

        let text = r#"Here is the result:
```json
{"name": "Diana", "age": 28}
```
Done!"#;
        let result = schema.parse_text(text).unwrap();
        assert_eq!(result.name, "Diana");
        assert_eq!(result.age, 28);
    }

    #[test]
    fn test_extract_json_code_block() {
        let text = r#"```json
{"key": "value"}
```"#;
        let result = extract_json(text).unwrap();
        assert_eq!(result, r#"{"key": "value"}"#);
    }

    #[test]
    fn test_extract_json_plain_code_block() {
        let text = r#"```
{"key": "value"}
```"#;
        let result = extract_json(text).unwrap();
        assert_eq!(result, r#"{"key": "value"}"#);
    }

    #[test]
    fn test_extract_json_embedded() {
        let text = r#"The result is: {"x": 1, "y": 2} and that's it."#;
        let result = extract_json(text).unwrap();
        assert_eq!(result, r#"{"x": 1, "y": 2}"#);
    }

    #[test]
    fn test_extract_json_array() {
        let text = r#"Here are the items: [1, 2, 3]"#;
        let result = extract_json(text).unwrap();
        assert_eq!(result, "[1, 2, 3]");
    }

    #[test]
    fn test_extract_json_not_found() {
        let text = "This is just plain text with no JSON.";
        let result = extract_json(text);
        assert!(result.is_err());
    }

    #[test]
    fn test_any_json_schema() {
        let schema = AnyJsonSchema::new();

        let value = serde_json::json!({"anything": [1, 2, 3]});
        let result = schema.parse_native(&value).unwrap();
        assert_eq!(result, value);
    }
}

//! Output specification types.
//!
//! This module provides `OutputSpec`, a flexible way to specify
//! output types for agent runs.

use serde::de::DeserializeOwned;
use serdes_ai_tools::ObjectJsonSchema;
use std::marker::PhantomData;

use crate::mode::OutputMode;
use crate::schema::{BoxedOutputSchema, OutputSchema};
use crate::structured::StructuredOutputSchema;
use crate::text::TextOutputSchema;

/// Specification for output type - allows multiple ways to specify.
///
/// This enum provides a flexible way to define what kind of output
/// you expect from an agent run.
pub enum OutputSpec<T> {
    /// Plain text output.
    Text(TextOutputSchema),
    /// Structured type with automatic schema.
    Structured(StructuredOutputSchema<T>),
    /// Custom schema.
    Custom(BoxedOutputSchema<T>),
}

impl<T> std::fmt::Debug for OutputSpec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputSpec::Text(s) => f.debug_tuple("Text").field(s).finish(),
            OutputSpec::Structured(_) => f.debug_tuple("Structured").field(&"...").finish(),
            OutputSpec::Custom(_) => f
                .debug_tuple("Custom")
                .field(&"<dyn OutputSchema>")
                .finish(),
        }
    }
}

impl OutputSpec<String> {
    /// Create a text output spec.
    #[must_use]
    pub fn text() -> Self {
        OutputSpec::Text(TextOutputSchema::new())
    }

    /// Create a text output spec with constraints.
    #[must_use]
    pub fn text_with_schema(schema: TextOutputSchema) -> Self {
        OutputSpec::Text(schema)
    }
}

impl<T: DeserializeOwned + Send + Sync + 'static> OutputSpec<T> {
    /// Create a structured output spec.
    #[must_use]
    pub fn structured(schema: ObjectJsonSchema) -> Self {
        OutputSpec::Structured(StructuredOutputSchema::new(schema))
    }

    /// Create a structured output spec with a custom schema.
    #[must_use]
    pub fn structured_with(schema: StructuredOutputSchema<T>) -> Self {
        OutputSpec::Structured(schema)
    }

    /// Create a custom output spec.
    pub fn custom<S: OutputSchema<T> + 'static>(schema: S) -> Self {
        OutputSpec::Custom(Box::new(schema))
    }

    /// Get the preferred output mode.
    #[must_use]
    pub fn mode(&self) -> OutputMode {
        match self {
            OutputSpec::Text(s) => s.mode(),
            OutputSpec::Structured(s) => s.mode(),
            OutputSpec::Custom(s) => s.mode(),
        }
    }

    /// Get tool definitions if using tool mode.
    #[must_use]
    pub fn tool_definitions(&self) -> Vec<serdes_ai_tools::ToolDefinition> {
        match self {
            OutputSpec::Text(s) => s.tool_definitions(),
            OutputSpec::Structured(s) => s.tool_definitions(),
            OutputSpec::Custom(s) => s.tool_definitions(),
        }
    }

    /// Get JSON schema if available.
    #[must_use]
    pub fn json_schema(&self) -> Option<ObjectJsonSchema> {
        match self {
            OutputSpec::Text(s) => s.json_schema(),
            OutputSpec::Structured(s) => s.json_schema(),
            OutputSpec::Custom(s) => s.json_schema(),
        }
    }
}

impl Default for OutputSpec<String> {
    fn default() -> Self {
        OutputSpec::text()
    }
}

/// Builder for output specifications.
#[derive(Debug)]
pub struct OutputSpecBuilder<T> {
    _phantom: PhantomData<T>,
}

impl<T> OutputSpecBuilder<T> {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T> Default for OutputSpecBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputSpecBuilder<String> {
    /// Build a text output spec.
    #[must_use]
    pub fn text(self) -> OutputSpec<String> {
        OutputSpec::text()
    }

    /// Build a text output spec with constraints.
    #[must_use]
    pub fn text_constrained(
        self,
        min_length: Option<usize>,
        max_length: Option<usize>,
    ) -> OutputSpec<String> {
        let mut schema = TextOutputSchema::new();
        if let Some(min) = min_length {
            schema = schema.with_min_length(min);
        }
        if let Some(max) = max_length {
            schema = schema.with_max_length(max);
        }
        OutputSpec::Text(schema)
    }
}

impl<T: DeserializeOwned + Send + Sync + 'static> OutputSpecBuilder<T> {
    /// Build a structured output spec.
    #[must_use]
    pub fn structured(self, schema: ObjectJsonSchema) -> OutputSpec<T> {
        OutputSpec::structured(schema)
    }

    /// Build a structured output spec with tool name.
    #[must_use]
    pub fn structured_with_tool(
        self,
        schema: ObjectJsonSchema,
        tool_name: impl Into<String>,
    ) -> OutputSpec<T> {
        OutputSpec::Structured(StructuredOutputSchema::new(schema).with_tool_name(tool_name))
    }
}

/// Utility trait for types that can be output specs.
pub trait IntoOutputSpec<T> {
    /// Convert into an output spec.
    fn into_output_spec(self) -> OutputSpec<T>;
}

impl<T> IntoOutputSpec<T> for OutputSpec<T> {
    fn into_output_spec(self) -> OutputSpec<T> {
        self
    }
}

impl IntoOutputSpec<String> for TextOutputSchema {
    fn into_output_spec(self) -> OutputSpec<String> {
        OutputSpec::Text(self)
    }
}

impl<T: DeserializeOwned + Send + Sync + 'static> IntoOutputSpec<T> for StructuredOutputSchema<T> {
    fn into_output_spec(self) -> OutputSpec<T> {
        OutputSpec::Structured(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serdes_ai_tools::PropertySchema;

    #[derive(Debug, Deserialize)]
    struct TestStruct {
        name: String,
    }

    #[test]
    fn test_output_spec_text() {
        let spec = OutputSpec::<String>::text();
        assert_eq!(spec.mode(), OutputMode::Text);
        assert!(spec.tool_definitions().is_empty());
    }

    #[test]
    fn test_output_spec_structured() {
        let schema = ObjectJsonSchema::new().with_property(
            "name",
            PropertySchema::string("Name").build(),
            true,
        );

        let spec = OutputSpec::<TestStruct>::structured(schema);
        assert_eq!(spec.mode(), OutputMode::Tool);
        assert_eq!(spec.tool_definitions().len(), 1);
    }

    #[test]
    fn test_output_spec_default() {
        let spec = OutputSpec::<String>::default();
        assert_eq!(spec.mode(), OutputMode::Text);
    }

    #[test]
    fn test_builder_text() {
        let spec = OutputSpecBuilder::<String>::new().text();
        assert_eq!(spec.mode(), OutputMode::Text);
    }

    #[test]
    fn test_builder_text_constrained() {
        let spec = OutputSpecBuilder::<String>::new().text_constrained(Some(10), Some(100));
        assert_eq!(spec.mode(), OutputMode::Text);
    }

    #[test]
    fn test_builder_structured() {
        let schema = ObjectJsonSchema::new().with_property(
            "name",
            PropertySchema::string("Name").build(),
            true,
        );

        let spec = OutputSpecBuilder::<TestStruct>::new().structured(schema);
        assert_eq!(spec.mode(), OutputMode::Tool);
    }

    #[test]
    fn test_into_output_spec() {
        let text_schema = TextOutputSchema::new();
        let spec: OutputSpec<String> = text_schema.into_output_spec();
        assert_eq!(spec.mode(), OutputMode::Text);
    }
}

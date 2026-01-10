//! Tool definition types for describing tools to LLMs.
//!
//! This module provides types for defining tools with JSON Schema parameters
//! that can be serialized and sent to language models.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// JSON Schema for an object type (tool parameters).
///
/// This represents the parameters that a tool accepts, using JSON Schema format
/// that is understood by language models for function calling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObjectJsonSchema {
    /// The schema type (always "object" for tool parameters).
    #[serde(rename = "type")]
    pub schema_type: String,

    /// Property definitions.
    pub properties: IndexMap<String, JsonValue>,

    /// List of required property names.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub required: Vec<String>,

    /// Description of the schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Whether additional properties are allowed.
    #[serde(
        rename = "additionalProperties",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_properties: Option<bool>,

    /// Extra schema properties.
    #[serde(flatten)]
    pub extra: HashMap<String, JsonValue>,
}

impl ObjectJsonSchema {
    /// Create a new empty object schema.
    #[must_use]
    pub fn new() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: IndexMap::new(),
            required: Vec::new(),
            description: None,
            additional_properties: None,
            extra: HashMap::new(),
        }
    }

    /// Add a property to the schema.
    #[must_use]
    pub fn with_property(mut self, name: &str, schema: JsonValue, required: bool) -> Self {
        self.properties.insert(name.to_string(), schema);
        if required {
            self.required.push(name.to_string());
        }
        self
    }

    /// Add a property without consuming self.
    pub fn add_property(&mut self, name: &str, schema: JsonValue, required: bool) {
        self.properties.insert(name.to_string(), schema);
        if required && !self.required.contains(&name.to_string()) {
            self.required.push(name.to_string());
        }
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Set whether additional properties are allowed.
    #[must_use]
    pub fn with_additional_properties(mut self, allowed: bool) -> Self {
        self.additional_properties = Some(allowed);
        self
    }

    /// Add an extra property to the schema.
    #[must_use]
    pub fn with_extra(mut self, key: &str, value: JsonValue) -> Self {
        self.extra.insert(key.to_string(), value);
        self
    }

    /// Check if a property is required.
    #[must_use]
    pub fn is_required(&self, name: &str) -> bool {
        self.required.contains(&name.to_string())
    }

    /// Get a property schema.
    #[must_use]
    pub fn get_property(&self, name: &str) -> Option<&JsonValue> {
        self.properties.get(name)
    }

    /// Get the number of properties.
    #[must_use]
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }

    /// Check if the schema has no properties.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.properties.is_empty()
    }

    /// Convert to a JSON value.
    pub fn to_json(&self) -> Result<JsonValue, serde_json::Error> {
        serde_json::to_value(self)
    }
}

impl Default for ObjectJsonSchema {
    fn default() -> Self {
        Self::new()
    }
}

impl TryFrom<JsonValue> for ObjectJsonSchema {
    type Error = serde_json::Error;

    fn try_from(value: JsonValue) -> Result<Self, Self::Error> {
        serde_json::from_value(value)
    }
}

impl From<ObjectJsonSchema> for JsonValue {
    fn from(schema: ObjectJsonSchema) -> Self {
        serde_json::to_value(schema).unwrap_or(JsonValue::Null)
    }
}

/// Complete tool definition sent to the model.
///
/// This contains all the information a language model needs to understand
/// and call a tool, including its name, description, and parameter schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    /// Tool name (must be a valid identifier).
    pub name: String,

    /// Human-readable description of what the tool does.
    pub description: String,

    /// JSON Schema for the tool's parameters.
    pub parameters_json_schema: JsonValue,

    /// Whether to use strict mode for schema validation (OpenAI feature).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,

    /// Key for outer typed dict (pydantic-ai compatibility).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outer_typed_dict_key: Option<String>,
}

impl ToolDefinition {
    /// Create a new tool definition.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters_json_schema: crate::schema::SchemaBuilder::new()
                .build()
                .expect("SchemaBuilder JSON serialization failed"),
            strict: None,
            outer_typed_dict_key: None,
        }
    }

    /// Set the parameters schema.
    #[must_use]
    pub fn with_parameters(mut self, schema: impl Into<JsonValue>) -> Self {
        self.parameters_json_schema = schema.into();
        self
    }

    /// Set strict mode.
    #[must_use]
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    /// Set the outer typed dict key.
    #[must_use]
    pub fn with_outer_typed_dict_key(mut self, key: impl Into<String>) -> Self {
        self.outer_typed_dict_key = Some(key.into());
        self
    }

    /// Get the tool name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the tool description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get the parameters schema.
    #[must_use]
    pub fn parameters(&self) -> &JsonValue {
        &self.parameters_json_schema
    }

    /// Check if strict mode is enabled.
    #[must_use]
    pub fn is_strict(&self) -> bool {
        self.strict.unwrap_or(false)
    }

    /// Convert to OpenAI function format.
    #[must_use]
    pub fn to_openai_function(&self) -> JsonValue {
        let mut func = serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_json_schema.clone()
            }
        });

        if let Some(strict) = self.strict {
            func["function"]["strict"] = JsonValue::Bool(strict);
        }

        func
    }

    /// Convert to Anthropic tool format.
    #[must_use]
    pub fn to_anthropic_tool(&self) -> JsonValue {
        serde_json::json!({
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_json_schema.clone()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_json_schema_new() {
        let schema = ObjectJsonSchema::new();
        assert_eq!(schema.schema_type, "object");
        assert!(schema.properties.is_empty());
        assert!(schema.required.is_empty());
    }

    #[test]
    fn test_object_json_schema_with_property() {
        let schema = ObjectJsonSchema::new()
            .with_property("name", serde_json::json!({"type": "string"}), true)
            .with_property("age", serde_json::json!({"type": "integer"}), false);

        assert_eq!(schema.property_count(), 2);
        assert!(schema.is_required("name"));
        assert!(!schema.is_required("age"));
    }

    #[test]
    fn test_tool_definition_new() {
        let def = ToolDefinition::new("get_weather", "Get the current weather");
        assert_eq!(def.name(), "get_weather");
        assert_eq!(def.description(), "Get the current weather");
        let properties = def
            .parameters()
            .get("properties")
            .and_then(|value| value.as_object())
            .unwrap();
        assert!(properties.is_empty());
    }

    #[test]
    fn test_tool_definition_with_parameters() {
        let params = crate::schema::SchemaBuilder::new()
            .string("location", "City name", true)
            .enum_values(
                "unit",
                "Temperature unit",
                &["celsius", "fahrenheit"],
                false,
            )
            .build()
            .expect("SchemaBuilder JSON serialization failed");

        let def = ToolDefinition::new("get_weather", "Get weather")
            .with_parameters(params)
            .with_strict(true);

        assert!(def.is_strict());
        let properties = def
            .parameters()
            .get("properties")
            .and_then(|value| value.as_object())
            .unwrap();
        assert_eq!(properties.len(), 2);
    }

    #[test]
    fn test_to_openai_function() {
        let def = ToolDefinition::new("test", "Test tool")
            .with_parameters(
                crate::schema::SchemaBuilder::new()
                    .string("x", "A value", true)
                    .build()
                    .expect("SchemaBuilder JSON serialization failed"),
            )
            .with_strict(true);

        let func = def.to_openai_function();
        assert_eq!(func["type"], "function");
        assert_eq!(func["function"]["name"], "test");
        assert_eq!(func["function"]["strict"], true);
    }

    #[test]
    fn test_to_anthropic_tool() {
        let def = ToolDefinition::new("test", "Test tool");
        let tool = def.to_anthropic_tool();
        assert_eq!(tool["name"], "test");
        assert!(tool.get("input_schema").is_some());
    }

    #[test]
    fn test_serde_roundtrip() {
        let schema = ObjectJsonSchema::new()
            .with_property("x", serde_json::json!({"type": "string"}), true)
            .with_description("Test schema");

        let json = serde_json::to_string(&schema).unwrap();
        let parsed: ObjectJsonSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(schema, parsed);
    }
}

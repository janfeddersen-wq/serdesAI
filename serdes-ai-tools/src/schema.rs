//! JSON schema generation utilities.
//!
//! This module provides utilities for generating JSON schemas from Rust types
//! and a builder API for manual schema construction.

use indexmap::IndexMap;
use serde_json::Value as JsonValue;

use crate::definition::ObjectJsonSchema;

/// Schema builder for manual schema construction.
///
/// Provides a fluent API for building JSON schemas for tool parameters.
///
/// # Example
///
/// ```rust
/// use serdes_ai_tools::SchemaBuilder;
///
/// let schema = SchemaBuilder::new()
///     .string("name", "The user's name", true)
///     .integer("age", "The user's age", false)
///     .enum_values("status", "User status", &["active", "inactive"], true)
///     .description("User information")
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct SchemaBuilder {
    properties: IndexMap<String, JsonValue>,
    required: Vec<String>,
    description: Option<String>,
}

impl SchemaBuilder {
    /// Create a new empty schema builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a string property.
    #[must_use]
    pub fn string(mut self, name: &str, desc: &str, required: bool) -> Self {
        self.properties.insert(
            name.to_string(),
            serde_json::json!({
                "type": "string",
                "description": desc
            }),
        );
        if required {
            self.required.push(name.to_string());
        }
        self
    }

    /// Add a string property with constraints.
    #[must_use]
    pub fn string_constrained(
        mut self,
        name: &str,
        desc: &str,
        required: bool,
        min_length: Option<usize>,
        max_length: Option<usize>,
        pattern: Option<&str>,
    ) -> Self {
        let mut prop = serde_json::json!({
            "type": "string",
            "description": desc
        });
        if let Some(min) = min_length {
            prop["minLength"] = JsonValue::from(min);
        }
        if let Some(max) = max_length {
            prop["maxLength"] = JsonValue::from(max);
        }
        if let Some(pat) = pattern {
            prop["pattern"] = JsonValue::String(pat.to_string());
        }
        self.properties.insert(name.to_string(), prop);
        if required {
            self.required.push(name.to_string());
        }
        self
    }

    /// Add an integer property.
    #[must_use]
    pub fn integer(mut self, name: &str, desc: &str, required: bool) -> Self {
        self.properties.insert(
            name.to_string(),
            serde_json::json!({
                "type": "integer",
                "description": desc
            }),
        );
        if required {
            self.required.push(name.to_string());
        }
        self
    }

    /// Add an integer property with constraints.
    #[must_use]
    pub fn integer_constrained(
        mut self,
        name: &str,
        desc: &str,
        required: bool,
        minimum: Option<i64>,
        maximum: Option<i64>,
    ) -> Self {
        let mut prop = serde_json::json!({
            "type": "integer",
            "description": desc
        });
        if let Some(min) = minimum {
            prop["minimum"] = JsonValue::from(min);
        }
        if let Some(max) = maximum {
            prop["maximum"] = JsonValue::from(max);
        }
        self.properties.insert(name.to_string(), prop);
        if required {
            self.required.push(name.to_string());
        }
        self
    }

    /// Add a number (float) property.
    #[must_use]
    pub fn number(mut self, name: &str, desc: &str, required: bool) -> Self {
        self.properties.insert(
            name.to_string(),
            serde_json::json!({
                "type": "number",
                "description": desc
            }),
        );
        if required {
            self.required.push(name.to_string());
        }
        self
    }

    /// Add a number property with constraints.
    #[must_use]
    pub fn number_constrained(
        mut self,
        name: &str,
        desc: &str,
        required: bool,
        minimum: Option<f64>,
        maximum: Option<f64>,
    ) -> Self {
        let mut prop = serde_json::json!({
            "type": "number",
            "description": desc
        });
        if let Some(min) = minimum {
            prop["minimum"] = JsonValue::from(min);
        }
        if let Some(max) = maximum {
            prop["maximum"] = JsonValue::from(max);
        }
        self.properties.insert(name.to_string(), prop);
        if required {
            self.required.push(name.to_string());
        }
        self
    }

    /// Add a boolean property.
    #[must_use]
    pub fn boolean(mut self, name: &str, desc: &str, required: bool) -> Self {
        self.properties.insert(
            name.to_string(),
            serde_json::json!({
                "type": "boolean",
                "description": desc
            }),
        );
        if required {
            self.required.push(name.to_string());
        }
        self
    }

    /// Add an array property.
    #[must_use]
    pub fn array(
        mut self,
        name: &str,
        desc: &str,
        items: JsonValue,
        required: bool,
    ) -> Self {
        self.properties.insert(
            name.to_string(),
            serde_json::json!({
                "type": "array",
                "description": desc,
                "items": items
            }),
        );
        if required {
            self.required.push(name.to_string());
        }
        self
    }

    /// Add a string array property.
    #[must_use]
    pub fn string_array(self, name: &str, desc: &str, required: bool) -> Self {
        self.array(name, desc, serde_json::json!({"type": "string"}), required)
    }

    /// Add an object property.
    #[must_use]
    pub fn object(
        mut self,
        name: &str,
        desc: &str,
        schema: ObjectJsonSchema,
        required: bool,
    ) -> Self {
        let mut obj = serde_json::to_value(&schema).unwrap_or_default();
        if let Some(obj_map) = obj.as_object_mut() {
            obj_map.insert(
                "description".to_string(),
                JsonValue::String(desc.to_string()),
            );
        }
        self.properties.insert(name.to_string(), obj);
        if required {
            self.required.push(name.to_string());
        }
        self
    }

    /// Add an enum property (string values).
    #[must_use]
    pub fn enum_values(
        mut self,
        name: &str,
        desc: &str,
        values: &[&str],
        required: bool,
    ) -> Self {
        self.properties.insert(
            name.to_string(),
            serde_json::json!({
                "type": "string",
                "description": desc,
                "enum": values
            }),
        );
        if required {
            self.required.push(name.to_string());
        }
        self
    }

    /// Add a raw JSON property.
    #[must_use]
    pub fn raw(mut self, name: &str, schema: JsonValue, required: bool) -> Self {
        self.properties.insert(name.to_string(), schema);
        if required {
            self.required.push(name.to_string());
        }
        self
    }

    /// Add a nullable property (anyOf with null).
    #[must_use]
    pub fn nullable(mut self, name: &str, schema: JsonValue, required: bool) -> Self {
        self.properties.insert(
            name.to_string(),
            serde_json::json!({
                "anyOf": [
                    schema,
                    {"type": "null"}
                ]
            }),
        );
        if required {
            self.required.push(name.to_string());
        }
        self
    }

    /// Set the schema description.
    #[must_use]
    pub fn description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Build the `ObjectJsonSchema`.
    #[must_use]
    pub fn build(self) -> ObjectJsonSchema {
        ObjectJsonSchema {
            schema_type: "object".to_string(),
            properties: self.properties,
            required: self.required,
            description: self.description,
            additional_properties: None,
            extra: std::collections::HashMap::new(),
        }
    }
}

/// Trait for types that can generate a JSON schema.
pub trait JsonSchemaGenerator {
    /// Generate a JSON schema for this type.
    fn json_schema() -> ObjectJsonSchema;
}

/// Common JSON schema types.
pub mod types {
    use serde_json::Value as JsonValue;

    /// Schema for a string.
    #[must_use]
    pub fn string() -> JsonValue {
        serde_json::json!({"type": "string"})
    }

    /// Schema for a string with description.
    #[must_use]
    pub fn string_with_desc(desc: &str) -> JsonValue {
        serde_json::json!({
            "type": "string",
            "description": desc
        })
    }

    /// Schema for an integer.
    #[must_use]
    pub fn integer() -> JsonValue {
        serde_json::json!({"type": "integer"})
    }

    /// Schema for an integer with description.
    #[must_use]
    pub fn integer_with_desc(desc: &str) -> JsonValue {
        serde_json::json!({
            "type": "integer",
            "description": desc
        })
    }

    /// Schema for a number.
    #[must_use]
    pub fn number() -> JsonValue {
        serde_json::json!({"type": "number"})
    }

    /// Schema for a number with description.
    #[must_use]
    pub fn number_with_desc(desc: &str) -> JsonValue {
        serde_json::json!({
            "type": "number",
            "description": desc
        })
    }

    /// Schema for a boolean.
    #[must_use]
    pub fn boolean() -> JsonValue {
        serde_json::json!({"type": "boolean"})
    }

    /// Schema for a boolean with description.
    #[must_use]
    pub fn boolean_with_desc(desc: &str) -> JsonValue {
        serde_json::json!({
            "type": "boolean",
            "description": desc
        })
    }

    /// Schema for an array of a given item type.
    #[must_use]
    pub fn array(items: JsonValue) -> JsonValue {
        serde_json::json!({
            "type": "array",
            "items": items
        })
    }

    /// Schema for a string array.
    #[must_use]
    pub fn string_array() -> JsonValue {
        array(string())
    }

    /// Schema for an integer array.
    #[must_use]
    pub fn integer_array() -> JsonValue {
        array(integer())
    }

    /// Schema for an enum.
    #[must_use]
    pub fn enum_values(values: &[&str]) -> JsonValue {
        serde_json::json!({
            "type": "string",
            "enum": values
        })
    }

    /// Schema for an enum with description.
    #[must_use]
    pub fn enum_with_desc(values: &[&str], desc: &str) -> JsonValue {
        serde_json::json!({
            "type": "string",
            "enum": values,
            "description": desc
        })
    }

    /// Schema for null.
    #[must_use]
    pub fn null() -> JsonValue {
        serde_json::json!({"type": "null"})
    }

    /// Schema for any of the given types.
    #[must_use]
    pub fn any_of(schemas: Vec<JsonValue>) -> JsonValue {
        serde_json::json!({"anyOf": schemas})
    }

    /// Schema for one of the given types.
    #[must_use]
    pub fn one_of(schemas: Vec<JsonValue>) -> JsonValue {
        serde_json::json!({"oneOf": schemas})
    }

    /// Schema for a nullable type.
    #[must_use]
    pub fn nullable(schema: JsonValue) -> JsonValue {
        any_of(vec![schema, null()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_builder() {
        let schema = SchemaBuilder::new()
            .string("name", "The name", true)
            .integer("age", "The age", false)
            .description("A person")
            .build();

        assert_eq!(schema.schema_type, "object");
        assert_eq!(schema.property_count(), 2);
        assert!(schema.is_required("name"));
        assert!(!schema.is_required("age"));
        assert_eq!(schema.description, Some("A person".to_string()));
    }

    #[test]
    fn test_schema_builder_all_types() {
        let schema = SchemaBuilder::new()
            .string("s", "string", true)
            .integer("i", "integer", true)
            .number("n", "number", true)
            .boolean("b", "boolean", true)
            .string_array("arr", "array", true)
            .enum_values("e", "enum", &["a", "b"], true)
            .build();

        assert_eq!(schema.property_count(), 6);
        assert_eq!(schema.required.len(), 6);
    }

    #[test]
    fn test_schema_builder_constraints() {
        let schema = SchemaBuilder::new()
            .string_constrained("s", "string", true, Some(1), Some(100), Some("^[a-z]+$"))
            .integer_constrained("i", "integer", true, Some(0), Some(100))
            .number_constrained("n", "number", true, Some(0.0), Some(1.0))
            .build();

        let s_prop = schema.get_property("s").unwrap();
        assert_eq!(s_prop["minLength"], 1);
        assert_eq!(s_prop["maxLength"], 100);
        assert_eq!(s_prop["pattern"], "^[a-z]+$");

        let i_prop = schema.get_property("i").unwrap();
        assert_eq!(i_prop["minimum"], 0);
        assert_eq!(i_prop["maximum"], 100);
    }

    #[test]
    fn test_schema_builder_nested() {
        let inner = SchemaBuilder::new()
            .string("street", "Street name", true)
            .string("city", "City name", true)
            .build();

        let schema = SchemaBuilder::new()
            .string("name", "Name", true)
            .object("address", "Address", inner, true)
            .build();

        assert!(schema.get_property("address").is_some());
    }

    #[test]
    fn test_schema_builder_nullable() {
        let schema = SchemaBuilder::new()
            .nullable("maybe", serde_json::json!({"type": "string"}), false)
            .build();

        let prop = schema.get_property("maybe").unwrap();
        assert!(prop.get("anyOf").is_some());
    }

    #[test]
    fn test_types_helpers() {
        assert_eq!(types::string()["type"], "string");
        assert_eq!(types::integer()["type"], "integer");
        assert_eq!(types::number()["type"], "number");
        assert_eq!(types::boolean()["type"], "boolean");
        assert!(types::string_array().get("items").is_some());
        assert!(types::enum_values(&["a", "b"]).get("enum").is_some());
        assert!(types::nullable(types::string()).get("anyOf").is_some());
    }

    #[test]
    fn test_types_with_desc() {
        let s = types::string_with_desc("A string");
        assert_eq!(s["description"], "A string");

        let e = types::enum_with_desc(&["x"], "An enum");
        assert_eq!(e["description"], "An enum");
    }
}

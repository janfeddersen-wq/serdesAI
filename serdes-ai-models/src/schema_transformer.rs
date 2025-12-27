//! JSON schema transformation for model compatibility.
//!
//! Different models have different requirements for JSON schemas.
//! This module provides transformers to convert schemas for compatibility.

use serde_json::Value as JsonValue;
use serdes_ai_tools::ObjectJsonSchema;
use std::collections::HashSet;

/// Transforms JSON schemas for model compatibility.
///
/// Different models have different requirements for JSON schemas.
/// This transformer can:
/// - Inline $ref definitions
/// - Remove unsupported keywords
/// - Convert types for compatibility
#[derive(Debug, Clone, Default)]
pub struct JsonSchemaTransformer {
    /// Inline all $ref definitions.
    pub inline_defs: bool,
    /// Remove unsupported keywords.
    pub remove_keywords: HashSet<String>,
    /// Convert additionalProperties: false to explicit property list.
    pub convert_additional_properties: bool,
    /// Remove default values.
    pub remove_defaults: bool,
    /// Remove format specifiers.
    pub remove_formats: bool,
    /// Remove examples.
    pub remove_examples: bool,
}

impl JsonSchemaTransformer {
    /// Create a new transformer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an OpenAI-compatible transformer.
    #[must_use]
    pub fn openai() -> Self {
        let mut remove_keywords = HashSet::new();
        // OpenAI strict mode doesn't support these
        remove_keywords.insert("examples".to_string());
        remove_keywords.insert("default".to_string());
        remove_keywords.insert("$id".to_string());
        remove_keywords.insert("$schema".to_string());

        Self {
            inline_defs: true,
            remove_keywords,
            convert_additional_properties: false,
            remove_defaults: true,
            remove_formats: false,
            remove_examples: true,
        }
    }

    /// Create an Anthropic-compatible transformer.
    #[must_use]
    pub fn anthropic() -> Self {
        let mut remove_keywords = HashSet::new();
        remove_keywords.insert("$id".to_string());
        remove_keywords.insert("$schema".to_string());

        Self {
            inline_defs: false,
            remove_keywords,
            convert_additional_properties: false,
            remove_defaults: false,
            remove_formats: false,
            remove_examples: false,
        }
    }

    /// Enable inlining of $ref definitions.
    #[must_use]
    pub fn with_inline_defs(mut self, inline: bool) -> Self {
        self.inline_defs = inline;
        self
    }

    /// Add a keyword to remove.
    #[must_use]
    pub fn remove_keyword(mut self, keyword: impl Into<String>) -> Self {
        self.remove_keywords.insert(keyword.into());
        self
    }

    /// Transform a JSON schema.
    pub fn transform(&self, schema: &ObjectJsonSchema) -> ObjectJsonSchema {
        let mut value = serde_json::to_value(schema).unwrap_or(JsonValue::Null);
        self.transform_value(&mut value);
        serde_json::from_value(value).unwrap_or_default()
    }

    /// Transform a JSON value in place.
    pub(crate) fn transform_value(&self, value: &mut JsonValue) {
        match value {
            JsonValue::Object(map) => {
                // Remove unsupported keywords
                for keyword in &self.remove_keywords {
                    map.remove(keyword);
                }

                // Remove defaults if configured
                if self.remove_defaults {
                    map.remove("default");
                }

                // Remove examples if configured
                if self.remove_examples {
                    map.remove("examples");
                    map.remove("example");
                }

                // Remove formats if configured
                if self.remove_formats {
                    map.remove("format");
                }

                // Inline $ref if configured
                if self.inline_defs {
                    if let Some(defs) = map.remove("$defs").or_else(|| map.remove("definitions")) {
                        // Store defs for reference resolution
                        self.inline_refs_in_object(map, &defs);
                    }
                }

                // Recursively transform nested values
                for (_, v) in map.iter_mut() {
                    self.transform_value(v);
                }
            }
            JsonValue::Array(arr) => {
                for v in arr.iter_mut() {
                    self.transform_value(v);
                }
            }
            _ => {}
        }
    }

    /// Inline $ref references in an object.
    fn inline_refs_in_object(
        &self,
        map: &mut serde_json::Map<String, JsonValue>,
        defs: &JsonValue,
    ) {
        // This is a simplified implementation
        // A full implementation would recursively resolve all $ref
        for (_, v) in map.iter_mut() {
            self.inline_refs_in_value(v, defs);
        }
    }

    /// Inline $ref references in a value.
    fn inline_refs_in_value(&self, value: &mut JsonValue, defs: &JsonValue) {
        match value {
            JsonValue::Object(map) => {
                if let Some(ref_path) = map.get("$ref").and_then(|v| v.as_str()) {
                    // Parse the $ref path and resolve it
                    if let Some(resolved) = self.resolve_ref(ref_path, defs) {
                        *value = resolved;
                    }
                } else {
                    for (_, v) in map.iter_mut() {
                        self.inline_refs_in_value(v, defs);
                    }
                }
            }
            JsonValue::Array(arr) => {
                for v in arr.iter_mut() {
                    self.inline_refs_in_value(v, defs);
                }
            }
            _ => {}
        }
    }

    /// Resolve a $ref path.
    fn resolve_ref(&self, ref_path: &str, defs: &JsonValue) -> Option<JsonValue> {
        // Handle #/$defs/Name or #/definitions/Name format
        let parts: Vec<&str> = ref_path.trim_start_matches('#').split('/').collect();

        if parts.len() >= 3 && (parts[1] == "$defs" || parts[1] == "definitions") {
            let name = parts[2];
            if let Some(def) = defs.get(name) {
                return Some(def.clone());
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_new() {
        let transformer = JsonSchemaTransformer::new();
        assert!(!transformer.inline_defs);
        assert!(transformer.remove_keywords.is_empty());
    }

    #[test]
    fn test_transformer_openai() {
        let transformer = JsonSchemaTransformer::openai();
        assert!(transformer.inline_defs);
        assert!(transformer.remove_keywords.contains("examples"));
        assert!(transformer.remove_keywords.contains("default"));
        assert!(transformer.remove_keywords.contains("$id"));
        assert!(transformer.remove_keywords.contains("$schema"));
        assert!(transformer.remove_defaults);
        assert!(transformer.remove_examples);
    }

    #[test]
    fn test_transformer_anthropic() {
        let transformer = JsonSchemaTransformer::anthropic();
        assert!(!transformer.inline_defs);
        assert!(transformer.remove_keywords.contains("$id"));
        assert!(transformer.remove_keywords.contains("$schema"));
        assert!(!transformer.remove_defaults);
        assert!(!transformer.remove_examples);
    }

    #[test]
    fn test_transformer_builder() {
        let transformer = JsonSchemaTransformer::new()
            .with_inline_defs(true)
            .remove_keyword("foo")
            .remove_keyword("bar");

        assert!(transformer.inline_defs);
        assert!(transformer.remove_keywords.contains("foo"));
        assert!(transformer.remove_keywords.contains("bar"));
    }

    #[test]
    fn test_transformer_remove_keywords() {
        let transformer = JsonSchemaTransformer::new()
            .remove_keyword("examples")
            .remove_keyword("$id");

        let mut value = serde_json::json!({
            "type": "object",
            "$id": "test",
            "examples": [{}],
            "properties": {
                "name": {
                    "type": "string",
                    "examples": ["Alice"]
                }
            }
        });

        transformer.transform_value(&mut value);

        assert!(value.get("$id").is_none());
        assert!(value.get("examples").is_none());
        assert!(value["properties"]["name"].get("examples").is_none());
    }

    #[test]
    fn test_transformer_remove_defaults() {
        let mut transformer = JsonSchemaTransformer::new();
        transformer.remove_defaults = true;

        let mut value = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "default": "John"
                }
            }
        });

        transformer.transform_value(&mut value);

        assert!(value["properties"]["name"].get("default").is_none());
    }

    #[test]
    fn test_transformer_remove_formats() {
        let mut transformer = JsonSchemaTransformer::new();
        transformer.remove_formats = true;

        let mut value = serde_json::json!({
            "type": "string",
            "format": "email"
        });

        transformer.transform_value(&mut value);

        assert!(value.get("format").is_none());
    }

    #[test]
    fn test_transformer_remove_examples_flag() {
        let mut transformer = JsonSchemaTransformer::new();
        transformer.remove_examples = true;

        let mut value = serde_json::json!({
            "type": "string",
            "examples": ["hello"],
            "example": "world"
        });

        transformer.transform_value(&mut value);

        assert!(value.get("examples").is_none());
        assert!(value.get("example").is_none());
    }

    #[test]
    fn test_transformer_array_recursion() {
        let transformer = JsonSchemaTransformer::new().remove_keyword("$id");

        let mut value = serde_json::json!({
            "anyOf": [
                { "type": "string", "$id": "str" },
                { "type": "number", "$id": "num" }
            ]
        });

        transformer.transform_value(&mut value);

        assert!(value["anyOf"][0].get("$id").is_none());
        assert!(value["anyOf"][1].get("$id").is_none());
    }
}

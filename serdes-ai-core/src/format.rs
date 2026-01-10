//! Format data as XML for LLM prompts.
//!
//! LLMs often work better with XML-formatted data for structured examples.
//! This module provides utilities to convert serializable Rust types into
//! XML format suitable for inclusion in prompts.
//!
//! # Example
//!
//! ```rust
//! use serdes_ai_core::format::{format_as_xml, XmlFormatOptions};
//!
//! let data = serde_json::json!({
//!     "name": "John",
//!     "age": 30,
//!     "hobbies": ["reading", "coding"]
//! });
//!
//! let xml = format_as_xml(&data, Some("user")).unwrap();
//! // <user>
//! //   <name>John</name>
//! //   <age>30</age>
//! //   <hobbies>
//! //     <item>reading</item>
//! //     <item>coding</item>
//! //   </hobbies>
//! // </user>
//! ```

use serde::Serialize;
use thiserror::Error;

/// Options for XML formatting.
#[derive(Debug, Clone)]
pub struct XmlFormatOptions {
    /// Root tag name. If None, no root tag is added.
    pub root_tag: Option<String>,
    /// Tag name for items in sequences.
    pub item_tag: String,
    /// String representation for None values.
    pub none_str: String,
    /// Indentation string (None for compact output).
    pub indent: Option<String>,
}

impl Default for XmlFormatOptions {
    fn default() -> Self {
        Self {
            root_tag: None,
            item_tag: "item".to_string(),
            none_str: "null".to_string(),
            indent: Some("  ".to_string()),
        }
    }
}

impl XmlFormatOptions {
    /// Create new options with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the root tag.
    #[must_use]
    pub fn with_root_tag(mut self, tag: impl Into<String>) -> Self {
        self.root_tag = Some(tag.into());
        self
    }

    /// Set the item tag for sequences.
    #[must_use]
    pub fn with_item_tag(mut self, tag: impl Into<String>) -> Self {
        self.item_tag = tag.into();
        self
    }

    /// Set the none string representation.
    #[must_use]
    pub fn with_none_str(mut self, s: impl Into<String>) -> Self {
        self.none_str = s.into();
        self
    }

    /// Set the indentation string. None for compact output.
    #[must_use]
    pub fn with_indent(mut self, indent: Option<String>) -> Self {
        self.indent = indent;
        self
    }

    /// Disable indentation for compact output.
    #[must_use]
    pub fn compact(mut self) -> Self {
        self.indent = None;
        self
    }
}

/// Error type for XML formatting operations.
#[derive(Debug, Error)]
pub enum XmlFormatError {
    /// Serialization error when converting to JSON intermediate.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Format a serializable value as XML.
///
/// This is the simple API that uses default options with an optional root tag.
///
/// # Arguments
///
/// * `value` - Any serializable value
/// * `root_tag` - Optional root tag name to wrap the output
///
/// # Example
///
/// ```rust
/// use serdes_ai_core::format::format_as_xml;
///
/// let data = serde_json::json!({
///     "name": "Alice",
///     "age": 25
/// });
///
/// let xml = format_as_xml(&data, Some("person")).unwrap();
/// assert!(xml.contains("<person>"));
/// assert!(xml.contains("<name>Alice</name>"));
/// ```
pub fn format_as_xml<T: Serialize>(
    value: &T,
    root_tag: Option<&str>,
) -> Result<String, XmlFormatError> {
    let options = XmlFormatOptions {
        root_tag: root_tag.map(String::from),
        ..Default::default()
    };
    format_as_xml_with_options(value, &options)
}

/// Format with full options control.
///
/// # Arguments
///
/// * `value` - Any serializable value
/// * `options` - Formatting options
///
/// # Example
///
/// ```rust
/// use serdes_ai_core::format::{format_as_xml_with_options, XmlFormatOptions};
///
/// let data = vec!["apple", "banana", "cherry"];
///
/// let options = XmlFormatOptions::new()
///     .with_root_tag("fruits")
///     .with_item_tag("fruit");
///
/// let xml = format_as_xml_with_options(&data, &options).unwrap();
/// assert!(xml.contains("<fruit>apple</fruit>"));
/// ```
pub fn format_as_xml_with_options<T: Serialize>(
    value: &T,
    options: &XmlFormatOptions,
) -> Result<String, XmlFormatError> {
    // Convert to serde_json::Value first for uniform handling
    let json_value = serde_json::to_value(value)?;

    let mut output = String::new();

    if let Some(ref root_tag) = options.root_tag {
        // With root tag, wrap the content
        output.push_str(&format!("<{root_tag}>"));
        if options.indent.is_some() {
            output.push('\n');
        }
        value_to_xml_inner(&json_value, options, 1, &mut output);
        output.push_str(&format!("</{root_tag}>"));
    } else {
        // No root tag, just output the content
        value_to_xml_inner(&json_value, options, 0, &mut output);
    }

    Ok(output)
}

/// Internal function to convert JSON value to XML string.
fn value_to_xml_inner(
    value: &serde_json::Value,
    options: &XmlFormatOptions,
    depth: usize,
    output: &mut String,
) {
    let indent = get_indent(options, depth);

    match value {
        serde_json::Value::Null => {
            output.push_str(&options.none_str);
        }
        serde_json::Value::Bool(b) => {
            output.push_str(if *b { "true" } else { "false" });
        }
        serde_json::Value::Number(n) => {
            output.push_str(&n.to_string());
        }
        serde_json::Value::String(s) => {
            output.push_str(&escape_xml(s));
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                output.push_str(&indent);
                output.push_str(&format!("<{}>", options.item_tag));

                if is_complex_value(item) {
                    if options.indent.is_some() {
                        output.push('\n');
                    }
                    value_to_xml_inner(item, options, depth + 1, output);
                    output.push_str(&indent);
                } else {
                    value_to_xml_inner(item, options, depth + 1, output);
                }

                output.push_str(&format!("</{}>", options.item_tag));
                if options.indent.is_some() {
                    output.push('\n');
                }
            }
        }
        serde_json::Value::Object(map) => {
            for (key, val) in map {
                let tag = sanitize_tag_name(key);
                output.push_str(&indent);
                output.push_str(&format!("<{tag}>"));

                if is_complex_value(val) {
                    if options.indent.is_some() {
                        output.push('\n');
                    }
                    value_to_xml_inner(val, options, depth + 1, output);
                    output.push_str(&indent);
                } else {
                    value_to_xml_inner(val, options, depth + 1, output);
                }

                output.push_str(&format!("</{tag}>"));
                if options.indent.is_some() {
                    output.push('\n');
                }
            }
        }
    }
}

/// Check if a value is complex (object or array) and needs nested formatting.
fn is_complex_value(value: &serde_json::Value) -> bool {
    matches!(
        value,
        serde_json::Value::Object(_) | serde_json::Value::Array(_)
    )
}

/// Get the indentation string for a given depth.
fn get_indent(options: &XmlFormatOptions, depth: usize) -> String {
    options
        .indent
        .as_ref()
        .map(|i| i.repeat(depth))
        .unwrap_or_default()
}

/// Sanitize a string to be a valid XML tag name.
///
/// XML tag names must:
/// - Start with a letter or underscore
/// - Contain only letters, digits, hyphens, underscores, and periods
/// - Not start with "xml" (case-insensitive)
fn sanitize_tag_name(name: &str) -> String {
    let mut result = String::with_capacity(name.len());

    for (i, c) in name.chars().enumerate() {
        if i == 0 {
            // First character must be letter or underscore
            if c.is_ascii_alphabetic() || c == '_' {
                result.push(c);
            } else {
                result.push('_');
                if c.is_ascii_alphanumeric() {
                    result.push(c);
                }
            }
        } else {
            // Subsequent characters can be letters, digits, hyphens, underscores, periods
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                result.push(c);
            } else {
                result.push('_');
            }
        }
    }

    // Handle empty result
    if result.is_empty() {
        return "_".to_string();
    }

    result
}

/// Escape special XML characters in a string.
fn escape_xml(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => result.push_str("&amp;"),
            '<' => result.push_str("&lt;"),
            '>' => result.push_str("&gt;"),
            '"' => result.push_str("&quot;"),
            '\'' => result.push_str("&apos;"),
            _ => result.push(c),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_simple_object() {
        let data = json!({
            "name": "John",
            "age": 30
        });

        let xml = format_as_xml(&data, Some("user")).unwrap();
        assert!(xml.contains("<user>"));
        assert!(xml.contains("</user>"));
        assert!(xml.contains("<name>John</name>"));
        assert!(xml.contains("<age>30</age>"));
    }

    #[test]
    fn test_nested_object() {
        let data = json!({
            "person": {
                "name": "Alice",
                "address": {
                    "city": "NYC"
                }
            }
        });

        let xml = format_as_xml(&data, Some("root")).unwrap();
        assert!(xml.contains("<city>NYC</city>"));
    }

    #[test]
    fn test_array() {
        let data = json!({
            "hobbies": ["reading", "coding", "gaming"]
        });

        let xml = format_as_xml(&data, None).unwrap();
        assert!(xml.contains("<item>reading</item>"));
        assert!(xml.contains("<item>coding</item>"));
        assert!(xml.contains("<item>gaming</item>"));
    }

    #[test]
    fn test_custom_item_tag() {
        let data = vec!["apple", "banana"];

        let options = XmlFormatOptions::new()
            .with_root_tag("fruits")
            .with_item_tag("fruit");

        let xml = format_as_xml_with_options(&data, &options).unwrap();
        assert!(xml.contains("<fruit>apple</fruit>"));
        assert!(xml.contains("<fruit>banana</fruit>"));
    }

    #[test]
    fn test_compact_output() {
        let data = json!({"a": 1, "b": 2});

        let options = XmlFormatOptions::new().with_root_tag("data").compact();

        let xml = format_as_xml_with_options(&data, &options).unwrap();
        // Should not have newlines
        assert!(!xml.contains("\n"));
    }

    #[test]
    fn test_null_value() {
        let data = json!({"value": null});

        let xml = format_as_xml(&data, None).unwrap();
        assert!(xml.contains("<value>null</value>"));
    }

    #[test]
    fn test_custom_none_str() {
        let data = json!({"value": null});

        let options = XmlFormatOptions::new().with_none_str("N/A");

        let xml = format_as_xml_with_options(&data, &options).unwrap();
        assert!(xml.contains("<value>N/A</value>"));
    }

    #[test]
    fn test_boolean_values() {
        let data = json!({"active": true, "disabled": false});

        let xml = format_as_xml(&data, None).unwrap();
        assert!(xml.contains("<active>true</active>"));
        assert!(xml.contains("<disabled>false</disabled>"));
    }

    #[test]
    fn test_xml_escape() {
        let data = json!({"text": "<script>alert('xss')</script>"});

        let xml = format_as_xml(&data, None).unwrap();
        assert!(xml.contains("&lt;script&gt;"));
        assert!(xml.contains("&apos;"));
    }

    #[test]
    fn test_sanitize_tag_name() {
        assert_eq!(sanitize_tag_name("valid_name"), "valid_name");
        assert_eq!(sanitize_tag_name("123start"), "_123start");
        assert_eq!(sanitize_tag_name("has space"), "has_space");
        assert_eq!(sanitize_tag_name("special@char"), "special_char");
        assert_eq!(sanitize_tag_name(""), "_");
    }

    #[test]
    fn test_escape_xml() {
        assert_eq!(escape_xml("hello"), "hello");
        assert_eq!(escape_xml("a & b"), "a &amp; b");
        assert_eq!(escape_xml("<tag>"), "&lt;tag&gt;");
        assert_eq!(escape_xml("\"quoted\""), "&quot;quoted&quot;");
        assert_eq!(escape_xml("it's"), "it&apos;s");
    }

    #[test]
    fn test_no_root_tag() {
        let data = json!({"key": "value"});

        let xml = format_as_xml(&data, None).unwrap();
        assert!(xml.contains("<key>value</key>"));
        // Should not have a root wrapper
        assert!(!xml.starts_with("<None"));
    }

    #[test]
    fn test_complex_nested_structure() {
        let data = json!({
            "users": [
                {"name": "Alice", "roles": ["admin", "user"]},
                {"name": "Bob", "roles": ["user"]}
            ],
            "metadata": {
                "version": "1.0",
                "count": 2
            }
        });

        let xml = format_as_xml(&data, Some("response")).unwrap();
        assert!(xml.contains("<response>"));
        assert!(xml.contains("</response>"));
        assert!(xml.contains("<name>Alice</name>"));
        assert!(xml.contains("<version>1.0</version>"));
    }

    #[test]
    fn test_options_builder() {
        let options = XmlFormatOptions::new()
            .with_root_tag("root")
            .with_item_tag("entry")
            .with_none_str("nil")
            .with_indent(Some("    ".to_string()));

        assert_eq!(options.root_tag, Some("root".to_string()));
        assert_eq!(options.item_tag, "entry");
        assert_eq!(options.none_str, "nil");
        assert_eq!(options.indent, Some("    ".to_string()));
    }

    #[test]
    fn test_struct_serialization() {
        #[derive(Serialize)]
        struct Person {
            name: String,
            age: u32,
        }

        let person = Person {
            name: "Charlie".to_string(),
            age: 35,
        };

        let xml = format_as_xml(&person, Some("person")).unwrap();
        assert!(xml.contains("<name>Charlie</name>"));
        assert!(xml.contains("<age>35</age>"));
    }
}

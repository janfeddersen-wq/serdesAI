//! Output parsing utilities.
//!
//! This module provides utilities for extracting and parsing structured
//! output from model responses.

use serde::de::DeserializeOwned;
use serde_json::Value as JsonValue;

use crate::error::OutputParseError;

/// Extract JSON from text that might contain markdown or prose.
///
/// This function handles various formats:
/// - Pure JSON objects/arrays
/// - JSON wrapped in markdown code blocks
/// - JSON embedded in prose text
/// - JSON with trailing comments or text
///
/// # Example
///
/// ```rust
/// use serdes_ai_output::parser::extract_json_from_text;
///
/// let text = r#"{"name": "Alice", "age": 30}"#;
/// let json = extract_json_from_text(text).unwrap();
/// assert!(json.contains("Alice"));
/// ```
pub fn extract_json_from_text(text: &str) -> Result<String, OutputParseError> {
    let text = text.trim();

    // Try markdown code block with json language
    if let Some(json) = extract_from_markdown_json(text) {
        return Ok(json);
    }

    // Try markdown code block without language
    if let Some(json) = extract_from_markdown_plain(text) {
        return Ok(json);
    }

    // Try to find JSON object
    if let Some(json) = find_json_object(text) {
        return Ok(json);
    }

    // Try to find JSON array
    if let Some(json) = find_json_array(text) {
        return Ok(json);
    }

    // Last resort: try parsing the whole thing
    if serde_json::from_str::<JsonValue>(text).is_ok() {
        return Ok(text.to_string());
    }

    Err(OutputParseError::NoJsonFound)
}

/// Extract JSON from a markdown ```json ... ``` block.
fn extract_from_markdown_json(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let start_markers = ["```json\n", "```json\r\n", "```json ", "```json"];

    for marker in start_markers {
        if let Some(start) = lower.find(marker) {
            let content_start = start + marker.len();
            if let Some(end) = text[content_start..].find("```") {
                let content = text[content_start..content_start + end].trim();
                if serde_json::from_str::<JsonValue>(content).is_ok() {
                    return Some(content.to_string());
                }
            }
        }
    }
    None
}

/// Extract JSON from a markdown ``` ... ``` block without language.
fn extract_from_markdown_plain(text: &str) -> Option<String> {
    if !text.starts_with("```") {
        return None;
    }

    // Skip the opening ``` and any language identifier
    let rest = &text[3..];
    let content_start = rest.find('\n').map(|i| i + 1).unwrap_or(0);
    let rest = &rest[content_start..];

    if let Some(end) = rest.find("```") {
        let content = rest[..end].trim();
        if serde_json::from_str::<JsonValue>(content).is_ok() {
            return Some(content.to_string());
        }
    }
    None
}

/// Find a JSON object in text using brace matching.
fn find_json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;

    // Count braces to find the matching closing brace
    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, c) in text[start..].char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }

        match c {
            '\\' if in_string => escape_next = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    let candidate = &text[start..=start + i];
                    if serde_json::from_str::<JsonValue>(candidate).is_ok() {
                        return Some(candidate.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Find a JSON array in text using bracket matching.
fn find_json_array(text: &str) -> Option<String> {
    let start = text.find('[')?;

    // Count brackets to find the matching closing bracket
    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, c) in text[start..].char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }

        match c {
            '\\' if in_string => escape_next = true,
            '"' => in_string = !in_string,
            '[' if !in_string => depth += 1,
            ']' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    let candidate = &text[start..=start + i];
                    if serde_json::from_str::<JsonValue>(candidate).is_ok() {
                        return Some(candidate.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse JSON from text into a typed value.
///
/// This is a convenience function that combines extraction and parsing.
pub fn parse_json_from_text<T: DeserializeOwned>(text: &str) -> Result<T, OutputParseError> {
    let json_str = extract_json_from_text(text)?;
    serde_json::from_str(&json_str).map_err(OutputParseError::JsonParse)
}

/// Parse a JSON value into a typed value.
pub fn parse_json_value<T: DeserializeOwned>(value: &JsonValue) -> Result<T, OutputParseError> {
    serde_json::from_value(value.clone()).map_err(OutputParseError::JsonParse)
}

/// Check if text appears to contain JSON.
pub fn looks_like_json(text: &str) -> bool {
    let text = text.trim();
    text.starts_with('{') || text.starts_with('[') || text.contains("```json")
}

/// Normalize JSON text by removing surrounding whitespace and common wrappers.
pub fn normalize_json(text: &str) -> String {
    let text = text.trim();

    // Remove markdown code block if present
    if let Ok(json) = extract_json_from_text(text) {
        return json;
    }

    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestStruct {
        name: String,
        value: i32,
    }

    #[test]
    fn test_extract_pure_json_object() {
        let text = r#"{"name": "test", "value": 42}"#;
        let result = extract_json_from_text(text).unwrap();
        assert_eq!(result, text);
    }

    #[test]
    fn test_extract_pure_json_array() {
        let text = r#"[1, 2, 3]"#;
        let result = extract_json_from_text(text).unwrap();
        assert_eq!(result, text);
    }

    #[test]
    fn test_extract_markdown_json_block() {
        let text = r#"Here is the result:
```json
{"name": "test", "value": 42}
```
Done!"#;
        let result = extract_json_from_text(text).unwrap();
        assert_eq!(result, r#"{"name": "test", "value": 42}"#);
    }

    #[test]
    fn test_extract_markdown_plain_block() {
        let text = r#"```
{"key": "value"}
```"#;
        let result = extract_json_from_text(text).unwrap();
        assert_eq!(result, r#"{"key": "value"}"#);
    }

    #[test]
    fn test_extract_embedded_object() {
        let text = r#"The answer is {"x": 1, "y": 2} and that's it."#;
        let result = extract_json_from_text(text).unwrap();
        assert_eq!(result, r#"{"x": 1, "y": 2}"#);
    }

    #[test]
    fn test_extract_embedded_array() {
        let text = r#"Items: ["a", "b", "c"] are listed."#;
        let result = extract_json_from_text(text).unwrap();
        assert_eq!(result, r#"["a", "b", "c"]"#);
    }

    #[test]
    fn test_extract_nested_objects() {
        let text = r#"{"outer": {"inner": {"deep": true}}}"#;
        let result = extract_json_from_text(text).unwrap();
        assert_eq!(result, text);
    }

    #[test]
    fn test_extract_with_escaped_quotes() {
        let text = r#"{"message": "He said \"hello\""}"#;
        let result = extract_json_from_text(text).unwrap();
        assert_eq!(result, text);
    }

    #[test]
    fn test_extract_no_json() {
        let text = "This is just plain text with no JSON at all.";
        let result = extract_json_from_text(text);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_json_from_text() {
        let text = r#"{"name": "Alice", "value": 100}"#;
        let result: TestStruct = parse_json_from_text(text).unwrap();
        assert_eq!(result.name, "Alice");
        assert_eq!(result.value, 100);
    }

    #[test]
    fn test_parse_json_from_markdown() {
        let text = r#"```json
{"name": "Bob", "value": 200}
```"#;
        let result: TestStruct = parse_json_from_text(text).unwrap();
        assert_eq!(result.name, "Bob");
        assert_eq!(result.value, 200);
    }

    #[test]
    fn test_looks_like_json() {
        assert!(looks_like_json("{\"key\": \"value\"}"));
        assert!(looks_like_json("[1, 2, 3]"));
        assert!(looks_like_json("```json\n{}\n```"));
        assert!(looks_like_json("  { \"spaced\": true } "));
        assert!(!looks_like_json("Just plain text"));
    }

    #[test]
    fn test_parse_json_value() {
        let value = serde_json::json!({"name": "Charlie", "value": 300});
        let result: TestStruct = parse_json_value(&value).unwrap();
        assert_eq!(result.name, "Charlie");
        assert_eq!(result.value, 300);
    }

    #[test]
    fn test_multiple_json_objects() {
        // Should find the first valid JSON object
        let text = r#"First: {"a": 1}, Second: {"b": 2}"#;
        let result = extract_json_from_text(text).unwrap();
        assert_eq!(result, r#"{"a": 1}"#);
    }

    #[test]
    fn test_json_with_braces_in_strings() {
        let text = r#"{"code": "if (x) { return y; }", "valid": true}"#;
        let result = extract_json_from_text(text).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["valid"], true);
    }
}

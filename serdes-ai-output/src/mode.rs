//! Output mode definitions.
//!
//! This module defines the different strategies for obtaining structured
//! output from language models.

use serde::{Deserialize, Serialize};
use std::fmt;

/// How the model should generate structured output.
///
/// Different models support different output modes, and the choice of mode
/// affects reliability, latency, and token usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    /// Let the model output free-form text, parse it ourselves.
    ///
    /// This is the most flexible but least reliable mode. The model
    /// is not constrained to any format, and parsing may fail.
    Text,

    /// Use native structured output (response_format in OpenAI).
    ///
    /// This uses the model's built-in JSON mode or structured output
    /// feature. Not all models support this.
    Native,

    /// Use prompted output with JSON in response.
    ///
    /// The model is instructed via the system prompt to output JSON,
    /// but there's no structural enforcement.
    Prompted,

    /// Use a tool call to return the result.
    ///
    /// This is the most reliable mode for structured output. A special
    /// "result" tool is added, and the model is instructed to call it
    /// with the final response.
    Tool,
}

impl Default for OutputMode {
    fn default() -> Self {
        OutputMode::Tool
    }
}

impl fmt::Display for OutputMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputMode::Text => write!(f, "text"),
            OutputMode::Native => write!(f, "native"),
            OutputMode::Prompted => write!(f, "prompted"),
            OutputMode::Tool => write!(f, "tool"),
        }
    }
}

impl OutputMode {
    /// Whether this mode requires tool support.
    #[must_use]
    pub fn requires_tools(&self) -> bool {
        matches!(self, OutputMode::Tool)
    }

    /// Whether this mode requires native JSON support.
    #[must_use]
    pub fn requires_native_json(&self) -> bool {
        matches!(self, OutputMode::Native)
    }

    /// Whether this mode produces structured output.
    #[must_use]
    pub fn is_structured(&self) -> bool {
        !matches!(self, OutputMode::Text)
    }

    /// Get all available output modes.
    #[must_use]
    pub fn all() -> &'static [OutputMode] {
        &[
            OutputMode::Text,
            OutputMode::Native,
            OutputMode::Prompted,
            OutputMode::Tool,
        ]
    }
}

/// Parse an output mode from a string.
impl std::str::FromStr for OutputMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" => Ok(OutputMode::Text),
            "native" => Ok(OutputMode::Native),
            "prompted" | "json" => Ok(OutputMode::Prompted),
            "tool" | "function" | "function_call" => Ok(OutputMode::Tool),
            _ => Err(format!("Unknown output mode: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_mode() {
        assert_eq!(OutputMode::default(), OutputMode::Tool);
    }

    #[test]
    fn test_display() {
        assert_eq!(OutputMode::Text.to_string(), "text");
        assert_eq!(OutputMode::Native.to_string(), "native");
        assert_eq!(OutputMode::Prompted.to_string(), "prompted");
        assert_eq!(OutputMode::Tool.to_string(), "tool");
    }

    #[test]
    fn test_from_str() {
        assert_eq!("text".parse::<OutputMode>().unwrap(), OutputMode::Text);
        assert_eq!("native".parse::<OutputMode>().unwrap(), OutputMode::Native);
        assert_eq!("prompted".parse::<OutputMode>().unwrap(), OutputMode::Prompted);
        assert_eq!("json".parse::<OutputMode>().unwrap(), OutputMode::Prompted);
        assert_eq!("tool".parse::<OutputMode>().unwrap(), OutputMode::Tool);
        assert_eq!("function".parse::<OutputMode>().unwrap(), OutputMode::Tool);
    }

    #[test]
    fn test_requires_tools() {
        assert!(!OutputMode::Text.requires_tools());
        assert!(!OutputMode::Native.requires_tools());
        assert!(!OutputMode::Prompted.requires_tools());
        assert!(OutputMode::Tool.requires_tools());
    }

    #[test]
    fn test_is_structured() {
        assert!(!OutputMode::Text.is_structured());
        assert!(OutputMode::Native.is_structured());
        assert!(OutputMode::Prompted.is_structured());
        assert!(OutputMode::Tool.is_structured());
    }

    #[test]
    fn test_serde() {
        let mode = OutputMode::Tool;
        let json = serde_json::to_string(&mode).unwrap();
        assert_eq!(json, "\"tool\"");

        let parsed: OutputMode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, mode);
    }
}

//! Error types for output parsing and validation.

use thiserror::Error;

/// Error during output parsing.
#[derive(Debug, Error)]
pub enum OutputParseError {
    /// Failed to parse JSON.
    #[error("Failed to parse JSON: {0}")]
    JsonParse(#[from] serde_json::Error),

    /// Missing required field in the output.
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Invalid value for a field.
    #[error("Invalid value for field '{field}': {message}")]
    InvalidField {
        /// The field name.
        field: String,
        /// The error message.
        message: String,
    },

    /// Unexpected tool call.
    #[error("Unexpected tool call: expected '{expected}', got '{actual}'")]
    UnexpectedTool {
        /// Expected tool name.
        expected: String,
        /// Actual tool name received.
        actual: String,
    },

    /// Output is not valid JSON.
    #[error("Output is not valid JSON")]
    NotJson,

    /// No JSON found in the output.
    #[error("No JSON object or array found in output")]
    NoJsonFound,

    /// Pattern mismatch.
    #[error("Output does not match required pattern: {pattern}")]
    PatternMismatch {
        /// The pattern that was expected.
        pattern: String,
    },

    /// Length constraint violation.
    #[error("Output length {actual} is {direction} allowed {limit}")]
    LengthViolation {
        /// Actual length.
        actual: usize,
        /// The limit.
        limit: usize,
        /// Direction ("below" or "above").
        direction: &'static str,
    },

    /// Regex compilation error.
    #[error("Invalid regex pattern: {0}")]
    RegexError(#[from] regex::Error),

    /// Custom parse error.
    #[error("Parse error: {0}")]
    Custom(String),
}

impl OutputParseError {
    /// Create a custom parse error.
    pub fn custom(msg: impl Into<String>) -> Self {
        Self::Custom(msg.into())
    }

    /// Create a missing field error.
    pub fn missing_field(field: impl Into<String>) -> Self {
        Self::MissingField(field.into())
    }

    /// Create an invalid field error.
    pub fn invalid_field(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InvalidField {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Create an unexpected tool error.
    pub fn unexpected_tool(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::UnexpectedTool {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create a length violation error for being too short.
    pub fn too_short(actual: usize, min: usize) -> Self {
        Self::LengthViolation {
            actual,
            limit: min,
            direction: "below",
        }
    }

    /// Create a length violation error for being too long.
    pub fn too_long(actual: usize, max: usize) -> Self {
        Self::LengthViolation {
            actual,
            limit: max,
            direction: "above",
        }
    }
}

/// Error during output validation.
#[derive(Debug, Error)]
pub enum OutputValidationError {
    /// Validation failed.
    #[error("Validation failed: {message}")]
    Failed {
        /// The error message.
        message: String,
        /// Whether the model should retry.
        retry: bool,
    },

    /// Model should retry with a different response.
    #[error("Model retry requested: {0}")]
    ModelRetry(String),

    /// Parse error during validation.
    #[error(transparent)]
    Parse(#[from] OutputParseError),
}

impl OutputValidationError {
    /// Create a retry error with a message for the model.
    pub fn retry(msg: impl Into<String>) -> Self {
        Self::ModelRetry(msg.into())
    }

    /// Create a failed validation error (no retry).
    pub fn failed(msg: impl Into<String>) -> Self {
        Self::Failed {
            message: msg.into(),
            retry: false,
        }
    }

    /// Create a failed validation error (with retry).
    pub fn failed_retry(msg: impl Into<String>) -> Self {
        Self::Failed {
            message: msg.into(),
            retry: true,
        }
    }

    /// Whether the model should retry.
    pub fn should_retry(&self) -> bool {
        match self {
            Self::Failed { retry, .. } => *retry,
            Self::ModelRetry(_) => true,
            Self::Parse(_) => false,
        }
    }

    /// Get the retry message if applicable.
    pub fn retry_message(&self) -> Option<&str> {
        match self {
            Self::ModelRetry(msg) => Some(msg),
            Self::Failed { message, retry: true } => Some(message),
            _ => None,
        }
    }
}

/// Result type for output parsing.
pub type ParseResult<T> = Result<T, OutputParseError>;

/// Result type for output validation.
pub type ValidationResult<T> = Result<T, OutputValidationError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error_custom() {
        let err = OutputParseError::custom("Something went wrong");
        assert!(err.to_string().contains("Something went wrong"));
    }

    #[test]
    fn test_parse_error_missing_field() {
        let err = OutputParseError::missing_field("name");
        assert!(err.to_string().contains("name"));
    }

    #[test]
    fn test_parse_error_invalid_field() {
        let err = OutputParseError::invalid_field("age", "must be positive");
        assert!(err.to_string().contains("age"));
        assert!(err.to_string().contains("must be positive"));
    }

    #[test]
    fn test_parse_error_unexpected_tool() {
        let err = OutputParseError::unexpected_tool("final_result", "search");
        assert!(err.to_string().contains("final_result"));
        assert!(err.to_string().contains("search"));
    }

    #[test]
    fn test_parse_error_too_short() {
        let err = OutputParseError::too_short(5, 10);
        assert!(err.to_string().contains("5"));
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains("below"));
    }

    #[test]
    fn test_validation_error_retry() {
        let err = OutputValidationError::retry("Please provide a valid email");
        assert!(err.should_retry());
        assert_eq!(err.retry_message(), Some("Please provide a valid email"));
    }

    #[test]
    fn test_validation_error_failed_no_retry() {
        let err = OutputValidationError::failed("Invalid data");
        assert!(!err.should_retry());
        assert!(err.retry_message().is_none());
    }

    #[test]
    fn test_validation_error_failed_retry() {
        let err = OutputValidationError::failed_retry("Try again");
        assert!(err.should_retry());
        assert_eq!(err.retry_message(), Some("Try again"));
    }
}

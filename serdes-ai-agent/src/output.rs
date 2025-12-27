//! Output validation and parsing.
//!
//! This module provides traits and implementations for validating
//! and transforming agent outputs.

use crate::context::RunContext;
use crate::errors::{OutputParseError, OutputValidationError};
use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value as JsonValue;
use std::marker::PhantomData;

/// Trait for validating agent outputs.
#[async_trait]
pub trait OutputValidator<Output, Deps>: Send + Sync {
    /// Validate and optionally transform the output.
    ///
    /// Returns the validated output or an error.
    async fn validate(
        &self,
        output: Output,
        ctx: &RunContext<Deps>,
    ) -> Result<Output, OutputValidationError>;
}

// ============================================================================
// Function-based Validators
// ============================================================================

/// Validator that uses an async function.
pub struct AsyncValidator<F, Deps, Output, Fut>
where
    F: Fn(Output, &RunContext<Deps>) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<Output, OutputValidationError>> + Send,
{
    func: F,
    _phantom: PhantomData<(Deps, Output, Fut)>,
}

impl<F, Deps, Output, Fut> AsyncValidator<F, Deps, Output, Fut>
where
    F: Fn(Output, &RunContext<Deps>) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<Output, OutputValidationError>> + Send,
{
    /// Create a new async validator.
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<F, Deps, Output, Fut> OutputValidator<Output, Deps> for AsyncValidator<F, Deps, Output, Fut>
where
    F: Fn(Output, &RunContext<Deps>) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<Output, OutputValidationError>> + Send + Sync,
    Deps: Send + Sync,
    Output: Send + Sync,
{
    async fn validate(
        &self,
        output: Output,
        ctx: &RunContext<Deps>,
    ) -> Result<Output, OutputValidationError> {
        (self.func)(output, ctx).await
    }
}

/// Validator that uses a sync function.
pub struct SyncValidator<F, Deps, Output>
where
    F: Fn(Output, &RunContext<Deps>) -> Result<Output, OutputValidationError> + Send + Sync,
{
    func: F,
    _phantom: PhantomData<(Deps, Output)>,
}

impl<F, Deps, Output> SyncValidator<F, Deps, Output>
where
    F: Fn(Output, &RunContext<Deps>) -> Result<Output, OutputValidationError> + Send + Sync,
{
    /// Create a new sync validator.
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<F, Deps, Output> OutputValidator<Output, Deps> for SyncValidator<F, Deps, Output>
where
    F: Fn(Output, &RunContext<Deps>) -> Result<Output, OutputValidationError> + Send + Sync,
    Deps: Send + Sync,
    Output: Send + Sync,
{
    async fn validate(
        &self,
        output: Output,
        ctx: &RunContext<Deps>,
    ) -> Result<Output, OutputValidationError> {
        (self.func)(output, ctx)
    }
}

// ============================================================================
// Common Validators
// ============================================================================

/// Validator that checks string outputs are not empty.
pub struct NonEmptyValidator;

#[async_trait]
impl<Deps: Send + Sync> OutputValidator<String, Deps> for NonEmptyValidator {
    async fn validate(
        &self,
        output: String,
        _ctx: &RunContext<Deps>,
    ) -> Result<String, OutputValidationError> {
        if output.trim().is_empty() {
            Err(OutputValidationError::failed("Output cannot be empty"))
        } else {
            Ok(output)
        }
    }
}

/// Validator that checks string length.
pub struct LengthValidator {
    min: Option<usize>,
    max: Option<usize>,
}

impl LengthValidator {
    /// Create a new length validator.
    pub fn new() -> Self {
        Self {
            min: None,
            max: None,
        }
    }

    /// Set minimum length.
    pub fn min(mut self, min: usize) -> Self {
        self.min = Some(min);
        self
    }

    /// Set maximum length.
    pub fn max(mut self, max: usize) -> Self {
        self.max = Some(max);
        self
    }
}

impl Default for LengthValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<Deps: Send + Sync> OutputValidator<String, Deps> for LengthValidator {
    async fn validate(
        &self,
        output: String,
        _ctx: &RunContext<Deps>,
    ) -> Result<String, OutputValidationError> {
        let len = output.len();

        if let Some(min) = self.min {
            if len < min {
                return Err(OutputValidationError::failed(format!(
                    "Output too short: {} < {}",
                    len, min
                )));
            }
        }

        if let Some(max) = self.max {
            if len > max {
                return Err(OutputValidationError::failed(format!(
                    "Output too long: {} > {}",
                    len, max
                )));
            }
        }

        Ok(output)
    }
}

/// Validator that applies a regex pattern.
#[cfg(feature = "regex")]
pub struct RegexValidator {
    pattern: regex::Regex,
    message: String,
}

#[cfg(feature = "regex")]
impl RegexValidator {
    /// Create a new regex validator.
    pub fn new(pattern: &str, message: impl Into<String>) -> Result<Self, regex::Error> {
        Ok(Self {
            pattern: regex::Regex::new(pattern)?,
            message: message.into(),
        })
    }
}

#[cfg(feature = "regex")]
#[async_trait]
impl<Deps: Send + Sync> OutputValidator<String, Deps> for RegexValidator {
    async fn validate(
        &self,
        output: String,
        _ctx: &RunContext<Deps>,
    ) -> Result<String, OutputValidationError> {
        if self.pattern.is_match(&output) {
            Ok(output)
        } else {
            Err(OutputValidationError::failed(&self.message))
        }
    }
}

// ============================================================================
// Chained Validators
// ============================================================================

/// Chain multiple validators together.
pub struct ChainedValidator<Output, Deps> {
    validators: Vec<Box<dyn OutputValidator<Output, Deps>>>,
}

impl<Output: Send + Sync + 'static, Deps: Send + Sync + 'static> ChainedValidator<Output, Deps> {
    /// Create a new chained validator.
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }

    /// Add a validator.
    pub fn add<V: OutputValidator<Output, Deps> + 'static>(mut self, validator: V) -> Self {
        self.validators.push(Box::new(validator));
        self
    }
}

impl<Output: Send + Sync + 'static, Deps: Send + Sync + 'static> Default
    for ChainedValidator<Output, Deps>
{
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<Output: Send + Sync, Deps: Send + Sync> OutputValidator<Output, Deps>
    for ChainedValidator<Output, Deps>
{
    async fn validate(
        &self,
        mut output: Output,
        ctx: &RunContext<Deps>,
    ) -> Result<Output, OutputValidationError> {
        for validator in &self.validators {
            output = validator.validate(output, ctx).await?;
        }
        Ok(output)
    }
}

// ============================================================================
// Output Schema
// ============================================================================

/// Output mode for the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputMode {
    /// Plain text output.
    #[default]
    Text,
    /// JSON output.
    Json,
    /// Tool call output.
    ToolCall,
}

/// Schema for parsing and validating output.
pub trait OutputSchema<Output>: Send + Sync {
    /// Get the JSON schema for structured output.
    fn json_schema(&self) -> Option<JsonValue> {
        None
    }

    /// Get the output mode.
    fn mode(&self) -> OutputMode {
        OutputMode::Text
    }

    /// Get the name of the output tool (if using tool mode).
    fn tool_name(&self) -> Option<&str> {
        None
    }

    /// Parse text output.
    fn parse_text(&self, text: &str) -> Result<Output, OutputParseError>;

    /// Parse tool call output.
    fn parse_tool_call(
        &self,
        _name: &str,
        _args: &JsonValue,
    ) -> Result<Output, OutputParseError> {
        Err(OutputParseError::ToolNotCalled)
    }
}

/// Text output schema (returns String).
#[derive(Debug, Clone, Default)]
pub struct TextOutputSchema;

impl OutputSchema<String> for TextOutputSchema {
    fn parse_text(&self, text: &str) -> Result<String, OutputParseError> {
        Ok(text.to_string())
    }
}

/// JSON output schema (parses JSON to type).
pub struct JsonOutputSchema<T> {
    schema: Option<JsonValue>,
    _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned> JsonOutputSchema<T> {
    /// Create a new JSON output schema.
    pub fn new() -> Self {
        Self {
            schema: None,
            _phantom: PhantomData,
        }
    }

    /// Set the JSON schema.
    pub fn with_schema(mut self, schema: JsonValue) -> Self {
        self.schema = Some(schema);
        self
    }
}

impl<T: DeserializeOwned> Default for JsonOutputSchema<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DeserializeOwned + Send + Sync> OutputSchema<T> for JsonOutputSchema<T> {
    fn json_schema(&self) -> Option<JsonValue> {
        self.schema.clone()
    }

    fn mode(&self) -> OutputMode {
        OutputMode::Json
    }

    fn parse_text(&self, text: &str) -> Result<T, OutputParseError> {
        // Try to extract JSON from the text
        let json_str = extract_json(text).unwrap_or(text);
        serde_json::from_str(json_str).map_err(OutputParseError::Json)
    }
}

/// Tool-based output schema.
pub struct ToolOutputSchema<T> {
    tool_name: String,
    schema: Option<JsonValue>,
    _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned> ToolOutputSchema<T> {
    /// Create a new tool output schema.
    pub fn new(tool_name: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            schema: None,
            _phantom: PhantomData,
        }
    }

    /// Set the JSON schema.
    pub fn with_schema(mut self, schema: JsonValue) -> Self {
        self.schema = Some(schema);
        self
    }
}

impl<T: DeserializeOwned + Send + Sync> OutputSchema<T> for ToolOutputSchema<T> {
    fn json_schema(&self) -> Option<JsonValue> {
        self.schema.clone()
    }

    fn mode(&self) -> OutputMode {
        OutputMode::ToolCall
    }

    fn tool_name(&self) -> Option<&str> {
        Some(&self.tool_name)
    }

    fn parse_text(&self, _text: &str) -> Result<T, OutputParseError> {
        Err(OutputParseError::ToolNotCalled)
    }

    fn parse_tool_call(&self, name: &str, args: &JsonValue) -> Result<T, OutputParseError> {
        if name != self.tool_name {
            return Err(OutputParseError::ToolNotCalled);
        }
        serde_json::from_value(args.clone()).map_err(OutputParseError::Json)
    }
}

/// Extract JSON from text (handles markdown code blocks).
fn extract_json(text: &str) -> Option<&str> {
    // Try to find JSON in code blocks
    if let Some(start) = text.find("```json") {
        let content_start = start + 7;
        if let Some(end) = text[content_start..].find("```") {
            return Some(&text[content_start..content_start + end].trim());
        }
    }

    // Try to find JSON in plain code blocks
    if let Some(start) = text.find("```") {
        let content_start = start + 3;
        // Skip any language identifier
        let line_end = text[content_start..].find('\n').unwrap_or(0);
        let content_start = content_start + line_end + 1;
        if let Some(end) = text[content_start..].find("```") {
            let potential = &text[content_start..content_start + end].trim();
            if potential.starts_with('{') || potential.starts_with('[') {
                return Some(potential);
            }
        }
    }

    // Try to find raw JSON
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            if end > start {
                return Some(&text[start..=end]);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::sync::Arc;

    fn make_context() -> RunContext<()> {
        RunContext {
            deps: Arc::new(()),
            run_id: "test".to_string(),
            start_time: Utc::now(),
            model_name: "test".to_string(),
            model_settings: Default::default(),
            tool_name: None,
            tool_call_id: None,
            retry_count: 0,
            metadata: None,
        }
    }

    #[tokio::test]
    async fn test_non_empty_validator() {
        let validator = NonEmptyValidator;
        let ctx = make_context();

        let result = validator.validate("hello".to_string(), &ctx).await;
        assert!(result.is_ok());

        let result = validator.validate("".to_string(), &ctx).await;
        assert!(result.is_err());

        let result = validator.validate("   ".to_string(), &ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_length_validator() {
        let validator = LengthValidator::new().min(5).max(10);
        let ctx = make_context();

        let result = validator.validate("hello".to_string(), &ctx).await;
        assert!(result.is_ok());

        let result = validator.validate("hi".to_string(), &ctx).await;
        assert!(result.is_err());

        let result = validator
            .validate("hello world!".to_string(), &ctx)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_chained_validator() {
        let validator = ChainedValidator::<String, ()>::new()
            .add(NonEmptyValidator)
            .add(LengthValidator::new().min(3));

        let ctx = make_context();

        let result = validator.validate("hello".to_string(), &ctx).await;
        assert!(result.is_ok());

        let result = validator.validate("hi".to_string(), &ctx).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_text_output_schema() {
        let schema = TextOutputSchema;
        let result = schema.parse_text("hello world");
        assert_eq!(result.unwrap(), "hello world");
    }

    #[test]
    fn test_json_output_schema() {
        use serde::Deserialize;

        #[derive(Debug, Deserialize, PartialEq)]
        struct Person {
            name: String,
            age: u32,
        }

        let schema = JsonOutputSchema::<Person>::new();

        // Plain JSON
        let result = schema.parse_text(r#"{"name": "Alice", "age": 30}"#);
        assert_eq!(
            result.unwrap(),
            Person {
                name: "Alice".to_string(),
                age: 30
            }
        );

        // JSON in code block
        let text = r#"Here's the person:
```json
{"name": "Bob", "age": 25}
```"#;
        let result = schema.parse_text(text);
        assert_eq!(
            result.unwrap(),
            Person {
                name: "Bob".to_string(),
                age: 25
            }
        );
    }

    #[test]
    fn test_extract_json() {
        let text = "Here's some JSON: {\"a\": 1}";
        assert_eq!(extract_json(text), Some("{\"a\": 1}"));

        let text = "```json\n{\"a\": 1}\n```";
        assert!(extract_json(text).is_some());
    }
}

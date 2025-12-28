//! Mock and function-based models for testing.
//!
//! This module provides testing utilities:
//!
//! - [`MockModel`]: A simple mock with pre-configured responses
//! - [`FunctionModel`]: A flexible model controlled by custom functions
//!
//! # Examples
//!
//! ## MockModel for simple response sequences
//!
//! ```rust
//! use serdes_ai_models::MockModel;
//!
//! let model = MockModel::new("test")
//!     .with_text_response("First response")
//!     .with_text_response("Second response");
//! ```
//!
//! ## FunctionModel for dynamic behavior
//!
//! ```rust,ignore
//! use serdes_ai_models::FunctionModel;
//! use serdes_ai_core::{ModelRequest, ModelResponse};
//!
//! let model = FunctionModel::new(|messages, _settings| {
//!     let user_count = messages.iter()
//!         .flat_map(|m| m.user_prompts())
//!         .count();
//!     
//!     ModelResponse::text(format!("Received {} user prompts", user_count))
//! });
//! ```

use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse};
use crate::profile::ModelProfile;
use async_trait::async_trait;
use serdes_ai_core::messages::TextPart;
use serdes_ai_core::{ModelRequest, ModelResponse, ModelResponsePart, ModelSettings};
use std::sync::Arc;

// ============================================================================
// MockModel - Simple pre-configured mock
// ============================================================================

/// A mock model for testing with pre-configured responses.
///
/// MockModel is the simplest testing model - you configure a queue of responses
/// and it returns them in order. Great for basic testing scenarios.
///
/// # Example
///
/// ```rust
/// use serdes_ai_models::MockModel;
///
/// let model = MockModel::new("test-model")
///     .with_text_response("Hello!")
///     .with_text_response("How can I help?");
/// ```
#[derive(Debug, Clone)]
pub struct MockModel {
    name: String,
    profile: ModelProfile,
    responses: Arc<std::sync::Mutex<Vec<ModelResponse>>>,
    requests: Arc<std::sync::Mutex<Vec<Vec<ModelRequest>>>>,
}

impl MockModel {
    /// Create a new mock model.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            profile: ModelProfile::default(),
            responses: Arc::new(std::sync::Mutex::new(Vec::new())),
            requests: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// Add a response to return.
    pub fn with_response(self, response: ModelResponse) -> Self {
        self.responses.lock().unwrap().push(response);
        self
    }

    /// Add a text response.
    pub fn with_text_response(self, text: impl Into<String>) -> Self {
        let response = ModelResponse {
            parts: vec![ModelResponsePart::Text(TextPart::new(text))],
            model_name: Some(self.name.clone()),
            timestamp: chrono::Utc::now(),
            finish_reason: Some(serdes_ai_core::FinishReason::Stop),
            usage: None,
            vendor_id: None,
            vendor_details: None,
            kind: "response".to_string(),
        };
        self.with_response(response)
    }

    /// Set custom profile.
    pub fn with_profile(mut self, profile: ModelProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Get recorded requests.
    pub fn recorded_requests(&self) -> Vec<Vec<ModelRequest>> {
        self.requests.lock().unwrap().clone()
    }

    /// Clear recorded requests.
    pub fn clear_requests(&self) {
        self.requests.lock().unwrap().clear();
    }
}

#[async_trait]
impl Model for MockModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn system(&self) -> &str {
        "mock"
    }

    fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    async fn request(
        &self,
        messages: &[ModelRequest],
        _settings: &ModelSettings,
        _params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        // Record request
        self.requests.lock().unwrap().push(messages.to_vec());

        // Return next response or default
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            Ok(ModelResponse {
                parts: vec![ModelResponsePart::Text(TextPart::new("Mock response"))],
                model_name: Some(self.name.clone()),
                timestamp: chrono::Utc::now(),
                finish_reason: Some(serdes_ai_core::FinishReason::Stop),
                usage: None,
                vendor_id: None,
                vendor_details: None,
                kind: "response".to_string(),
            })
        } else {
            Ok(responses.remove(0))
        }
    }

    async fn request_stream(
        &self,
        _messages: &[ModelRequest],
        _settings: &ModelSettings,
        _params: &ModelRequestParameters,
    ) -> Result<StreamedResponse, ModelError> {
        Err(ModelError::Other(anyhow::anyhow!(
            "Streaming not implemented for MockModel"
        )))
    }
}

// ============================================================================
// FunctionModel - Dynamic function-based model
// ============================================================================

/// Type alias for function model callback.
///
/// The function receives the message history and settings, and returns a response.
pub type FunctionDef = Box<dyn Fn(&[ModelRequest], &ModelSettings) -> ModelResponse + Send + Sync>;

/// Type alias for stream function callback.
///
/// The function receives the message history and settings, and returns a streaming response.
pub type StreamFunctionDef =
    Box<dyn Fn(&[ModelRequest], &ModelSettings) -> StreamedResponse + Send + Sync>;

/// A model controlled by a local function.
///
/// This is more flexible than [`MockModel`] - instead of pre-configured responses,
/// you provide a function that receives the full message history and can return
/// dynamic responses based on the conversation state.
///
/// This is inspired by pydantic-ai's `FunctionModel` and is extremely useful
/// for testing complex agent behaviors.
///
/// # Example
///
/// ```rust,ignore
/// use serdes_ai_models::FunctionModel;
/// use serdes_ai_core::{ModelRequest, ModelResponse};
///
/// // Count user messages and respond with the count
/// let model = FunctionModel::new(|messages, _settings| {
///     let user_count = messages.iter()
///         .flat_map(|m| m.user_prompts())
///         .count();
///     
///     ModelResponse::text(format!("Received {} user prompts", user_count))
/// });
/// ```
///
/// # Streaming
///
/// You can also provide a streaming function:
///
/// ```rust,ignore
/// use serdes_ai_models::FunctionModel;
///
/// let model = FunctionModel::with_stream(|messages, settings| {
///     // Return a StreamedResponse
///     todo!("implement streaming")
/// });
/// ```
#[derive(Clone)]
pub struct FunctionModel {
    name: String,
    profile: ModelProfile,
    function: Option<Arc<FunctionDef>>,
    stream_function: Option<Arc<StreamFunctionDef>>,
}

impl std::fmt::Debug for FunctionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionModel")
            .field("name", &self.name)
            .field("profile", &self.profile)
            .field("has_function", &self.function.is_some())
            .field("has_stream_function", &self.stream_function.is_some())
            .finish()
    }
}

impl FunctionModel {
    /// Create a new FunctionModel with a response function.
    ///
    /// The function receives the full message history and settings,
    /// allowing you to implement dynamic response logic.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use serdes_ai_models::FunctionModel;
    /// use serdes_ai_core::{ModelRequest, ModelResponse};
    ///
    /// let model = FunctionModel::new(|messages, _settings| {
    ///     if messages.is_empty() {
    ///         ModelResponse::text("No messages received")
    ///     } else {
    ///         ModelResponse::text(format!("Got {} requests", messages.len()))
    ///     }
    /// });
    /// ```
    pub fn new<F>(function: F) -> Self
    where
        F: Fn(&[ModelRequest], &ModelSettings) -> ModelResponse + Send + Sync + 'static,
    {
        Self {
            name: "function-model".to_string(),
            profile: ModelProfile::default(),
            function: Some(Arc::new(Box::new(function))),
            stream_function: None,
        }
    }

    /// Create a new FunctionModel with a streaming function.
    ///
    /// Use this when you need to test streaming behavior.
    pub fn with_stream<F>(stream_function: F) -> Self
    where
        F: Fn(&[ModelRequest], &ModelSettings) -> StreamedResponse + Send + Sync + 'static,
    {
        Self {
            name: "function-model".to_string(),
            profile: ModelProfile::default(),
            function: None,
            stream_function: Some(Arc::new(Box::new(stream_function))),
        }
    }

    /// Create a FunctionModel with both regular and streaming functions.
    ///
    /// This allows the model to handle both `request()` and `request_stream()` calls.
    pub fn with_both<F, SF>(function: F, stream_function: SF) -> Self
    where
        F: Fn(&[ModelRequest], &ModelSettings) -> ModelResponse + Send + Sync + 'static,
        SF: Fn(&[ModelRequest], &ModelSettings) -> StreamedResponse + Send + Sync + 'static,
    {
        Self {
            name: "function-model".to_string(),
            profile: ModelProfile::default(),
            function: Some(Arc::new(Box::new(function))),
            stream_function: Some(Arc::new(Box::new(stream_function))),
        }
    }

    /// Set a custom model name.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = FunctionModel::constant_text("hi")
    ///     .with_name("my-test-model");
    /// assert_eq!(model.name(), "my-test-model");
    /// ```
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set a custom profile.
    ///
    /// Use this to test behavior with specific model capabilities.
    #[must_use]
    pub fn with_profile(mut self, profile: ModelProfile) -> Self {
        self.profile = profile;
        self
    }
}

// ============================================================================
// FunctionModel convenience constructors
// ============================================================================

impl FunctionModel {
    /// Create a model that always returns the same text.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use serdes_ai_models::FunctionModel;
    ///
    /// let model = FunctionModel::constant_text("I always say this!");
    /// ```
    pub fn constant_text(text: impl Into<String>) -> Self {
        let text = text.into();
        Self::new(move |_, _| ModelResponse::text(text.clone()))
    }

    /// Create a model that echoes the last user prompt.
    ///
    /// Useful for testing that messages are being passed correctly.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use serdes_ai_models::FunctionModel;
    ///
    /// let model = FunctionModel::echo();
    /// // When user says "hello", model responds "Echo: hello"
    /// ```
    pub fn echo() -> Self {
        Self::new(|messages, _| {
            // Find the last user prompt
            let last_user_text = messages
                .iter()
                .rev()
                .flat_map(|m| m.user_prompts())
                .next()
                .and_then(|p| p.as_text())
                .unwrap_or("No user message");

            ModelResponse::text(format!("Echo: {}", last_user_text))
        })
    }

    /// Create a model that always returns a tool call.
    ///
    /// Useful for testing tool handling in agents.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use serdes_ai_models::FunctionModel;
    /// use serde_json::json;
    ///
    /// let model = FunctionModel::tool_call(
    ///     "get_weather",
    ///     json!({"city": "NYC"})
    /// );
    /// ```
    pub fn tool_call(tool_name: impl Into<String>, args: serde_json::Value) -> Self {
        let tool_name = tool_name.into();
        Self::new(move |_, _| {
            ModelResponse::with_parts(vec![ModelResponsePart::tool_call(
                tool_name.clone(),
                args.clone(),
            )])
            .with_finish_reason(serdes_ai_core::FinishReason::ToolCall)
        })
    }

    /// Create a model that cycles through multiple responses.
    ///
    /// Each call returns the next response in the sequence, cycling back
    /// to the start when exhausted.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use serdes_ai_models::FunctionModel;
    ///
    /// let model = FunctionModel::cycle(vec![
    ///     "First response".to_string(),
    ///     "Second response".to_string(),
    /// ]);
    /// ```
    pub fn cycle(responses: Vec<String>) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let counter = Arc::new(AtomicUsize::new(0));
        let responses = Arc::new(responses);

        Self::new(move |_, _| {
            let idx = counter.fetch_add(1, Ordering::SeqCst) % responses.len();
            ModelResponse::text(responses[idx].clone())
        })
    }

    /// Create a model that counts the number of requests.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use serdes_ai_models::FunctionModel;
    ///
    /// let model = FunctionModel::counter();
    /// // First call returns "Request #1"
    /// // Second call returns "Request #2"
    /// ```
    pub fn counter() -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let counter = Arc::new(AtomicUsize::new(0));

        Self::new(move |_, _| {
            let count = counter.fetch_add(1, Ordering::SeqCst) + 1;
            ModelResponse::text(format!("Request #{}", count))
        })
    }
}

#[async_trait]
impl Model for FunctionModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn system(&self) -> &str {
        "function"
    }

    fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    async fn request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        _params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        match &self.function {
            Some(f) => Ok(f(messages, settings)),
            None => Err(ModelError::Other(anyhow::anyhow!(
                "FunctionModel has no request function defined. \
                 Use FunctionModel::new() or FunctionModel::with_both() to provide one."
            ))),
        }
    }

    async fn request_stream(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        _params: &ModelRequestParameters,
    ) -> Result<StreamedResponse, ModelError> {
        match &self.stream_function {
            Some(f) => Ok(f(messages, settings)),
            None => Err(ModelError::Other(anyhow::anyhow!(
                "FunctionModel has no stream function defined. \
                 Use FunctionModel::with_stream() or FunctionModel::with_both() to provide one."
            ))),
        }
    }
}

// ============================================================================
// Type alias for convenience
// ============================================================================

/// Alias for [`FunctionModel`].
///
/// Some folks prefer `TestModel` as a name - both work!
pub type TestModel = FunctionModel;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_model_new() {
        let model = MockModel::new("test-model");
        assert_eq!(model.name(), "test-model");
        assert_eq!(model.system(), "mock");
    }

    #[tokio::test]
    async fn test_mock_model_request() {
        let model = MockModel::new("test").with_text_response("Hello!");

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let response = model.request(&messages, &settings, &params).await.unwrap();

        assert_eq!(response.parts.len(), 1);
        if let ModelResponsePart::Text(text) = &response.parts[0] {
            assert_eq!(text.content, "Hello!");
        } else {
            panic!("Expected text part");
        }
    }

    #[tokio::test]
    async fn test_mock_model_records_requests() {
        let model = MockModel::new("test");

        let mut req = ModelRequest::new();
        req.add_user_prompt("Hello");
        let messages = vec![req];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        model.request(&messages, &settings, &params).await.unwrap();

        let recorded = model.recorded_requests();
        assert_eq!(recorded.len(), 1);
    }

    // ========================================================================
    // FunctionModel tests
    // ========================================================================

    #[test]
    fn test_function_model_debug() {
        let model = FunctionModel::constant_text("hi");
        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("FunctionModel"));
        assert!(debug_str.contains("has_function: true"));
    }

    #[tokio::test]
    async fn test_function_model_constant() {
        let model = FunctionModel::constant_text("Always this!");

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let response = model.request(&messages, &settings, &params).await.unwrap();
        assert_eq!(response.text_content(), "Always this!");

        // Second call should return the same
        let response2 = model.request(&messages, &settings, &params).await.unwrap();
        assert_eq!(response2.text_content(), "Always this!");
    }

    #[tokio::test]
    async fn test_function_model_echo() {
        let model = FunctionModel::echo();

        let mut req = ModelRequest::new();
        req.add_user_prompt("Hello world!");
        let messages = vec![req];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let response = model.request(&messages, &settings, &params).await.unwrap();
        assert_eq!(response.text_content(), "Echo: Hello world!");
    }

    #[tokio::test]
    async fn test_function_model_counter() {
        let model = FunctionModel::counter();

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let r1 = model.request(&messages, &settings, &params).await.unwrap();
        assert_eq!(r1.text_content(), "Request #1");

        let r2 = model.request(&messages, &settings, &params).await.unwrap();
        assert_eq!(r2.text_content(), "Request #2");

        let r3 = model.request(&messages, &settings, &params).await.unwrap();
        assert_eq!(r3.text_content(), "Request #3");
    }

    #[tokio::test]
    async fn test_function_model_cycle() {
        let model = FunctionModel::cycle(vec!["One".to_string(), "Two".to_string()]);

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let r1 = model.request(&messages, &settings, &params).await.unwrap();
        assert_eq!(r1.text_content(), "One");

        let r2 = model.request(&messages, &settings, &params).await.unwrap();
        assert_eq!(r2.text_content(), "Two");

        // Should cycle back
        let r3 = model.request(&messages, &settings, &params).await.unwrap();
        assert_eq!(r3.text_content(), "One");
    }

    #[tokio::test]
    async fn test_function_model_tool_call() {
        let model = FunctionModel::tool_call("get_weather", serde_json::json!({"city": "NYC"}));

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let response = model.request(&messages, &settings, &params).await.unwrap();
        assert!(response.has_tool_calls());

        let tool_calls: Vec<_> = response.tool_call_parts().collect();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].tool_name, "get_weather");
    }

    #[tokio::test]
    async fn test_function_model_custom_name() {
        let model = FunctionModel::constant_text("hi").with_name("custom-name");
        assert_eq!(model.name(), "custom-name");
        assert_eq!(model.system(), "function");
    }

    #[tokio::test]
    async fn test_function_model_no_function_error() {
        // Create a stream-only model and try to call request()
        let model = FunctionModel {
            name: "stream-only".to_string(),
            profile: ModelProfile::default(),
            function: None,
            stream_function: None,
        };

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let result = model.request(&messages, &settings, &params).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.to_string().contains("no request function defined"));
    }

    #[tokio::test]
    async fn test_function_model_no_stream_function_error() {
        let model = FunctionModel::constant_text("hi");

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let result = model.request_stream(&messages, &settings, &params).await;
        match result {
            Err(err) => {
                assert!(err.to_string().contains("no stream function defined"));
            }
            Ok(_) => panic!("Expected error for missing stream function"),
        }
    }

    #[tokio::test]
    async fn test_function_model_dynamic_behavior() {
        // Test a truly dynamic function that inspects messages
        let model = FunctionModel::new(|messages, _| {
            let total_user_prompts: usize =
                messages.iter().map(|m| m.user_prompts().count()).sum();

            if total_user_prompts > 2 {
                ModelResponse::text("That's a lot of messages!")
            } else {
                ModelResponse::text(format!("Got {} user prompts", total_user_prompts))
            }
        });

        let mut req = ModelRequest::new();
        req.add_user_prompt("Hi");
        let messages = vec![req];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let response = model.request(&messages, &settings, &params).await.unwrap();
        assert_eq!(response.text_content(), "Got 1 user prompts");
    }

    #[test]
    fn test_type_alias() {
        // TestModel should be usable as an alias
        let _model: TestModel = FunctionModel::constant_text("hi");
    }
}

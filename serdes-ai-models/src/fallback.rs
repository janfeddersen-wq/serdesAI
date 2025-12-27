//! Fallback model that wraps multiple models and tries them in sequence.
//!
//! This module provides a [`FallbackModel`] that implements resilient model access
//! by trying multiple models in order until one succeeds.
//!
//! # Example
//!
//! ```rust,ignore
//! use serdes_ai_models::fallback::{FallbackModel, RetryOn};
//! use serdes_ai_models::MockModel;
//!
//! let fallback = FallbackModel::new(vec![
//!     Box::new(primary_model),
//!     Box::new(backup_model),
//! ])
//! .with_retry_on(RetryOn::RateLimits);
//!
//! // If primary fails with rate limit, automatically tries backup
//! let response = fallback.request(&messages, &settings, &params).await?;
//! ```

use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse};
use crate::profile::ModelProfile;
use async_trait::async_trait;
use serdes_ai_core::{ModelRequest, ModelResponse, ModelSettings};
use tracing::{debug, warn};

/// Policy for determining when to retry with the next model.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum RetryOn {
    /// Retry on any error.
    #[default]
    AnyError,
    /// Only retry on rate limit errors.
    RateLimits,
    /// Only retry on transient errors (timeout, connection, server errors).
    Transient,
}

impl RetryOn {
    /// Check if the given error should trigger a retry.
    #[must_use]
    pub fn should_retry(&self, error: &ModelError) -> bool {
        match self {
            RetryOn::AnyError => true,
            RetryOn::RateLimits => matches!(error, ModelError::RateLimited { .. }),
            RetryOn::Transient => match error {
                ModelError::Timeout(_)
                | ModelError::Connection(_)
                | ModelError::Network(_) => true,
                ModelError::Http { status, .. } => *status >= 500,
                _ => false,
            },
        }
    }
}

/// A model that tries multiple models in order until one succeeds.
///
/// This is useful for:
/// - Implementing fallback strategies (e.g., try Claude first, fall back to GPT-4)
/// - Handling rate limits by falling back to alternative models
/// - Testing model behavior with mock fallbacks
pub struct FallbackModel {
    models: Vec<Box<dyn Model>>,
    retry_on: RetryOn,
    profile: ModelProfile,
}

impl std::fmt::Debug for FallbackModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FallbackModel")
            .field("model_count", &self.models.len())
            .field("retry_on", &self.retry_on)
            .finish()
    }
}

impl FallbackModel {
    /// Create a new fallback model with the given models.
    ///
    /// Models are tried in order; the first model that succeeds returns its response.
    ///
    /// # Arguments
    ///
    /// * `models` - List of models to try in order
    ///
    /// # Panics
    ///
    /// Does not panic, but returns an error on requests if the list is empty.
    #[must_use]
    pub fn new(models: Vec<Box<dyn Model>>) -> Self {
        // Use the first model's profile as default, or a default profile if empty
        let profile = models
            .first()
            .map(|m| m.profile().clone())
            .unwrap_or_default();

        Self {
            models,
            retry_on: RetryOn::default(),
            profile,
        }
    }

    /// Set the retry policy.
    ///
    /// # Arguments
    ///
    /// * `retry_on` - When to retry with the next model
    #[must_use]
    pub fn with_retry_on(mut self, retry_on: RetryOn) -> Self {
        self.retry_on = retry_on;
        self
    }

    /// Add another model to the fallback chain.
    ///
    /// # Arguments
    ///
    /// * `model` - Model to add to the end of the chain
    #[must_use]
    pub fn with_model(mut self, model: impl Model + 'static) -> Self {
        self.models.push(Box::new(model));
        self
    }

    /// Set a custom profile for this fallback model.
    ///
    /// By default, uses the first model's profile.
    #[must_use]
    pub fn with_profile(mut self, profile: ModelProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Get the number of models in the fallback chain.
    #[must_use]
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Check if the fallback chain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Check if we should retry with the next model for the given error.
    fn should_retry(&self, error: &ModelError) -> bool {
        self.retry_on.should_retry(error)
    }
}

#[async_trait]
impl Model for FallbackModel {
    fn name(&self) -> &str {
        "fallback"
    }

    fn system(&self) -> &str {
        "fallback"
    }

    fn identifier(&self) -> String {
        let model_names: Vec<_> = self.models.iter().map(|m| m.identifier()).collect();
        format!("fallback:[{}]", model_names.join(","))
    }

    fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    async fn request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        if self.models.is_empty() {
            return Err(ModelError::configuration("No models in fallback chain"));
        }

        let mut last_error: Option<ModelError> = None;

        for (i, model) in self.models.iter().enumerate() {
            let is_last = i == self.models.len() - 1;

            debug!(
                model = %model.identifier(),
                attempt = i + 1,
                total = self.models.len(),
                "Trying model in fallback chain"
            );

            match model.request(messages, settings, params).await {
                Ok(response) => {
                    if i > 0 {
                        debug!(
                            model = %model.identifier(),
                            "Fallback model succeeded after {} previous attempts",
                            i
                        );
                    }
                    return Ok(response);
                }
                Err(e) => {
                    warn!(
                        model = %model.identifier(),
                        error = %e,
                        "Model request failed"
                    );

                    if is_last {
                        // No more models to try
                        return Err(e);
                    }

                    if self.should_retry(&e) {
                        debug!(
                            error_type = ?std::mem::discriminant(&e),
                            "Error is retryable, trying next model"
                        );
                        last_error = Some(e);
                        continue;
                    }

                    // Non-retryable error, propagate immediately
                    return Err(e);
                }
            }
        }

        // This shouldn't be reached due to the is_last check above,
        // but handle it gracefully just in case
        Err(last_error.unwrap_or_else(|| {
            ModelError::configuration("No models in fallback chain")
        }))
    }

    async fn request_stream(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<StreamedResponse, ModelError> {
        if self.models.is_empty() {
            return Err(ModelError::configuration("No models in fallback chain"));
        }

        let mut last_error: Option<ModelError> = None;

        for (i, model) in self.models.iter().enumerate() {
            let is_last = i == self.models.len() - 1;

            debug!(
                model = %model.identifier(),
                attempt = i + 1,
                total = self.models.len(),
                "Trying model in fallback chain (streaming)"
            );

            match model.request_stream(messages, settings, params).await {
                Ok(stream) => {
                    if i > 0 {
                        debug!(
                            model = %model.identifier(),
                            "Fallback model succeeded after {} previous attempts",
                            i
                        );
                    }
                    return Ok(stream);
                }
                Err(e) => {
                    warn!(
                        model = %model.identifier(),
                        error = %e,
                        "Model stream request failed"
                    );

                    if is_last {
                        return Err(e);
                    }

                    if self.should_retry(&e) {
                        debug!(
                            error_type = ?std::mem::discriminant(&e),
                            "Error is retryable, trying next model"
                        );
                        last_error = Some(e);
                        continue;
                    }

                    return Err(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            ModelError::configuration("No models in fallback chain")
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockModel;
    use serdes_ai_core::messages::TextPart;
    use serdes_ai_core::{FinishReason, ModelResponsePart};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    /// A mock model that can fail with configurable errors.
    struct FailingMockModel {
        name: String,
        error: ModelError,
        call_count: Arc<AtomicUsize>,
        profile: ModelProfile,
    }

    impl FailingMockModel {
        fn new(name: impl Into<String>, error: ModelError) -> Self {
            Self {
                name: name.into(),
                error,
                call_count: Arc::new(AtomicUsize::new(0)),
                profile: ModelProfile::default(),
            }
        }

        #[allow(dead_code)]
        fn call_count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Model for FailingMockModel {
        fn name(&self) -> &str {
            &self.name
        }

        fn system(&self) -> &str {
            "failing-mock"
        }

        fn profile(&self) -> &ModelProfile {
            &self.profile
        }

        async fn request(
            &self,
            _messages: &[ModelRequest],
            _settings: &ModelSettings,
            _params: &ModelRequestParameters,
        ) -> Result<ModelResponse, ModelError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            // Return a new instance of the error type
            match &self.error {
                ModelError::RateLimited { retry_after } => {
                    Err(ModelError::RateLimited { retry_after: *retry_after })
                }
                ModelError::Timeout(d) => Err(ModelError::Timeout(*d)),
                ModelError::Connection(msg) => Err(ModelError::Connection(msg.clone())),
                ModelError::Authentication(msg) => Err(ModelError::Authentication(msg.clone())),
                ModelError::Http { status, body, headers } => {
                    Err(ModelError::Http {
                        status: *status,
                        body: body.clone(),
                        headers: headers.clone(),
                    })
                }
                _ => Err(ModelError::api("Generic error")),
            }
        }

        async fn request_stream(
            &self,
            _messages: &[ModelRequest],
            _settings: &ModelSettings,
            _params: &ModelRequestParameters,
        ) -> Result<StreamedResponse, ModelError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Err(ModelError::api("Stream error"))
        }
    }

    /// A mock model that succeeds and tracks calls.
    struct SucceedingMockModel {
        name: String,
        response_text: String,
        call_count: Arc<AtomicUsize>,
        profile: ModelProfile,
    }

    impl SucceedingMockModel {
        fn new(name: impl Into<String>, response: impl Into<String>) -> Self {
            Self {
                name: name.into(),
                response_text: response.into(),
                call_count: Arc::new(AtomicUsize::new(0)),
                profile: ModelProfile::default(),
            }
        }

        #[allow(dead_code)]
        fn call_count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Model for SucceedingMockModel {
        fn name(&self) -> &str {
            &self.name
        }

        fn system(&self) -> &str {
            "succeeding-mock"
        }

        fn profile(&self) -> &ModelProfile {
            &self.profile
        }

        async fn request(
            &self,
            _messages: &[ModelRequest],
            _settings: &ModelSettings,
            _params: &ModelRequestParameters,
        ) -> Result<ModelResponse, ModelError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(ModelResponse {
                parts: vec![ModelResponsePart::Text(TextPart::new(&self.response_text))],
                model_name: Some(self.name.clone()),
                timestamp: chrono::Utc::now(),
                finish_reason: Some(FinishReason::Stop),
                usage: None,
                vendor_id: None,
                vendor_details: None,
                kind: "response".to_string(),
            })
        }

        async fn request_stream(
            &self,
            _messages: &[ModelRequest],
            _settings: &ModelSettings,
            _params: &ModelRequestParameters,
        ) -> Result<StreamedResponse, ModelError> {
            Err(ModelError::not_supported("Streaming"))
        }
    }

    #[test]
    fn test_retry_on_should_retry() {
        // AnyError retries on everything
        assert!(RetryOn::AnyError.should_retry(&ModelError::api("test")));
        assert!(RetryOn::AnyError.should_retry(&ModelError::rate_limited(None)));
        assert!(RetryOn::AnyError.should_retry(&ModelError::Timeout(Duration::from_secs(30))));

        // RateLimits only retries on rate limit errors
        assert!(RetryOn::RateLimits.should_retry(&ModelError::rate_limited(None)));
        assert!(!RetryOn::RateLimits.should_retry(&ModelError::api("test")));
        assert!(!RetryOn::RateLimits.should_retry(&ModelError::Timeout(Duration::from_secs(30))));

        // Transient retries on timeout, connection, network, and 5xx errors
        assert!(RetryOn::Transient.should_retry(&ModelError::Timeout(Duration::from_secs(30))));
        assert!(RetryOn::Transient.should_retry(&ModelError::Connection("failed".into())));
        assert!(RetryOn::Transient.should_retry(&ModelError::Network("failed".into())));
        assert!(RetryOn::Transient.should_retry(&ModelError::http(500, "Server error")));
        assert!(RetryOn::Transient.should_retry(&ModelError::http(502, "Bad gateway")));
        assert!(!RetryOn::Transient.should_retry(&ModelError::http(400, "Bad request")));
        assert!(!RetryOn::Transient.should_retry(&ModelError::api("test")));
        assert!(!RetryOn::Transient.should_retry(&ModelError::rate_limited(None)));
    }

    #[test]
    fn test_fallback_model_new() {
        let model1 = MockModel::new("model1");
        let model2 = MockModel::new("model2");

        let fallback = FallbackModel::new(vec![Box::new(model1), Box::new(model2)]);

        assert_eq!(fallback.name(), "fallback");
        assert_eq!(fallback.system(), "fallback");
        assert_eq!(fallback.model_count(), 2);
        assert!(!fallback.is_empty());
    }

    #[test]
    fn test_fallback_model_with_model() {
        let model1 = MockModel::new("model1");
        let model2 = MockModel::new("model2");

        let fallback = FallbackModel::new(vec![Box::new(model1)]).with_model(model2);

        assert_eq!(fallback.model_count(), 2);
    }

    #[test]
    fn test_fallback_model_identifier() {
        let model1 = MockModel::new("model1");
        let model2 = MockModel::new("model2");

        let fallback = FallbackModel::new(vec![Box::new(model1), Box::new(model2)]);

        assert_eq!(fallback.identifier(), "fallback:[mock:model1,mock:model2]");
    }

    #[test]
    fn test_fallback_empty() {
        let fallback: FallbackModel = FallbackModel::new(vec![]);
        assert!(fallback.is_empty());
        assert_eq!(fallback.model_count(), 0);
    }

    #[tokio::test]
    async fn test_fallback_first_model_succeeds() {
        let model1 = SucceedingMockModel::new("model1", "response1");
        let model2 = SucceedingMockModel::new("model2", "response2");

        let call_count1 = model1.call_count.clone();
        let call_count2 = model2.call_count.clone();

        let fallback = FallbackModel::new(vec![Box::new(model1), Box::new(model2)]);

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let response = fallback.request(&messages, &settings, &params).await.unwrap();

        // First model should succeed, second should not be called
        assert_eq!(call_count1.load(Ordering::SeqCst), 1);
        assert_eq!(call_count2.load(Ordering::SeqCst), 0);

        // Should get first model's response
        if let ModelResponsePart::Text(text) = &response.parts[0] {
            assert_eq!(text.content, "response1");
        } else {
            panic!("Expected text response");
        }
    }

    #[tokio::test]
    async fn test_fallback_first_fails_second_succeeds() {
        let model1 = FailingMockModel::new("model1", ModelError::rate_limited(None));
        let model2 = SucceedingMockModel::new("model2", "response2");

        let call_count1 = model1.call_count.clone();
        let call_count2 = model2.call_count.clone();

        let fallback = FallbackModel::new(vec![Box::new(model1), Box::new(model2)]);

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let response = fallback.request(&messages, &settings, &params).await.unwrap();

        // Both models should be called
        assert_eq!(call_count1.load(Ordering::SeqCst), 1);
        assert_eq!(call_count2.load(Ordering::SeqCst), 1);

        // Should get second model's response
        if let ModelResponsePart::Text(text) = &response.parts[0] {
            assert_eq!(text.content, "response2");
        } else {
            panic!("Expected text response");
        }
    }

    #[tokio::test]
    async fn test_fallback_all_models_fail() {
        let model1 = FailingMockModel::new("model1", ModelError::rate_limited(None));
        let model2 = FailingMockModel::new("model2", ModelError::rate_limited(None));

        let call_count1 = model1.call_count.clone();
        let call_count2 = model2.call_count.clone();

        let fallback = FallbackModel::new(vec![Box::new(model1), Box::new(model2)]);

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let result = fallback.request(&messages, &settings, &params).await;

        // Both models should be called
        assert_eq!(call_count1.load(Ordering::SeqCst), 1);
        assert_eq!(call_count2.load(Ordering::SeqCst), 1);

        // Should return error
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ModelError::RateLimited { .. }));
    }

    #[tokio::test]
    async fn test_fallback_empty_chain_error() {
        let fallback: FallbackModel = FallbackModel::new(vec![]);

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let result = fallback.request(&messages, &settings, &params).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ModelError::Configuration(_)));
        assert!(err.to_string().contains("No models in fallback chain"));
    }

    #[tokio::test]
    async fn test_fallback_retry_on_rate_limits_only() {
        // First model fails with auth error (not retryable with RateLimits policy)
        let model1 = FailingMockModel::new("model1", ModelError::auth("Invalid key"));
        let model2 = SucceedingMockModel::new("model2", "response2");

        let call_count1 = model1.call_count.clone();
        let call_count2 = model2.call_count.clone();

        let fallback = FallbackModel::new(vec![Box::new(model1), Box::new(model2)])
            .with_retry_on(RetryOn::RateLimits);

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let result = fallback.request(&messages, &settings, &params).await;

        // First model should be called, second should NOT (auth error is not retryable)
        assert_eq!(call_count1.load(Ordering::SeqCst), 1);
        assert_eq!(call_count2.load(Ordering::SeqCst), 0);

        // Should return auth error
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ModelError::Authentication(_)));
    }

    #[tokio::test]
    async fn test_fallback_retry_on_rate_limits_succeeds() {
        // First model fails with rate limit (retryable)
        let model1 = FailingMockModel::new("model1", ModelError::rate_limited(None));
        let model2 = SucceedingMockModel::new("model2", "response2");

        let call_count1 = model1.call_count.clone();
        let call_count2 = model2.call_count.clone();

        let fallback = FallbackModel::new(vec![Box::new(model1), Box::new(model2)])
            .with_retry_on(RetryOn::RateLimits);

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let response = fallback.request(&messages, &settings, &params).await.unwrap();

        // Both models should be called
        assert_eq!(call_count1.load(Ordering::SeqCst), 1);
        assert_eq!(call_count2.load(Ordering::SeqCst), 1);

        if let ModelResponsePart::Text(text) = &response.parts[0] {
            assert_eq!(text.content, "response2");
        } else {
            panic!("Expected text response");
        }
    }

    #[tokio::test]
    async fn test_fallback_retry_on_transient() {
        // First model fails with timeout (transient)
        let model1 = FailingMockModel::new("model1", ModelError::Timeout(Duration::from_secs(30)));
        let model2 = SucceedingMockModel::new("model2", "response2");

        let call_count1 = model1.call_count.clone();
        let call_count2 = model2.call_count.clone();

        let fallback = FallbackModel::new(vec![Box::new(model1), Box::new(model2)])
            .with_retry_on(RetryOn::Transient);

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let response = fallback.request(&messages, &settings, &params).await.unwrap();

        assert_eq!(call_count1.load(Ordering::SeqCst), 1);
        assert_eq!(call_count2.load(Ordering::SeqCst), 1);

        if let ModelResponsePart::Text(text) = &response.parts[0] {
            assert_eq!(text.content, "response2");
        } else {
            panic!("Expected text response");
        }
    }

    #[tokio::test]
    async fn test_fallback_transient_does_not_retry_on_rate_limit() {
        // First model fails with rate limit (NOT transient)
        let model1 = FailingMockModel::new("model1", ModelError::rate_limited(None));
        let model2 = SucceedingMockModel::new("model2", "response2");

        let call_count1 = model1.call_count.clone();
        let call_count2 = model2.call_count.clone();

        let fallback = FallbackModel::new(vec![Box::new(model1), Box::new(model2)])
            .with_retry_on(RetryOn::Transient);

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let result = fallback.request(&messages, &settings, &params).await;

        // Only first model should be called
        assert_eq!(call_count1.load(Ordering::SeqCst), 1);
        assert_eq!(call_count2.load(Ordering::SeqCst), 0);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ModelError::RateLimited { .. }));
    }

    #[tokio::test]
    async fn test_fallback_three_models() {
        let model1 = FailingMockModel::new("model1", ModelError::rate_limited(None));
        let model2 = FailingMockModel::new("model2", ModelError::Timeout(Duration::from_secs(30)));
        let model3 = SucceedingMockModel::new("model3", "response3");

        let call_count1 = model1.call_count.clone();
        let call_count2 = model2.call_count.clone();
        let call_count3 = model3.call_count.clone();

        let fallback = FallbackModel::new(vec![
            Box::new(model1),
            Box::new(model2),
            Box::new(model3),
        ]);

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let response = fallback.request(&messages, &settings, &params).await.unwrap();

        // All three models should be called
        assert_eq!(call_count1.load(Ordering::SeqCst), 1);
        assert_eq!(call_count2.load(Ordering::SeqCst), 1);
        assert_eq!(call_count3.load(Ordering::SeqCst), 1);

        if let ModelResponsePart::Text(text) = &response.parts[0] {
            assert_eq!(text.content, "response3");
        } else {
            panic!("Expected text response");
        }
    }

    #[test]
    fn test_default_retry_on() {
        let retry_on = RetryOn::default();
        assert_eq!(retry_on, RetryOn::AnyError);
    }

    #[tokio::test]
    async fn test_fallback_stream_empty_chain() {
        let fallback: FallbackModel = FallbackModel::new(vec![]);

        let messages = vec![ModelRequest::new()];
        let settings = ModelSettings::default();
        let params = ModelRequestParameters::new();

        let result = fallback.request_stream(&messages, &settings, &params).await;

        assert!(result.is_err());
        match result {
            Err(ModelError::Configuration(msg)) => {
                assert!(msg.contains("No models in fallback chain"));
            }
            _ => panic!("Expected Configuration error"),
        }
    }
}

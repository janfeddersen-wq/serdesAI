//! Output validators.
//!
//! This module provides the `OutputValidator` trait and implementations
//! for validating parsed output with custom logic.

use async_trait::async_trait;
use serdes_ai_tools::RunContext;
use std::sync::Arc;

use crate::error::OutputValidationError;

/// Trait for output validators.
///
/// Validators are applied after parsing to add custom validation logic.
/// They can transform the value, return errors, or request model retries.
#[async_trait]
pub trait OutputValidator<T: Send + 'static, Deps: Send + Sync + 'static = ()>:
    Send + Sync
{
    /// Validate the output, returning it or an error.
    async fn validate(&self, value: T, ctx: &RunContext<Deps>) -> Result<T, OutputValidationError>;
}

/// Boxed validator for dynamic dispatch.
pub type BoxedValidator<T, Deps = ()> = Arc<dyn OutputValidator<T, Deps>>;

/// Simple sync validator.
///
/// Wraps a synchronous function as a validator.
pub struct SyncValidator<F> {
    func: F,
}

impl<F> SyncValidator<F> {
    /// Create a new sync validator.
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

#[async_trait]
impl<F, T, Deps> OutputValidator<T, Deps> for SyncValidator<F>
where
    F: Fn(T) -> Result<T, OutputValidationError> + Send + Sync,
    T: Send + 'static,
    Deps: Send + Sync + 'static,
{
    async fn validate(
        &self,
        value: T,
        _ctx: &RunContext<Deps>,
    ) -> Result<T, OutputValidationError> {
        (self.func)(value)
    }
}

impl<F> std::fmt::Debug for SyncValidator<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncValidator").finish()
    }
}

/// Chain multiple validators.
///
/// Validators are applied in order, with each receiving the output
/// of the previous validator.
pub struct ValidatorChain<T: Send + 'static, Deps: Send + Sync + 'static = ()> {
    validators: Vec<BoxedValidator<T, Deps>>,
}

impl<T: Send + 'static, Deps: Send + Sync + 'static> ValidatorChain<T, Deps> {
    /// Create a new empty validator chain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }

    /// Add a validator to the chain.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn add<V: OutputValidator<T, Deps> + 'static>(mut self, validator: V) -> Self {
        self.validators.push(Arc::new(validator));
        self
    }

    /// Add an Arc-wrapped validator to the chain.
    #[must_use]
    pub fn add_arc(mut self, validator: BoxedValidator<T, Deps>) -> Self {
        self.validators.push(validator);
        self
    }

    /// Get the number of validators.
    #[must_use]
    pub fn len(&self) -> usize {
        self.validators.len()
    }

    /// Check if the chain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.validators.is_empty()
    }
}

impl<T: Send + 'static, Deps: Send + Sync + 'static> Default for ValidatorChain<T, Deps> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<T: Send + 'static, Deps: Send + Sync + 'static> OutputValidator<T, Deps>
    for ValidatorChain<T, Deps>
{
    async fn validate(
        &self,
        mut value: T,
        ctx: &RunContext<Deps>,
    ) -> Result<T, OutputValidationError> {
        for validator in &self.validators {
            value = validator.validate(value, ctx).await?;
        }
        Ok(value)
    }
}

impl<T: Send + 'static, Deps: Send + Sync + 'static> std::fmt::Debug for ValidatorChain<T, Deps> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidatorChain")
            .field("count", &self.validators.len())
            .finish()
    }
}

/// Validator that always passes.
#[derive(Debug, Clone, Default)]
pub struct NoOpValidator;

impl NoOpValidator {
    /// Create a new no-op validator.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl<T: Send + 'static, Deps: Send + Sync + 'static> OutputValidator<T, Deps> for NoOpValidator {
    async fn validate(
        &self,
        value: T,
        _ctx: &RunContext<Deps>,
    ) -> Result<T, OutputValidationError> {
        Ok(value)
    }
}

/// Validator that rejects all values.
#[derive(Debug, Clone)]
pub struct RejectValidator {
    message: String,
}

impl RejectValidator {
    /// Create a new reject validator.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Default for RejectValidator {
    fn default() -> Self {
        Self::new("Validation rejected")
    }
}

#[async_trait]
impl<T: Send + 'static, Deps: Send + Sync + 'static> OutputValidator<T, Deps> for RejectValidator {
    async fn validate(
        &self,
        _value: T,
        _ctx: &RunContext<Deps>,
    ) -> Result<T, OutputValidationError> {
        Err(OutputValidationError::failed(&self.message))
    }
}

/// Validator that requests a model retry.
#[derive(Debug, Clone)]
pub struct RetryValidator {
    message: String,
}

impl RetryValidator {
    /// Create a new retry validator.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[async_trait]
impl<T: Send + 'static, Deps: Send + Sync + 'static> OutputValidator<T, Deps> for RetryValidator {
    async fn validate(
        &self,
        _value: T,
        _ctx: &RunContext<Deps>,
    ) -> Result<T, OutputValidationError> {
        Err(OutputValidationError::retry(&self.message))
    }
}

/// Helper function to create a sync validator from a closure.
pub fn sync_validator<F, T>(func: F) -> SyncValidator<F>
where
    F: Fn(T) -> Result<T, OutputValidationError> + Send + Sync,
    T: Send + 'static,
{
    SyncValidator::new(func)
}

/// Helper function to create an async validator from a closure.
///
/// Note: For async validators, consider implementing the trait directly
/// or using `SyncValidator` with blocking operations.
pub fn async_validator<F, T>(func: F) -> SyncValidator<F>
where
    F: Fn(T) -> Result<T, OutputValidationError> + Send + Sync,
    T: Send + 'static,
{
    SyncValidator::new(func)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sync_validator() {
        let validator = SyncValidator::new(|value: i32| {
            if value > 0 {
                Ok(value)
            } else {
                Err(OutputValidationError::failed("Value must be positive"))
            }
        });

        let ctx = RunContext::<()>::minimal("test");

        let result = validator.validate(5, &ctx).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 5);

        let result = validator.validate(-1, &ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validator_chain() {
        let chain = ValidatorChain::new()
            .add(SyncValidator::new(|v: i32| Ok(v * 2)))
            .add(SyncValidator::new(|v: i32| Ok(v + 1)));

        let ctx = RunContext::<()>::minimal("test");
        let result = chain.validate(5, &ctx).await.unwrap();

        // 5 * 2 = 10, 10 + 1 = 11
        assert_eq!(result, 11);
    }

    #[tokio::test]
    async fn test_validator_chain_short_circuit() {
        let chain = ValidatorChain::new()
            .add(SyncValidator::new(|v: i32| {
                if v > 0 {
                    Ok(v)
                } else {
                    Err(OutputValidationError::failed("Must be positive"))
                }
            }))
            .add(SyncValidator::new(|v: i32| Ok(v * 2)));

        let ctx = RunContext::<()>::minimal("test");

        // Positive value passes through
        let result = chain.validate(5, &ctx).await.unwrap();
        assert_eq!(result, 10);

        // Negative value fails at first validator
        let result = chain.validate(-1, &ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_noop_validator() {
        let validator = NoOpValidator::new();
        let ctx = RunContext::<()>::minimal("test");

        let result = validator.validate(42, &ctx).await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_reject_validator() {
        let validator = RejectValidator::new("Always fails");
        let ctx = RunContext::<()>::minimal("test");

        let result = validator.validate(42, &ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_retry_validator() {
        let validator = RetryValidator::new("Please try again");
        let ctx = RunContext::<()>::minimal("test");

        let result = validator.validate(42, &ctx).await;
        assert!(result.is_err());

        if let Err(e) = result {
            assert!(e.should_retry());
            assert_eq!(e.retry_message(), Some("Please try again"));
        }
    }

    #[test]
    fn test_sync_validator_helper() {
        let _validator = sync_validator(|v: String| Ok(v.to_uppercase()));
    }
}

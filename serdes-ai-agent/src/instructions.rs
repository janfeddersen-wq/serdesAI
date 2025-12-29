//! System prompts and dynamic instruction generation.
//!
//! This module provides traits and implementations for generating
//! system prompts and instructions dynamically based on context.

use crate::context::RunContext;
use async_trait::async_trait;
use std::future::Future;
use std::marker::PhantomData;

/// Trait for generating dynamic instructions.
///
/// Instructions are combined with the system prompt to form the
/// complete system message sent to the model.
#[async_trait]
pub trait InstructionFn<Deps>: Send + Sync {
    /// Generate instruction text based on the run context.
    ///
    /// Returns `None` if no instruction should be added.
    async fn generate(&self, ctx: &RunContext<Deps>) -> Option<String>;
}

/// Trait for generating dynamic system prompts.
///
/// System prompts can be static strings or dynamically generated
/// based on the run context and dependencies.
#[async_trait]
pub trait SystemPromptFn<Deps>: Send + Sync {
    /// Generate system prompt text based on the run context.
    ///
    /// Returns `None` if no prompt should be added.
    async fn generate(&self, ctx: &RunContext<Deps>) -> Option<String>;
}

// ============================================================================
// Async Function Wrappers
// ============================================================================

/// Wrapper for async instruction functions.
pub struct AsyncInstructionFn<F, Deps, Fut>
where
    F: Fn(&RunContext<Deps>) -> Fut + Send + Sync,
    Fut: Future<Output = Option<String>> + Send,
{
    func: F,
    _phantom: PhantomData<fn(Deps) -> Fut>,
}

impl<F, Deps, Fut> AsyncInstructionFn<F, Deps, Fut>
where
    F: Fn(&RunContext<Deps>) -> Fut + Send + Sync,
    Fut: Future<Output = Option<String>> + Send,
{
    /// Create a new async instruction function.
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<F, Deps, Fut> InstructionFn<Deps> for AsyncInstructionFn<F, Deps, Fut>
where
    F: Fn(&RunContext<Deps>) -> Fut + Send + Sync,
    Fut: Future<Output = Option<String>> + Send,
    Deps: Send + Sync,
{
    async fn generate(&self, ctx: &RunContext<Deps>) -> Option<String> {
        (self.func)(ctx).await
    }
}

/// Wrapper for async system prompt functions.
pub struct AsyncSystemPromptFn<F, Deps, Fut>
where
    F: Fn(&RunContext<Deps>) -> Fut + Send + Sync,
    Fut: Future<Output = Option<String>> + Send,
{
    func: F,
    _phantom: PhantomData<fn(Deps) -> Fut>,
}

impl<F, Deps, Fut> AsyncSystemPromptFn<F, Deps, Fut>
where
    F: Fn(&RunContext<Deps>) -> Fut + Send + Sync,
    Fut: Future<Output = Option<String>> + Send,
{
    /// Create a new async system prompt function.
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<F, Deps, Fut> SystemPromptFn<Deps> for AsyncSystemPromptFn<F, Deps, Fut>
where
    F: Fn(&RunContext<Deps>) -> Fut + Send + Sync,
    Fut: Future<Output = Option<String>> + Send,
    Deps: Send + Sync,
{
    async fn generate(&self, ctx: &RunContext<Deps>) -> Option<String> {
        (self.func)(ctx).await
    }
}

// ============================================================================
// Sync Function Wrappers
// ============================================================================

/// Wrapper for sync instruction functions.
pub struct SyncInstructionFn<F, Deps>
where
    F: Fn(&RunContext<Deps>) -> Option<String> + Send + Sync,
{
    func: F,
    _phantom: PhantomData<Deps>,
}

impl<F, Deps> SyncInstructionFn<F, Deps>
where
    F: Fn(&RunContext<Deps>) -> Option<String> + Send + Sync,
{
    /// Create a new sync instruction function.
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<F, Deps> InstructionFn<Deps> for SyncInstructionFn<F, Deps>
where
    F: Fn(&RunContext<Deps>) -> Option<String> + Send + Sync,
    Deps: Send + Sync,
{
    async fn generate(&self, ctx: &RunContext<Deps>) -> Option<String> {
        (self.func)(ctx)
    }
}

/// Wrapper for sync system prompt functions.
pub struct SyncSystemPromptFn<F, Deps>
where
    F: Fn(&RunContext<Deps>) -> Option<String> + Send + Sync,
{
    func: F,
    _phantom: PhantomData<Deps>,
}

impl<F, Deps> SyncSystemPromptFn<F, Deps>
where
    F: Fn(&RunContext<Deps>) -> Option<String> + Send + Sync,
{
    /// Create a new sync system prompt function.
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<F, Deps> SystemPromptFn<Deps> for SyncSystemPromptFn<F, Deps>
where
    F: Fn(&RunContext<Deps>) -> Option<String> + Send + Sync,
    Deps: Send + Sync,
{
    async fn generate(&self, ctx: &RunContext<Deps>) -> Option<String> {
        (self.func)(ctx)
    }
}

// ============================================================================
// Static Wrappers
// ============================================================================

/// Static instruction that always returns the same text.
pub struct StaticInstruction {
    text: String,
}

impl StaticInstruction {
    /// Create a new static instruction.
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

#[async_trait]
impl<Deps: Send + Sync> InstructionFn<Deps> for StaticInstruction {
    async fn generate(&self, _ctx: &RunContext<Deps>) -> Option<String> {
        Some(self.text.clone())
    }
}

/// Static system prompt that always returns the same text.
pub struct StaticSystemPrompt {
    text: String,
}

impl StaticSystemPrompt {
    /// Create a new static system prompt.
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

#[async_trait]
impl<Deps: Send + Sync> SystemPromptFn<Deps> for StaticSystemPrompt {
    async fn generate(&self, _ctx: &RunContext<Deps>) -> Option<String> {
        Some(self.text.clone())
    }
}

// ============================================================================
// Instruction Builder
// ============================================================================

/// Builder for combining multiple instructions.
pub struct InstructionBuilder<Deps> {
    parts: Vec<Box<dyn InstructionFn<Deps>>>,
    separator: String,
}

impl<Deps: Send + Sync + 'static> InstructionBuilder<Deps> {
    /// Create a new instruction builder.
    pub fn new() -> Self {
        Self {
            parts: Vec::new(),
            separator: "\n\n".to_string(),
        }
    }

    /// Set the separator between instruction parts.
    pub fn separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }

    /// Add a static instruction.
    pub fn add(mut self, text: impl Into<String>) -> Self {
        self.parts.push(Box::new(StaticInstruction::new(text)));
        self
    }

    /// Add a dynamic instruction function.
    pub fn add_fn<F>(mut self, func: F) -> Self
    where
        F: Fn(&RunContext<Deps>) -> Option<String> + Send + Sync + 'static,
    {
        self.parts.push(Box::new(SyncInstructionFn::new(func)));
        self
    }

    /// Add a custom instruction.
    pub fn add_instruction(mut self, instruction: Box<dyn InstructionFn<Deps>>) -> Self {
        self.parts.push(instruction);
        self
    }

    /// Build the combined instruction generator.
    pub fn build(self) -> CombinedInstruction<Deps> {
        CombinedInstruction {
            parts: self.parts,
            separator: self.separator,
        }
    }
}

impl<Deps: Send + Sync + 'static> Default for InstructionBuilder<Deps> {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined instruction from multiple sources.
pub struct CombinedInstruction<Deps> {
    parts: Vec<Box<dyn InstructionFn<Deps>>>,
    separator: String,
}

#[async_trait]
impl<Deps: Send + Sync> InstructionFn<Deps> for CombinedInstruction<Deps> {
    async fn generate(&self, ctx: &RunContext<Deps>) -> Option<String> {
        let mut results = Vec::new();

        for part in &self.parts {
            if let Some(text) = part.generate(ctx).await {
                if !text.is_empty() {
                    results.push(text);
                }
            }
        }

        if results.is_empty() {
            None
        } else {
            Some(results.join(&self.separator))
        }
    }
}

// ============================================================================
// Common Instruction Functions
// ============================================================================

/// Instruction that includes the current date/time.
pub struct DateTimeInstruction {
    format: String,
    prefix: String,
}

impl DateTimeInstruction {
    /// Create with default format.
    pub fn new() -> Self {
        Self {
            format: "%Y-%m-%d %H:%M:%S UTC".to_string(),
            prefix: "Current date and time:".to_string(),
        }
    }

    /// Set custom format.
    pub fn format(mut self, fmt: impl Into<String>) -> Self {
        self.format = fmt.into();
        self
    }

    /// Set prefix text.
    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }
}

impl Default for DateTimeInstruction {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<Deps: Send + Sync> InstructionFn<Deps> for DateTimeInstruction {
    async fn generate(&self, ctx: &RunContext<Deps>) -> Option<String> {
        let formatted = ctx.start_time.format(&self.format).to_string();
        Some(format!("{} {}", self.prefix, formatted))
    }
}

/// Instruction that includes user information.
pub struct UserInfoInstruction<F, Deps>
where
    F: Fn(&Deps) -> Option<String> + Send + Sync,
{
    extractor: F,
    _phantom: PhantomData<Deps>,
}

impl<F, Deps> UserInfoInstruction<F, Deps>
where
    F: Fn(&Deps) -> Option<String> + Send + Sync,
{
    /// Create with a user info extractor.
    pub fn new(extractor: F) -> Self {
        Self {
            extractor,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<F, Deps> InstructionFn<Deps> for UserInfoInstruction<F, Deps>
where
    F: Fn(&Deps) -> Option<String> + Send + Sync,
    Deps: Send + Sync,
{
    async fn generate(&self, ctx: &RunContext<Deps>) -> Option<String> {
        (self.extractor)(&ctx.deps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::sync::Arc;

    fn make_test_context() -> RunContext<()> {
        RunContext {
            deps: Arc::new(()),
            run_id: "test-run".to_string(),
            start_time: Utc::now(),
            model_name: "test-model".to_string(),
            model_settings: Default::default(),
            tool_name: None,
            tool_call_id: None,
            retry_count: 0,
            metadata: None,
        }
    }

    #[tokio::test]
    async fn test_static_instruction() {
        let instruction = StaticInstruction::new("Be helpful.");
        let ctx = make_test_context();
        let result = instruction.generate(&ctx).await;
        assert_eq!(result, Some("Be helpful.".to_string()));
    }

    #[tokio::test]
    async fn test_sync_instruction_fn() {
        let instruction = SyncInstructionFn::new(|ctx: &RunContext<()>| {
            Some(format!("Run ID: {}", ctx.run_id))
        });
        let ctx = make_test_context();
        let result = instruction.generate(&ctx).await;
        assert_eq!(result, Some("Run ID: test-run".to_string()));
    }

    #[tokio::test]
    async fn test_instruction_builder() {
        let instruction = InstructionBuilder::<()>::new()
            .add("First instruction.")
            .add("Second instruction.")
            .build();

        let ctx = make_test_context();
        let result = instruction.generate(&ctx).await.unwrap();

        assert!(result.contains("First instruction."));
        assert!(result.contains("Second instruction."));
    }

    #[tokio::test]
    async fn test_datetime_instruction() {
        let instruction = DateTimeInstruction::new();
        let ctx = make_test_context();
        let result = instruction.generate(&ctx).await.unwrap();

        assert!(result.contains("Current date and time:"));
    }

    #[tokio::test]
    async fn test_combined_instruction_skips_empty() {
        let instruction = InstructionBuilder::<()>::new()
            .add("Has content.")
            .add_fn(|_| None) // Returns None
            .add("") // Empty
            .add("Also has content.")
            .build();

        let ctx = make_test_context();
        let result = instruction.generate(&ctx).await.unwrap();

        let parts: Vec<_> = result.split("\n\n").collect();
        assert_eq!(parts.len(), 2);
    }
}

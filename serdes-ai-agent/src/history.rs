//! Message history processing.
//!
//! History processors can modify the message history before it's sent to the model.
//! Common use cases include truncation, summarization, and filtering.

use crate::context::RunContext;
use async_trait::async_trait;
use serdes_ai_core::ModelRequest;
use std::marker::PhantomData;

/// Trait for processing message history before model calls.
#[async_trait]
pub trait HistoryProcessor<Deps>: Send + Sync {
    /// Process the message history.
    ///
    /// Returns the modified history.
    async fn process(
        &self,
        ctx: &RunContext<Deps>,
        messages: Vec<ModelRequest>,
    ) -> Vec<ModelRequest>;
}

// ============================================================================
// Truncation Processors
// ============================================================================

/// Truncate history to keep only the most recent messages.
#[derive(Debug, Clone)]
pub struct TruncateHistory {
    /// Maximum number of messages to keep.
    max_messages: usize,
    /// Always keep the first message (usually system prompt).
    keep_first: bool,
}

impl TruncateHistory {
    /// Create a new truncation processor.
    pub fn new(max_messages: usize) -> Self {
        Self {
            max_messages,
            keep_first: true,
        }
    }

    /// Set whether to always keep the first message.
    pub fn keep_first(mut self, keep: bool) -> Self {
        self.keep_first = keep;
        self
    }
}

#[async_trait]
impl<Deps: Send + Sync> HistoryProcessor<Deps> for TruncateHistory {
    async fn process(
        &self,
        _ctx: &RunContext<Deps>,
        mut messages: Vec<ModelRequest>,
    ) -> Vec<ModelRequest> {
        if messages.len() <= self.max_messages {
            return messages;
        }

        if self.keep_first && !messages.is_empty() {
            // Keep first message, truncate the rest
            let first = messages.remove(0);
            let keep_count = self.max_messages.saturating_sub(1);
            let start = messages.len().saturating_sub(keep_count);
            let mut result = vec![first];
            result.extend(messages.drain(start..));
            result
        } else {
            // Just keep the most recent
            let start = messages.len().saturating_sub(self.max_messages);
            messages.drain(start..).collect()
        }
    }
}

/// Truncate based on token count.
#[derive(Debug, Clone)]
pub struct TruncateByTokens {
    /// Maximum tokens to keep.
    max_tokens: u64,
    /// Token estimator (chars per token).
    chars_per_token: f64,
    /// Always keep the first message.
    keep_first: bool,
}

impl TruncateByTokens {
    /// Create a new token-based truncation processor.
    pub fn new(max_tokens: u64) -> Self {
        Self {
            max_tokens,
            chars_per_token: 4.0, // Reasonable default for English
            keep_first: true,
        }
    }

    /// Set chars per token ratio.
    pub fn chars_per_token(mut self, ratio: f64) -> Self {
        self.chars_per_token = ratio;
        self
    }

    /// Set whether to keep the first message.
    pub fn keep_first(mut self, keep: bool) -> Self {
        self.keep_first = keep;
        self
    }

    fn estimate_tokens(&self, message: &ModelRequest) -> u64 {
        let chars: usize = message.parts.iter().map(|p| {
            match p {
                serdes_ai_core::ModelRequestPart::SystemPrompt(s) => s.content.len(),
                serdes_ai_core::ModelRequestPart::UserPrompt(u) => {
                    // Estimate based on content
                    match &u.content {
                        serdes_ai_core::messages::UserContent::Text(t) => t.len(),
                        serdes_ai_core::messages::UserContent::Parts(parts) => {
                            parts.iter().map(|p| {
                                match p {
                                    serdes_ai_core::messages::UserContentPart::Text { text } => text.len(),
                                    _ => 100, // Estimate for non-text
                                }
                            }).sum()
                        }
                    }
                }
                serdes_ai_core::ModelRequestPart::ToolReturn(t) => {
                    t.content.to_string_content().len()
                }
                serdes_ai_core::ModelRequestPart::RetryPrompt(r) => {
                    r.content.message().len()
                }
                serdes_ai_core::ModelRequestPart::BuiltinToolReturn(b) => {
                    // Estimate based on content type
                    b.content_type().len() + 100
                }
            }
        }).sum();

        (chars as f64 / self.chars_per_token).ceil() as u64
    }
}

#[async_trait]
impl<Deps: Send + Sync> HistoryProcessor<Deps> for TruncateByTokens {
    async fn process(
        &self,
        _ctx: &RunContext<Deps>,
        messages: Vec<ModelRequest>,
    ) -> Vec<ModelRequest> {
        if messages.is_empty() {
            return messages;
        }

        let mut result = Vec::new();
        let mut total_tokens = 0u64;

        // If keeping first, add it unconditionally
        let iter: Box<dyn Iterator<Item = _>> = if self.keep_first && !messages.is_empty() {
            let first = &messages[0];
            let tokens = self.estimate_tokens(first);
            result.push(first.clone());
            total_tokens += tokens;
            Box::new(messages.iter().skip(1).rev())
        } else {
            Box::new(messages.iter().rev())
        };

        // Add messages from the end until we hit the limit
        let mut to_prepend = Vec::new();
        for msg in iter {
            let tokens = self.estimate_tokens(msg);
            if total_tokens + tokens > self.max_tokens {
                break;
            }
            total_tokens += tokens;
            to_prepend.push(msg.clone());
        }

        // Reverse and add to result (we iterated backwards)
        to_prepend.reverse();
        if self.keep_first {
            result.extend(to_prepend);
        } else {
            result = to_prepend;
        }

        result
    }
}

// ============================================================================
// Filter Processors
// ============================================================================

/// Filter out specific message types.
#[derive(Debug, Clone)]
pub struct FilterHistory {
    /// Remove system prompts.
    remove_system: bool,
    /// Remove tool returns.
    remove_tool_returns: bool,
    /// Remove retry prompts.
    remove_retries: bool,
}

impl FilterHistory {
    /// Create a new filter processor.
    pub fn new() -> Self {
        Self {
            remove_system: false,
            remove_tool_returns: false,
            remove_retries: false,
        }
    }

    /// Remove system prompts.
    pub fn remove_system(mut self, remove: bool) -> Self {
        self.remove_system = remove;
        self
    }

    /// Remove tool returns.
    pub fn remove_tool_returns(mut self, remove: bool) -> Self {
        self.remove_tool_returns = remove;
        self
    }

    /// Remove retry prompts.
    pub fn remove_retries(mut self, remove: bool) -> Self {
        self.remove_retries = remove;
        self
    }
}

impl Default for FilterHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<Deps: Send + Sync> HistoryProcessor<Deps> for FilterHistory {
    async fn process(
        &self,
        _ctx: &RunContext<Deps>,
        messages: Vec<ModelRequest>,
    ) -> Vec<ModelRequest> {
        messages
            .into_iter()
            .map(|mut msg| {
                msg.parts.retain(|part| {
                    use serdes_ai_core::ModelRequestPart::*;
                    match part {
                        SystemPrompt(_) => !self.remove_system,
                        ToolReturn(_) => !self.remove_tool_returns,
                        RetryPrompt(_) => !self.remove_retries,
                        _ => true,
                    }
                });
                msg
            })
            .filter(|msg| !msg.parts.is_empty())
            .collect()
    }
}

// ============================================================================
// Combination Processors
// ============================================================================

/// Chain multiple processors together.
pub struct ChainedProcessor<Deps> {
    processors: Vec<Box<dyn HistoryProcessor<Deps>>>,
}

impl<Deps: Send + Sync + 'static> ChainedProcessor<Deps> {
    /// Create a new chained processor.
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    /// Add a processor to the chain.
    pub fn add<P: HistoryProcessor<Deps> + 'static>(mut self, processor: P) -> Self {
        self.processors.push(Box::new(processor));
        self
    }
}

impl<Deps: Send + Sync + 'static> Default for ChainedProcessor<Deps> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<Deps: Send + Sync> HistoryProcessor<Deps> for ChainedProcessor<Deps> {
    async fn process(
        &self,
        ctx: &RunContext<Deps>,
        mut messages: Vec<ModelRequest>,
    ) -> Vec<ModelRequest> {
        for processor in &self.processors {
            messages = processor.process(ctx, messages).await;
        }
        messages
    }
}

// ============================================================================
// Summarization (Placeholder)
// ============================================================================

/// Summarize old messages using a model.
///
/// This is a placeholder - actual implementation would require
/// calling a model to generate summaries.
#[derive(Debug, Clone)]
pub struct SummarizeHistory {
    /// Number of recent messages to keep.
    keep_recent: usize,
    /// Token threshold before summarization.
    threshold_tokens: u64,
}

impl SummarizeHistory {
    /// Create a new summarization processor.
    pub fn new(keep_recent: usize, threshold_tokens: u64) -> Self {
        Self {
            keep_recent,
            threshold_tokens,
        }
    }
}

#[async_trait]
impl<Deps: Send + Sync> HistoryProcessor<Deps> for SummarizeHistory {
    async fn process(
        &self,
        _ctx: &RunContext<Deps>,
        messages: Vec<ModelRequest>,
    ) -> Vec<ModelRequest> {
        // For now, just truncate. Full implementation would:
        // 1. Estimate tokens
        // 2. If above threshold, summarize older messages
        // 3. Replace old messages with summary
        if messages.len() <= self.keep_recent {
            return messages;
        }

        // Keep the most recent messages
        let start = messages.len().saturating_sub(self.keep_recent);
        messages[start..].to_vec()
    }
}

// ============================================================================
// Custom Processor
// ============================================================================

/// Processor that uses a custom function.
pub struct FnProcessor<F, Deps>
where
    F: Fn(&RunContext<Deps>, Vec<ModelRequest>) -> Vec<ModelRequest> + Send + Sync,
{
    func: F,
    _phantom: PhantomData<Deps>,
}

impl<F, Deps> FnProcessor<F, Deps>
where
    F: Fn(&RunContext<Deps>, Vec<ModelRequest>) -> Vec<ModelRequest> + Send + Sync,
{
    /// Create a new function processor.
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<F, Deps> HistoryProcessor<Deps> for FnProcessor<F, Deps>
where
    F: Fn(&RunContext<Deps>, Vec<ModelRequest>) -> Vec<ModelRequest> + Send + Sync,
    Deps: Send + Sync,
{
    async fn process(
        &self,
        ctx: &RunContext<Deps>,
        messages: Vec<ModelRequest>,
    ) -> Vec<ModelRequest> {
        (self.func)(ctx, messages)
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

    fn make_messages(count: usize) -> Vec<ModelRequest> {
        (0..count)
            .map(|i| {
                let mut req = ModelRequest::new();
                req.add_user_prompt(format!("Message {}", i));
                req
            })
            .collect()
    }

    #[tokio::test]
    async fn test_truncate_history() {
        let processor = TruncateHistory::new(3).keep_first(false);
        let ctx = make_test_context();
        let messages = make_messages(5);

        let result = processor.process(&ctx, messages).await;
        assert_eq!(result.len(), 3);
    }

    #[tokio::test]
    async fn test_truncate_keep_first() {
        let processor = TruncateHistory::new(3).keep_first(true);
        let ctx = make_test_context();
        let messages = make_messages(5);

        let result = processor.process(&ctx, messages).await;
        assert_eq!(result.len(), 3);
    }

    #[tokio::test]
    async fn test_truncate_no_change() {
        let processor = TruncateHistory::new(10);
        let ctx = make_test_context();
        let messages = make_messages(5);

        let result = processor.process(&ctx, messages).await;
        assert_eq!(result.len(), 5);
    }

    #[tokio::test]
    async fn test_chained_processor() {
        let processor = ChainedProcessor::<()>::new()
            .add(TruncateHistory::new(5))
            .add(TruncateHistory::new(3));

        let ctx = make_test_context();
        let messages = make_messages(10);

        let result = processor.process(&ctx, messages).await;
        assert_eq!(result.len(), 3);
    }

    #[tokio::test]
    async fn test_fn_processor() {
        let processor = FnProcessor::new(|_ctx: &RunContext<()>, mut msgs: Vec<ModelRequest>| {
            msgs.pop();
            msgs
        });

        let ctx = make_test_context();
        let messages = make_messages(5);

        let result = processor.process(&ctx, messages).await;
        assert_eq!(result.len(), 4);
    }
}

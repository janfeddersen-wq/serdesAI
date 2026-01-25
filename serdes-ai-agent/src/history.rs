//! Message history processing.
//!
//! History processors can modify the message history before it's sent to the model.
//! Common use cases include truncation, summarization, and filtering.

use crate::context::RunContext;
use async_trait::async_trait;
use serdes_ai_core::{ModelRequest, ModelRequestPart, ModelResponsePart};
use std::collections::HashSet;
use std::marker::PhantomData;

// Conditional tracing - use no-op macros when tracing feature is disabled
#[cfg(feature = "tracing-integration")]
use tracing::debug;

#[cfg(not(feature = "tracing-integration"))]
macro_rules! debug {
    ($($arg:tt)*) => {};
}

// ============================================================================
// Tool Pair Helpers
// ============================================================================

/// Extract all tool_call_ids from ToolCallPart in ModelResponse parts.
///
/// This looks for `ModelRequestPart::ModelResponse` and extracts `tool_call_id`
/// from any `ToolCallPart` or `BuiltinToolCallPart` within.
fn extract_tool_use_ids(message: &ModelRequest) -> Vec<String> {
    let mut ids = Vec::new();
    for part in &message.parts {
        if let ModelRequestPart::ModelResponse(response) = part {
            for response_part in &response.parts {
                match response_part {
                    ModelResponsePart::ToolCall(tc) => {
                        if let Some(id) = &tc.tool_call_id {
                            ids.push(id.clone());
                        }
                    }
                    ModelResponsePart::BuiltinToolCall(btc) => {
                        if let Some(id) = &btc.tool_call_id {
                            ids.push(id.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    ids
}

/// Extract all tool_call_ids from ToolReturnPart and BuiltinToolReturnPart.
///
/// This looks for `ModelRequestPart::ToolReturn` and `ModelRequestPart::BuiltinToolReturn`
/// and extracts their `tool_call_id` values.
#[cfg(test)]
fn extract_tool_result_ids(message: &ModelRequest) -> Vec<String> {
    let mut ids = Vec::new();
    for part in &message.parts {
        match part {
            ModelRequestPart::ToolReturn(tr) => {
                if let Some(id) = &tr.tool_call_id {
                    ids.push(id.clone());
                }
            }
            ModelRequestPart::BuiltinToolReturn(btr) => {
                // BuiltinToolReturnPart has non-optional tool_call_id
                ids.push(btr.tool_call_id.clone());
            }
            ModelRequestPart::RetryPrompt(rp) => {
                // RetryPrompt can also have a tool_call_id
                if let Some(id) = &rp.tool_call_id {
                    ids.push(id.clone());
                }
            }
            _ => {}
        }
    }
    ids
}

/// Collect all tool_use IDs from a list of messages.
fn collect_all_tool_use_ids(messages: &[ModelRequest]) -> HashSet<String> {
    messages
        .iter()
        .flat_map(extract_tool_use_ids)
        .collect()
}

/// Collect all tool_result IDs from a list of messages.
#[cfg(test)]
fn collect_all_tool_result_ids(messages: &[ModelRequest]) -> HashSet<String> {
    messages
        .iter()
        .flat_map(extract_tool_result_ids)
        .collect()
}

/// Remove orphaned tool results from messages.
///
/// An orphaned tool_result is one whose `tool_call_id` has no corresponding
/// `tool_use` (ToolCallPart) in the message history. This can happen when
/// truncation removes earlier messages containing the tool_use.
///
/// This function modifies messages in-place, removing orphaned ToolReturn,
/// BuiltinToolReturn, and RetryPrompt parts. If a message becomes empty
/// after removal, it is filtered out entirely.
///
/// # Behavior Notes
///
/// **Asymmetric handling of `tool_call_id` is intentional:**
///
/// - `ToolReturn` and `RetryPrompt`: If `tool_call_id` is `None`, the part is KEPT.
///   This handles edge cases where tools may return results without IDs (e.g., legacy
///   tools, certain provider quirks). Keeping these is safe - the worst case is a
///   slightly larger context, but it won't cause API errors.
///
/// - `BuiltinToolReturn`: Has non-optional `tool_call_id: String`. Empty strings are
///   treated as invalid (removed) since they indicate malformed data that would likely
///   cause API errors anyway.
///
/// This design prioritizes avoiding false-positive removals over strict validation.
fn remove_orphaned_tool_results(
    messages: Vec<ModelRequest>,
    valid_tool_ids: &HashSet<String>,
) -> Vec<ModelRequest> {
    messages
        .into_iter()
        .filter_map(|mut msg| {
            msg.parts.retain(|part| {
                match part {
                    ModelRequestPart::ToolReturn(tr) => {
                        // INTENTIONAL: Keep if no tool_call_id (None) to handle edge cases
                        // gracefully. Only remove if we have an ID that doesn't match.
                        // See function docs for rationale.
                        let dominated = tr.tool_call_id
                            .as_ref()
                            .map_or(true, |id| valid_tool_ids.contains(id));
                        if !dominated {
                            debug!(
                                tool_name = %tr.tool_name,
                                tool_call_id = ?tr.tool_call_id,
                                "Removing orphaned ToolReturn: no matching tool_use found"
                            );
                        }
                        dominated
                    }
                    ModelRequestPart::BuiltinToolReturn(btr) => {
                        // BuiltinToolReturn has non-optional tool_call_id: String
                        // Empty string is treated as invalid (likely malformed data)
                        let dominated = !btr.tool_call_id.is_empty() 
                            && valid_tool_ids.contains(&btr.tool_call_id);
                        if !dominated {
                            debug!(
                                tool_name = %btr.tool_name,
                                tool_call_id = %btr.tool_call_id,
                                "Removing orphaned BuiltinToolReturn: no matching tool_use found"
                            );
                        }
                        dominated
                    }
                    ModelRequestPart::RetryPrompt(rp) => {
                        // INTENTIONAL: Keep if no tool_call_id (None) - same rationale as ToolReturn
                        let keep = rp.tool_call_id
                            .as_ref()
                            .map_or(true, |id| valid_tool_ids.contains(id));
                        if !keep {
                            debug!(
                                tool_name = ?rp.tool_name,
                                tool_call_id = ?rp.tool_call_id,
                                "Removing orphaned RetryPrompt: no matching tool_use found"
                            );
                        }
                        keep
                    }
                    // Keep all other parts
                    _ => true,
                }
            });
            // Only keep messages that still have parts
            if msg.parts.is_empty() {
                None
            } else {
                Some(msg)
            }
        })
        .collect()
}

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

        let result = if self.keep_first && !messages.is_empty() {
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
        };

        // Post-processing: Remove orphaned tool results
        let valid_tool_ids = collect_all_tool_use_ids(&result);
        remove_orphaned_tool_results(result, &valid_tool_ids)
    }
}

/// Truncate based on token count.
#[derive(Debug, Clone)]
pub struct TruncateByTokens {
    /// Maximum tokens to keep.
    max_tokens: u64,
    /// Token estimator (chars per token).
    chars_per_token: f64,
    /// Number of messages to always keep at the beginning (e.g., system prompt + first user message).
    keep_first_n: usize,
}

impl TruncateByTokens {
    /// Create a new token-based truncation processor.
    ///
    /// By default, keeps the first 2 messages (system prompt + first user message).
    pub fn new(max_tokens: u64) -> Self {
        Self {
            max_tokens,
            chars_per_token: 4.0, // Reasonable default for English
            keep_first_n: 2,
        }
    }

    /// Set chars per token ratio.
    pub fn chars_per_token(mut self, ratio: f64) -> Self {
        self.chars_per_token = ratio;
        self
    }

    /// Set the number of messages to keep at the beginning.
    ///
    /// Common values:
    /// - 0: Don't preserve any messages at the start
    /// - 1: Keep just the system prompt
    /// - 2: Keep system prompt + first user message (default)
    pub fn keep_first_n(mut self, n: usize) -> Self {
        self.keep_first_n = n;
        self
    }

    /// Set whether to keep the first message.
    ///
    /// **Deprecated:** Use `keep_first_n()` instead for more control.
    ///
    /// - `keep: true` sets `keep_first_n` to 1
    /// - `keep: false` sets `keep_first_n` to 0
    pub fn keep_first(mut self, keep: bool) -> Self {
        self.keep_first_n = if keep { 1 } else { 0 };
        self
    }

    fn estimate_tokens(&self, message: &ModelRequest) -> u64 {
        let chars: usize = message
            .parts
            .iter()
            .map(|p| {
                match p {
                    serdes_ai_core::ModelRequestPart::SystemPrompt(s) => s.content.len(),
                    serdes_ai_core::ModelRequestPart::UserPrompt(u) => {
                        // Estimate based on content
                        match &u.content {
                            serdes_ai_core::messages::UserContent::Text(t) => t.len(),
                            serdes_ai_core::messages::UserContent::Parts(parts) => {
                                parts
                                    .iter()
                                    .map(|p| {
                                        match p {
                                            serdes_ai_core::messages::UserContentPart::Text {
                                                text,
                                            } => text.len(),
                                            _ => 100, // Estimate for non-text
                                        }
                                    })
                                    .sum()
                            }
                        }
                    }
                    serdes_ai_core::ModelRequestPart::ToolReturn(t) => {
                        t.content.to_string_content().len()
                    }
                    serdes_ai_core::ModelRequestPart::RetryPrompt(r) => r.content.message().len(),
                    serdes_ai_core::ModelRequestPart::BuiltinToolReturn(b) => {
                        // Estimate based on content type
                        b.content_type().len() + 100
                    }
                    serdes_ai_core::ModelRequestPart::ModelResponse(r) => {
                        // Estimate based on response parts
                        r.parts
                            .iter()
                            .map(|p| match p {
                                serdes_ai_core::ModelResponsePart::Text(t) => t.content.len(),
                                serdes_ai_core::ModelResponsePart::ToolCall(tc) => {
                                    tc.tool_name.len()
                                        + tc.args.to_json_string().map(|s| s.len()).unwrap_or(50)
                                }
                                serdes_ai_core::ModelResponsePart::Thinking(t) => t.content.len(),
                                serdes_ai_core::ModelResponsePart::File(_) => 100,
                                serdes_ai_core::ModelResponsePart::BuiltinToolCall(_) => 100,
                            })
                            .sum::<usize>()
                    }
                }
            })
            .sum();

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

        // How many messages to unconditionally keep at the start
        let keep_n = self.keep_first_n.min(messages.len());

        // Add the first N messages unconditionally
        for msg in messages.iter().take(keep_n) {
            let tokens = self.estimate_tokens(msg);
            result.push(msg.clone());
            total_tokens += tokens;
        }

        // Iterate through remaining messages from the end
        let remaining = &messages[keep_n..];
        let mut to_append = Vec::new();

        for msg in remaining.iter().rev() {
            let tokens = self.estimate_tokens(msg);
            if total_tokens + tokens > self.max_tokens {
                break;
            }
            total_tokens += tokens;
            to_append.push(msg.clone());
        }

        // Reverse and append (we iterated backwards)
        to_append.reverse();
        result.extend(to_append);

        // Post-processing: Remove orphaned tool results
        // This prevents Claude API errors like:
        // "unexpected `tool_use_id` found in `tool_result` blocks. Each `tool_result`
        // block must have a corresponding `tool_use` block in the previous message."
        let valid_tool_ids = collect_all_tool_use_ids(&result);
        let result = remove_orphaned_tool_results(result, &valid_tool_ids);

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
    #[allow(clippy::should_implement_trait)]
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
    #[allow(dead_code)]
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

    #[tokio::test]
    async fn test_truncate_by_tokens_default_keeps_first_two() {
        // Default behavior: keep_first_n = 2 (system prompt + first user message)
        let processor = TruncateByTokens::new(1); // Very low token limit
        let ctx = make_test_context();
        let messages = make_messages(5);

        let result = processor.process(&ctx, messages).await;
        // Should keep at least the first 2 messages even with low token limit
        assert!(result.len() >= 2);
    }

    #[tokio::test]
    async fn test_truncate_by_tokens_keep_first_n() {
        // Explicitly set keep_first_n to 3
        let processor = TruncateByTokens::new(1).keep_first_n(3);
        let ctx = make_test_context();
        let messages = make_messages(5);

        let result = processor.process(&ctx, messages).await;
        // Should keep at least the first 3 messages
        assert!(result.len() >= 3);
    }

    #[tokio::test]
    async fn test_truncate_by_tokens_keep_first_n_zero() {
        // Set keep_first_n to 0 (don't preserve any)
        let processor = TruncateByTokens::new(1).keep_first_n(0);
        let ctx = make_test_context();
        let messages = make_messages(5);

        let result = processor.process(&ctx, messages).await;
        // With very low token limit and no preserved messages, might get 0 or 1
        assert!(result.len() <= 1);
    }

    #[tokio::test]
    async fn test_truncate_by_tokens_backwards_compat_keep_first_true() {
        // Using deprecated keep_first(true) should set keep_first_n to 1
        let processor = TruncateByTokens::new(1).keep_first(true);
        let ctx = make_test_context();
        let messages = make_messages(5);

        let result = processor.process(&ctx, messages).await;
        // Should keep at least the first message
        assert!(!result.is_empty());
    }

    #[tokio::test]
    async fn test_truncate_by_tokens_backwards_compat_keep_first_false() {
        // Using deprecated keep_first(false) should set keep_first_n to 0
        let processor = TruncateByTokens::new(1).keep_first(false);
        let ctx = make_test_context();
        let messages = make_messages(5);

        let result = processor.process(&ctx, messages).await;
        // With very low token limit and keep_first_n=0, might get 0 or 1
        assert!(result.len() <= 1);
    }

    #[tokio::test]
    async fn test_truncate_by_tokens_with_sufficient_tokens() {
        // With enough tokens, should keep all messages
        let processor = TruncateByTokens::new(10000);
        let ctx = make_test_context();
        let messages = make_messages(5);

        let result = processor.process(&ctx, messages).await;
        assert_eq!(result.len(), 5);
    }

    #[tokio::test]
    async fn test_truncate_by_tokens_keeps_most_recent() {
        // Should keep the first N + most recent messages
        let processor = TruncateByTokens::new(100).keep_first_n(1); // Keep first + some recent
        let ctx = make_test_context();
        let messages = make_messages(10);

        let result = processor.process(&ctx, messages).await;
        // First message should always be present
        assert!(!result.is_empty());
    }

    // ========================================================================
    // Tool Pair Aware Truncation Tests
    // ========================================================================

    use serdes_ai_core::{
        messages::tool_return::ToolReturnContent,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
    };

    /// Create a message with a tool call (ModelResponse containing ToolCallPart)
    fn make_tool_call_message(tool_call_id: &str) -> ModelRequest {
        let mut response = ModelResponse::new();
        let tool_call = ToolCallPart::new("test_tool", serde_json::json!({"arg": "value"}))
            .with_tool_call_id(tool_call_id);
        response.add_part(ModelResponsePart::ToolCall(tool_call));
        
        ModelRequest::with_parts(vec![
            ModelRequestPart::ModelResponse(Box::new(response)),
        ])
    }

    /// Create a message with a tool return (ToolReturnPart)
    fn make_tool_return_message(tool_call_id: &str) -> ModelRequest {
        let tool_return = ToolReturnPart::new("test_tool", ToolReturnContent::text("result"))
            .with_tool_call_id(tool_call_id);
        
        ModelRequest::with_parts(vec![
            ModelRequestPart::ToolReturn(tool_return),
        ])
    }

    #[test]
    fn test_extract_tool_use_ids() {
        let msg = make_tool_call_message("call_123");
        let ids = extract_tool_use_ids(&msg);
        assert_eq!(ids, vec!["call_123"]);
    }

    #[test]
    fn test_extract_tool_use_ids_empty() {
        let msg = make_messages(1).pop().unwrap();
        let ids = extract_tool_use_ids(&msg);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_extract_tool_result_ids() {
        let msg = make_tool_return_message("call_456");
        let ids = extract_tool_result_ids(&msg);
        assert_eq!(ids, vec!["call_456"]);
    }

    #[test]
    fn test_extract_tool_result_ids_empty() {
        let msg = make_messages(1).pop().unwrap();
        let ids = extract_tool_result_ids(&msg);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_remove_orphaned_tool_results() {
        // Create messages with a tool call and a tool return
        let tool_call_msg = make_tool_call_message("call_abc");
        let tool_return_msg = make_tool_return_message("call_abc");
        let orphan_return_msg = make_tool_return_message("call_orphan");

        let messages = vec![tool_call_msg, tool_return_msg, orphan_return_msg];
        let valid_ids = collect_all_tool_use_ids(&messages);
        
        // valid_ids should only contain "call_abc"
        assert!(valid_ids.contains("call_abc"));
        assert!(!valid_ids.contains("call_orphan"));

        let result = remove_orphaned_tool_results(messages, &valid_ids);
        
        // Should have 2 messages: tool_call and matching tool_return
        // The orphan_return_msg should be removed entirely (it only had the orphan part)
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_remove_orphaned_preserves_mixed_messages() {
        // Create a message that has both a user prompt AND an orphaned tool return
        let mut mixed_msg = ModelRequest::new();
        mixed_msg.add_user_prompt("This is a user message");
        let orphan_return = ToolReturnPart::new("test_tool", ToolReturnContent::text("orphan result"))
            .with_tool_call_id("orphan_id");
        mixed_msg.add_part(ModelRequestPart::ToolReturn(orphan_return));

        let messages = vec![mixed_msg];
        let valid_ids: HashSet<String> = HashSet::new(); // No valid IDs
        
        let result = remove_orphaned_tool_results(messages, &valid_ids);
        
        // Message should still exist because it has a user prompt
        assert_eq!(result.len(), 1);
        // But it should only have 1 part (the user prompt, not the orphan return)
        assert_eq!(result[0].parts.len(), 1);
        assert!(matches!(result[0].parts[0], ModelRequestPart::UserPrompt(_)));
    }

    #[tokio::test]
    async fn test_truncate_history_removes_orphaned_tool_results() {
        // Create a conversation with tool interactions
        // Message 0: User prompt
        // Message 1: Tool call (call_1)
        // Message 2: Tool return (call_1)
        // Message 3: Tool call (call_2)
        // Message 4: Tool return (call_2)
        let mut messages = Vec::new();
        
        let mut user_msg = ModelRequest::new();
        user_msg.add_user_prompt("Hello");
        messages.push(user_msg);
        
        messages.push(make_tool_call_message("call_1"));
        messages.push(make_tool_return_message("call_1"));
        messages.push(make_tool_call_message("call_2"));
        messages.push(make_tool_return_message("call_2"));

        // Truncate to 3 messages (keeping last 3)
        // This would normally keep: tool_return(call_1), tool_call(call_2), tool_return(call_2)
        // But tool_return(call_1) is orphaned because tool_call(call_1) was truncated
        let processor = TruncateHistory::new(3).keep_first(false);
        let ctx = make_test_context();
        
        let result = processor.process(&ctx, messages).await;
        
        // The orphaned tool_return(call_1) should be removed
        // So we should have: tool_call(call_2), tool_return(call_2)
        assert_eq!(result.len(), 2);
        
        // Verify no orphaned tool results
        let tool_use_ids = collect_all_tool_use_ids(&result);
        let tool_result_ids = collect_all_tool_result_ids(&result);
        
        // All tool_result_ids should have matching tool_use_ids
        for id in &tool_result_ids {
            assert!(tool_use_ids.contains(id), "Orphaned tool_result found: {}", id);
        }
    }

    #[tokio::test]
    async fn test_truncate_by_tokens_removes_orphaned_tool_results() {
        // Similar test but for TruncateByTokens
        let mut messages = Vec::new();
        
        let mut user_msg = ModelRequest::new();
        user_msg.add_user_prompt("Hello");
        messages.push(user_msg);
        
        messages.push(make_tool_call_message("call_a"));
        messages.push(make_tool_return_message("call_a"));
        messages.push(make_tool_call_message("call_b"));
        messages.push(make_tool_return_message("call_b"));

        // Use a small token limit to force truncation
        let processor = TruncateByTokens::new(200).keep_first_n(0);
        let ctx = make_test_context();
        
        let result = processor.process(&ctx, messages).await;
        
        // Verify no orphaned tool results
        let tool_use_ids = collect_all_tool_use_ids(&result);
        let tool_result_ids = collect_all_tool_result_ids(&result);
        
        for id in &tool_result_ids {
            assert!(tool_use_ids.contains(id), "Orphaned tool_result found: {}", id);
        }
    }

    #[tokio::test]
    async fn test_tool_pair_aware_truncation_keeps_complete_pairs() {
        // Test that when we have complete pairs, they're preserved
        let mut messages = Vec::new();
        
        messages.push(make_tool_call_message("call_x"));
        messages.push(make_tool_return_message("call_x"));

        let processor = TruncateByTokens::new(10000).keep_first_n(0);
        let ctx = make_test_context();
        
        let result = processor.process(&ctx, messages).await;
        
        // Both should be preserved (no truncation needed)
        assert_eq!(result.len(), 2);
        
        let tool_use_ids = collect_all_tool_use_ids(&result);
        let tool_result_ids = collect_all_tool_result_ids(&result);
        
        assert_eq!(tool_use_ids.len(), 1);
        assert_eq!(tool_result_ids.len(), 1);
        assert!(tool_use_ids.contains("call_x"));
        assert!(tool_result_ids.contains("call_x"));
    }

    #[test]
    fn test_collect_all_tool_use_ids() {
        let messages = vec![
            make_tool_call_message("id_1"),
            make_tool_call_message("id_2"),
            make_tool_return_message("id_1"),
        ];
        
        let ids = collect_all_tool_use_ids(&messages);
        
        assert_eq!(ids.len(), 2);
        assert!(ids.contains("id_1"));
        assert!(ids.contains("id_2"));
    }

    #[test]
    fn test_collect_all_tool_result_ids() {
        let messages = vec![
            make_tool_call_message("id_1"),
            make_tool_return_message("id_1"),
            make_tool_return_message("id_2"),
        ];
        
        let ids = collect_all_tool_result_ids(&messages);
        
        assert_eq!(ids.len(), 2);
        assert!(ids.contains("id_1"));
        assert!(ids.contains("id_2"));
    }

    #[test]
    fn test_tool_return_with_none_id_is_kept() {
        // ToolReturn with None tool_call_id should be kept (intentional edge case handling)
        let tool_return_no_id = ToolReturnPart::new("test_tool", ToolReturnContent::text("result"));
        // Note: no .with_tool_call_id() - so it's None
        
        let msg = ModelRequest::with_parts(vec![
            ModelRequestPart::ToolReturn(tool_return_no_id),
        ]);

        let messages = vec![msg];
        let valid_ids: HashSet<String> = HashSet::new(); // No valid IDs
        
        let result = remove_orphaned_tool_results(messages, &valid_ids);
        
        // Should be kept because tool_call_id is None
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].parts.len(), 1);
    }

    #[test]
    fn test_builtin_tool_return_with_empty_string_id_is_removed() {
        use serdes_ai_core::messages::parts::{BuiltinToolReturnContent, WebSearchResults};
        
        // BuiltinToolReturn with empty string tool_call_id should be removed
        let empty_results = WebSearchResults::new("query", vec![]);
        let content = BuiltinToolReturnContent::web_search(empty_results);
        // Create with empty string ID - this is malformed data
        let builtin_return = serdes_ai_core::BuiltinToolReturnPart::new(
            "web_search",
            content,
            "", // Empty string!
        );
        
        let msg = ModelRequest::with_parts(vec![
            ModelRequestPart::BuiltinToolReturn(builtin_return),
        ]);

        let messages = vec![msg];
        // Even with an empty valid_ids set, empty string should be treated as invalid
        let valid_ids: HashSet<String> = HashSet::new();
        
        let result = remove_orphaned_tool_results(messages, &valid_ids);
        
        // Should be removed because empty string ID is invalid
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_builtin_tool_return_with_valid_id_is_kept() {
        use serdes_ai_core::messages::parts::{BuiltinToolReturnContent, WebSearchResults};
        
        // BuiltinToolReturn with matching ID should be kept
        let empty_results = WebSearchResults::new("query", vec![]);
        let content = BuiltinToolReturnContent::web_search(empty_results);
        let builtin_return = serdes_ai_core::BuiltinToolReturnPart::new(
            "web_search",
            content,
            "valid_call_id",
        );
        
        let msg = ModelRequest::with_parts(vec![
            ModelRequestPart::BuiltinToolReturn(builtin_return),
        ]);

        let messages = vec![msg];
        let mut valid_ids: HashSet<String> = HashSet::new();
        valid_ids.insert("valid_call_id".to_string());
        
        let result = remove_orphaned_tool_results(messages, &valid_ids);
        
        // Should be kept because ID matches
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].parts.len(), 1);
    }
}

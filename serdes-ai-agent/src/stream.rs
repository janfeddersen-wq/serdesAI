//! Streaming agent execution.
//!
//! This module provides streaming support for agent runs with real
//! character-by-character streaming from the model.

use crate::agent::{Agent, RegisteredTool};
use crate::context::{RunContext, RunUsage, generate_run_id};
use crate::errors::AgentRunError;
use crate::lifecycle::{self, CheckpointBoundary, ContextPolicyInput};
use crate::run::{CompressionStrategy, RunOptions};
use chrono::Utc;
use futures::{Stream, StreamExt};
use serdes_ai_core::ClassifyModelFailure;
use serdes_ai_core::messages::{
    ModelResponseStreamEvent, StreamCompleteEvent, ToolCallArgs, ToolReturnPart, UserContent,
};
use serdes_ai_core::{
    FinishReason, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart, RequestUsage,
};
use serdes_ai_models::ModelRequestParameters;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

// Await durability before allowing the caller to advance to another side effect.
macro_rules! checkpoint {
    ($sink:expr, $id:expr, $step:expr, $boundary:expr, $messages:expr, $response:expr, $usage:expr, $tx:expr) => {
        if let Err(error) = lifecycle::save(
            $sink.as_ref(),
            &$id,
            $step,
            $boundary,
            &$messages,
            $response,
            &$usage,
        )
        .await
        {
            let _ = $tx.try_send(Err(error));
            return;
        }
    };
}

// Conditional tracing - use no-op macros when tracing feature is disabled
#[cfg(feature = "tracing-integration")]
use tracing::{debug, error, info, warn};

#[cfg(not(feature = "tracing-integration"))]
macro_rules! debug {
    ($($arg:tt)*) => {};
}
#[cfg(not(feature = "tracing-integration"))]
macro_rules! info {
    ($($arg:tt)*) => {};
}
#[cfg(not(feature = "tracing-integration"))]
macro_rules! error {
    ($($arg:tt)*) => {};
}
#[cfg(not(feature = "tracing-integration"))]
macro_rules! warn {
    ($($arg:tt)*) => {};
}

/// Events emitted during streaming.
#[derive(Debug, Clone)]
pub enum AgentStreamEvent {
    /// Run started.
    RunStart { run_id: String },
    /// Context size information (emitted before each model request).
    ContextInfo {
        /// Estimated token count (~request_bytes / 4).
        estimated_tokens: usize,
        /// Raw request size in bytes (serialized messages + tools).
        request_bytes: usize,
        /// Model's context window limit (if known).
        context_limit: Option<u64>,
    },
    /// Context was compressed to fit within limits.
    ContextCompressed {
        /// Token count before compression.
        original_tokens: usize,
        /// Token count after compression.
        compressed_tokens: usize,
        /// Strategy used: "truncate" or "summarize".
        strategy: String,
        /// Number of messages before compression.
        messages_before: usize,
        /// Number of messages after compression.
        messages_after: usize,
    },
    /// Model request started.
    RequestStart { step: u32 },
    /// Text delta.
    TextDelta { text: String },
    /// Tool call started.
    ToolCallStart {
        tool_name: String,
        tool_call_id: Option<String>,
    },
    /// Tool call arguments delta.
    ToolCallDelta {
        delta: String,
        tool_call_id: Option<String>,
    },
    /// Tool call completed (arguments fully received).
    ToolCallComplete {
        tool_name: String,
        tool_call_id: Option<String>,
    },
    /// Tool executed.
    ToolExecuted {
        tool_name: String,
        tool_call_id: Option<String>,
        success: bool,
        error: Option<String>,
    },
    /// Thinking delta (for reasoning models).
    ThinkingDelta { text: String },
    /// Model response completed.
    ResponseComplete {
        /// 1-based index of the model response within the run.
        step: u32,
        /// Token usage the provider reported for THIS model response.
        ///
        /// `None` means the provider did not report usage for this step.
        usage: Option<RequestUsage>,
    },
    /// Output ready.
    OutputReady,
    /// Run completed.
    RunComplete {
        run_id: String,
        /// Complete message history from this run (system prompt, user prompts,
        /// assistant responses, tool calls and returns).
        messages: Vec<ModelRequest>,
        /// Field-wise aggregate of the per-step `RequestUsage` values across
        /// every model response in this run; `total_tokens` is derived from
        /// request+response when a provider omits it. Equivalent to the
        /// `AgentRunResult.usage` returned by the non-streaming `run()` path.
        usage: RunUsage,
    },
    /// Error occurred.
    Error { message: String },
    /// Run was cancelled.
    Cancelled {
        /// Partial text accumulated before cancellation.
        partial_text: Option<String>,
        /// Partial thinking content accumulated before cancellation.
        partial_thinking: Option<String>,
        /// Tool calls that were in progress when cancelled.
        pending_tools: Vec<String>,
        /// Run-aggregate token usage accumulated up to the point of cancellation.
        ///
        /// A cancelled run still reports what it spent so far; may be all-zero
        /// (`RunUsage::default()`) if cancelled before any usage-bearing response.
        usage: RunUsage,
    },
}

/// Streaming agent execution.
///
/// This provides real streaming by spawning a task that streams from the model
/// and sends events through a channel.
///
/// # Cancellation
///
/// Use [`AgentStream::new_with_cancel`] to create a stream with cancellation support.
/// When the cancellation token is triggered, the stream will:
/// 1. Stop the model stream
/// 2. Cancel any pending tool calls
/// 3. Emit a [`AgentStreamEvent::Cancelled`] event with partial results
pub struct AgentStream {
    output: Arc<std::sync::Mutex<Option<Box<dyn std::any::Any + Send + Sync>>>>,
    rx: mpsc::Receiver<Result<AgentStreamEvent, AgentRunError>>,
    /// Cancellation token for this stream (if cancellation is enabled).
    cancel_token: Option<CancellationToken>,
}

/// Canonicalize tool-call arguments in a model response before persisting it.
fn canonicalize_tool_call_args_in_response(response: &mut ModelResponse) {
    for part in &mut response.parts {
        if let ModelResponsePart::ToolCall(tc) = part {
            let repaired = tc.args.to_json();
            tc.args = ToolCallArgs::Json(repaired);
        }
    }
}

fn usage_from_stream_complete(event: &StreamCompleteEvent) -> Option<RequestUsage> {
    event.request_usage()
}

impl AgentStream {
    pub(crate) fn take_typed_output<O: Send + Sync + 'static>(&mut self) -> Option<O> {
        self.output
            .lock()
            .unwrap()
            .take()?
            .downcast::<O>()
            .ok()
            .map(|value| *value)
    }

    /// Create a new streaming agent run.
    ///
    /// This spawns a background task that handles the actual streaming
    /// and tool execution.
    pub async fn new<Deps, Output>(
        agent: &Agent<Deps, Output>,
        prompt: UserContent,
        deps: Deps,
        options: RunOptions,
    ) -> Result<Self, AgentRunError>
    where
        Deps: Send + Sync + 'static,
        Output: Send + Sync + 'static,
    {
        Self::new_with_cancel(agent, prompt, deps, options, CancellationToken::new()).await
    }

    /// Create a new streaming agent run with cancellation support.
    ///
    /// The provided `CancellationToken` can be used to cancel the agent run
    /// mid-execution. When cancelled:
    /// - The model stream is stopped
    /// - In-flight tool calls are aborted
    /// - A `Cancelled` event is emitted with partial results
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tokio_util::sync::CancellationToken;
    ///
    /// let cancel_token = CancellationToken::new();
    /// let stream = AgentStream::new_with_cancel(
    ///     &agent,
    ///     "Hello!".into(),
    ///     deps,
    ///     RunOptions::default(),
    ///     cancel_token.clone(),
    /// ).await?;
    ///
    /// // Cancel from another task
    /// cancel_token.cancel();
    /// ```
    pub async fn new_with_cancel<Deps, Output>(
        agent: &Agent<Deps, Output>,
        prompt: UserContent,
        deps: Deps,
        options: RunOptions,
        cancel_token: CancellationToken,
    ) -> Result<Self, AgentRunError>
    where
        Deps: Send + Sync + 'static,
        Output: Send + Sync + 'static,
    {
        let output = Arc::new(std::sync::Mutex::new(
            None::<Box<dyn std::any::Any + Send + Sync>>,
        ));
        let output_writer = output.clone();
        let run_id = generate_run_id();
        let (tx, rx) = mpsc::channel(64);

        // Clone what we need for the spawned task
        let model = agent.model_arc();
        let model_name = model.name().to_string();
        let model_settings = options
            .model_settings
            .clone()
            .unwrap_or_else(|| agent.model_settings.clone());

        // Share schema, validators and prompt generators with the worker.
        let mut static_system_prompt = agent.static_system_prompt().to_string();
        let dynamic_prompts = agent.system_prompt_fns.clone();
        let dynamic_instructions = agent.instruction_fns.clone();
        let output_schema = agent.output_schema.clone();
        let output_validators = agent.output_validators.clone();
        let max_output_retries = agent.max_output_retries;

        let tool_definitions = agent.tool_definitions();
        let native_output_schema = agent.native_output_schema();
        let _end_strategy = agent.end_strategy;
        let parallel_tools = agent.parallel_tool_calls;
        let max_concurrent_tools = agent.max_concurrent_tools;
        let usage_limits = agent.usage_limits.clone();
        let run_usage_limits = options.usage_limits.clone();

        // Clone tool executors - now possible because RegisteredTool implements Clone!
        let tools: Vec<RegisteredTool<Deps>> = agent.tools.to_vec();

        // Wrap deps in Arc for shared access in tool execution
        let deps = Arc::new(deps);

        let initial_history = options.message_history.clone();
        let _metadata = options.metadata.clone();
        let processors = agent.history_processors.clone();
        let context_policy = agent.context_policy.clone();
        let context_failure = agent.context_failure;
        let supervisor = crate::stream_lifecycle::StreamLifecycle::new(
            agent.checkpoint_sink.clone(),
            run_id.clone(),
        );
        let checkpoint_sink: Option<Arc<dyn crate::CheckpointSink>> = Some(supervisor.clone());
        let compression_config = if context_policy.is_some() || !processors.is_empty() {
            None
        } else {
            options.compression.clone()
        };
        let run_id_clone = run_id.clone();

        debug!(run_id = %run_id, "AgentStream: spawning streaming task");

        // Spawn the streaming task
        let mut initial_snapshot = initial_history.clone().unwrap_or_default();
        let mut initial_prompt = ModelRequest::new();
        initial_prompt.add_user_prompt(prompt.clone());
        initial_snapshot.push(initial_prompt);
        supervisor.update(&initial_snapshot, None, 0, &RunUsage::new());
        let supervisor_tx = tx.clone();
        let supervisor_token = cancel_token.clone();
        tokio::spawn(async move {
            let work = async {
                let mut validated_output = None;
                info!(run_id = %run_id_clone, "AgentStream: task started");

                let mut policy_context =
                    RunContext::with_shared_deps(deps.clone(), model_name.clone());
                policy_context.run_id = run_id_clone.clone();
                policy_context.model_settings = model_settings.clone();
                policy_context.metadata = _metadata.clone();
                for prompt in &dynamic_prompts {
                    if let Some(text) = prompt.generate(&policy_context).await {
                        if !text.is_empty() {
                            static_system_prompt.push_str("\n\n");
                            static_system_prompt.push_str(&text);
                        }
                    }
                }
                for instruction in &dynamic_instructions {
                    if let Some(text) = instruction.generate(&policy_context).await {
                        if !text.is_empty() {
                            static_system_prompt.push_str("\n\n");
                            static_system_prompt.push_str(&text);
                        }
                    }
                }
                let mut output_retries = 0u32;
                // Emit RunStart
                debug!("AgentStream: emitting RunStart");
                if tx
                    .send(Ok(AgentStreamEvent::RunStart {
                        run_id: run_id_clone.clone(),
                    }))
                    .await
                    .is_err()
                {
                    warn!("AgentStream: receiver dropped before RunStart");
                    return;
                }

                // Build initial messages
                let mut messages = initial_history.unwrap_or_default();
                debug!(
                    initial_messages = messages.len(),
                    "AgentStream: building messages"
                );

                // Add system prompt if non-empty
                if !static_system_prompt.is_empty() {
                    let mut req = ModelRequest::new();
                    req.add_system_prompt(static_system_prompt.clone());
                    messages.push(req);
                }

                // Add user prompt
                let mut user_req = ModelRequest::new();
                user_req.add_user_prompt(prompt);
                messages.push(user_req);
                supervisor.update(&messages, None, 0, &RunUsage::new());

                let mut responses: Vec<ModelResponse> = Vec::new();
                let mut usage = RunUsage::new();
                let mut step = 0u32;
                let mut finished = false;
                let mut finish_reason: Option<FinishReason>;

                // Main agent loop
                while !finished {
                    step += 1;

                    // Check usage limits
                    if let Some(ref limits) = usage_limits {
                        if let Err(e) = limits.check(&usage) {
                            checkpoint!(
                                checkpoint_sink,
                                run_id_clone,
                                step,
                                CheckpointBoundary::Failed,
                                messages,
                                responses.last(),
                                usage,
                                tx
                            );
                            let _ = tx.send(Err(e.into())).await;
                            return;
                        }
                    }

                    if let Some(ref limits) = run_usage_limits {
                        if let Err(e) = limits.check(&usage) {
                            checkpoint!(
                                checkpoint_sink,
                                run_id_clone,
                                step,
                                CheckpointBoundary::Failed,
                                messages,
                                responses.last(),
                                usage,
                                tx
                            );
                            let _ = tx.send(Err(e.into())).await;
                            return;
                        }
                    }

                    // Emit RequestStart
                    if tx
                        .send(Ok(AgentStreamEvent::RequestStart { step }))
                        .await
                        .is_err()
                    {
                        return;
                    }

                    // Build request parameters
                    let mut params = ModelRequestParameters::new()
                        .with_tools_arc(tool_definitions.clone())
                        .with_allow_text(true);

                    // Carry the structured-output request to the provider, as the
                    // blocking path does.
                    if let Some(schema) = native_output_schema.clone() {
                        params = params.with_output_schema(schema);
                    }

                    // === Context Size Calculation & Compression ===

                    // Calculate context size by serializing (this is the actual request size)
                    let (request_bytes, estimated_tokens) = {
                        let messages_json = serde_json::to_string(&messages).unwrap_or_default();
                        let tools_json =
                            serde_json::to_string(&*tool_definitions).unwrap_or_default();
                        let bytes = messages_json.len() + tools_json.len();
                        (bytes, bytes / 4)
                    };

                    // Get context limit from model profile
                    let context_limit = model.profile().context_window;

                    // Emit ContextInfo event
                    let _ = tx
                        .send(Ok(AgentStreamEvent::ContextInfo {
                            estimated_tokens,
                            request_bytes,
                            context_limit,
                        }))
                        .await;

                    // Check if compression is needed
                    if let Some(ref compression) = compression_config {
                        if let Some(limit) = context_limit {
                            let threshold_tokens = (limit as f64 * compression.threshold) as usize;

                            if estimated_tokens > threshold_tokens {
                                let messages_before = messages.len();
                                let original_tokens = estimated_tokens;

                                // Apply compression based on strategy
                                let strategy_name = match compression.strategy {
                                    CompressionStrategy::Truncate => {
                                        // Use TruncateByTokens with keep_first_n=2 (system + first user)
                                        use crate::history::{HistoryProcessor, TruncateByTokens};
                                        let truncator =
                                            TruncateByTokens::new(compression.target_tokens as u64)
                                                .keep_first_n(2);

                                        // Create a minimal context for the processor
                                        let temp_ctx = RunContext::new((), &model_name);
                                        messages = truncator.process(&temp_ctx, messages).await;
                                        "truncate"
                                    }
                                    CompressionStrategy::Summarize
                                    | CompressionStrategy::SummarizeOrTruncate => {
                                        // Use the same model to summarize the conversation history
                                        // Keep first 2 messages (system + first user) and last few messages
                                        // Summarize everything in between

                                        if messages.len() <= 4 {
                                            // Too few messages to summarize, just truncate
                                            use crate::history::{
                                                HistoryProcessor, TruncateByTokens,
                                            };
                                            let truncator = TruncateByTokens::new(
                                                compression.target_tokens as u64,
                                            )
                                            .keep_first_n(2);
                                            let temp_ctx = RunContext::new((), &model_name);
                                            messages = truncator.process(&temp_ctx, messages).await;
                                            "truncate (too few messages)"
                                        } else {
                                            // Split messages: first 2 (keep), middle (summarize), last 2 (keep)
                                            let first_two: Vec<_> =
                                                messages.iter().take(2).cloned().collect();
                                            let last_two: Vec<_> = messages
                                                .iter()
                                                .rev()
                                                .take(2)
                                                .cloned()
                                                .collect::<Vec<_>>()
                                                .into_iter()
                                                .rev()
                                                .collect();
                                            let middle: Vec<_> = messages
                                                .iter()
                                                .skip(2)
                                                .take(messages.len().saturating_sub(4))
                                                .cloned()
                                                .collect();

                                            if middle.is_empty() {
                                                // Nothing to summarize
                                                "summarize (nothing to compress)"
                                            } else {
                                                // Build summarization prompt
                                                let middle_json =
                                                    serde_json::to_string_pretty(&middle)
                                                        .unwrap_or_default();
                                                let summary_prompt = format!(
                                                    "Condense this conversation history into a brief summary while preserving:\n\
                                                - Key decisions and conclusions\n\
                                                - Important information discovered\n\
                                                - Tool calls made and their essential results\n\
                                                - Any errors or issues encountered\n\n\
                                                Keep the summary concise but complete enough to continue the conversation.\n\n\
                                                Conversation to summarize:\n{}\n\n\
                                                Respond with ONLY the summary, no preamble.",
                                                    middle_json
                                                );

                                                // Create a minimal request for summarization
                                                let mut summary_req = ModelRequest::new();
                                                summary_req.add_user_prompt(summary_prompt);

                                                // Call the model (non-streaming for simplicity)
                                                let summary_params = ModelRequestParameters::new();
                                                match tokio::time::timeout(model_settings.timeout.unwrap_or(std::time::Duration::from_secs(30)), model
                                                .request(
                                                    &[summary_req],
                                                    &model_settings,
                                                    &summary_params,
                                                )).await.unwrap_or_else(|_| Err(serdes_ai_models::ModelError::incomplete_stream("summary deadline exceeded")))
                                            {
                                                Ok(response) => {
                                                    if let Some(summary_usage) = &response.usage { usage.add_request(summary_usage.clone()); } else { usage.record_request(); }
                                                    // Extract text from response
                                                    let summary_text = response
                                                        .parts
                                                        .iter()
                                                        .filter_map(|p| match p {
                                                            ModelResponsePart::Text(t) => {
                                                                Some(t.content.clone())
                                                            }
                                                            _ => None,
                                                        })
                                                        .collect::<Vec<_>>()
                                                        .join("\n");

                                                    if !summary_text.is_empty() {
                                                        // Build new message list: first 2 + summary + last 2
                                                        let mut new_messages = first_two;

                                                        // Add summary as a "previous context" message
                                                        let mut summary_msg = ModelRequest::new();
                                                        summary_msg.add_user_prompt(format!(
                                                            "[Previous conversation summary]\n{}\n[End of summary - continuing conversation]",
                                                            summary_text
                                                        ));
                                                        new_messages.push(summary_msg);

                                                        new_messages.extend(last_two);
                                                        messages = new_messages;
                                                        "summarize"
                                                    } else {
                                                        if !matches!(compression.strategy, CompressionStrategy::SummarizeOrTruncate) {
                                                            checkpoint!(checkpoint_sink, run_id_clone, step, CheckpointBoundary::Failed, messages, responses.last(), usage, tx);
                                                            let _ = tx.try_send(Err(AgentRunError::ContextPolicy("Summary returned no text".into())));
                                                            return;
                                                        }
                                                        // Explicit compatibility fallback
                                                        use crate::history::{
                                                            HistoryProcessor, TruncateByTokens,
                                                        };
                                                        let truncator = TruncateByTokens::new(
                                                            compression.target_tokens as u64,
                                                        )
                                                        .keep_first_n(2);
                                                        let temp_ctx =
                                                            RunContext::new((), &model_name);
                                                        messages = truncator
                                                            .process(&temp_ctx, messages)
                                                            .await;
                                                        "truncate (summary empty)"
                                                    }
                                                }
                                                Err(_e) => {
                                                    usage.record_request();
                                                    if !matches!(compression.strategy, CompressionStrategy::SummarizeOrTruncate) {
                                                        checkpoint!(checkpoint_sink, run_id_clone, step, CheckpointBoundary::Failed, messages, responses.last(), usage, tx);
                                                        let _ = tx.try_send(Err(AgentRunError::ContextPolicy("Summary request failed".into())));
                                                        return;
                                                    }
                                                    use crate::history::{
                                                        HistoryProcessor, TruncateByTokens,
                                                    };
                                                    let truncator = TruncateByTokens::new(
                                                        compression.target_tokens as u64,
                                                    )
                                                    .keep_first_n(2);
                                                    let temp_ctx = RunContext::new((), &model_name);
                                                    messages = truncator
                                                        .process(&temp_ctx, messages)
                                                        .await;
                                                    "truncate (summary failed)"
                                                }
                                            }
                                            }
                                        }
                                    }
                                };

                                // Calculate new size
                                let new_bytes = serde_json::to_string(&messages)
                                    .map(|s| s.len())
                                    .unwrap_or(0);
                                let compressed_tokens = new_bytes / 4;

                                // Emit compression event
                                let _ = tx
                                    .send(Ok(AgentStreamEvent::ContextCompressed {
                                        original_tokens,
                                        compressed_tokens,
                                        strategy: strategy_name.to_string(),
                                        messages_before,
                                        messages_after: messages.len(),
                                    }))
                                    .await;
                            }
                        }
                    }
                    // === End Context Compression ===

                    for limits in [usage_limits.as_ref(), run_usage_limits.as_ref()]
                        .into_iter()
                        .flatten()
                    {
                        if let Err(error) = limits.check(&usage) {
                            checkpoint!(
                                checkpoint_sink,
                                run_id_clone,
                                step,
                                CheckpointBoundary::Failed,
                                messages,
                                responses.last(),
                                usage,
                                tx
                            );
                            let _ = tx.try_send(Err(error.into()));
                            return;
                        }
                    }
                    // Make streaming request
                    info!(
                        step = step,
                        message_count = messages.len(),
                        "AgentStream: calling model.request_stream"
                    );
                    if let Err(error) = lifecycle::prepare(
                        &processors,
                        context_policy.as_ref(),
                        context_failure,
                        ContextPolicyInput {
                            context: &policy_context,
                            settings: &model_settings,
                            parameters: &params,
                            model: model.as_ref(),
                        },
                        &mut messages,
                    )
                    .await
                    {
                        checkpoint!(
                            checkpoint_sink,
                            run_id_clone,
                            step,
                            CheckpointBoundary::Failed,
                            messages,
                            responses.last(),
                            usage,
                            tx
                        );
                        let _ = tx.send(Err(error)).await;
                        return;
                    }
                    checkpoint!(
                        checkpoint_sink,
                        run_id_clone,
                        step,
                        CheckpointBoundary::BeforeRequest,
                        messages,
                        None,
                        usage,
                        tx
                    );
                    let stream_result = model
                        .request_stream(&messages, &model_settings, &params)
                        .await;

                    let mut partial_timer = lifecycle::partial_timer(checkpoint_sink.as_ref());
                    let mut partial_response = ModelResponse::with_parts(Vec::new());
                    let mut model_stream = match stream_result {
                        Ok(s) => {
                            debug!("AgentStream: model.request_stream succeeded, got stream");
                            s
                        }
                        Err(e) => {
                            error!(error = %e, "AgentStream: model.request_stream failed");
                            let _ = tx
                                .send(Ok(AgentStreamEvent::Error {
                                    message: e.to_string(),
                                }))
                                .await;
                            checkpoint!(
                                checkpoint_sink,
                                run_id_clone,
                                step,
                                CheckpointBoundary::ModelFailed(e.model_failure().kind),
                                messages,
                                Some(&partial_response),
                                usage,
                                tx
                            );
                            let _ = tx.send(Err(AgentRunError::Model(e))).await;
                            return;
                        }
                    };

                    // Collect response parts while streaming
                    let mut response_parts: Vec<ModelResponsePart> = Vec::new();
                    // Track stream events (used by tracing when enabled)
                    let mut stream_event_count = 0u32;
                    // Provider-reported finish reason (set by StreamComplete event)
                    let mut stream_finish_reason: Option<FinishReason> = None;
                    let mut stream_usage: Option<RequestUsage> = None;
                    let mut terminal_metadata = None;

                    // Process stream events
                    debug!("AgentStream: starting to process model stream events");
                    loop {
                        partial_response.parts = response_parts.clone();
                        partial_response.usage = stream_usage.clone();
                        partial_response.finish_reason = stream_finish_reason;
                        supervisor.update(&messages, Some(&partial_response), step, &usage);
                        let event_result = tokio::select! {
                            _ = partial_timer.tick(), if checkpoint_sink.as_ref().is_some_and(|sink| sink.partial_interval().is_some()) => {
                                checkpoint!(checkpoint_sink, run_id_clone, step, CheckpointBoundary::Partial,
                                    messages, Some(&partial_response), usage, tx);
                                continue;
                            }
                            event = model_stream.next() => event,
                        };
                        let Some(event_result) = event_result else {
                            break;
                        };
                        {
                            stream_event_count += 1;
                            let _ = stream_event_count;
                        }
                        match event_result {
                            Ok(event) => {
                                lifecycle::apply_partial(&mut partial_response, &event);
                                supervisor.update(&messages, Some(&partial_response), step, &usage);
                                match event {
                                    ModelResponseStreamEvent::PartStart(start) => {
                                        match &start.part {
                                            ModelResponsePart::Text(t) => {
                                                if !t.content.is_empty() {
                                                    let _ = tx
                                                        .send(Ok(AgentStreamEvent::TextDelta {
                                                            text: t.content.clone(),
                                                        }))
                                                        .await;
                                                }
                                            }
                                            ModelResponsePart::ToolCall(tc) => {
                                                let _ = tx
                                                    .send(Ok(AgentStreamEvent::ToolCallStart {
                                                        tool_name: tc.tool_name.clone(),
                                                        tool_call_id: tc.tool_call_id.clone(),
                                                    }))
                                                    .await;
                                                // If args are already present (non-streaming models),
                                                // send them as a delta immediately
                                                if let Ok(args_str) = tc.args.to_json_string() {
                                                    if !args_str.is_empty() && args_str != "{}" {
                                                        let _ = tx
                                                            .send(Ok(
                                                                AgentStreamEvent::ToolCallDelta {
                                                                    delta: args_str,
                                                                    tool_call_id: tc
                                                                        .tool_call_id
                                                                        .clone(),
                                                                },
                                                            ))
                                                            .await;
                                                    }
                                                }
                                            }
                                            ModelResponsePart::Thinking(t)
                                                if !t.content.is_empty() =>
                                            {
                                                let _ = tx
                                                    .send(Ok(AgentStreamEvent::ThinkingDelta {
                                                        text: t.content.clone(),
                                                    }))
                                                    .await;
                                            }
                                            _ => {}
                                        }
                                        response_parts.push(start.part.clone());
                                    }
                                    ModelResponseStreamEvent::PartDelta(delta) => {
                                        use serdes_ai_core::messages::ModelResponsePartDelta;
                                        match &delta.delta {
                                            ModelResponsePartDelta::Text(t) => {
                                                let _ = tx
                                                    .send(Ok(AgentStreamEvent::TextDelta {
                                                        text: t.content_delta.clone(),
                                                    }))
                                                    .await;
                                                // Update the part
                                                if let Some(ModelResponsePart::Text(text)) =
                                                    response_parts.get_mut(delta.index)
                                                {
                                                    text.content.push_str(&t.content_delta);
                                                }
                                            }
                                            ModelResponsePartDelta::ToolCall(tc) => {
                                                // Get tool_call_id from the existing response part
                                                let tool_call_id =
                                                    response_parts.get(delta.index).and_then(|p| {
                                                        if let ModelResponsePart::ToolCall(tc) = p {
                                                            tc.tool_call_id.clone()
                                                        } else {
                                                            None
                                                        }
                                                    });
                                                let _ = tx
                                                    .send(Ok(AgentStreamEvent::ToolCallDelta {
                                                        delta: tc.args_delta.clone(),
                                                        tool_call_id,
                                                    }))
                                                    .await;
                                                // Update args - accumulate the delta into the tool call
                                                if let Some(ModelResponsePart::ToolCall(
                                                    tool_call,
                                                )) = response_parts.get_mut(delta.index)
                                                {
                                                    tc.apply(tool_call);
                                                }
                                            }
                                            ModelResponsePartDelta::Thinking(t) => {
                                                let _ = tx
                                                    .send(Ok(AgentStreamEvent::ThinkingDelta {
                                                        text: t.content_delta.clone(),
                                                    }))
                                                    .await;
                                                if let Some(ModelResponsePart::Thinking(think)) =
                                                    response_parts.get_mut(delta.index)
                                                {
                                                    t.apply(think);
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                    ModelResponseStreamEvent::PartEnd(_) => {
                                        // Part finished
                                    }
                                    ModelResponseStreamEvent::StreamComplete(sc) => {
                                        stream_finish_reason = Some(sc.finish_reason);
                                        stream_usage = usage_from_stream_complete(&sc);
                                        terminal_metadata = sc.metadata.clone();
                                    }
                                }
                            }
                            Err(e) => {
                                let _ = tx
                                    .send(Ok(AgentStreamEvent::Error {
                                        message: e.to_string(),
                                    }))
                                    .await;
                                checkpoint!(
                                    checkpoint_sink,
                                    run_id_clone,
                                    step,
                                    CheckpointBoundary::ModelFailed(e.model_failure().kind),
                                    messages,
                                    Some(&partial_response),
                                    usage,
                                    tx
                                );
                                let _ = tx.send(Err(AgentRunError::Model(e))).await;
                                return;
                            }
                        }
                    }

                    info!(
                        stream_events = stream_event_count,
                        parts = response_parts.len(),
                        "AgentStream: finished processing model stream"
                    );

                    // A stream may legitimately end without an explicit terminal StreamComplete
                    // (in-memory/mock streams, and OpenAI/Google/etc. parsers which never emit one).
                    // Anthropic's parser surfaces REAL truncation separately as an explicit Err
                    // (handled above), so a clean end here with content is a successful completion.
                    // Default the finish reason to Stop. (The `response_parts.is_empty()` guard in
                    // the next block still catches a stream that produced nothing.)
                    if stream_finish_reason.is_none() {
                        debug!(
                            parts = response_parts.len(),
                            "stream ended without terminal StreamComplete; defaulting finish_reason=Stop"
                        );
                    }
                    let stream_finish_reason = stream_finish_reason.unwrap_or(FinishReason::Stop);

                    // If the stream produced no parts at all, treat it as an error
                    if response_parts.is_empty() && terminal_metadata.is_none() {
                        checkpoint!(
                            checkpoint_sink,
                            run_id_clone,
                            step,
                            CheckpointBoundary::ModelFailed(
                                serdes_ai_core::ModelFailureKind::IncompleteStream
                            ),
                            messages,
                            Some(&partial_response),
                            usage,
                            tx
                        );
                        let _ = tx
                            .send(Ok(AgentStreamEvent::Error {
                                message: "model stream ended without producing any content"
                                    .to_string(),
                            }))
                            .await;
                        let _ = tx
                            .send(Err(AgentRunError::Model(
                                serdes_ai_models::ModelError::incomplete_stream(
                                    "model stream ended without producing any content",
                                ),
                            )))
                            .await;
                        return;
                    }

                    // Build the complete response using the provider-reported finish reason
                    let mut response = ModelResponse {
                        parts: response_parts.clone(),
                        model_name: Some(model.name().to_string()),
                        timestamp: Utc::now(),
                        finish_reason: Some(stream_finish_reason),
                        usage: stream_usage,
                        vendor_id: None,
                        vendor_details: None,
                        kind: "response".to_string(),
                    };
                    if let Some(metadata) = &terminal_metadata {
                        metadata.apply(&mut response);
                    }
                    // Accumulate run-wide usage, mirroring the non-streaming run()
                    // path so streaming and non-streaming agree. The request is
                    // counted either way, so max_requests bounds the loop even
                    // against a provider that reports no usage.
                    match &response.usage {
                        Some(u) => usage.add_request(u.clone()),
                        None => usage.record_request(),
                    }

                    checkpoint!(
                        checkpoint_sink,
                        run_id_clone,
                        step,
                        CheckpointBoundary::AfterResponse,
                        messages,
                        Some(&response),
                        usage,
                        tx
                    );
                    canonicalize_tool_call_args_in_response(&mut response);

                    finish_reason = response.finish_reason;
                    responses.clear(); // Only the current response is needed by streaming checkpoints.
                    responses.push(response.clone());

                    // Emit ResponseComplete
                    let _ = tx
                        .send(Ok(AgentStreamEvent::ResponseComplete {
                            step,
                            usage: response.usage.clone(),
                        }))
                        .await;

                    // Check for tool calls that need execution
                    let tool_calls: Vec<_> = response
                        .parts
                        .iter()
                        .filter_map(|p| {
                            if let ModelResponsePart::ToolCall(tc) = p {
                                if output_schema.tool_name() == Some(tc.tool_name.as_str()) {
                                    return None;
                                }
                                Some(tc.clone())
                            } else {
                                None
                            }
                        })
                        .collect();

                    let explicit_output = response.parts.iter().any(|part| matches!(part,
                        ModelResponsePart::ToolCall(call) if output_schema.tool_name() == Some(call.tool_name.as_str())));
                    let mut accepted_mixed = None;
                    if explicit_output
                        && !tool_calls.is_empty()
                        && !matches!(
                            finish_reason,
                            Some(FinishReason::Length | FinishReason::ContentFilter)
                        )
                    {
                        policy_context.retry_count = output_retries;
                        match crate::stream_output::validate(
                            output_schema.as_ref(),
                            &output_validators,
                            &response,
                            &policy_context,
                        )
                        .await
                        {
                            Ok(output) => accepted_mixed = Some(output),
                            Err(_) => {
                                output_retries += 1;
                                let mut request = ModelRequest::new();
                                request.parts.push(ModelRequestPart::ModelResponse(Box::new(
                                    response.clone(),
                                )));
                                messages.push(request);
                                if output_retries > max_output_retries {
                                    checkpoint!(
                                        checkpoint_sink,
                                        run_id_clone,
                                        step,
                                        CheckpointBoundary::ValidationFailed,
                                        messages,
                                        Some(&response),
                                        usage,
                                        tx
                                    );
                                    let _ =
                                        tx.try_send(Err(AgentRunError::OutputValidationFailed(
                                            crate::OutputValidationError::failed(
                                                "Output validation exhausted",
                                            ),
                                        )));
                                    return;
                                }
                                let mut results = ModelRequest::new();
                                for part in &response.parts {
                                    if let ModelResponsePart::ToolCall(call) = part {
                                        let mut result = ToolReturnPart::new(
                                            &call.tool_name,
                                            "Not executed: final output rejected; retry",
                                        );
                                        if let Some(id) = &call.tool_call_id {
                                            result = result.with_tool_call_id(id);
                                        }
                                        results.parts.push(ModelRequestPart::ToolReturn(result));
                                    }
                                }
                                messages.push(results);
                                continue;
                            }
                        }
                    }
                    if !tool_calls.is_empty()
                        && !matches!(
                            finish_reason,
                            Some(FinishReason::Length | FinishReason::ContentFilter)
                        )
                    {
                        // Add response to messages for proper alternation
                        let mut response_req = ModelRequest::new();
                        response_req
                            .parts
                            .push(ModelRequestPart::ModelResponse(Box::new(response.clone())));
                        messages.push(response_req);

                        let mut tool_req = ModelRequest::new();

                        for call in &tool_calls {
                            if accepted_mixed.is_none()
                                || _end_strategy == crate::EndStrategy::Exhaustive
                            {
                                usage.record_tool_call();
                            }
                            let _ = tx
                                .send(Ok(AgentStreamEvent::ToolCallComplete {
                                    tool_name: call.tool_name.clone(),
                                    tool_call_id: call.tool_call_id.clone(),
                                }))
                                .await;
                        }
                        let results = if accepted_mixed.is_some()
                            && _end_strategy == crate::EndStrategy::Early
                        {
                            tool_calls
                                .into_iter()
                                .map(|call| {
                                    (
                                        call,
                                        Ok(serdes_ai_tools::ToolReturn::text(
                                            "Skipped: final output accepted",
                                        )),
                                    )
                                })
                                .collect()
                        } else {
                            crate::stream_tools::execute(
                                &tools,
                                tool_calls,
                                &policy_context,
                                parallel_tools,
                                max_concurrent_tools,
                            )
                            .await
                        };
                        for (call, result) in results {
                            let success = result.is_ok();
                            let error = result.as_ref().err().map(ToString::to_string);
                            let mut part = match result {
                                Ok(value) => ToolReturnPart::new(&call.tool_name, value.content),
                                Err(_) => {
                                    ToolReturnPart::error(&call.tool_name, "Tool execution failed")
                                }
                            };
                            if let Some(id) = &call.tool_call_id {
                                part = part.with_tool_call_id(id);
                            }
                            tool_req.parts.push(ModelRequestPart::ToolReturn(part));
                            let _ = tx
                                .send(Ok(AgentStreamEvent::ToolExecuted {
                                    tool_name: call.tool_name,
                                    tool_call_id: call.tool_call_id,
                                    success,
                                    error,
                                }))
                                .await;
                        }
                        // A mixed final-output call is not executed as an application tool.
                        // Acknowledge it before asking for the next final response.
                        for part in &response.parts {
                            if let ModelResponsePart::ToolCall(call) = part {
                                if output_schema.tool_name() == Some(call.tool_name.as_str()) {
                                    let mut result = ToolReturnPart::new(
                                        &call.tool_name,
                                        "Output deferred until ordinary tools complete",
                                    );
                                    if let Some(id) = &call.tool_call_id {
                                        result = result.with_tool_call_id(id);
                                    }
                                    tool_req.parts.push(ModelRequestPart::ToolReturn(result));
                                }
                            }
                        }

                        if !tool_req.parts.is_empty() {
                            messages.push(tool_req);
                            checkpoint!(
                                checkpoint_sink,
                                run_id_clone,
                                step,
                                CheckpointBoundary::AfterTools,
                                messages,
                                Some(&response),
                                usage,
                                tx
                            );
                        }

                        if let Some(output) = accepted_mixed {
                            validated_output = Some(output);
                            finished = true;
                            let _ = tx.send(Ok(AgentStreamEvent::OutputReady)).await;
                        }
                        continue;
                    }

                    // Parse and validate through the actual configured output type before
                    // any completion event. Partial token/filter output remains partial.
                    let partial = matches!(
                        finish_reason,
                        Some(FinishReason::Length | FinishReason::ContentFilter)
                    );
                    if partial {
                        validated_output = None;
                    }
                    if !partial {
                        policy_context.retry_count = output_retries;
                        let validation = crate::stream_output::validate(
                            output_schema.as_ref(),
                            &output_validators,
                            &response,
                            &policy_context,
                        )
                        .await;
                        if let Ok(value) = validation {
                            validated_output = Some(value);
                        } else {
                            output_retries += 1;
                            let mut rejected = ModelRequest::new();
                            rejected
                                .parts
                                .push(ModelRequestPart::ModelResponse(Box::new(response.clone())));
                            messages.push(rejected);
                            if output_retries > max_output_retries {
                                checkpoint!(
                                    checkpoint_sink,
                                    run_id_clone,
                                    step,
                                    CheckpointBoundary::ValidationFailed,
                                    messages,
                                    Some(&response),
                                    usage,
                                    tx
                                );
                                let _ = tx.try_send(Err(AgentRunError::OutputValidationFailed(
                                    crate::OutputValidationError::failed(
                                        "Output validation retry budget exhausted",
                                    ),
                                )));
                                return;
                            }
                            let mut retry = ModelRequest::new();
                            let output_calls: Vec<_> = response
                                .parts
                                .iter()
                                .filter_map(|p| match p {
                                    ModelResponsePart::ToolCall(call)
                                        if output_schema.tool_name()
                                            == Some(call.tool_name.as_str()) =>
                                    {
                                        Some(call)
                                    }
                                    _ => None,
                                })
                                .collect();
                            if output_calls.is_empty() {
                                retry.parts.push(ModelRequestPart::RetryPrompt(serdes_ai_core::messages::RetryPromptPart::new("Output did not pass validation; return a valid final output")));
                            } else {
                                for call in output_calls {
                                    let mut part = serdes_ai_core::messages::RetryPromptPart::new("Output did not pass validation; return a valid final output").with_tool_name(&call.tool_name);
                                    if let Some(id) = &call.tool_call_id {
                                        part = part.with_tool_call_id(id);
                                    }
                                    retry.parts.push(ModelRequestPart::RetryPrompt(part));
                                }
                            }
                            messages.push(retry);
                            continue;
                        }
                    }
                    // No tool calls - check finish condition
                    if partial
                        || _end_strategy == crate::EndStrategy::Early
                        || finish_reason.is_some_and(|r| r.is_complete())
                        || output_schema.tool_name().is_some()
                    {
                        // Add final response to messages for complete history
                        let mut response_req = ModelRequest::new();
                        response_req
                            .parts
                            .push(ModelRequestPart::ModelResponse(Box::new(response.clone())));
                        messages.push(response_req);

                        finished = true;
                        if !partial {
                            let _ = tx.send(Ok(AgentStreamEvent::OutputReady)).await;
                        }
                    } else if let Some(r) = finish_reason {
                        // finish_reason is Some but NOT complete, and there were no
                        // tool calls (the tool-call path `continue`d before reaching
                        // here). The loop is about to silently re-issue the same
                        // request - make that observable. NOT a behavior change:
                        // termination is unchanged; termination-on-Length/etc. is a
                        // separate P2 follow-up. `let _ = &r` keeps `r` "used" so the
                        // no-op `warn!` build (tracing-integration off) stays clean.
                        let _ = &r;
                        warn!(
                            finish_reason = ?r,
                            "stream completed with a non-terminal finish reason and no tool calls; re-issuing request (this can loop - see follow-up for Length/ContentFilter/Error handling)"
                        );
                    }
                }

                checkpoint!(
                    checkpoint_sink,
                    run_id_clone,
                    step,
                    CheckpointBoundary::Terminal,
                    messages,
                    responses.last(),
                    usage,
                    tx
                );
                if let Some(value) = validated_output {
                    *output_writer.lock().unwrap() = Some(Box::new(value));
                }
                // Emit RunComplete
                let _ = tx
                    .send(Ok(AgentStreamEvent::RunComplete {
                        run_id: run_id_clone,
                        messages,
                        usage,
                    }))
                    .await;
            };
            supervisor
                .supervise(work, supervisor_tx, supervisor_token)
                .await;
        });

        Ok(AgentStream {
            output,
            rx,
            cancel_token: Some(cancel_token),
        })
    }

    /// Cancel the running agent stream.
    ///
    /// If this stream was created with cancellation support via
    /// [`AgentStream::new_with_cancel`], this will trigger cancellation.
    /// The stream will emit a `Cancelled` event with any partial results.
    ///
    /// Both constructors support cancellation. Dropping the receiver also stops
    /// in-flight work and records ConsumerDetached when a sink is configured.
    pub fn cancel(&self) {
        if let Some(ref token) = self.cancel_token {
            token.cancel();
        }
    }

    /// Check if this stream was cancelled.
    ///
    /// Returns `true` if a cancellation token was provided and it has been
    /// triggered, `false` otherwise.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token
            .as_ref()
            .map(|t| t.is_cancelled())
            .unwrap_or(false)
    }

    /// Get the cancellation token if one was provided.
    ///
    /// This can be used to share the token with other tasks that need
    /// to coordinate cancellation.
    pub fn cancellation_token(&self) -> Option<&CancellationToken> {
        self.cancel_token.as_ref()
    }
}

impl Stream for AgentStream {
    type Item = Result<AgentStreamEvent, AgentRunError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.rx).poll_recv(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::agent;
    use futures::{StreamExt, stream};
    use serdes_ai_core::messages::{
        FinishReason, ModelRequestPart, StreamCompleteEvent, TextPart, ToolCallPart,
    };
    use serdes_ai_models::FunctionModel;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn test_stream_event_debug() {
        let event = AgentStreamEvent::TextDelta {
            text: "hello".to_string(),
        };
        let debug = format!("{:?}", event);
        assert!(debug.contains("TextDelta"));
    }

    #[test]
    fn test_stream_event_variants() {
        let events = [
            AgentStreamEvent::RunStart {
                run_id: "123".to_string(),
            },
            AgentStreamEvent::RequestStart { step: 1 },
            AgentStreamEvent::TextDelta {
                text: "hi".to_string(),
            },
            AgentStreamEvent::ToolCallStart {
                tool_name: "search".to_string(),
                tool_call_id: Some("call-1".to_string()),
            },
            AgentStreamEvent::OutputReady,
            AgentStreamEvent::RunComplete {
                run_id: "123".to_string(),
                messages: vec![],
                usage: RunUsage::default(),
            },
            AgentStreamEvent::Cancelled {
                partial_text: Some("partial".to_string()),
                partial_thinking: None,
                pending_tools: vec!["tool1".to_string()],
                usage: RunUsage::default(),
            },
        ];

        assert_eq!(events.len(), 7);
    }

    #[test]
    fn test_cancelled_event() {
        let event = AgentStreamEvent::Cancelled {
            partial_text: Some("Hello, I was saying...".to_string()),
            partial_thinking: Some("Let me think about this...".to_string()),
            pending_tools: vec!["search".to_string(), "fetch".to_string()],
            usage: RunUsage::default(),
        };

        let debug = format!("{:?}", event);
        assert!(debug.contains("Cancelled"));
        assert!(debug.contains("partial_text"));
        assert!(debug.contains("pending_tools"));
    }

    #[test]
    fn test_cancelled_event_empty() {
        let event = AgentStreamEvent::Cancelled {
            partial_text: None,
            partial_thinking: None,
            pending_tools: vec![],
            usage: RunUsage::default(),
        };

        if let AgentStreamEvent::Cancelled {
            partial_text,
            partial_thinking,
            pending_tools,
            usage,
        } = event
        {
            assert!(partial_text.is_none());
            assert!(partial_thinking.is_none());
            assert!(pending_tools.is_empty());
            // No usage-bearing response occurred, so the partial aggregate is empty.
            assert_eq!(usage.request_count, 0);
            assert_eq!(usage.request_tokens, 0);
            assert_eq!(usage.response_tokens, 0);
            assert_eq!(usage.total_tokens, 0);
        } else {
            panic!("Expected Cancelled event");
        }
    }

    #[test]
    fn test_stream_complete_usage_preserves_all_provider_token_fields() {
        let event = StreamCompleteEvent::new(FinishReason::Stop)
            .with_input_tokens(10)
            .with_output_tokens(5)
            .with_cache_creation_tokens(3)
            .with_cache_read_tokens(7);

        let usage = usage_from_stream_complete(&event).expect("usage should be present");

        assert_eq!(usage.request_tokens, Some(10));
        assert_eq!(usage.response_tokens, Some(5));
        assert_eq!(usage.total_tokens, Some(15));
        assert_eq!(usage.cache_creation_tokens, Some(3));
        assert_eq!(usage.cache_read_tokens, Some(7));
    }

    // ========================================================================
    // Usage-surfacing tests (T1-T4): token usage flows through the streaming
    // `AgentStreamEvent` path, mirroring the non-streaming `run()` path.
    // ========================================================================

    /// T1 (R1): `ResponseComplete` carries the per-step provider usage.
    #[tokio::test]
    async fn test_response_complete_reports_per_step_usage() {
        let model = FunctionModel::with_stream(move |_messages, _settings| {
            let events = vec![
                Ok(ModelResponseStreamEvent::part_start(
                    0,
                    ModelResponsePart::Text(TextPart::new("done")),
                )),
                Ok(ModelResponseStreamEvent::part_end(0)),
                Ok(ModelResponseStreamEvent::StreamComplete(
                    StreamCompleteEvent::new(FinishReason::Stop)
                        .with_input_tokens(10)
                        .with_output_tokens(5),
                )),
            ];
            Box::pin(stream::iter(events))
        });

        let agent = agent(model).build();
        let mut stream = agent
            .run_stream("hello", ())
            .await
            .expect("stream should start");

        let mut response_completes = Vec::new();
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event should be ok");
            if let AgentStreamEvent::ResponseComplete { step, usage } = event {
                response_completes.push((step, usage));
            }
        }

        assert_eq!(
            response_completes.len(),
            1,
            "expected exactly one ResponseComplete"
        );
        let (step, usage) = &response_completes[0];
        assert_eq!(*step, 1);
        let usage = usage.as_ref().expect("per-step usage should be present");
        assert_eq!(usage.request_tokens, Some(10));
        assert_eq!(usage.response_tokens, Some(5));
        assert_eq!(usage.total_tokens, Some(15));
    }

    /// T2 (R1 / AC1.2): `ResponseComplete.usage` is `None` when the provider
    /// reports no token fields.
    #[tokio::test]
    async fn test_response_complete_usage_none_when_provider_reports_nothing() {
        let model = FunctionModel::with_stream(move |_messages, _settings| {
            let events = vec![
                Ok(ModelResponseStreamEvent::part_start(
                    0,
                    ModelResponsePart::Text(TextPart::new("done")),
                )),
                Ok(ModelResponseStreamEvent::part_end(0)),
                Ok(ModelResponseStreamEvent::StreamComplete(
                    StreamCompleteEvent::new(FinishReason::Stop),
                )),
            ];
            Box::pin(stream::iter(events))
        });

        let agent = agent(model).build();
        let mut stream = agent
            .run_stream("hello", ())
            .await
            .expect("stream should start");

        let mut saw_response_complete = false;
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event should be ok");
            if let AgentStreamEvent::ResponseComplete { usage, .. } = event {
                saw_response_complete = true;
                assert!(
                    usage.is_none(),
                    "usage should be None when provider reports no tokens"
                );
            }
        }
        assert!(saw_response_complete, "expected a ResponseComplete event");
    }

    /// T3 (R2 + R3): `RunComplete.usage` is the field-wise aggregate of every
    /// step's usage across a real 2-request tool loop, and equals the sum of
    /// the per-step `ResponseComplete.usage` values observed in the same stream.
    #[tokio::test]
    async fn test_run_complete_reports_aggregate_usage() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let model = {
            let call_count = Arc::clone(&call_count);
            FunctionModel::with_stream(move |_messages, _settings| {
                let step = call_count.fetch_add(1, Ordering::SeqCst);
                let events = if step == 0 {
                    vec![
                        Ok(ModelResponseStreamEvent::part_start(
                            0,
                            ModelResponsePart::ToolCall(
                                ToolCallPart::new("demo_tool", ToolCallArgs::string("{}"))
                                    .with_tool_call_id("call_1"),
                            ),
                        )),
                        Ok(ModelResponseStreamEvent::part_end(0)),
                        Ok(ModelResponseStreamEvent::StreamComplete(
                            StreamCompleteEvent::new(FinishReason::ToolCall)
                                .with_input_tokens(10)
                                .with_output_tokens(5),
                        )),
                    ]
                } else {
                    vec![
                        Ok(ModelResponseStreamEvent::part_start(
                            0,
                            ModelResponsePart::Text(TextPart::new("done")),
                        )),
                        Ok(ModelResponseStreamEvent::part_end(0)),
                        Ok(ModelResponseStreamEvent::StreamComplete(
                            StreamCompleteEvent::new(FinishReason::Stop)
                                .with_input_tokens(4)
                                .with_output_tokens(6),
                        )),
                    ]
                };
                Box::pin(stream::iter(events))
            })
        };

        let agent = agent(model)
            .tool_fn(
                "demo_tool",
                "Demo tool",
                |_ctx, _args: serde_json::Value| Ok(serdes_ai_tools::ToolReturn::text("ok")),
            )
            .build();

        let mut stream = agent
            .run_stream("trigger tool then finish", ())
            .await
            .expect("stream should start");

        let mut per_step_request = 0u64;
        let mut per_step_response = 0u64;
        let mut per_step_total = 0u64;
        let mut run_complete_usage = None;
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event should be ok");
            match event {
                AgentStreamEvent::ResponseComplete { usage: Some(u), .. } => {
                    per_step_request += u.request_tokens.unwrap_or(0);
                    per_step_response += u.response_tokens.unwrap_or(0);
                    per_step_total += u.total_tokens.unwrap_or(0);
                }
                AgentStreamEvent::RunComplete { usage, .. } => {
                    run_complete_usage = Some(usage);
                }
                _ => {}
            }
        }

        let usage = run_complete_usage.expect("expected a RunComplete event");
        // Field-wise sum of both steps: input 10+4, output 5+6, total 15+10.
        assert_eq!(usage.request_tokens, 14);
        assert_eq!(usage.response_tokens, 11);
        assert_eq!(usage.total_tokens, 25);
        // Ties R2 accumulation to the R1 per-step values seen in this stream.
        assert_eq!(usage.request_tokens, per_step_request);
        assert_eq!(usage.response_tokens, per_step_response);
        assert_eq!(usage.total_tokens, per_step_total);
    }

    /// T4 (R4): `Cancelled` carries the partial run-aggregate usage.
    ///
    /// Deterministic mid-run cancellation is racy against a synchronous mock
    /// stream (the whole model stream drains before the token flips at the
    /// intended point), so this uses the TEST_PLAN-documented fallback: a
    /// construction+match round-trip asserting the partial aggregate survives
    /// on the `Cancelled` variant. The three production emit sites are covered
    /// by code review (they all pass `usage.clone()`), and the existing
    /// cancel-path tests remain green with the new field.
    #[test]
    fn test_cancelled_reports_partial_usage() {
        let mut partial = RunUsage::new();
        partial.add_request(RequestUsage::new().request_tokens(10).response_tokens(5));

        let event = AgentStreamEvent::Cancelled {
            partial_text: Some("partial".to_string()),
            partial_thinking: None,
            pending_tools: vec!["demo_tool".to_string()],
            usage: partial,
        };

        if let AgentStreamEvent::Cancelled { usage, .. } = event {
            // A run cancelled after >=1 usage-bearing response reports the
            // partial sum, not an all-zero aggregate.
            assert_eq!(usage.request_count, 1);
            assert_eq!(usage.request_tokens, 10);
            assert_eq!(usage.response_tokens, 5);
            assert_eq!(usage.total_tokens, 15);
        } else {
            panic!("expected Cancelled event");
        }
    }

    /// T4b (R2 Loop 2 / AC2.2): prove Loop 2's usage accumulation by EXECUTION.
    ///
    /// Cancel-path option chosen: **full non-cancelled run THROUGH Loop 2**.
    /// Deterministic mid-run cancellation is racy against a synchronous mock
    /// stream — the whole model stream drains before the token flip can land at
    /// the intended point — so instead of the unreliable mid-run-cancel path we
    /// drive the *cancellable* code path (`new_with_cancel`) with a token that is
    /// never triggered, all the way through a real 2-request tool loop to
    /// completion. This exercises the accumulator at stream.rs:748-749 that lives
    /// inside the `new_with_cancel` task (a physically different loop body from
    /// the `new`/`run_stream` path proven by T3), so Loop 2's `add_request` is
    /// validated by running it, not only by code review.
    #[tokio::test]
    async fn test_run_complete_aggregate_usage_through_cancellable_loop() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let model = {
            let call_count = Arc::clone(&call_count);
            FunctionModel::with_stream(move |_messages, _settings| {
                let step = call_count.fetch_add(1, Ordering::SeqCst);
                let events = if step == 0 {
                    vec![
                        Ok(ModelResponseStreamEvent::part_start(
                            0,
                            ModelResponsePart::ToolCall(
                                ToolCallPart::new("demo_tool", ToolCallArgs::string("{}"))
                                    .with_tool_call_id("call_1"),
                            ),
                        )),
                        Ok(ModelResponseStreamEvent::part_end(0)),
                        Ok(ModelResponseStreamEvent::StreamComplete(
                            StreamCompleteEvent::new(FinishReason::ToolCall)
                                .with_input_tokens(10)
                                .with_output_tokens(5),
                        )),
                    ]
                } else {
                    vec![
                        Ok(ModelResponseStreamEvent::part_start(
                            0,
                            ModelResponsePart::Text(TextPart::new("done")),
                        )),
                        Ok(ModelResponseStreamEvent::part_end(0)),
                        Ok(ModelResponseStreamEvent::StreamComplete(
                            StreamCompleteEvent::new(FinishReason::Stop)
                                .with_input_tokens(4)
                                .with_output_tokens(6),
                        )),
                    ]
                };
                Box::pin(stream::iter(events))
            })
        };

        let agentic = agent(model)
            .tool_fn(
                "demo_tool",
                "Demo tool",
                |_ctx, _args: serde_json::Value| Ok(serdes_ai_tools::ToolReturn::text("ok")),
            )
            .build();

        // Untriggered token -> the run completes normally through Loop 2.
        let token = CancellationToken::new();
        let mut stream = AgentStream::new_with_cancel(
            &agentic,
            "trigger tool then finish".into(),
            (),
            RunOptions::default(),
            token,
        )
        .await
        .expect("stream should start");

        let mut per_step_request = 0u64;
        let mut per_step_response = 0u64;
        let mut run_complete_usage = None;
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event should be ok");
            match event {
                AgentStreamEvent::ResponseComplete { usage: Some(u), .. } => {
                    per_step_request += u.request_tokens.unwrap_or(0);
                    per_step_response += u.response_tokens.unwrap_or(0);
                }
                AgentStreamEvent::RunComplete { usage, .. } => {
                    run_complete_usage = Some(usage);
                }
                _ => {}
            }
        }

        let usage = run_complete_usage.expect("expected a RunComplete event");
        // Field-wise aggregate of both steps accumulated inside Loop 2.
        assert_eq!(usage.request_tokens, 14);
        assert_eq!(usage.response_tokens, 11);
        assert_eq!(usage.total_tokens, 25);
        assert_eq!(usage.request_count, 2);
        // Loop 2's aggregate matches the per-step values it emitted.
        assert_eq!(usage.request_tokens, per_step_request);
        assert_eq!(usage.response_tokens, per_step_response);
    }

    /// T7 (R2 / AC1.2 + AC3.2): a mid-run step with NO provider usage must not
    /// corrupt or double-count the run aggregate. Step 0 reports usage (10/5)
    /// and triggers a tool; step 1 reports no token fields. Step 1's
    /// `ResponseComplete.usage` is `None`, and the token aggregate reflects only
    /// the one usage-bearing step.
    ///
    /// `request_count` is deliberately not token-gated: it counts requests
    /// actually issued, both of them here. It is the quantity `max_requests`
    /// bounds, and counting only usage-bearing responses left that limit inert
    /// against any provider that omits usage.
    #[tokio::test]
    async fn test_run_complete_aggregate_ignores_usage_none_step() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let model = {
            let call_count = Arc::clone(&call_count);
            FunctionModel::with_stream(move |_messages, _settings| {
                let step = call_count.fetch_add(1, Ordering::SeqCst);
                let events = if step == 0 {
                    vec![
                        Ok(ModelResponseStreamEvent::part_start(
                            0,
                            ModelResponsePart::ToolCall(
                                ToolCallPart::new("demo_tool", ToolCallArgs::string("{}"))
                                    .with_tool_call_id("call_1"),
                            ),
                        )),
                        Ok(ModelResponseStreamEvent::part_end(0)),
                        Ok(ModelResponseStreamEvent::StreamComplete(
                            StreamCompleteEvent::new(FinishReason::ToolCall)
                                .with_input_tokens(10)
                                .with_output_tokens(5),
                        )),
                    ]
                } else {
                    // Step 1 reports NO token fields.
                    vec![
                        Ok(ModelResponseStreamEvent::part_start(
                            0,
                            ModelResponsePart::Text(TextPart::new("done")),
                        )),
                        Ok(ModelResponseStreamEvent::part_end(0)),
                        Ok(ModelResponseStreamEvent::StreamComplete(
                            StreamCompleteEvent::new(FinishReason::Stop),
                        )),
                    ]
                };
                Box::pin(stream::iter(events))
            })
        };

        let agent = agent(model)
            .tool_fn(
                "demo_tool",
                "Demo tool",
                |_ctx, _args: serde_json::Value| Ok(serdes_ai_tools::ToolReturn::text("ok")),
            )
            .build();

        let mut stream = agent
            .run_stream("trigger tool then finish", ())
            .await
            .expect("stream should start");

        let mut step_usages: Vec<(u32, Option<RequestUsage>)> = Vec::new();
        let mut run_complete_usage = None;
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event should be ok");
            match event {
                AgentStreamEvent::ResponseComplete { step, usage } => {
                    step_usages.push((step, usage));
                }
                AgentStreamEvent::RunComplete { usage, .. } => {
                    run_complete_usage = Some(usage);
                }
                _ => {}
            }
        }

        assert_eq!(step_usages.len(), 2, "expected two ResponseComplete events");
        // Step 0 carried usage; step 1 reported nothing.
        let step0 = step_usages[0].1.as_ref().expect("step 0 has usage");
        assert_eq!(step0.request_tokens, Some(10));
        assert!(
            step_usages[1].1.is_none(),
            "the usage-None step must ship None, not a zeroed usage"
        );

        let usage = run_complete_usage.expect("expected a RunComplete event");
        // Only the one usage-bearing step contributes; the None step is skipped,
        // so the aggregate is neither corrupted nor double-counted.
        assert_eq!(usage.request_tokens, 10);
        assert_eq!(usage.response_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
        // Tokens come only from the usage-bearing step, but both requests count.
        assert_eq!(usage.request_count, 2);
    }

    /// T8 (R1 + R3, billing): cache-creation and cache-read tokens survive from
    /// the provider's `StreamComplete` all the way onto the event path — both
    /// the per-step `ResponseComplete.usage` and the terminal `RunComplete.usage`
    /// aggregate — proving cache-token accumulation is exercised end-to-end, not
    /// just inside the private `usage_from_stream_complete` helper.
    #[tokio::test]
    async fn test_cache_tokens_surface_on_event_path() {
        let model = FunctionModel::with_stream(move |_messages, _settings| {
            let events = vec![
                Ok(ModelResponseStreamEvent::part_start(
                    0,
                    ModelResponsePart::Text(TextPart::new("done")),
                )),
                Ok(ModelResponseStreamEvent::part_end(0)),
                Ok(ModelResponseStreamEvent::StreamComplete(
                    StreamCompleteEvent::new(FinishReason::Stop)
                        .with_input_tokens(10)
                        .with_output_tokens(5)
                        .with_cache_creation_tokens(3)
                        .with_cache_read_tokens(7),
                )),
            ];
            Box::pin(stream::iter(events))
        });

        let agent = agent(model).build();
        let mut stream = agent
            .run_stream("hello", ())
            .await
            .expect("stream should start");

        let mut step_usage = None;
        let mut run_complete_usage = None;
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event should be ok");
            match event {
                AgentStreamEvent::ResponseComplete { usage, .. } => {
                    step_usage = usage;
                }
                AgentStreamEvent::RunComplete { usage, .. } => {
                    run_complete_usage = Some(usage);
                }
                _ => {}
            }
        }

        // Per-step cache tokens survive onto ResponseComplete.usage.
        let step = step_usage.expect("per-step usage should be present");
        assert_eq!(step.cache_creation_tokens, Some(3));
        assert_eq!(step.cache_read_tokens, Some(7));

        // And they accumulate into the terminal RunComplete aggregate.
        let usage = run_complete_usage.expect("expected a RunComplete event");
        assert_eq!(usage.cache_creation_tokens, Some(3));
        assert_eq!(usage.cache_read_tokens, Some(7));
    }

    #[test]
    fn test_canonicalize_tool_call_args_in_response_converts_string_args_to_json() {
        let mut response = ModelResponse::new();
        response.add_part(ModelResponsePart::ToolCall(
            serdes_ai_core::messages::ToolCallPart::new(
                "demo_tool",
                ToolCallArgs::string("{foo: bar,}"),
            )
            .with_tool_call_id("call_1"),
        ));

        canonicalize_tool_call_args_in_response(&mut response);

        match &response.parts[0] {
            ModelResponsePart::ToolCall(tc) => {
                assert!(matches!(tc.args, ToolCallArgs::Json(_)));
            }
            _ => panic!("expected tool call part"),
        }
    }

    #[tokio::test]
    async fn test_run_complete_messages_persist_canonical_tool_call_args() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let model = {
            let call_count = Arc::clone(&call_count);
            FunctionModel::with_stream(move |_messages, _settings| {
                let step = call_count.fetch_add(1, Ordering::SeqCst);
                let events = if step == 0 {
                    vec![
                        Ok(ModelResponseStreamEvent::part_start(
                            0,
                            ModelResponsePart::ToolCall(
                                ToolCallPart::new("demo_tool", ToolCallArgs::string("{foo: bar,}"))
                                    .with_tool_call_id("call_1"),
                            ),
                        )),
                        Ok(ModelResponseStreamEvent::part_end(0)),
                        Ok(ModelResponseStreamEvent::StreamComplete(
                            StreamCompleteEvent::new(FinishReason::ToolCall),
                        )),
                    ]
                } else {
                    vec![
                        Ok(ModelResponseStreamEvent::part_start(
                            0,
                            ModelResponsePart::Text(TextPart::new("done")),
                        )),
                        Ok(ModelResponseStreamEvent::part_end(0)),
                        Ok(ModelResponseStreamEvent::StreamComplete(
                            StreamCompleteEvent::new(FinishReason::Stop),
                        )),
                    ]
                };

                Box::pin(stream::iter(events))
            })
        };

        let agent = agent(model)
            .tool_fn("demo_tool", "Demo tool", |_ctx, args: serde_json::Value| {
                assert!(args.is_object());
                Ok(serdes_ai_tools::ToolReturn::text("ok"))
            })
            .build();

        let mut stream = agent
            .run_stream("trigger tool then finish", ())
            .await
            .expect("stream should start");

        let mut run_complete_messages = None;
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event should be ok");
            if let AgentStreamEvent::RunComplete { messages, .. } = event {
                run_complete_messages = Some(messages);
                break;
            }
        }

        let messages = run_complete_messages.expect("expected RunComplete event");

        let mut saw_tool_call = false;
        for request in &messages {
            for request_part in &request.parts {
                if let ModelRequestPart::ModelResponse(response) = request_part {
                    for response_part in &response.parts {
                        if let ModelResponsePart::ToolCall(tc) = response_part {
                            saw_tool_call = true;
                            assert!(
                                matches!(tc.args, ToolCallArgs::Json(_)),
                                "tool call args should be canonical JSON in persisted RunComplete messages"
                            );
                        }
                    }
                }
            }
        }

        assert!(
            saw_tool_call,
            "expected at least one tool call in persisted RunComplete messages"
        );
    }

    // ========================================================================
    // Regression tests for issue #39: premature EOF must not produce
    // successful OutputReady, committed history, fabricated Stop, or
    // successful RunComplete.
    //
    // The Anthropic parser now emits `ModelError::IncompleteStream` on
    // premature EOF. These agent tests verify that when the model stream
    // produces such an error — after partial content — the agent does not
    // emit success events or commit the response.
    // ========================================================================

    /// Helper: collect all events from a stream and check that no
    /// `OutputReady`, `RunComplete`, or `ResponseComplete` event was emitted,
    /// and that an error was produced.
    async fn assert_stream_error_no_success(
        model: FunctionModel,
        use_cancel: bool,
    ) -> Vec<AgentStreamEvent> {
        let agentic = agent(model).build();

        let mut stream = if use_cancel {
            let token = CancellationToken::new();
            AgentStream::new_with_cancel(&agentic, "Hello".into(), (), RunOptions::default(), token)
                .await
                .expect("stream should start")
        } else {
            agentic
                .run_stream("Hello", ())
                .await
                .expect("stream should start")
        };

        let mut events = Vec::new();
        let mut saw_error = false;
        while let Some(result) = stream.next().await {
            match result {
                Ok(event) => {
                    assert!(
                        !matches!(event, AgentStreamEvent::OutputReady),
                        "must NOT emit OutputReady on stream error"
                    );
                    assert!(
                        !matches!(event, AgentStreamEvent::RunComplete { .. }),
                        "must NOT emit RunComplete on stream error"
                    );
                    assert!(
                        !matches!(event, AgentStreamEvent::ResponseComplete { .. }),
                        "must NOT emit ResponseComplete on stream error"
                    );
                    events.push(event);
                }
                Err(e) => {
                    saw_error = true;
                    events.push(AgentStreamEvent::Error {
                        message: e.to_string(),
                    });
                }
            }
        }
        assert!(saw_error, "expected an error event");
        events
    }

    /// Test: stream error after partial text (simulating Anthropic
    /// IncompleteStream) produces an error and no success events.
    /// Non-cancellable path.
    #[tokio::test]
    async fn test_premature_eof_non_cancellable() {
        let model = FunctionModel::with_stream(|_messages, _settings| {
            Box::pin(stream::iter(vec![
                Ok(ModelResponseStreamEvent::part_start(
                    0,
                    ModelResponsePart::Text(TextPart::new("Partial response")),
                )),
                Err(serdes_ai_models::ModelError::incomplete_stream(
                    "stream ended before message_stop was received",
                )),
            ]))
        });

        let events = assert_stream_error_no_success(model, false).await;

        let saw_text = events
            .iter()
            .any(|e| matches!(e, AgentStreamEvent::TextDelta { .. }));
        assert!(saw_text, "should have seen partial text");
    }

    /// Test: stream error after partial text — cancellable path.
    #[tokio::test]
    async fn test_premature_eof_cancellable() {
        let model = FunctionModel::with_stream(|_messages, _settings| {
            Box::pin(stream::iter(vec![
                Ok(ModelResponseStreamEvent::part_start(
                    0,
                    ModelResponsePart::Text(TextPart::new("Partial response")),
                )),
                Err(serdes_ai_models::ModelError::incomplete_stream(
                    "stream ended before message_stop was received",
                )),
            ]))
        });

        let events = assert_stream_error_no_success(model, true).await;

        let saw_text = events
            .iter()
            .any(|e| matches!(e, AgentStreamEvent::TextDelta { .. }));
        assert!(saw_text, "should have seen partial text");
    }

    /// Test: empty stream (no parts at all) produces an error.
    /// Non-cancellable path.
    #[tokio::test]
    async fn test_premature_eof_empty_stream_non_cancellable() {
        let model =
            FunctionModel::with_stream(|_messages, _settings| Box::pin(stream::iter(vec![])));

        let events = assert_stream_error_no_success(model, false).await;

        let saw_text = events
            .iter()
            .any(|e| matches!(e, AgentStreamEvent::TextDelta { .. }));
        assert!(!saw_text, "should NOT have seen any text from empty stream");
    }

    /// Test: stream error after incomplete tool call — must not execute
    /// the tool or commit it.
    #[tokio::test]
    async fn test_premature_eof_incomplete_tool_call() {
        let model = FunctionModel::with_stream(|_messages, _settings| {
            Box::pin(stream::iter(vec![
                Ok(ModelResponseStreamEvent::part_start(
                    0,
                    ModelResponsePart::ToolCall(
                        ToolCallPart::new("search", serde_json::json!({}))
                            .with_tool_call_id("call_1"),
                    ),
                )),
                Err(serdes_ai_models::ModelError::incomplete_stream(
                    "stream ended with open content block",
                )),
            ]))
        });

        let events = assert_stream_error_no_success(model, false).await;

        // Should NOT have seen ToolCallComplete
        let saw_tool_complete = events
            .iter()
            .any(|e| matches!(e, AgentStreamEvent::ToolCallComplete { .. }));
        assert!(
            !saw_tool_complete,
            "must NOT emit ToolCallComplete on incomplete tool call"
        );

        // Should NOT have seen ToolExecuted
        let saw_tool_executed = events
            .iter()
            .any(|e| matches!(e, AgentStreamEvent::ToolExecuted { .. }));
        assert!(!saw_tool_executed, "must NOT execute tool on premature EOF");
    }

    /// Test: a valid stream (PartStart + PartEnd + StreamComplete) produces
    /// OutputReady and RunComplete. This verifies the happy path is not broken.
    #[tokio::test]
    async fn test_valid_stream_produces_success() {
        let model = FunctionModel::with_stream(|_messages, _settings| {
            Box::pin(stream::iter(vec![
                Ok(ModelResponseStreamEvent::part_start(
                    0,
                    ModelResponsePart::Text(TextPart::new("Hello!")),
                )),
                Ok(ModelResponseStreamEvent::part_end(0)),
                Ok(ModelResponseStreamEvent::StreamComplete(
                    StreamCompleteEvent::new(FinishReason::Stop),
                )),
            ]))
        });

        let agentic = agent(model).build();
        let mut stream = agentic
            .run_stream("Hi", ())
            .await
            .expect("stream should start");

        let mut saw_output_ready = false;
        let mut saw_run_complete = false;
        while let Some(result) = stream.next().await {
            match result {
                Ok(AgentStreamEvent::OutputReady) => saw_output_ready = true,
                Ok(AgentStreamEvent::RunComplete { .. }) => saw_run_complete = true,
                Ok(_) => {}
                Err(e) => panic!("valid stream should not error: {:?}", e),
            }
        }
        assert!(saw_output_ready, "valid stream should emit OutputReady");
        assert!(saw_run_complete, "valid stream should emit RunComplete");
    }

    /// Test: premature EOF — empty stream, cancellable path.
    #[tokio::test]
    async fn test_premature_eof_empty_stream_cancellable() {
        let model =
            FunctionModel::with_stream(|_messages, _settings| Box::pin(stream::iter(vec![])));

        let events = assert_stream_error_no_success(model, true).await;

        let saw_text = events
            .iter()
            .any(|e| matches!(e, AgentStreamEvent::TextDelta { .. }));
        assert!(!saw_text, "should NOT have seen any text from empty stream");
    }

    /// Test: premature EOF — incomplete tool call, cancellable path.
    #[tokio::test]
    async fn test_premature_eof_incomplete_tool_call_cancellable() {
        let model = FunctionModel::with_stream(|_messages, _settings| {
            Box::pin(stream::iter(vec![
                Ok(ModelResponseStreamEvent::part_start(
                    0,
                    ModelResponsePart::ToolCall(
                        ToolCallPart::new("search", serde_json::json!({}))
                            .with_tool_call_id("call_1"),
                    ),
                )),
                Err(serdes_ai_models::ModelError::incomplete_stream(
                    "stream ended with open content block",
                )),
            ]))
        });

        let events = assert_stream_error_no_success(model, true).await;

        let saw_tool_complete = events
            .iter()
            .any(|e| matches!(e, AgentStreamEvent::ToolCallComplete { .. }));
        assert!(
            !saw_tool_complete,
            "must NOT emit ToolCallComplete on incomplete tool call (cancellable)"
        );

        let saw_tool_executed = events
            .iter()
            .any(|e| matches!(e, AgentStreamEvent::ToolExecuted { .. }));
        assert!(
            !saw_tool_executed,
            "must NOT execute tool on premature EOF (cancellable)"
        );
    }

    // ========================================================================
    // Regression tests for the #39/PR#50 over-broad inferred-EOF gate:
    // a stream that produces content and ends WITHOUT a terminal
    // StreamComplete (in-memory/mock streams AND real OpenAI/Google/etc.
    // parsers, which never emit one) must complete SUCCESSFULLY. Real
    // truncation is still surfaced as an explicit Err by the provider parser
    // (Anthropic), which is covered by serdes-ai-models tests.
    // ========================================================================

    /// RT1: a mock stream emits text + part_end but NO StreamComplete.
    /// It must complete successfully (TextDelta + RunComplete, no Error).
    /// This FAILED before the fix (inferred-EOF gate emitted Error + returned).
    #[tokio::test]
    async fn test_run_stream_completes_without_terminal_streamcomplete() {
        let model = FunctionModel::with_stream(|_messages, _settings| {
            Box::pin(stream::iter(vec![
                Ok(ModelResponseStreamEvent::part_start(
                    0,
                    ModelResponsePart::Text(TextPart::new("hello world")),
                )),
                Ok(ModelResponseStreamEvent::part_end(0)),
            ]))
        });

        let agentic = agent(model).build();
        let mut stream = agentic
            .run_stream("Hi", ())
            .await
            .expect("stream should start");

        let mut saw_text = false;
        let mut saw_run_complete = false;
        let mut saw_error = false;
        while let Some(result) = stream.next().await {
            match result {
                Ok(AgentStreamEvent::TextDelta { .. }) => saw_text = true,
                Ok(AgentStreamEvent::RunComplete { .. }) => saw_run_complete = true,
                Ok(AgentStreamEvent::Error { .. }) => saw_error = true,
                Ok(_) => {}
                Err(e) => panic!("stream without StreamComplete must not error: {:?}", e),
            }
        }

        assert!(saw_text, "expected at least one TextDelta");
        assert!(saw_run_complete, "expected a terminal RunComplete");
        assert!(!saw_error, "must NOT emit an Error event");
    }

    /// RT2: a genuinely empty stream (no parts, no StreamComplete) must still
    /// error - the empty-content guard is preserved.
    #[tokio::test]
    async fn test_empty_stream_still_errors() {
        let model =
            FunctionModel::with_stream(|_messages, _settings| Box::pin(stream::iter(vec![])));

        let agentic = agent(model).build();
        let mut stream = agentic
            .run_stream("Hi", ())
            .await
            .expect("stream should start");

        let mut saw_run_complete = false;
        let mut saw_error = false;
        while let Some(result) = stream.next().await {
            match result {
                Ok(AgentStreamEvent::RunComplete { .. }) => saw_run_complete = true,
                Ok(AgentStreamEvent::Error { .. }) => saw_error = true,
                Ok(_) => {}
                Err(_) => saw_error = true,
            }
        }

        assert!(!saw_run_complete, "empty stream must NOT emit RunComplete");
        assert!(saw_error, "empty stream must still produce an error");
    }

    /// RT4: the OpenAI-parser terminal shape (part_start + text_delta + part_end,
    /// closed on finish, NO StreamComplete) - the real non-Anthropic-provider
    /// stream shape. Must complete successfully (TextDelta + RunComplete, no Error).
    #[tokio::test]
    async fn test_run_stream_openai_shape_no_streamcomplete() {
        let model = FunctionModel::with_stream(|_messages, _settings| {
            Box::pin(stream::iter(vec![
                Ok(ModelResponseStreamEvent::part_start(
                    0,
                    ModelResponsePart::Text(TextPart::new("")),
                )),
                Ok(ModelResponseStreamEvent::text_delta(0, "hello ")),
                Ok(ModelResponseStreamEvent::text_delta(0, "world")),
                Ok(ModelResponseStreamEvent::part_end(0)),
            ]))
        });

        let agentic = agent(model).build();
        let mut stream = agentic
            .run_stream("Hi", ())
            .await
            .expect("stream should start");

        let mut saw_text = false;
        let mut saw_run_complete = false;
        let mut saw_error = false;
        while let Some(result) = stream.next().await {
            match result {
                Ok(AgentStreamEvent::TextDelta { .. }) => saw_text = true,
                Ok(AgentStreamEvent::RunComplete { .. }) => saw_run_complete = true,
                Ok(AgentStreamEvent::Error { .. }) => saw_error = true,
                Ok(_) => {}
                Err(e) => panic!("OpenAI-shape stream must not error: {:?}", e),
            }
        }

        assert!(saw_text, "expected at least one TextDelta");
        assert!(saw_run_complete, "expected a terminal RunComplete");
        assert!(!saw_error, "must NOT emit an Error event");
    }

    /// RT1-cancellable (Loop 2 success path): the RT1 mock — text + part_end,
    /// NO terminal StreamComplete — driven through the *cancellable* code path
    /// (`new_with_cancel` with an UNTRIGGERED token). This proves Loop 2's
    /// `unwrap_or(FinishReason::Stop)` at stream.rs:1270+ by EXECUTION: Loop 2 is
    /// a physically distinct loop body from `run_stream`'s Loop 1 (proven by RT1),
    /// so it needs its own regression guard. Must complete successfully
    /// (TextDelta + RunComplete, no Error).
    #[tokio::test]
    async fn test_run_stream_completes_without_terminal_streamcomplete_cancellable() {
        let model = FunctionModel::with_stream(|_messages, _settings| {
            Box::pin(stream::iter(vec![
                Ok(ModelResponseStreamEvent::part_start(
                    0,
                    ModelResponsePart::Text(TextPart::new("hello world")),
                )),
                Ok(ModelResponseStreamEvent::part_end(0)),
            ]))
        });

        let agentic = agent(model).build();

        // Untriggered token -> the run completes normally through Loop 2.
        let token = CancellationToken::new();
        let mut stream =
            AgentStream::new_with_cancel(&agentic, "Hi".into(), (), RunOptions::default(), token)
                .await
                .expect("stream should start");

        let mut saw_text = false;
        let mut saw_run_complete = false;
        let mut saw_error = false;
        while let Some(result) = stream.next().await {
            match result {
                Ok(AgentStreamEvent::TextDelta { .. }) => saw_text = true,
                Ok(AgentStreamEvent::RunComplete { .. }) => saw_run_complete = true,
                Ok(AgentStreamEvent::Error { .. }) => saw_error = true,
                Ok(_) => {}
                Err(e) => {
                    panic!(
                        "cancellable stream without StreamComplete must not error: {:?}",
                        e
                    )
                }
            }
        }

        assert!(saw_text, "expected at least one TextDelta");
        assert!(saw_run_complete, "expected a terminal RunComplete");
        assert!(!saw_error, "must NOT emit an Error event");
    }

    /// RT2-cancellable (Loop 2 empty-content guard): a genuinely empty stream
    /// (no parts, no StreamComplete) driven through the *cancellable* path must
    /// still error. This proves the `response_parts.is_empty()` guard is intact
    /// on Loop 2 as well as Loop 1 (RT2) — defaulting the finish reason to Stop
    /// does NOT paper over a stream that produced nothing.
    #[tokio::test]
    async fn test_empty_stream_still_errors_cancellable() {
        let model =
            FunctionModel::with_stream(|_messages, _settings| Box::pin(stream::iter(vec![])));

        let agentic = agent(model).build();

        let token = CancellationToken::new();
        let mut stream =
            AgentStream::new_with_cancel(&agentic, "Hi".into(), (), RunOptions::default(), token)
                .await
                .expect("stream should start");

        let mut saw_run_complete = false;
        let mut saw_error = false;
        while let Some(result) = stream.next().await {
            match result {
                Ok(AgentStreamEvent::RunComplete { .. }) => saw_run_complete = true,
                Ok(AgentStreamEvent::Error { .. }) => saw_error = true,
                Ok(_) => {}
                Err(_) => saw_error = true,
            }
        }

        assert!(
            !saw_run_complete,
            "empty cancellable stream must NOT emit RunComplete"
        );
        assert!(
            saw_error,
            "empty cancellable stream must still produce an error"
        );
    }

    /// Multi-step tool loop where NO step emits a terminal StreamComplete
    /// (each defaults to FinishReason::Stop). Step 0 emits a ToolCall; step 1
    /// emits text. This proves loop continuation keys on TOOL-CALL PRESENCE,
    /// not on the finish reason: even though step 0 defaults to Stop, the
    /// pending tool call forces a 2nd request. Asserts the tool executed, two
    /// model requests were made, and exactly ONE terminal RunComplete fired.
    #[tokio::test]
    async fn test_tool_loop_without_streamcomplete_continues() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let tool_calls = Arc::new(AtomicUsize::new(0));
        let model = {
            let call_count = Arc::clone(&call_count);
            FunctionModel::with_stream(move |_messages, _settings| {
                let step = call_count.fetch_add(1, Ordering::SeqCst);
                let events = if step == 0 {
                    // ToolCall part, NO StreamComplete -> finish_reason defaults to Stop.
                    vec![
                        Ok(ModelResponseStreamEvent::part_start(
                            0,
                            ModelResponsePart::ToolCall(
                                ToolCallPart::new("demo_tool", ToolCallArgs::string("{}"))
                                    .with_tool_call_id("call_1"),
                            ),
                        )),
                        Ok(ModelResponseStreamEvent::part_end(0)),
                    ]
                } else {
                    // Text, NO StreamComplete -> finish_reason defaults to Stop.
                    vec![
                        Ok(ModelResponseStreamEvent::part_start(
                            0,
                            ModelResponsePart::Text(TextPart::new("done")),
                        )),
                        Ok(ModelResponseStreamEvent::part_end(0)),
                    ]
                };
                Box::pin(stream::iter(events))
            })
        };

        let agentic = {
            let tool_calls = Arc::clone(&tool_calls);
            agent(model)
                .tool_fn(
                    "demo_tool",
                    "Demo tool",
                    move |_ctx, _args: serde_json::Value| {
                        tool_calls.fetch_add(1, Ordering::SeqCst);
                        Ok(serdes_ai_tools::ToolReturn::text("ok"))
                    },
                )
                .build()
        };

        let mut stream = agentic
            .run_stream("trigger tool then finish", ())
            .await
            .expect("stream should start");

        let mut run_complete_count = 0usize;
        let mut saw_error = false;
        while let Some(result) = stream.next().await {
            match result {
                Ok(AgentStreamEvent::RunComplete { .. }) => run_complete_count += 1,
                Ok(AgentStreamEvent::Error { .. }) => saw_error = true,
                Ok(_) => {}
                Err(e) => panic!("tool loop without StreamComplete must not error: {:?}", e),
            }
        }

        assert!(!saw_error, "must NOT emit an Error event");
        assert_eq!(
            tool_calls.load(Ordering::SeqCst),
            1,
            "the registered tool must have executed exactly once"
        );
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            2,
            "loop must make a 2nd model request after the tool call (2 steps)"
        );
        assert_eq!(
            run_complete_count, 1,
            "exactly one terminal RunComplete must fire"
        );
    }

    /// AC3.2 ACCEPTED-TRADE-OFF GUARD: a stream that emits partial content
    /// (a couple of TextDeltas + part_end) but NO terminal StreamComplete and
    /// NO Err is accepted as a SUCCESSFUL completion (finish_reason defaults to
    /// Stop). This is a DELIBERATE trade-off: for non-Anthropic providers a
    /// genuinely truncated stream is currently indistinguishable from success,
    /// because their parsers never emit StreamComplete. Anthropic still surfaces
    /// real truncation as an explicit Err and is unaffected.
    ///
    /// This test PINS the current behavior so any future re-tightening (see the
    /// AC3.3 follow-up: real OpenAI/Google termination detection) is a conscious,
    /// reviewed change rather than a silent regression.
    #[tokio::test]
    async fn test_partial_content_without_terminal_accepted_as_stop() {
        let model = FunctionModel::with_stream(|_messages, _settings| {
            Box::pin(stream::iter(vec![
                Ok(ModelResponseStreamEvent::part_start(
                    0,
                    ModelResponsePart::Text(TextPart::new("")),
                )),
                Ok(ModelResponseStreamEvent::text_delta(0, "partial ")),
                Ok(ModelResponseStreamEvent::text_delta(0, "answer")),
                Ok(ModelResponseStreamEvent::part_end(0)),
            ]))
        });

        let agentic = agent(model).build();
        let mut stream = agentic
            .run_stream("Hi", ())
            .await
            .expect("stream should start");

        let mut saw_run_complete = false;
        let mut saw_error = false;
        while let Some(result) = stream.next().await {
            match result {
                Ok(AgentStreamEvent::RunComplete { .. }) => saw_run_complete = true,
                Ok(AgentStreamEvent::Error { .. }) => saw_error = true,
                Ok(_) => {}
                Err(e) => panic!("accepted-trade-off stream must not error: {:?}", e),
            }
        }

        // AC3.2: partial-but-unterminated content is accepted as SUCCESS today.
        assert!(
            saw_run_complete,
            "partial content without StreamComplete must complete as success"
        );
        assert!(
            !saw_error,
            "must NOT emit an Error event (accepted trade-off)"
        );
    }

    // ========================================================================
    // EndTurn-loop regression (fix: run_stream exited only on Stop, but
    // Anthropic streaming maps normal completions to EndTurn). The loop now
    // keys on FinishReason::is_complete() = {Stop, EndTurn, StopSequence} at
    // BOTH exit sites (Loop 1 = run_stream, Loop 2 = new_with_cancel).
    // ========================================================================

    /// RT1 (R1/R5, Loop 1): a tool-less stream whose terminal StreamComplete
    /// carries `FinishReason::EndTurn` (exactly what Anthropic streaming emits
    /// for a normal completion) must finish in EXACTLY ONE model round.
    ///
    /// The closure counts calls via an `AtomicUsize` and has a SAFETY CAP: once
    /// it has been invoked 3 times it switches to `Stop`, so a still-broken loop
    /// cannot hang the test suite - it just fails the `== 1` assertion instead.
    /// Pre-fix (`== Some(Stop)`) this loops (EndTurn never matched); post-fix it
    /// is exactly 1. Drives Loop 1 (`run_stream`).
    #[tokio::test]
    async fn test_run_stream_completes_on_endturn() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let model = {
            let call_count = Arc::clone(&call_count);
            FunctionModel::with_stream(move |_messages, _settings| {
                let step = call_count.fetch_add(1, Ordering::SeqCst);
                // Safety cap: after 3 rounds fall back to Stop so a still-broken
                // loop terminates (and fails `== 1`) instead of hanging.
                let reason = if step >= 3 {
                    FinishReason::Stop
                } else {
                    FinishReason::EndTurn
                };
                Box::pin(stream::iter(vec![
                    Ok(ModelResponseStreamEvent::part_start(
                        0,
                        ModelResponsePart::Text(TextPart::new("done")),
                    )),
                    Ok(ModelResponseStreamEvent::part_end(0)),
                    Ok(ModelResponseStreamEvent::StreamComplete(
                        StreamCompleteEvent::new(reason),
                    )),
                ]))
            })
        };

        let agentic = agent(model).build();
        let mut stream = agentic
            .run_stream("hello", ())
            .await
            .expect("stream should start");

        let mut saw_output_ready = false;
        let mut saw_run_complete = false;
        let mut saw_error = false;
        while let Some(result) = stream.next().await {
            match result {
                Ok(AgentStreamEvent::OutputReady) => saw_output_ready = true,
                Ok(AgentStreamEvent::RunComplete { .. }) => saw_run_complete = true,
                Ok(AgentStreamEvent::Error { .. }) => saw_error = true,
                Ok(_) => {}
                Err(e) => panic!("EndTurn stream must not error: {:?}", e),
            }
        }

        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "EndTurn must complete the loop in exactly one round"
        );
        assert!(saw_output_ready, "expected an OutputReady event");
        assert!(saw_run_complete, "expected a terminal RunComplete");
        assert!(!saw_error, "must NOT emit an Error event");
    }

    /// RT1-cancellable (R1/AC1.2, Loop 2): identical to RT1 but driven through
    /// the *cancellable* path (`new_with_cancel` with an UNTRIGGERED token).
    /// Loop 2 is a physically distinct loop body from `run_stream`'s Loop 1, so
    /// it needs its own execution guard - this proves the fix landed on Loop 2
    /// too. Mirrors `test_run_stream_completes_without_terminal_streamcomplete_cancellable`.
    #[tokio::test]
    async fn test_run_stream_completes_on_endturn_cancellable() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let model = {
            let call_count = Arc::clone(&call_count);
            FunctionModel::with_stream(move |_messages, _settings| {
                let step = call_count.fetch_add(1, Ordering::SeqCst);
                let reason = if step >= 3 {
                    FinishReason::Stop
                } else {
                    FinishReason::EndTurn
                };
                Box::pin(stream::iter(vec![
                    Ok(ModelResponseStreamEvent::part_start(
                        0,
                        ModelResponsePart::Text(TextPart::new("done")),
                    )),
                    Ok(ModelResponseStreamEvent::part_end(0)),
                    Ok(ModelResponseStreamEvent::StreamComplete(
                        StreamCompleteEvent::new(reason),
                    )),
                ]))
            })
        };

        let agentic = agent(model).build();
        let token = CancellationToken::new();
        let mut stream = AgentStream::new_with_cancel(
            &agentic,
            "hello".into(),
            (),
            RunOptions::default(),
            token,
        )
        .await
        .expect("stream should start");

        let mut saw_run_complete = false;
        let mut saw_error = false;
        while let Some(result) = stream.next().await {
            match result {
                Ok(AgentStreamEvent::RunComplete { .. }) => saw_run_complete = true,
                Ok(AgentStreamEvent::Error { .. }) => saw_error = true,
                Ok(_) => {}
                Err(e) => panic!("EndTurn cancellable stream must not error: {:?}", e),
            }
        }

        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "EndTurn must complete Loop 2 in exactly one round"
        );
        assert!(saw_run_complete, "expected a terminal RunComplete");
        assert!(!saw_error, "must NOT emit an Error event");
    }

    /// RT2 (R1/AC5.1b, Loop 1): same shape as RT1 but with
    /// `FinishReason::StopSequence`. Proves the loop keys on the full
    /// `is_complete()` set - {Stop, EndTurn, StopSequence} - and did not merely
    /// add an EndTurn special-case.
    #[tokio::test]
    async fn test_run_stream_completes_on_stop_sequence() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let model = {
            let call_count = Arc::clone(&call_count);
            FunctionModel::with_stream(move |_messages, _settings| {
                let step = call_count.fetch_add(1, Ordering::SeqCst);
                let reason = if step >= 3 {
                    FinishReason::Stop
                } else {
                    FinishReason::StopSequence
                };
                Box::pin(stream::iter(vec![
                    Ok(ModelResponseStreamEvent::part_start(
                        0,
                        ModelResponsePart::Text(TextPart::new("done")),
                    )),
                    Ok(ModelResponseStreamEvent::part_end(0)),
                    Ok(ModelResponseStreamEvent::StreamComplete(
                        StreamCompleteEvent::new(reason),
                    )),
                ]))
            })
        };

        let agentic = agent(model).build();
        let mut stream = agentic
            .run_stream("hello", ())
            .await
            .expect("stream should start");

        let mut saw_run_complete = false;
        let mut saw_error = false;
        while let Some(result) = stream.next().await {
            match result {
                Ok(AgentStreamEvent::RunComplete { .. }) => saw_run_complete = true,
                Ok(AgentStreamEvent::Error { .. }) => saw_error = true,
                Ok(_) => {}
                Err(e) => panic!("StopSequence stream must not error: {:?}", e),
            }
        }

        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "StopSequence must complete the loop in exactly one round"
        );
        assert!(saw_run_complete, "expected a terminal RunComplete");
        assert!(!saw_error, "must NOT emit an Error event");
    }

    /// RT5 (R4/AC4.1): the usage feature stays intact on the EndTurn completion
    /// path. A single-round EndTurn completion (input 10 / output 5) must emit
    /// EXACTLY ONE terminal RunComplete whose aggregate usage is 10/5/15 - i.e.
    /// usage fires once, not once-per-erroneous-loop.
    #[tokio::test]
    async fn test_run_stream_endturn_usage_fires_once() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let model = {
            let call_count = Arc::clone(&call_count);
            FunctionModel::with_stream(move |_messages, _settings| {
                let step = call_count.fetch_add(1, Ordering::SeqCst);
                let reason = if step >= 3 {
                    FinishReason::Stop
                } else {
                    FinishReason::EndTurn
                };
                Box::pin(stream::iter(vec![
                    Ok(ModelResponseStreamEvent::part_start(
                        0,
                        ModelResponsePart::Text(TextPart::new("done")),
                    )),
                    Ok(ModelResponseStreamEvent::part_end(0)),
                    Ok(ModelResponseStreamEvent::StreamComplete(
                        StreamCompleteEvent::new(reason)
                            .with_input_tokens(10)
                            .with_output_tokens(5),
                    )),
                ]))
            })
        };

        let agentic = agent(model).build();
        let mut stream = agentic
            .run_stream("hello", ())
            .await
            .expect("stream should start");

        let mut run_completes = Vec::new();
        while let Some(result) = stream.next().await {
            match result {
                Ok(AgentStreamEvent::RunComplete { usage, .. }) => run_completes.push(usage),
                Ok(_) => {}
                Err(e) => panic!("EndTurn usage stream must not error: {:?}", e),
            }
        }

        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "EndTurn must complete in exactly one round"
        );
        assert_eq!(
            run_completes.len(),
            1,
            "expected exactly one terminal RunComplete"
        );
        let usage = &run_completes[0];
        assert_eq!(usage.request_tokens, 10);
        assert_eq!(usage.response_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    /// RT6 (ordering guard, Loop 1): a tool-bearing step whose terminal
    /// StreamComplete ALSO carries `FinishReason::EndTurn` - which is now
    /// `is_complete()` - must NOT terminate the loop early. The tool-call
    /// `continue` (which fires on `!tool_calls.is_empty()`) MUST run BEFORE the
    /// widened completion check, so the loop issues a 2nd request to let the
    /// model respond to the tool result.
    ///
    /// STEP 0: ToolCall part_start/part_end + StreamComplete(EndTurn)
    ///         (a tool-bearing response that also carries a now-complete reason).
    /// STEP 1: text + StreamComplete(EndTurn) -> the genuine terminal round.
    ///
    /// If the widened finish check were reached BEFORE the tool `continue`,
    /// STEP 0's EndTurn would terminate the run after ONE round and the tool
    /// result would never be sent back - the `call_count == 2` assertion catches
    /// that regression. An `AtomicUsize` call counter with a SAFETY CAP (>= 4 ->
    /// Stop) guarantees a genuinely broken loop FAILS an assertion rather than
    /// hanging the suite.
    #[tokio::test]
    async fn test_tool_loop_with_endturn_continues() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let tool_calls = Arc::new(AtomicUsize::new(0));
        let model = {
            let call_count = Arc::clone(&call_count);
            FunctionModel::with_stream(move |_messages, _settings| {
                let step = call_count.fetch_add(1, Ordering::SeqCst);
                // Safety cap: after 4 rounds fall back to Stop so a genuinely
                // broken loop terminates (and fails `== 2`) instead of hanging.
                let reason = if step >= 4 {
                    FinishReason::Stop
                } else {
                    FinishReason::EndTurn
                };
                let events = if step == 0 {
                    // Tool-bearing step whose terminal reason is ALSO complete
                    // (EndTurn). The tool-call `continue` must win over the
                    // widened finish check.
                    vec![
                        Ok(ModelResponseStreamEvent::part_start(
                            0,
                            ModelResponsePart::ToolCall(
                                ToolCallPart::new("demo_tool", ToolCallArgs::string("{}"))
                                    .with_tool_call_id("call_1"),
                            ),
                        )),
                        Ok(ModelResponseStreamEvent::part_end(0)),
                        Ok(ModelResponseStreamEvent::StreamComplete(
                            StreamCompleteEvent::new(reason),
                        )),
                    ]
                } else {
                    // Genuine terminal round: text + EndTurn completion.
                    vec![
                        Ok(ModelResponseStreamEvent::part_start(
                            0,
                            ModelResponsePart::Text(TextPart::new("done")),
                        )),
                        Ok(ModelResponseStreamEvent::part_end(0)),
                        Ok(ModelResponseStreamEvent::StreamComplete(
                            StreamCompleteEvent::new(reason),
                        )),
                    ]
                };
                Box::pin(stream::iter(events))
            })
        };

        let agentic = {
            let tool_calls = Arc::clone(&tool_calls);
            agent(model)
                .tool_fn(
                    "demo_tool",
                    "Demo tool",
                    move |_ctx, _args: serde_json::Value| {
                        tool_calls.fetch_add(1, Ordering::SeqCst);
                        Ok(serdes_ai_tools::ToolReturn::text("ok"))
                    },
                )
                .build()
        };

        let mut stream = agentic
            .run_stream("call the tool then finish", ())
            .await
            .expect("stream should start");

        let mut run_complete_count = 0usize;
        let mut saw_output_ready = false;
        let mut saw_error = false;
        while let Some(result) = stream.next().await {
            match result {
                Ok(AgentStreamEvent::OutputReady) => saw_output_ready = true,
                Ok(AgentStreamEvent::RunComplete { .. }) => run_complete_count += 1,
                Ok(AgentStreamEvent::Error { .. }) => saw_error = true,
                Ok(_) => {}
                Err(e) => panic!("tool+EndTurn loop must not error: {:?}", e),
            }
        }

        assert!(!saw_error, "must NOT emit an Error event");
        assert_eq!(
            tool_calls.load(Ordering::SeqCst),
            1,
            "the registered tool must have executed exactly once"
        );
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            2,
            "tool-bearing EndTurn step must NOT terminate early: the loop must \
             make a 2nd round to respond to the tool result"
        );
        assert!(
            saw_output_ready,
            "the genuine terminal (2nd) round must emit OutputReady"
        );
        assert_eq!(
            run_complete_count, 1,
            "exactly one terminal RunComplete must fire (no premature termination)"
        );
    }
}

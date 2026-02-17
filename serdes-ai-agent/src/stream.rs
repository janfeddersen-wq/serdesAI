//! Streaming agent execution.
//!
//! This module provides streaming support for agent runs with real
//! character-by-character streaming from the model.

use crate::agent::{Agent, RegisteredTool};
use crate::context::{generate_run_id, RunContext, RunUsage};
use crate::errors::AgentRunError;
use crate::run::{CompressionStrategy, RunOptions};
use chrono::Utc;
use futures::{Stream, StreamExt};
use serdes_ai_core::messages::{
    ModelResponseStreamEvent, ToolCallArgs, ToolReturnPart, UserContent,
};
use serdes_ai_core::{
    FinishReason, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart,
};
use serdes_ai_models::ModelRequestParameters;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

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
    ResponseComplete { step: u32 },
    /// Output ready.
    OutputReady,
    /// Run completed.
    RunComplete {
        run_id: String,
        /// Complete message history from this run (system prompt, user prompts,
        /// assistant responses, tool calls and returns).
        messages: Vec<ModelRequest>,
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

impl AgentStream {
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
        let run_id = generate_run_id();
        let (tx, rx) = mpsc::channel(64);

        // Clone what we need for the spawned task
        let model = agent.model_arc();
        let model_name = model.name().to_string();
        let model_settings = options
            .model_settings
            .clone()
            .unwrap_or_else(|| agent.model_settings.clone());

        // Get the static system prompt - for streaming we use just the static part
        // Dynamic prompts are not supported in streaming mode for simplicity
        let static_system_prompt = agent.static_system_prompt().to_string();

        let tool_definitions = agent.tool_definitions();
        let _end_strategy = agent.end_strategy;
        let usage_limits = agent.usage_limits.clone();
        let run_usage_limits = options.usage_limits.clone();

        // Clone tool executors - now possible because RegisteredTool implements Clone!
        let tools: Vec<RegisteredTool<Deps>> = agent.tools.to_vec();

        // Wrap deps in Arc for shared access in tool execution
        let deps = Arc::new(deps);

        let initial_history = options.message_history.clone();
        let _metadata = options.metadata.clone();
        let compression_config = options.compression.clone();
        let run_id_clone = run_id.clone();

        debug!(run_id = %run_id, "AgentStream: spawning streaming task");

        // Spawn the streaming task
        tokio::spawn(async move {
            info!(run_id = %run_id_clone, "AgentStream: task started");

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
                        let _ = tx.send(Err(e.into())).await;
                        return;
                    }
                }

                if let Some(ref limits) = run_usage_limits {
                    if let Err(e) = limits.check(&usage) {
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
                let params = ModelRequestParameters::new()
                    .with_tools_arc(tool_definitions.clone())
                    .with_allow_text(true);

                // === Context Size Calculation & Compression ===

                // Calculate context size by serializing (this is the actual request size)
                let (request_bytes, estimated_tokens) = {
                    let messages_json = serde_json::to_string(&messages).unwrap_or_default();
                    let tools_json = serde_json::to_string(&*tool_definitions).unwrap_or_default();
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
                                CompressionStrategy::Summarize => {
                                    // Use the same model to summarize the conversation history
                                    // Keep first 2 messages (system + first user) and last few messages
                                    // Summarize everything in between

                                    if messages.len() <= 4 {
                                        // Too few messages to summarize, just truncate
                                        use crate::history::{HistoryProcessor, TruncateByTokens};
                                        let truncator =
                                            TruncateByTokens::new(compression.target_tokens as u64)
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
                                            let middle_json = serde_json::to_string_pretty(&middle)
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
                                            match model
                                                .request(
                                                    &[summary_req],
                                                    &model_settings,
                                                    &summary_params,
                                                )
                                                .await
                                            {
                                                Ok(response) => {
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
                                                        // Fallback to truncate if summary failed
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
                                                    warn!(
                                                        "Summarization failed, falling back to truncate: {}",
                                                        _e
                                                    );
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

                // Make streaming request
                info!(
                    step = step,
                    message_count = messages.len(),
                    "AgentStream: calling model.request_stream"
                );
                let stream_result = model
                    .request_stream(&messages, &model_settings, &params)
                    .await;

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
                        let _ = tx.send(Err(AgentRunError::Model(e))).await;
                        return;
                    }
                };

                // Collect response parts while streaming
                let mut response_parts: Vec<ModelResponsePart> = Vec::new();
                // Track stream events (used by tracing when enabled)
                let mut stream_event_count = 0u32;

                // Process stream events
                debug!("AgentStream: starting to process model stream events");
                while let Some(event_result) = model_stream.next().await {
                    {
                        stream_event_count += 1;
                        let _ = stream_event_count;
                    }
                    match event_result {
                        Ok(event) => {
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
                                                        .send(Ok(AgentStreamEvent::ToolCallDelta {
                                                            delta: args_str,
                                                            tool_call_id: tc.tool_call_id.clone(),
                                                        }))
                                                        .await;
                                                }
                                            }
                                        }
                                        ModelResponsePart::Thinking(t) => {
                                            if !t.content.is_empty() {
                                                let _ = tx
                                                    .send(Ok(AgentStreamEvent::ThinkingDelta {
                                                        text: t.content.clone(),
                                                    }))
                                                    .await;
                                            }
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
                                            if let Some(ModelResponsePart::Text(ref mut text)) =
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
                                                ref mut tool_call,
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
                                            if let Some(ModelResponsePart::Thinking(
                                                ref mut think,
                                            )) = response_parts.get_mut(delta.index)
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
                            }
                        }
                        Err(e) => {
                            let _ = tx
                                .send(Ok(AgentStreamEvent::Error {
                                    message: e.to_string(),
                                }))
                                .await;
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

                // Build the complete response
                let mut response = ModelResponse {
                    parts: response_parts.clone(),
                    model_name: Some(model.name().to_string()),
                    timestamp: Utc::now(),
                    finish_reason: Some(FinishReason::Stop),
                    usage: None,
                    vendor_id: None,
                    vendor_details: None,
                    kind: "response".to_string(),
                };
                canonicalize_tool_call_args_in_response(&mut response);

                finish_reason = response.finish_reason;
                responses.push(response.clone());

                // Emit ResponseComplete
                let _ = tx
                    .send(Ok(AgentStreamEvent::ResponseComplete { step }))
                    .await;

                // Check for tool calls that need execution
                let tool_calls: Vec<_> = response
                    .parts
                    .iter()
                    .filter_map(|p| {
                        if let ModelResponsePart::ToolCall(tc) = p {
                            Some(tc.clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                if !tool_calls.is_empty() {
                    // Add response to messages for proper alternation
                    let mut response_req = ModelRequest::new();
                    response_req
                        .parts
                        .push(ModelRequestPart::ModelResponse(Box::new(response.clone())));
                    messages.push(response_req);

                    let mut tool_req = ModelRequest::new();

                    for tc in tool_calls {
                        let _ = tx
                            .send(Ok(AgentStreamEvent::ToolCallComplete {
                                tool_name: tc.tool_name.clone(),
                                tool_call_id: tc.tool_call_id.clone(),
                            }))
                            .await;

                        usage.record_tool_call();

                        // Find the tool by name
                        let tool = tools.iter().find(|t| t.definition.name == tc.tool_name);

                        match tool {
                            Some(tool) => {
                                // Create a RunContext for tool execution
                                let tool_ctx =
                                    RunContext::with_shared_deps(deps.clone(), model_name.clone())
                                        .for_tool(&tc.tool_name, tc.tool_call_id.clone());

                                // Execute the tool
                                let result =
                                    tool.executor.execute(tc.args.to_json(), &tool_ctx).await;

                                match result {
                                    Ok(ret) => {
                                        let _ = tx
                                            .send(Ok(AgentStreamEvent::ToolExecuted {
                                                tool_name: tc.tool_name.clone(),
                                                tool_call_id: tc.tool_call_id.clone(),
                                                success: true,
                                                error: None,
                                            }))
                                            .await;

                                        // Use ToolReturnPart for successful execution
                                        let mut part =
                                            ToolReturnPart::new(&tc.tool_name, ret.content);
                                        if let Some(id) = tc.tool_call_id.clone() {
                                            part = part.with_tool_call_id(id);
                                        }
                                        tool_req.parts.push(ModelRequestPart::ToolReturn(part));
                                    }
                                    Err(e) => {
                                        let error_msg = e.to_string();
                                        let _ = tx
                                            .send(Ok(AgentStreamEvent::ToolExecuted {
                                                tool_name: tc.tool_name.clone(),
                                                tool_call_id: tc.tool_call_id.clone(),
                                                success: false,
                                                error: Some(error_msg.clone()),
                                            }))
                                            .await;

                                        // Use ToolReturnPart with error content for tool errors
                                        let mut part = ToolReturnPart::error(
                                            &tc.tool_name,
                                            format!("Tool error: {}", e),
                                        );
                                        if let Some(id) = tc.tool_call_id.clone() {
                                            part = part.with_tool_call_id(id);
                                        }
                                        tool_req.parts.push(ModelRequestPart::ToolReturn(part));
                                    }
                                }
                            }
                            None => {
                                let error_msg = format!("Unknown tool: {}", tc.tool_name);
                                let _ = tx
                                    .send(Ok(AgentStreamEvent::ToolExecuted {
                                        tool_name: tc.tool_name.clone(),
                                        tool_call_id: tc.tool_call_id.clone(),
                                        success: false,
                                        error: Some(error_msg.clone()),
                                    }))
                                    .await;

                                // Unknown tool - use ToolReturnPart with error
                                let mut part = ToolReturnPart::error(
                                    &tc.tool_name,
                                    format!("Unknown tool: {}", tc.tool_name),
                                );
                                if let Some(id) = tc.tool_call_id.clone() {
                                    part = part.with_tool_call_id(id);
                                }
                                tool_req.parts.push(ModelRequestPart::ToolReturn(part));
                            }
                        }
                    }

                    if !tool_req.parts.is_empty() {
                        messages.push(tool_req);
                    }

                    // Continue to let model respond to tool "error"
                    continue;
                }

                // No tool calls - check finish condition
                if finish_reason == Some(FinishReason::Stop) {
                    // Add final response to messages for complete history
                    let mut response_req = ModelRequest::new();
                    response_req
                        .parts
                        .push(ModelRequestPart::ModelResponse(Box::new(response.clone())));
                    messages.push(response_req);

                    finished = true;
                    let _ = tx.send(Ok(AgentStreamEvent::OutputReady)).await;
                }
            }

            // Emit RunComplete
            let _ = tx
                .send(Ok(AgentStreamEvent::RunComplete {
                    run_id: run_id_clone,
                    messages,
                }))
                .await;
        });

        Ok(AgentStream {
            rx,
            cancel_token: None,
        })
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
        let run_id = generate_run_id();
        let (tx, rx) = mpsc::channel(64);

        // Clone what we need for the spawned task
        let model = agent.model_arc();
        let model_name = model.name().to_string();
        let model_settings = options
            .model_settings
            .clone()
            .unwrap_or_else(|| agent.model_settings.clone());

        let static_system_prompt = agent.static_system_prompt().to_string();
        let tool_definitions = agent.tool_definitions();
        let _end_strategy = agent.end_strategy;
        let usage_limits = agent.usage_limits.clone();
        let run_usage_limits = options.usage_limits.clone();
        let tools: Vec<RegisteredTool<Deps>> = agent.tools.to_vec();
        let deps = Arc::new(deps);

        let initial_history = options.message_history.clone();
        let _metadata = options.metadata.clone();
        let compression_config = options.compression.clone();
        let run_id_clone = run_id.clone();
        let cancel_token_clone = cancel_token.clone();

        debug!(run_id = %run_id, "AgentStream: spawning streaming task with cancellation support");

        tokio::spawn(async move {
            info!(run_id = %run_id_clone, "AgentStream: task started with cancellation support");

            // Track partial content for cancellation reporting
            let mut accumulated_text = String::new();
            let mut accumulated_thinking = String::new();
            let mut pending_tool_names: Vec<String> = Vec::new();

            // Emit RunStart
            if tx
                .send(Ok(AgentStreamEvent::RunStart {
                    run_id: run_id_clone.clone(),
                }))
                .await
                .is_err()
            {
                return;
            }

            // Build initial messages
            let mut messages = initial_history.unwrap_or_default();

            if !static_system_prompt.is_empty() {
                let mut req = ModelRequest::new();
                req.add_system_prompt(static_system_prompt.clone());
                messages.push(req);
            }

            let mut user_req = ModelRequest::new();
            user_req.add_user_prompt(prompt);
            messages.push(user_req);

            let mut responses: Vec<ModelResponse> = Vec::new();
            let mut usage = RunUsage::new();
            let mut step = 0u32;
            let mut finished = false;
            let mut finish_reason: Option<FinishReason>;

            // Main agent loop with cancellation support
            while !finished {
                // Check for cancellation at the start of each iteration
                if cancel_token_clone.is_cancelled() {
                    info!(run_id = %run_id_clone, "AgentStream: cancelled at loop start");
                    let _ = tx
                        .send(Ok(AgentStreamEvent::Cancelled {
                            partial_text: if accumulated_text.is_empty() {
                                None
                            } else {
                                Some(accumulated_text)
                            },
                            partial_thinking: if accumulated_thinking.is_empty() {
                                None
                            } else {
                                Some(accumulated_thinking)
                            },
                            pending_tools: pending_tool_names,
                        }))
                        .await;
                    let _ = tx.send(Err(AgentRunError::Cancelled)).await;
                    return;
                }

                step += 1;

                // Check usage limits
                if let Some(ref limits) = usage_limits {
                    if let Err(e) = limits.check(&usage) {
                        let _ = tx.send(Err(e.into())).await;
                        return;
                    }
                }

                if let Some(ref limits) = run_usage_limits {
                    if let Err(e) = limits.check(&usage) {
                        let _ = tx.send(Err(e.into())).await;
                        return;
                    }
                }

                if tx
                    .send(Ok(AgentStreamEvent::RequestStart { step }))
                    .await
                    .is_err()
                {
                    return;
                }

                let params = ModelRequestParameters::new()
                    .with_tools_arc(tool_definitions.clone())
                    .with_allow_text(true);

                // Context size calculation (simplified - full version in main new())
                let (request_bytes, estimated_tokens) = {
                    let messages_json = serde_json::to_string(&messages).unwrap_or_default();
                    let tools_json = serde_json::to_string(&*tool_definitions).unwrap_or_default();
                    let bytes = messages_json.len() + tools_json.len();
                    (bytes, bytes / 4)
                };

                let context_limit = model.profile().context_window;

                let _ = tx
                    .send(Ok(AgentStreamEvent::ContextInfo {
                        estimated_tokens,
                        request_bytes,
                        context_limit,
                    }))
                    .await;

                // Context compression (simplified version)
                if let Some(ref compression) = compression_config {
                    if let Some(limit) = context_limit {
                        let threshold_tokens = (limit as f64 * compression.threshold) as usize;
                        if estimated_tokens > threshold_tokens {
                            use crate::history::{HistoryProcessor, TruncateByTokens};
                            let truncator = TruncateByTokens::new(compression.target_tokens as u64)
                                .keep_first_n(2);
                            let temp_ctx = RunContext::new((), &model_name);
                            messages = truncator.process(&temp_ctx, messages).await;
                        }
                    }
                }

                // Make streaming request with cancellation support
                let stream_result = model
                    .request_stream(&messages, &model_settings, &params)
                    .await;

                let mut model_stream = match stream_result {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = tx
                            .send(Ok(AgentStreamEvent::Error {
                                message: e.to_string(),
                            }))
                            .await;
                        let _ = tx.send(Err(AgentRunError::Model(e))).await;
                        return;
                    }
                };

                let mut response_parts: Vec<ModelResponsePart> = Vec::new();

                // Process stream events with cancellation check
                loop {
                    tokio::select! {
                        biased;

                        _ = cancel_token_clone.cancelled() => {
                            info!(run_id = %run_id_clone, "AgentStream: cancelled during model stream");
                            let _ = tx
                                .send(Ok(AgentStreamEvent::Cancelled {
                                    partial_text: if accumulated_text.is_empty() {
                                        None
                                    } else {
                                        Some(accumulated_text)
                                    },
                                    partial_thinking: if accumulated_thinking.is_empty() {
                                        None
                                    } else {
                                        Some(accumulated_thinking)
                                    },
                                    pending_tools: pending_tool_names,
                                }))
                                .await;
                            let _ = tx.send(Err(AgentRunError::Cancelled)).await;
                            return;
                        }

                        event_result = model_stream.next() => {
                            match event_result {
                                Some(Ok(event)) => {
                                    match event {
                                        ModelResponseStreamEvent::PartStart(start) => {
                                            match &start.part {
                                                ModelResponsePart::Text(t) => {
                                                    if !t.content.is_empty() {
                                                        accumulated_text.push_str(&t.content);
                                                        let _ = tx
                                                            .send(Ok(AgentStreamEvent::TextDelta {
                                                                text: t.content.clone(),
                                                            }))
                                                            .await;
                                                    }
                                                }
                                                ModelResponsePart::ToolCall(tc) => {
                                                    pending_tool_names.push(tc.tool_name.clone());
                                                    let _ = tx
                                                        .send(Ok(AgentStreamEvent::ToolCallStart {
                                                            tool_name: tc.tool_name.clone(),
                                                            tool_call_id: tc.tool_call_id.clone(),
                                                        }))
                                                        .await;
                                                    if let Ok(args_str) = tc.args.to_json_string() {
                                                        if !args_str.is_empty() && args_str != "{}" {
                                                            let _ = tx
                                                                .send(Ok(AgentStreamEvent::ToolCallDelta {
                                                                    delta: args_str,
                                                                    tool_call_id: tc.tool_call_id.clone(),
                                                                }))
                                                                .await;
                                                        }
                                                    }
                                                }
                                                ModelResponsePart::Thinking(t) => {
                                                    if !t.content.is_empty() {
                                                        accumulated_thinking.push_str(&t.content);
                                                        let _ = tx
                                                            .send(Ok(AgentStreamEvent::ThinkingDelta {
                                                                text: t.content.clone(),
                                                            }))
                                                            .await;
                                                    }
                                                }
                                                _ => {}
                                            }
                                            response_parts.push(start.part.clone());
                                        }
                                        ModelResponseStreamEvent::PartDelta(delta) => {
                                            use serdes_ai_core::messages::ModelResponsePartDelta;
                                            match &delta.delta {
                                                ModelResponsePartDelta::Text(t) => {
                                                    accumulated_text.push_str(&t.content_delta);
                                                    let _ = tx
                                                        .send(Ok(AgentStreamEvent::TextDelta {
                                                            text: t.content_delta.clone(),
                                                        }))
                                                        .await;
                                                    if let Some(ModelResponsePart::Text(ref mut text)) =
                                                        response_parts.get_mut(delta.index)
                                                    {
                                                        text.content.push_str(&t.content_delta);
                                                    }
                                                }
                                                ModelResponsePartDelta::ToolCall(tc) => {
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
                                                    if let Some(ModelResponsePart::ToolCall(
                                                        ref mut tool_call,
                                                    )) = response_parts.get_mut(delta.index)
                                                    {
                                                        tc.apply(tool_call);
                                                    }
                                                }
                                                ModelResponsePartDelta::Thinking(t) => {
                                                    accumulated_thinking.push_str(&t.content_delta);
                                                    let _ = tx
                                                        .send(Ok(AgentStreamEvent::ThinkingDelta {
                                                            text: t.content_delta.clone(),
                                                        }))
                                                        .await;
                                                    if let Some(ModelResponsePart::Thinking(
                                                        ref mut think,
                                                    )) = response_parts.get_mut(delta.index)
                                                    {
                                                        t.apply(think);
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                        ModelResponseStreamEvent::PartEnd(_) => {}
                                    }
                                }
                                Some(Err(e)) => {
                                    let _ = tx
                                        .send(Ok(AgentStreamEvent::Error {
                                            message: e.to_string(),
                                        }))
                                        .await;
                                    let _ = tx.send(Err(AgentRunError::Model(e))).await;
                                    return;
                                }
                                None => {
                                    // Stream ended normally
                                    break;
                                }
                            }
                        }
                    }
                }

                // Build the complete response
                let mut response = ModelResponse {
                    parts: response_parts.clone(),
                    model_name: Some(model.name().to_string()),
                    timestamp: Utc::now(),
                    finish_reason: Some(FinishReason::Stop),
                    usage: None,
                    vendor_id: None,
                    vendor_details: None,
                    kind: "response".to_string(),
                };
                canonicalize_tool_call_args_in_response(&mut response);

                finish_reason = response.finish_reason;
                responses.push(response.clone());

                let _ = tx
                    .send(Ok(AgentStreamEvent::ResponseComplete { step }))
                    .await;

                // Check for tool calls
                let tool_calls: Vec<_> = response
                    .parts
                    .iter()
                    .filter_map(|p| {
                        if let ModelResponsePart::ToolCall(tc) = p {
                            Some(tc.clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                if !tool_calls.is_empty() {
                    let mut response_req = ModelRequest::new();
                    response_req
                        .parts
                        .push(ModelRequestPart::ModelResponse(Box::new(response.clone())));
                    messages.push(response_req);

                    let mut tool_req = ModelRequest::new();

                    for tc in tool_calls {
                        // Check for cancellation before each tool execution
                        if cancel_token_clone.is_cancelled() {
                            info!(run_id = %run_id_clone, "AgentStream: cancelled before tool execution");
                            let _ = tx
                                .send(Ok(AgentStreamEvent::Cancelled {
                                    partial_text: if accumulated_text.is_empty() {
                                        None
                                    } else {
                                        Some(accumulated_text)
                                    },
                                    partial_thinking: if accumulated_thinking.is_empty() {
                                        None
                                    } else {
                                        Some(accumulated_thinking)
                                    },
                                    pending_tools: pending_tool_names,
                                }))
                                .await;
                            let _ = tx.send(Err(AgentRunError::Cancelled)).await;
                            return;
                        }

                        let _ = tx
                            .send(Ok(AgentStreamEvent::ToolCallComplete {
                                tool_name: tc.tool_name.clone(),
                                tool_call_id: tc.tool_call_id.clone(),
                            }))
                            .await;

                        usage.record_tool_call();
                        // Remove from pending after completion
                        pending_tool_names.retain(|n| n != &tc.tool_name);

                        let tool = tools.iter().find(|t| t.definition.name == tc.tool_name);

                        match tool {
                            Some(tool) => {
                                let tool_ctx =
                                    RunContext::with_shared_deps(deps.clone(), model_name.clone())
                                        .for_tool(&tc.tool_name, tc.tool_call_id.clone());

                                let result =
                                    tool.executor.execute(tc.args.to_json(), &tool_ctx).await;

                                match result {
                                    Ok(ret) => {
                                        let _ = tx
                                            .send(Ok(AgentStreamEvent::ToolExecuted {
                                                tool_name: tc.tool_name.clone(),
                                                tool_call_id: tc.tool_call_id.clone(),
                                                success: true,
                                                error: None,
                                            }))
                                            .await;

                                        let mut part =
                                            ToolReturnPart::new(&tc.tool_name, ret.content);
                                        if let Some(id) = tc.tool_call_id.clone() {
                                            part = part.with_tool_call_id(id);
                                        }
                                        tool_req.parts.push(ModelRequestPart::ToolReturn(part));
                                    }
                                    Err(e) => {
                                        let error_msg = e.to_string();
                                        let _ = tx
                                            .send(Ok(AgentStreamEvent::ToolExecuted {
                                                tool_name: tc.tool_name.clone(),
                                                tool_call_id: tc.tool_call_id.clone(),
                                                success: false,
                                                error: Some(error_msg.clone()),
                                            }))
                                            .await;

                                        let mut part = ToolReturnPart::error(
                                            &tc.tool_name,
                                            format!("Tool error: {}", e),
                                        );
                                        if let Some(id) = tc.tool_call_id.clone() {
                                            part = part.with_tool_call_id(id);
                                        }
                                        tool_req.parts.push(ModelRequestPart::ToolReturn(part));
                                    }
                                }
                            }
                            None => {
                                let error_msg = format!("Unknown tool: {}", tc.tool_name);
                                let _ = tx
                                    .send(Ok(AgentStreamEvent::ToolExecuted {
                                        tool_name: tc.tool_name.clone(),
                                        tool_call_id: tc.tool_call_id.clone(),
                                        success: false,
                                        error: Some(error_msg.clone()),
                                    }))
                                    .await;

                                let mut part = ToolReturnPart::error(
                                    &tc.tool_name,
                                    format!("Unknown tool: {}", tc.tool_name),
                                );
                                if let Some(id) = tc.tool_call_id.clone() {
                                    part = part.with_tool_call_id(id);
                                }
                                tool_req.parts.push(ModelRequestPart::ToolReturn(part));
                            }
                        }
                    }

                    if !tool_req.parts.is_empty() {
                        messages.push(tool_req);
                    }

                    continue;
                }

                if finish_reason == Some(FinishReason::Stop) {
                    // Add final response to messages for complete history
                    let mut response_req = ModelRequest::new();
                    response_req
                        .parts
                        .push(ModelRequestPart::ModelResponse(Box::new(response.clone())));
                    messages.push(response_req);

                    finished = true;
                    let _ = tx.send(Ok(AgentStreamEvent::OutputReady)).await;
                }
            }

            let _ = tx
                .send(Ok(AgentStreamEvent::RunComplete {
                    run_id: run_id_clone,
                    messages,
                }))
                .await;
        });

        Ok(AgentStream {
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
    /// If this stream was created without cancellation support (via `new`),
    /// this method does nothing.
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
    use futures::{stream, StreamExt};
    use serdes_ai_core::messages::{ModelRequestPart, TextPart, ToolCallPart};
    use serdes_ai_models::FunctionModel;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
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
            },
            AgentStreamEvent::Cancelled {
                partial_text: Some("partial".to_string()),
                partial_thinking: None,
                pending_tools: vec!["tool1".to_string()],
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
        };

        if let AgentStreamEvent::Cancelled {
            partial_text,
            partial_thinking,
            pending_tools,
        } = event
        {
            assert!(partial_text.is_none());
            assert!(partial_thinking.is_none());
            assert!(pending_tools.is_empty());
        } else {
            panic!("Expected Cancelled event");
        }
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
                                ToolCallPart::new(
                                    "demo_tool",
                                    ToolCallArgs::string("{foo: bar,}"),
                                )
                                .with_tool_call_id("call_1"),
                            ),
                        )),
                        Ok(ModelResponseStreamEvent::part_end(0)),
                    ]
                } else {
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

        let agent = agent(model)
            .tool_fn(
                "demo_tool",
                "Demo tool",
                |_ctx, args: serde_json::Value| {
                    assert!(args.is_object());
                    Ok(serdes_ai_tools::ToolReturn::text("ok"))
                },
            )
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
}

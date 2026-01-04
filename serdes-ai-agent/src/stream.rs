//! Streaming agent execution.
//!
//! This module provides streaming support for agent runs with real
//! character-by-character streaming from the model.

use crate::agent::{Agent, RegisteredTool};
use crate::context::{generate_run_id, RunContext, RunUsage};
use crate::errors::AgentRunError;
use crate::run::RunOptions;
use chrono::Utc;
use futures::{Stream, StreamExt};
use serdes_ai_core::messages::{ModelResponseStreamEvent, ToolReturnPart, UserContent};
use serdes_ai_core::{
    FinishReason, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart,
};
use serdes_ai_models::ModelRequestParameters;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::mpsc;

// Conditional tracing - use no-op macros when tracing feature is disabled
#[cfg(feature = "tracing-integration")]
use tracing::{debug, error, info, warn};

#[cfg(not(feature = "tracing-integration"))]
macro_rules! debug { ($($arg:tt)*) => {} }
#[cfg(not(feature = "tracing-integration"))]
macro_rules! info { ($($arg:tt)*) => {} }
#[cfg(not(feature = "tracing-integration"))]
macro_rules! error { ($($arg:tt)*) => {} }
#[cfg(not(feature = "tracing-integration"))]
macro_rules! warn { ($($arg:tt)*) => {} }

/// Events emitted during streaming.
#[derive(Debug, Clone)]
pub enum AgentStreamEvent {
    /// Run started.
    RunStart { run_id: String },
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
    RunComplete { run_id: String },
    /// Error occurred.
    Error { message: String },
}



/// Streaming agent execution.
///
/// This provides real streaming by spawning a task that streams from the model
/// and sends events through a channel.
pub struct AgentStream {
    rx: mpsc::Receiver<Result<AgentStreamEvent, AgentRunError>>,
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
        let (tx, rx): (
            mpsc::Sender<Result<AgentStreamEvent, AgentRunError>>,
            mpsc::Receiver<Result<AgentStreamEvent, AgentRunError>>,
        ) = mpsc::channel(64);

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
        let tools: Vec<RegisteredTool<Deps>> = agent.tools.iter().cloned().collect();

        // Wrap deps in Arc for shared access in tool execution
        let deps = Arc::new(deps);

        let initial_history = options.message_history.clone();
        let _metadata = options.metadata.clone();
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
            debug!(initial_messages = messages.len(), "AgentStream: building messages");

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

                // Make streaming request
                info!(step = step, message_count = messages.len(), "AgentStream: calling model.request_stream");
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
                    { stream_event_count += 1; let _ = stream_event_count; }
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
                                            let tool_call_id = response_parts.get(delta.index)
                                                .and_then(|p| {
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
                                            if let Some(ModelResponsePart::ToolCall(ref mut tool_call)) =
                                                response_parts.get_mut(delta.index)
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
                                            if let Some(ModelResponsePart::Thinking(ref mut think)) =
                                                response_parts.get_mut(delta.index)
                                            {
                                                think.content.push_str(&t.content_delta);
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
                
                info!(stream_events = stream_event_count, parts = response_parts.len(), "AgentStream: finished processing model stream");

                // Build the complete response
                let response = ModelResponse {
                    parts: response_parts.clone(),
                    model_name: Some(model.name().to_string()),
                    timestamp: Utc::now(),
                    finish_reason: Some(FinishReason::Stop),
                    usage: None,
                    vendor_id: None,
                    vendor_details: None,
                    kind: "response".to_string(),
                };

                finish_reason = response.finish_reason.clone();
                responses.push(response.clone());

                // Emit ResponseComplete
                let _ = tx
                    .send(Ok(AgentStreamEvent::ResponseComplete { step }))
                    .await;

                // Check for tool calls that need execution
                let tool_calls: Vec<_> = response.parts.iter().filter_map(|p| {
                    if let ModelResponsePart::ToolCall(tc) = p {
                        Some(tc.clone())
                    } else {
                        None
                    }
                }).collect();

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
                                let tool_ctx = RunContext::with_shared_deps(
                                    deps.clone(),
                                    model_name.clone(),
                                ).for_tool(&tc.tool_name, tc.tool_call_id.clone());

                                // Execute the tool
                                let result = tool.executor.execute(tc.args.to_json(), &tool_ctx).await;

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
                                        let mut part = ToolReturnPart::new(&tc.tool_name, ret.content);
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
                    finished = true;
                    let _ = tx.send(Ok(AgentStreamEvent::OutputReady)).await;
                }
            }

            // Emit RunComplete
            let _ = tx
                .send(Ok(AgentStreamEvent::RunComplete {
                    run_id: run_id_clone,
                }))
                .await;
        });

        Ok(AgentStream { rx })
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
        let events = vec![
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
            },
        ];

        assert_eq!(events.len(), 6);
    }
}

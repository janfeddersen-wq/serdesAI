//! Agent run execution.
//!
//! This module contains the core execution logic for agent runs.

use crate::agent::{Agent, EndStrategy};
use crate::context::{generate_run_id, RunContext, RunUsage, UsageLimits};
use crate::errors::{AgentRunError, OutputParseError, OutputValidationError};
use chrono::Utc;
use serde_json::Value as JsonValue;
use serdes_ai_core::messages::{RetryPromptPart, ToolReturnPart, UserContent};
use serdes_ai_core::{
    FinishReason, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart, ModelSettings,
};
use serdes_ai_models::ModelRequestParameters;
use serdes_ai_tools::{ToolError, ToolReturn};
use std::sync::Arc;

/// Context compression strategy.
#[derive(Debug, Clone, Default)]
pub enum CompressionStrategy {
    /// Keep only the last ~30k tokens worth of messages.
    #[default]
    Truncate,
    /// Use LLM to summarize older messages into condensed form.
    Summarize,
}

/// Context compression configuration.
#[derive(Debug, Clone)]
pub struct ContextCompression {
    /// Compression strategy to use.
    pub strategy: CompressionStrategy,
    /// Trigger threshold (0.0-1.0). Default: 0.75
    pub threshold: f64,
    /// Target token count for truncation/summarization. Default: 30_000
    pub target_tokens: usize,
}

impl Default for ContextCompression {
    fn default() -> Self {
        Self {
            strategy: CompressionStrategy::Truncate,
            threshold: 0.75,
            target_tokens: 30_000,
        }
    }
}

/// Options for a run.
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    /// Override model settings.
    pub model_settings: Option<ModelSettings>,
    /// Message history to continue from.
    pub message_history: Option<Vec<ModelRequest>>,
    /// Usage limits for this run.
    pub usage_limits: Option<crate::context::UsageLimits>,
    /// Custom metadata.
    pub metadata: Option<JsonValue>,
    /// Context compression configuration.
    pub compression: Option<ContextCompression>,
}

impl RunOptions {
    /// Create new default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set model settings override.
    pub fn model_settings(mut self, settings: ModelSettings) -> Self {
        self.model_settings = Some(settings);
        self
    }

    /// Set message history.
    pub fn message_history(mut self, history: Vec<ModelRequest>) -> Self {
        self.message_history = Some(history);
        self
    }

    /// Set metadata.
    pub fn metadata(mut self, metadata: JsonValue) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set context compression configuration.
    pub fn with_compression(mut self, config: ContextCompression) -> Self {
        self.compression = Some(config);
        self
    }
}

/// Result of an agent run.
#[derive(Debug, Clone)]
pub struct AgentRunResult<Output> {
    /// The output data.
    pub output: Output,
    /// Message history.
    pub messages: Vec<ModelRequest>,
    /// All model responses.
    pub responses: Vec<ModelResponse>,
    /// Usage for this run.
    pub usage: RunUsage,
    /// Run ID.
    pub run_id: String,
    /// Finish reason.
    pub finish_reason: FinishReason,
    /// Metadata.
    pub metadata: Option<JsonValue>,
}

impl<Output> AgentRunResult<Output> {
    /// Get the output.
    pub fn output(&self) -> &Output {
        &self.output
    }

    /// Consume and return output.
    pub fn into_output(self) -> Output {
        self.output
    }
}

/// Active agent run that can be iterated.
pub struct AgentRun<'a, Deps, Output> {
    agent: &'a Agent<Deps, Output>,
    #[allow(dead_code)]
    deps: Arc<Deps>,
    state: AgentRunState<Output>,
    ctx: RunContext<Deps>,
    run_usage_limits: Option<UsageLimits>,
}

struct AgentRunState<Output> {
    messages: Vec<ModelRequest>,
    responses: Vec<ModelResponse>,
    usage: RunUsage,
    run_id: String,
    step: u32,
    output_retries: u32,
    final_output: Option<Output>,
    finished: bool,
    finish_reason: Option<FinishReason>,
}

/// Result of a single step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepResult {
    /// Continue to next step.
    Continue,
    /// Tools were executed.
    ToolsExecuted(usize),
    /// Output is ready.
    OutputReady,
    /// Retrying output validation.
    RetryingOutput,
    /// Run is finished.
    Finished,
}

impl<'a, Deps, Output> AgentRun<'a, Deps, Output>
where
    Deps: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    /// Create a new agent run.
    pub async fn new(
        agent: &'a Agent<Deps, Output>,
        prompt: UserContent,
        deps: Deps,
        options: RunOptions,
    ) -> Result<Self, AgentRunError> {
        let run_id = generate_run_id();
        let deps = Arc::new(deps);

        let model_settings = options
            .model_settings
            .unwrap_or_else(|| agent.model_settings.clone());

        let ctx = RunContext {
            deps: deps.clone(),
            run_id: run_id.clone(),
            start_time: Utc::now(),
            model_name: agent.model().name().to_string(),
            model_settings: model_settings.clone(),
            tool_name: None,
            tool_call_id: None,
            retry_count: 0,
            metadata: options.metadata.clone(),
        };

        // Build initial messages
        let mut messages = options.message_history.unwrap_or_default();

        // Build system prompt
        let system_prompt = agent.build_system_prompt(&ctx).await;
        if !system_prompt.is_empty() {
            let mut req = ModelRequest::new();
            req.add_system_prompt(system_prompt);
            messages.push(req);
        }

        // Add user prompt
        let mut user_req = ModelRequest::new();
        user_req.add_user_prompt(prompt);
        messages.push(user_req);

        Ok(Self {
            agent,
            deps,
            state: AgentRunState {
                messages,
                responses: Vec::new(),
                usage: RunUsage::new(),
                run_id,
                step: 0,
                output_retries: 0,
                final_output: None,
                finished: false,
                finish_reason: None,
            },
            ctx,
            run_usage_limits: options.usage_limits,
        })
    }

    /// Run to completion.
    pub async fn run_to_completion(mut self) -> Result<AgentRunResult<Output>, AgentRunError> {
        while !self.state.finished {
            self.step().await?;
        }
        self.finalize()
    }

    /// Execute one step.
    pub async fn step(&mut self) -> Result<StepResult, AgentRunError> {
        if self.state.finished {
            return Ok(StepResult::Finished);
        }

        self.state.step += 1;

        // Check usage limits
        if let Some(limits) = &self.agent.usage_limits {
            limits.check(&self.state.usage)?;
            limits.check_time(self.ctx.elapsed_seconds() as u64)?;
        }

        if let Some(limits) = &self.run_usage_limits {
            limits.check(&self.state.usage)?;
            limits.check_time(self.ctx.elapsed_seconds() as u64)?;
        }

        // Get cached tool definitions (pre-computed at build time - no cloning!)
        let tool_defs = self.agent.tool_definitions();

        // Build request parameters
        let params = ModelRequestParameters::new()
            .with_tools_arc(tool_defs)
            .with_allow_text(true);

        // Process message history
        let messages = self.process_history().await;

        // Make model request
        let response = self
            .agent
            .model()
            .request(&messages, &self.ctx.model_settings, &params)
            .await?;

        // Update usage
        if let Some(usage) = &response.usage {
            self.state.usage.add_request(usage.clone());
        }

        // Store response
        if response.finish_reason.is_some() {
            self.state.finish_reason = response.finish_reason;
        }
        self.state.responses.push(response.clone());

        // Process response
        self.process_response(response).await
    }

    async fn process_history(&self) -> Vec<ModelRequest> {
        let mut messages = self.state.messages.clone();

        // Apply history processors
        for processor in &self.agent.history_processors {
            messages = processor.process(&self.ctx, messages).await;
        }

        messages
    }

    async fn process_response(
        &mut self,
        response: ModelResponse,
    ) -> Result<StepResult, AgentRunError> {
        let mut tool_calls = Vec::new();
        let mut found_output = None;

        for part in &response.parts {
            match part {
                ModelResponsePart::Text(text) => {
                    if !text.content.is_empty() {
                        // Try to parse as output
                        match self.agent.output_schema.parse_text(&text.content) {
                            Ok(output) => found_output = Some(output),
                            Err(OutputParseError::NotFound) => {}
                            Err(_) => {} // Try other parts
                        }
                    }
                }
                ModelResponsePart::ToolCall(tc) => {
                    // Check if this is the output tool
                    if self.agent.is_output_tool(&tc.tool_name) {
                        let args = tc.args.to_json();
                        if let Ok(output) = self
                            .agent
                            .output_schema
                            .parse_tool_call(&tc.tool_name, &args)
                        {
                            found_output = Some(output);
                            continue;
                        }
                    }

                    // Regular tool call
                    tool_calls.push(tc.clone());
                }
                ModelResponsePart::Thinking(_) => {
                    // Thinking parts are recorded but not processed
                }
                ModelResponsePart::File(_) => {
                    // File parts are recorded but not processed as output
                }
                ModelResponsePart::BuiltinToolCall(_) => {
                    // Builtin tool calls are handled by the provider
                }
            }
        }

        // Execute tool calls FIRST - they take priority over text output.
        // This matches the behavior in stream.rs and prevents the agent from
        // stopping early when the model returns both explanatory text AND tool
        // calls in the same response. This is especially important when
        // Output=String, since any text would be valid "output".
        if !tool_calls.is_empty() {
            let count = tool_calls.len();
            let returns = self.execute_tool_calls(tool_calls).await;
            self.add_tool_returns(returns)?;
            return Ok(StepResult::ToolsExecuted(count));
        }

        // Handle output if found (only when no tool calls are pending)
        if let Some(output) = found_output {
            match self.validate_output(output).await {
                Ok(validated) => {
                    self.state.final_output = Some(validated);

                    // Early strategy: stop immediately
                    if self.agent.end_strategy == EndStrategy::Early {
                        self.state.finished = true;
                        return Ok(StepResult::OutputReady);
                    }
                }
                Err(e) => {
                    self.state.output_retries += 1;
                    if self.state.output_retries > self.agent.max_output_retries {
                        return Err(AgentRunError::OutputValidationFailed(e));
                    }

                    // Add retry message
                    self.add_retry_message(e)?;
                    return Ok(StepResult::RetryingOutput);
                }
            }
        }

        // Check if we should finish
        if response.finish_reason == Some(FinishReason::Stop) {
            if self.state.final_output.is_some() {
                self.state.finished = true;
                return Ok(StepResult::Finished);
            }

            // No output and model stopped - try to use text content as output
            if let Some(text) = response.parts.iter().find_map(|p| match p {
                ModelResponsePart::Text(t) if !t.content.is_empty() => Some(&t.content),
                _ => None,
            }) {
                // Try one more time to parse
                if let Ok(output) = self.agent.output_schema.parse_text(text) {
                    match self.validate_output(output).await {
                        Ok(validated) => {
                            self.state.final_output = Some(validated);
                            self.state.finished = true;
                            return Ok(StepResult::Finished);
                        }
                        Err(e) => {
                            return Err(AgentRunError::OutputValidationFailed(e));
                        }
                    }
                }
            }

            return Err(AgentRunError::UnexpectedStop);
        }

        Ok(StepResult::Continue)
    }

    async fn execute_tool_calls(
        &mut self,
        calls: Vec<serdes_ai_core::messages::ToolCallPart>,
    ) -> Vec<(String, Option<String>, Result<ToolReturn, ToolError>)> {
        if self.agent.parallel_tool_calls {
            self.execute_tools_parallel(calls).await
        } else {
            self.execute_tools_sequential(calls).await
        }
    }

    /// Execute tool calls sequentially (original behavior).
    async fn execute_tools_sequential(
        &mut self,
        calls: Vec<serdes_ai_core::messages::ToolCallPart>,
    ) -> Vec<(String, Option<String>, Result<ToolReturn, ToolError>)> {
        let mut returns = Vec::new();

        for tc in calls {
            self.state.usage.record_tool_call();

            let tool = match self.agent.find_tool(&tc.tool_name) {
                Some(t) => t,
                None => {
                    returns.push((
                        tc.tool_name.clone(),
                        tc.tool_call_id.clone(),
                        Err(ToolError::NotFound(tc.tool_name.clone())),
                    ));
                    continue;
                }
            };

            // Create tool context
            let tool_ctx = self.ctx.for_tool(&tc.tool_name, tc.tool_call_id.clone());

            // Execute with retries
            let args = tc.args.to_json();
            let mut retries = 0;
            let result = loop {
                match tool.executor.execute(args.clone(), &tool_ctx).await {
                    Ok(r) => break Ok(r),
                    Err(e) if e.is_retryable() && retries < tool.max_retries => {
                        retries += 1;
                        continue;
                    }
                    Err(e) => break Err(e),
                }
            };

            returns.push((tc.tool_name.clone(), tc.tool_call_id.clone(), result));
        }

        returns
    }

    /// Execute tool calls in parallel.
    async fn execute_tools_parallel(
        &mut self,
        calls: Vec<serdes_ai_core::messages::ToolCallPart>,
    ) -> Vec<(String, Option<String>, Result<ToolReturn, ToolError>)> {
        use futures::future::join_all;

        // Record all tool calls upfront
        for _ in &calls {
            self.state.usage.record_tool_call();
        }

        // Build futures for each tool call
        let futures: Vec<_> = calls
            .into_iter()
            .map(|tc| {
                let tool_name = tc.tool_name.clone();
                let tool_call_id = tc.tool_call_id.clone();
                let args = tc.args.to_json();

                // Look up tool (we need to clone Arc references for async move)
                let tool = self.agent.find_tool(&tc.tool_name).cloned();
                let tool_ctx = self.ctx.for_tool(&tc.tool_name, tc.tool_call_id.clone());

                async move {
                    let tool = match tool {
                        Some(t) => t,
                        None => {
                            return (
                                tool_name.clone(),
                                tool_call_id,
                                Err(ToolError::NotFound(tool_name)),
                            );
                        }
                    };

                    // Execute with retries
                    let max_retries = tool.max_retries;
                    let executor = tool.executor;
                    let mut retries = 0;

                    let result = loop {
                        match executor.execute(args.clone(), &tool_ctx).await {
                            Ok(r) => break Ok(r),
                            Err(e) if e.is_retryable() && retries < max_retries => {
                                retries += 1;
                                continue;
                            }
                            Err(e) => break Err(e),
                        }
                    };

                    (tool_name, tool_call_id, result)
                }
            })
            .collect();

        // Execute all futures, respecting concurrency limit if set
        if let Some(max_concurrent) = self.agent.max_concurrent_tools {
            self.execute_with_semaphore(futures, max_concurrent).await
        } else {
            join_all(futures).await
        }
    }

    /// Execute futures with a concurrency limit using a semaphore.
    ///
    /// Uses `join_all` to preserve the order of results while limiting
    /// how many futures execute concurrently via a semaphore.
    async fn execute_with_semaphore<F, T>(&self, futures: Vec<F>, max_concurrent: usize) -> Vec<T>
    where
        F: std::future::Future<Output = T> + Send,
        T: Send,
    {
        use futures::future::join_all;
        use std::sync::Arc;
        use tokio::sync::Semaphore;

        let semaphore = Arc::new(Semaphore::new(max_concurrent));

        let wrapped_futures: Vec<_> = futures
            .into_iter()
            .map(|fut| {
                let sem = Arc::clone(&semaphore);
                async move {
                    // Acquire permit before executing - this limits concurrency
                    let _permit = sem.acquire().await.expect("Semaphore closed unexpectedly");
                    fut.await
                    // Permit is dropped here, allowing another future to proceed
                }
            })
            .collect();

        // join_all preserves order - results[i] corresponds to futures[i]
        join_all(wrapped_futures).await
    }

    fn add_tool_returns(
        &mut self,
        returns: Vec<(String, Option<String>, Result<ToolReturn, ToolError>)>,
    ) -> Result<(), AgentRunError> {
        // CRITICAL: First add the previous response as a model response part.
        // This ensures proper user/assistant alternation for Anthropic and other providers.
        // Without this, we'd send consecutive user messages which violates the API contract.
        if let Some(last_response) = self.state.responses.last() {
            let mut response_req = ModelRequest::new();
            response_req
                .parts
                .push(ModelRequestPart::ModelResponse(Box::new(
                    last_response.clone(),
                )));
            self.state.messages.push(response_req);
        }

        let mut req = ModelRequest::new();

        for (tool_name, tool_call_id, result) in returns {
            match result {
                Ok(ret) => {
                    let mut part = ToolReturnPart::new(&tool_name, ret.content);
                    if let Some(id) = tool_call_id {
                        part = part.with_tool_call_id(id);
                    }
                    req.parts.push(ModelRequestPart::ToolReturn(part));
                }
                Err(e) => {
                    let mut part = RetryPromptPart::new(format!("Tool error: {}", e));
                    part = part.with_tool_name(&tool_name);
                    if let Some(id) = tool_call_id {
                        part = part.with_tool_call_id(id);
                    }
                    req.parts.push(ModelRequestPart::RetryPrompt(part));
                }
            }
        }

        if !req.parts.is_empty() {
            self.state.messages.push(req);
        }

        Ok(())
    }

    fn add_retry_message(&mut self, error: OutputValidationError) -> Result<(), AgentRunError> {
        let mut req = ModelRequest::new();
        let part = RetryPromptPart::new(error.retry_message());
        req.parts.push(ModelRequestPart::RetryPrompt(part));
        self.state.messages.push(req);
        Ok(())
    }

    async fn validate_output(&self, output: Output) -> Result<Output, OutputValidationError> {
        let mut output = output;
        for validator in &self.agent.output_validators {
            output = validator.validate(output, &self.ctx).await?;
        }
        Ok(output)
    }

    fn finalize(self) -> Result<AgentRunResult<Output>, AgentRunError> {
        let output = self.state.final_output.ok_or(AgentRunError::NoOutput)?;

        Ok(AgentRunResult {
            output,
            messages: self.state.messages,
            responses: self.state.responses,
            usage: self.state.usage,
            run_id: self.state.run_id,
            finish_reason: self.state.finish_reason.unwrap_or(FinishReason::Stop),
            metadata: self.ctx.metadata.clone(),
        })
    }

    /// Get current messages.
    pub fn messages(&self) -> &[ModelRequest] {
        &self.state.messages
    }

    /// Get current usage.
    pub fn usage(&self) -> &RunUsage {
        &self.state.usage
    }

    /// Get run ID.
    pub fn run_id(&self) -> &str {
        &self.state.run_id
    }

    /// Check if finished.
    pub fn is_finished(&self) -> bool {
        self.state.finished
    }

    /// Get current step number.
    pub fn step_number(&self) -> u32 {
        self.state.step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_options_default() {
        let options = RunOptions::default();
        assert!(options.model_settings.is_none());
        assert!(options.message_history.is_none());
    }

    #[test]
    fn test_run_options_builder() {
        let options = RunOptions::new()
            .model_settings(ModelSettings::new().temperature(0.5))
            .metadata(serde_json::json!({"key": "value"}));

        assert!(options.model_settings.is_some());
        assert!(options.metadata.is_some());
    }

    #[test]
    fn test_step_result_eq() {
        assert_eq!(StepResult::Continue, StepResult::Continue);
        assert_eq!(StepResult::ToolsExecuted(2), StepResult::ToolsExecuted(2));
        assert_ne!(StepResult::ToolsExecuted(1), StepResult::ToolsExecuted(2));
    }
}

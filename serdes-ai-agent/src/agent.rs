//! Main Agent type.
//!
//! The Agent is the core type of serdes-ai. It orchestrates model calls,
//! tool execution, and output validation.

use crate::context::{RunContext, RunUsage, UsageLimits};
use crate::errors::AgentRunError;
use crate::history::HistoryProcessor;
use crate::instructions::{InstructionFn, SystemPromptFn};
use crate::output::{OutputMode, OutputSchema, OutputValidator, TextOutputSchema};
use crate::run::{AgentRun, AgentRunResult, RunOptions};
use crate::stream::AgentStream;
use serdes_ai_core::messages::UserContent;
use serdes_ai_core::ModelSettings;
use serdes_ai_models::Model;
use serdes_ai_tools::ToolDefinition;
use std::marker::PhantomData;
use std::sync::Arc;

/// Strategy for handling tool calls when output is ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EndStrategy {
    /// Stop as soon as valid output is found (skip remaining tools).
    #[default]
    Early,
    /// Execute all tool calls even if output is ready.
    Exhaustive,
}

/// Instrumentation settings for tracing/logging.
#[derive(Debug, Clone, Default)]
pub struct InstrumentationSettings {
    /// Enable OpenTelemetry tracing.
    pub enable_tracing: bool,
    /// Log level for agent events.
    pub log_level: Option<String>,
    /// Custom span name.
    pub span_name: Option<String>,
}

/// The main agent type.
///
/// An agent wraps a model and provides:
/// - System prompts and instructions
/// - Tool registration and execution
/// - Structured output parsing and validation
/// - Retry logic for failures
/// - Usage tracking and limits
///
/// # Type Parameters
///
/// - `Deps`: Dependencies injected into tools and instruction functions.
/// - `Output`: The output type (default: `String`).
pub struct Agent<Deps = (), Output = String> {
    /// Model to use.
    pub(crate) model: Arc<dyn Model>,
    /// Agent name for identification.
    pub(crate) name: Option<String>,
    /// Default model settings.
    pub(crate) model_settings: ModelSettings,
    /// Static instructions.
    pub(crate) instructions: Vec<String>,
    /// Dynamic instruction functions.
    pub(crate) instruction_fns: Vec<Box<dyn InstructionFn<Deps>>>,
    /// Static system prompts.
    pub(crate) system_prompts: Vec<String>,
    /// Dynamic system prompt functions.
    pub(crate) system_prompt_fns: Vec<Box<dyn SystemPromptFn<Deps>>>,
    /// Registered tool definitions.
    pub(crate) tools: Vec<RegisteredTool<Deps>>,
    /// Output schema.
    pub(crate) output_schema: Box<dyn OutputSchema<Output>>,
    /// Output validators.
    pub(crate) output_validators: Vec<Box<dyn OutputValidator<Output, Deps>>>,
    /// End strategy for tool calls.
    pub(crate) end_strategy: EndStrategy,
    /// Maximum retries for output validation.
    pub(crate) max_output_retries: u32,
    /// Maximum retries for tools.
    pub(crate) max_tool_retries: u32,
    /// Usage limits.
    pub(crate) usage_limits: Option<UsageLimits>,
    /// History processors.
    pub(crate) history_processors: Vec<Box<dyn HistoryProcessor<Deps>>>,
    /// Instrumentation settings.
    pub(crate) instrument: Option<InstrumentationSettings>,
    pub(crate) _phantom: PhantomData<(Deps, Output)>,
}

/// A registered tool with its executor.
pub struct RegisteredTool<Deps> {
    /// Tool definition.
    pub definition: ToolDefinition,
    /// Tool executor.
    pub executor: Box<dyn ToolExecutor<Deps>>,
    /// Max retries for this tool.
    pub max_retries: u32,
}

/// Trait for executing tools.
#[async_trait::async_trait]
pub trait ToolExecutor<Deps>: Send + Sync {
    /// Execute the tool.
    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &RunContext<Deps>,
    ) -> Result<serdes_ai_tools::ToolReturn, serdes_ai_tools::ToolError>;
}

impl<Deps, Output> Agent<Deps, Output>
where
    Deps: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    /// Get the model.
    pub fn model(&self) -> &dyn Model {
        self.model.as_ref()
    }

    /// Get agent name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get model settings.
    pub fn model_settings(&self) -> &ModelSettings {
        &self.model_settings
    }

    /// Get registered tools.
    pub fn tools(&self) -> Vec<&ToolDefinition> {
        self.tools.iter().map(|t| &t.definition).collect()
    }

    /// Get the output mode.
    pub fn output_mode(&self) -> OutputMode {
        self.output_schema.mode()
    }

    /// Check if the agent has tools.
    pub fn has_tools(&self) -> bool {
        !self.tools.is_empty()
    }

    /// Get usage limits.
    pub fn usage_limits(&self) -> Option<&UsageLimits> {
        self.usage_limits.as_ref()
    }

    /// Run the agent with a prompt.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The user prompt to send to the model.
    /// * `deps` - Dependencies to inject into tools and instructions.
    ///
    /// # Returns
    ///
    /// The agent's output after completing the conversation.
    pub async fn run(
        &self,
        prompt: impl Into<UserContent>,
        deps: Deps,
    ) -> Result<AgentRunResult<Output>, AgentRunError> {
        self.run_with_options(prompt, deps, RunOptions::default())
            .await
    }

    /// Run with options.
    pub async fn run_with_options(
        &self,
        prompt: impl Into<UserContent>,
        deps: Deps,
        options: RunOptions,
    ) -> Result<AgentRunResult<Output>, AgentRunError> {
        let run = self.start_run(prompt, deps, options).await?;
        run.run_to_completion().await
    }

    /// Run synchronously (blocking).
    ///
    /// Note: This requires a Tokio runtime to be available.
    pub fn run_sync(
        &self,
        prompt: impl Into<UserContent>,
        deps: Deps,
    ) -> Result<AgentRunResult<Output>, AgentRunError> {
        tokio::runtime::Handle::current().block_on(self.run(prompt, deps))
    }

    /// Start a run that can be iterated.
    ///
    /// This allows stepping through the agent's execution manually.
    pub async fn start_run(
        &self,
        prompt: impl Into<UserContent>,
        deps: Deps,
        options: RunOptions,
    ) -> Result<AgentRun<Deps, Output>, AgentRunError> {
        AgentRun::new(self, prompt.into(), deps, options).await
    }

    /// Run with streaming output.
    pub async fn run_stream(
        &self,
        prompt: impl Into<UserContent>,
        deps: Deps,
    ) -> Result<AgentStream<Deps, Output>, AgentRunError> {
        self.run_stream_with_options(prompt, deps, RunOptions::default())
            .await
    }

    /// Run stream with options.
    pub async fn run_stream_with_options(
        &self,
        prompt: impl Into<UserContent>,
        deps: Deps,
        options: RunOptions,
    ) -> Result<AgentStream<Deps, Output>, AgentRunError> {
        AgentStream::new(self, prompt.into(), deps, options).await
    }

    /// Build the system prompt for a run.
    pub(crate) async fn build_system_prompt(&self, ctx: &RunContext<Deps>) -> String {
        let mut parts = Vec::new();

        // Static system prompts
        for prompt in &self.system_prompts {
            if !prompt.is_empty() {
                parts.push(prompt.clone());
            }
        }

        // Dynamic system prompts
        for prompt_fn in &self.system_prompt_fns {
            if let Some(prompt) = prompt_fn.generate(ctx).await {
                if !prompt.is_empty() {
                    parts.push(prompt);
                }
            }
        }

        // Static instructions
        for instruction in &self.instructions {
            if !instruction.is_empty() {
                parts.push(instruction.clone());
            }
        }

        // Dynamic instructions
        for instruction_fn in &self.instruction_fns {
            if let Some(instruction) = instruction_fn.generate(ctx).await {
                if !instruction.is_empty() {
                    parts.push(instruction);
                }
            }
        }

        parts.join("\n\n")
    }

    /// Find a tool by name.
    pub(crate) fn find_tool(&self, name: &str) -> Option<&RegisteredTool<Deps>> {
        self.tools.iter().find(|t| t.definition.name == name)
    }

    /// Check if this is the output tool.
    pub(crate) fn is_output_tool(&self, name: &str) -> bool {
        self.output_schema
            .tool_name()
            .map(|n| n == name)
            .unwrap_or(false)
    }
}

// Default for String output
impl<Deps: Send + Sync + 'static> Default for Agent<Deps, String> {
    fn default() -> Self {
        // Create a dummy model for default - users should always use builder
        panic!("Agent must be created using Agent::builder() or AgentBuilder")
    }
}

impl<Deps, Output> std::fmt::Debug for Agent<Deps, Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("name", &self.name)
            .field("model", &self.model.name())
            .field("tools", &self.tools.len())
            .field("end_strategy", &self.end_strategy)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_end_strategy_default() {
        assert_eq!(EndStrategy::default(), EndStrategy::Early);
    }

    #[test]
    fn test_instrumentation_settings_default() {
        let settings = InstrumentationSettings::default();
        assert!(!settings.enable_tracing);
        assert!(settings.log_level.is_none());
    }
}

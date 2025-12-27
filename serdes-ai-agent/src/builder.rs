//! Agent builder pattern.
//!
//! The builder provides a fluent interface for configuring agents.

use crate::agent::{Agent, EndStrategy, InstrumentationSettings, RegisteredTool, ToolExecutor};
use crate::context::{RunContext, UsageLimits};
use crate::errors::OutputValidationError;
use crate::history::HistoryProcessor;
use crate::instructions::{
    AsyncInstructionFn, AsyncSystemPromptFn, InstructionFn, SyncInstructionFn, SyncSystemPromptFn,
    SystemPromptFn,
};
use crate::output::{
    JsonOutputSchema, OutputSchema, OutputValidator, SyncValidator, TextOutputSchema,
    ToolOutputSchema,
};
use serde::de::DeserializeOwned;
use serdes_ai_core::ModelSettings;
use serdes_ai_models::Model;
use serdes_ai_tools::{ToolDefinition, ToolError, ToolReturn};
use serde_json::Value as JsonValue;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

/// Builder for creating agents.
pub struct AgentBuilder<Deps = (), Output = String> {
    model: Arc<dyn Model>,
    name: Option<String>,
    model_settings: ModelSettings,
    instructions: Vec<String>,
    instruction_fns: Vec<Box<dyn InstructionFn<Deps>>>,
    system_prompts: Vec<String>,
    system_prompt_fns: Vec<Box<dyn SystemPromptFn<Deps>>>,
    tools: Vec<RegisteredTool<Deps>>,
    output_schema: Option<Box<dyn OutputSchema<Output>>>,
    output_validators: Vec<Box<dyn OutputValidator<Output, Deps>>>,
    end_strategy: EndStrategy,
    max_output_retries: u32,
    max_tool_retries: u32,
    usage_limits: Option<UsageLimits>,
    history_processors: Vec<Box<dyn HistoryProcessor<Deps>>>,
    instrument: Option<InstrumentationSettings>,
    _phantom: PhantomData<(Deps, Output)>,
}

impl<Deps, Output> AgentBuilder<Deps, Output>
where
    Deps: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    /// Create a new agent builder with the given model.
    pub fn new<M: Model + 'static>(model: M) -> Self {
        Self {
            model: Arc::new(model),
            name: None,
            model_settings: ModelSettings::default(),
            instructions: Vec::new(),
            instruction_fns: Vec::new(),
            system_prompts: Vec::new(),
            system_prompt_fns: Vec::new(),
            tools: Vec::new(),
            output_schema: None,
            output_validators: Vec::new(),
            end_strategy: EndStrategy::Early,
            max_output_retries: 3,
            max_tool_retries: 3,
            usage_limits: None,
            history_processors: Vec::new(),
            instrument: None,
            _phantom: PhantomData,
        }
    }

    /// Set agent name.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set model settings.
    #[must_use]
    pub fn model_settings(mut self, settings: ModelSettings) -> Self {
        self.model_settings = settings;
        self
    }

    /// Set temperature.
    #[must_use]
    pub fn temperature(mut self, temp: f64) -> Self {
        self.model_settings = self.model_settings.temperature(temp);
        self
    }

    /// Set max tokens.
    #[must_use]
    pub fn max_tokens(mut self, tokens: u64) -> Self {
        self.model_settings = self.model_settings.max_tokens(tokens);
        self
    }

    /// Set top-p.
    #[must_use]
    pub fn top_p(mut self, p: f64) -> Self {
        self.model_settings = self.model_settings.top_p(p);
        self
    }

    /// Add static instructions.
    #[must_use]
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions.push(instructions.into());
        self
    }

    /// Add dynamic instructions function (async).
    #[must_use]
    pub fn instructions_fn<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(&RunContext<Deps>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<String>> + Send + 'static,
    {
        self.instruction_fns
            .push(Box::new(AsyncInstructionFn::new(f)));
        self
    }

    /// Add dynamic instructions function (sync).
    #[must_use]
    pub fn instructions_fn_sync<F>(mut self, f: F) -> Self
    where
        F: Fn(&RunContext<Deps>) -> Option<String> + Send + Sync + 'static,
    {
        self.instruction_fns
            .push(Box::new(SyncInstructionFn::new(f)));
        self
    }

    /// Add system prompt.
    #[must_use]
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompts.push(prompt.into());
        self
    }

    /// Add dynamic system prompt function (async).
    #[must_use]
    pub fn system_prompt_fn<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(&RunContext<Deps>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<String>> + Send + 'static,
    {
        self.system_prompt_fns
            .push(Box::new(AsyncSystemPromptFn::new(f)));
        self
    }

    /// Add dynamic system prompt function (sync).
    #[must_use]
    pub fn system_prompt_fn_sync<F>(mut self, f: F) -> Self
    where
        F: Fn(&RunContext<Deps>) -> Option<String> + Send + Sync + 'static,
    {
        self.system_prompt_fns
            .push(Box::new(SyncSystemPromptFn::new(f)));
        self
    }

    /// Add a tool with a custom executor.
    #[must_use]
    pub fn tool_with_executor<E>(mut self, definition: ToolDefinition, executor: E) -> Self
    where
        E: ToolExecutor<Deps> + 'static,
    {
        self.tools.push(RegisteredTool {
            definition,
            executor: Box::new(executor),
            max_retries: self.max_tool_retries,
        });
        self
    }

    /// Add a tool from a sync function.
    #[must_use]
    pub fn tool_fn<F, Args>(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        f: F,
    ) -> Self
    where
        F: Fn(&RunContext<Deps>, Args) -> Result<ToolReturn, ToolError> + Send + Sync + 'static,
        Args: DeserializeOwned + Send + 'static,
    {
        let definition = ToolDefinition::new(name.into(), description.into());

        let executor = SyncFnExecutor {
            func: Arc::new(move |ctx, args: JsonValue| {
                let parsed: Args = serde_json::from_value(args)
                    .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
                f(ctx, parsed)
            }),
            _phantom: PhantomData,
        };

        self.tools.push(RegisteredTool {
            definition,
            executor: Box::new(executor),
            max_retries: self.max_tool_retries,
        });
        self
    }

    /// Add a tool from an async function.
    #[must_use]
    pub fn tool_fn_async<F, Fut, Args>(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        f: F,
    ) -> Self
    where
        F: Fn(&RunContext<Deps>, Args) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolReturn, ToolError>> + Send + Sync + 'static,
        Args: DeserializeOwned + Send + Sync + 'static,
    {
        let definition = ToolDefinition::new(name.into(), description.into());

        let executor = AsyncFnExecutor {
            func: Arc::new(f),
            _phantom: PhantomData,
        };

        self.tools.push(RegisteredTool {
            definition,
            executor: Box::new(executor),
            max_retries: self.max_tool_retries,
        });
        self
    }

    /// Set custom output schema.
    #[must_use]
    pub fn output_schema<S: OutputSchema<Output> + 'static>(mut self, schema: S) -> Self {
        self.output_schema = Some(Box::new(schema));
        self
    }

    /// Add output validator.
    #[must_use]
    pub fn output_validator<V: OutputValidator<Output, Deps> + 'static>(
        mut self,
        validator: V,
    ) -> Self {
        self.output_validators.push(Box::new(validator));
        self
    }

    /// Add output validator from sync function.
    #[must_use]
    pub fn output_validator_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(Output, &RunContext<Deps>) -> Result<Output, OutputValidationError>
            + Send
            + Sync
            + 'static,
    {
        self.output_validators
            .push(Box::new(SyncValidator::new(f)));
        self
    }

    /// Set end strategy.
    #[must_use]
    pub fn end_strategy(mut self, strategy: EndStrategy) -> Self {
        self.end_strategy = strategy;
        self
    }

    /// Set max output retries.
    #[must_use]
    pub fn max_output_retries(mut self, retries: u32) -> Self {
        self.max_output_retries = retries;
        self
    }

    /// Set max tool retries.
    #[must_use]
    pub fn max_tool_retries(mut self, retries: u32) -> Self {
        self.max_tool_retries = retries;
        self
    }

    /// Set usage limits.
    #[must_use]
    pub fn usage_limits(mut self, limits: UsageLimits) -> Self {
        self.usage_limits = Some(limits);
        self
    }

    /// Add history processor.
    #[must_use]
    pub fn history_processor<P: HistoryProcessor<Deps> + 'static>(mut self, processor: P) -> Self {
        self.history_processors.push(Box::new(processor));
        self
    }

    /// Enable instrumentation.
    #[must_use]
    pub fn instrument(mut self, settings: InstrumentationSettings) -> Self {
        self.instrument = Some(settings);
        self
    }

    /// Build the agent.
    pub fn build(self) -> Agent<Deps, Output>
    where
        Output: serde::de::DeserializeOwned,
    {
        let output_schema = self
            .output_schema
            .unwrap_or_else(|| Box::new(JsonOutputSchema::<Output>::new()) as Box<dyn OutputSchema<Output>>);

        Agent {
            model: self.model,
            name: self.name,
            model_settings: self.model_settings,
            instructions: self.instructions,
            instruction_fns: self.instruction_fns,
            system_prompts: self.system_prompts,
            system_prompt_fns: self.system_prompt_fns,
            tools: self.tools,
            output_schema,
            output_validators: self.output_validators,
            end_strategy: self.end_strategy,
            max_output_retries: self.max_output_retries,
            max_tool_retries: self.max_tool_retries,
            usage_limits: self.usage_limits,
            history_processors: self.history_processors,
            instrument: self.instrument,
            _phantom: PhantomData,
        }
    }
}

// Specialized builders for output types

impl<Deps: Send + Sync + 'static> AgentBuilder<Deps, String> {
    /// Change output type to a JSON-parsed type.
    #[must_use]
    pub fn output_type<T: DeserializeOwned + Send + Sync + 'static>(
        self,
    ) -> AgentBuilder<Deps, T> {
        AgentBuilder {
            model: self.model,
            name: self.name,
            model_settings: self.model_settings,
            instructions: self.instructions,
            instruction_fns: self.instruction_fns,
            system_prompts: self.system_prompts,
            system_prompt_fns: self.system_prompt_fns,
            tools: self.tools,
            output_schema: Some(Box::new(JsonOutputSchema::<T>::new())),
            output_validators: Vec::new(),
            end_strategy: self.end_strategy,
            max_output_retries: self.max_output_retries,
            max_tool_retries: self.max_tool_retries,
            usage_limits: self.usage_limits,
            history_processors: self.history_processors,
            instrument: self.instrument,
            _phantom: PhantomData,
        }
    }

    /// Change output type with JSON schema.
    #[must_use]
    pub fn output_type_with_schema<T: DeserializeOwned + Send + Sync + 'static>(
        self,
        schema: JsonValue,
    ) -> AgentBuilder<Deps, T> {
        AgentBuilder {
            model: self.model,
            name: self.name,
            model_settings: self.model_settings,
            instructions: self.instructions,
            instruction_fns: self.instruction_fns,
            system_prompts: self.system_prompts,
            system_prompt_fns: self.system_prompt_fns,
            tools: self.tools,
            output_schema: Some(Box::new(JsonOutputSchema::<T>::new().with_schema(schema))),
            output_validators: Vec::new(),
            end_strategy: self.end_strategy,
            max_output_retries: self.max_output_retries,
            max_tool_retries: self.max_tool_retries,
            usage_limits: self.usage_limits,
            history_processors: self.history_processors,
            instrument: self.instrument,
            _phantom: PhantomData,
        }
    }

    /// Use tool-based output.
    #[must_use]
    pub fn output_tool<T: DeserializeOwned + Send + Sync + 'static>(
        self,
        tool_name: impl Into<String>,
        schema: JsonValue,
    ) -> AgentBuilder<Deps, T> {
        AgentBuilder {
            model: self.model,
            name: self.name,
            model_settings: self.model_settings,
            instructions: self.instructions,
            instruction_fns: self.instruction_fns,
            system_prompts: self.system_prompts,
            system_prompt_fns: self.system_prompt_fns,
            tools: self.tools,
            output_schema: Some(Box::new(
                ToolOutputSchema::<T>::new(tool_name).with_schema(schema),
            )),
            output_validators: Vec::new(),
            end_strategy: self.end_strategy,
            max_output_retries: self.max_output_retries,
            max_tool_retries: self.max_tool_retries,
            usage_limits: self.usage_limits,
            history_processors: self.history_processors,
            instrument: self.instrument,
            _phantom: PhantomData,
        }
    }
}

// ============================================================================
// Tool Executors
// ============================================================================

/// Sync function executor.
struct SyncFnExecutor<Deps> {
    func:
        Arc<dyn Fn(&RunContext<Deps>, JsonValue) -> Result<ToolReturn, ToolError> + Send + Sync>,
    _phantom: PhantomData<Deps>,
}

#[async_trait::async_trait]
impl<Deps: Send + Sync> ToolExecutor<Deps> for SyncFnExecutor<Deps> {
    async fn execute(
        &self,
        args: JsonValue,
        ctx: &RunContext<Deps>,
    ) -> Result<ToolReturn, ToolError> {
        (self.func)(ctx, args)
    }
}

/// Async function executor.
struct AsyncFnExecutor<F, Deps, Args, Fut>
where
    F: Fn(&RunContext<Deps>, Args) -> Fut + Send + Sync,
    Fut: Future<Output = Result<ToolReturn, ToolError>> + Send,
    Args: DeserializeOwned + Send,
{
    func: Arc<F>,
    _phantom: PhantomData<(Deps, Args, Fut)>,
}

#[async_trait::async_trait]
impl<F, Deps, Args, Fut> ToolExecutor<Deps> for AsyncFnExecutor<F, Deps, Args, Fut>
where
    F: Fn(&RunContext<Deps>, Args) -> Fut + Send + Sync,
    Fut: Future<Output = Result<ToolReturn, ToolError>> + Send + Sync,
    Args: DeserializeOwned + Send + Sync,
    Deps: Send + Sync,
{
    async fn execute(
        &self,
        args: JsonValue,
        ctx: &RunContext<Deps>,
    ) -> Result<ToolReturn, ToolError> {
        let parsed: Args = serde_json::from_value(args)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        (self.func)(ctx, parsed).await
    }
}

/// Convenience function to create a builder.
pub fn agent<M: Model + 'static>(model: M) -> AgentBuilder<(), String> {
    AgentBuilder::new(model)
}

/// Convenience function to create a builder with dependencies.
pub fn agent_with_deps<Deps: Send + Sync + 'static, M: Model + 'static>(
    model: M,
) -> AgentBuilder<Deps, String> {
    AgentBuilder::new(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serdes_ai_models::MockModel;

    fn create_mock_model() -> MockModel {
        MockModel::new("test-model")
    }

    #[test]
    fn test_builder_basic() {
        let model = create_mock_model();
        let agent = AgentBuilder::<(), String>::new(model)
            .name("test-agent")
            .temperature(0.7)
            .build();

        assert_eq!(agent.name(), Some("test-agent"));
        assert_eq!(agent.model_settings().temperature, Some(0.7));
    }

    #[test]
    fn test_builder_with_instructions() {
        let model = create_mock_model();
        let agent = AgentBuilder::<(), String>::new(model)
            .system_prompt("You are helpful.")
            .instructions("Be concise.")
            .build();

        assert_eq!(agent.system_prompts.len(), 1);
        assert_eq!(agent.instructions.len(), 1);
    }

    #[test]
    fn test_builder_with_tool() {
        let model = create_mock_model();
        let agent = AgentBuilder::<(), String>::new(model)
            .tool_fn(
                "greet",
                "Greet someone",
                |_ctx: &RunContext<()>, args: serde_json::Value| {
                    let name = args["name"].as_str().unwrap_or("World");
                    Ok(ToolReturn::text(format!("Hello, {}!", name)))
                },
            )
            .build();

        assert_eq!(agent.tools.len(), 1);
        assert_eq!(agent.tools[0].definition.name, "greet");
    }

    #[test]
    fn test_builder_usage_limits() {
        let model = create_mock_model();
        let agent = AgentBuilder::<(), String>::new(model)
            .usage_limits(UsageLimits::new().total_tokens(1000).requests(10))
            .build();

        let limits = agent.usage_limits().unwrap();
        assert_eq!(limits.max_total_tokens, Some(1000));
        assert_eq!(limits.max_requests, Some(10));
    }

    #[test]
    fn test_builder_end_strategy() {
        let model = create_mock_model();
        let agent = AgentBuilder::<(), String>::new(model)
            .end_strategy(EndStrategy::Exhaustive)
            .build();

        assert_eq!(agent.end_strategy, EndStrategy::Exhaustive);
    }

    #[test]
    fn test_agent_convenience() {
        let model = create_mock_model();
        let agent = agent(model)
            .name("quick-agent")
            .build();

        assert_eq!(agent.name(), Some("quick-agent"));
    }
}

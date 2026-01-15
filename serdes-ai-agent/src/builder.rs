//! Agent builder pattern.
//!
//! The builder provides a fluent interface for configuring agents.
//!
//! # Examples
//!
//! ## Using a model spec string (simplest)
//!
//! ```ignore
//! use serdes_ai_agent::AgentBuilder;
//!
//! // Uses environment variables for API keys
//! let agent = AgentBuilder::from_model("openai:gpt-4o")?
//!     .system_prompt("You are helpful.")
//!     .build();
//! ```
//!
//! ## With explicit API key
//!
//! ```ignore
//! use serdes_ai_agent::{AgentBuilder, ModelConfig};
//!
//! let config = ModelConfig::new("openai:gpt-4o")
//!     .with_api_key("sk-your-api-key");
//!
//! let agent = AgentBuilder::from_config(config)?
//!     .system_prompt("You are helpful.")
//!     .build();
//! ```
//!
//! ## With concrete model type (most control)
//!
//! ```ignore
//! use serdes_ai_agent::AgentBuilder;
//! use serdes_ai_models::openai::OpenAIChatModel;
//!
//! let model = OpenAIChatModel::new("gpt-4o", "sk-your-api-key")
//!     .with_base_url("https://custom-endpoint.com/v1");
//!
//! let agent = AgentBuilder::new(model)
//!     .system_prompt("You are helpful.")
//!     .build();
//! ```

use crate::agent::{Agent, EndStrategy, InstrumentationSettings, RegisteredTool, ToolExecutor};
use crate::context::{RunContext, UsageLimits};
use crate::errors::OutputValidationError;
use crate::history::HistoryProcessor;
use crate::instructions::{
    AsyncInstructionFn, AsyncSystemPromptFn, InstructionFn, SyncInstructionFn, SyncSystemPromptFn,
    SystemPromptFn,
};
use crate::output::{
    DefaultOutputSchema, JsonOutputSchema, OutputSchema, OutputValidator, SyncValidator,
    ToolOutputSchema,
};
use serde::de::DeserializeOwned;
use serde_json::Value as JsonValue;
use serdes_ai_core::ModelSettings;
use serdes_ai_models::{Model, ModelError};
use serdes_ai_tools::{ToolDefinition, ToolError, ToolReturn};
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// Model Configuration
// ============================================================================

/// Configuration for creating a model from a string spec.
///
/// This allows specifying a model using the standard `provider:model` format
/// while also providing custom API keys, base URLs, and other options.
///
/// # Examples
///
/// ```ignore
/// use serdes_ai_agent::ModelConfig;
///
/// // Simple: just a model spec (uses env vars for keys)
/// let config = ModelConfig::new("openai:gpt-4o");
///
/// // With explicit API key
/// let config = ModelConfig::new("anthropic:claude-3-5-sonnet-20241022")
///     .with_api_key("sk-ant-your-key");
///
/// // With custom base URL (for proxies or compatible APIs)
/// let config = ModelConfig::new("openai:gpt-4o")
///     .with_api_key("your-key")
///     .with_base_url("https://your-proxy.com/v1");
/// ```
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model spec in `provider:model` format (e.g., "openai:gpt-4o")
    pub spec: String,
    /// Optional API key (overrides environment variable)
    pub api_key: Option<String>,
    /// Optional base URL (for custom endpoints)
    pub base_url: Option<String>,
    /// Optional request timeout
    pub timeout: Option<Duration>,
}

impl ModelConfig {
    /// Create a new model config from a spec string.
    ///
    /// The spec should be in `provider:model` format, e.g.:
    /// - `"openai:gpt-4o"`
    /// - `"anthropic:claude-3-5-sonnet-20241022"`
    /// - `"groq:llama-3.1-70b-versatile"`
    /// - `"ollama:llama3.1"`
    ///
    /// If no provider prefix is given, OpenAI is assumed.
    #[must_use]
    pub fn new(spec: impl Into<String>) -> Self {
        Self {
            spec: spec.into(),
            api_key: None,
            base_url: None,
            timeout: None,
        }
    }

    /// Set an explicit API key (overrides environment variable).
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set a custom base URL (for proxies or compatible APIs).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set a request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Parse the provider and model name from the spec.
    fn parse_spec(&self) -> (&str, &str) {
        if self.spec.contains(':') {
            let parts: Vec<&str> = self.spec.splitn(2, ':').collect();
            (parts[0], parts[1])
        } else {
            ("openai", self.spec.as_str())
        }
    }

    /// Build a model from this configuration.
    ///
    /// This creates the appropriate model type based on the provider,
    /// applying any custom API key, base URL, or timeout settings.
    ///
    /// # Note
    ///
    /// This method delegates to `serdes_ai_models::infer_model_with_config` when
    /// using default settings (no custom API key/base URL), or creates the model
    /// directly when custom configuration is provided.
    ///
    /// The available providers depend on the features enabled in `serdes-ai-models`:
    /// - `openai` (default) - OpenAI models (gpt-4o, gpt-4, etc.)
    /// - `anthropic` - Anthropic models (claude-3-5-sonnet, etc.)
    /// - `groq` - Groq models
    /// - `mistral` - Mistral models
    /// - `ollama` - Local Ollama models
    /// - `google` - Google/Gemini models
    pub fn build_model(&self) -> Result<Arc<dyn Model>, ModelError> {
        // If no custom config, use infer_model which handles feature flags
        if self.api_key.is_none() && self.base_url.is_none() && self.timeout.is_none() {
            return serdes_ai_models::infer_model(&self.spec);
        }

        // Custom config requires building the model directly
        let (provider, model_name) = self.parse_spec();

        // We need to build the model with custom settings
        // This requires the concrete model types which are behind feature flags
        // in serdes-ai-models. We use a helper function pattern.
        self.build_model_with_config(provider, model_name)
    }

    fn build_model_with_config(
        &self,
        provider: &str,
        model_name: &str,
    ) -> Result<Arc<dyn Model>, ModelError> {
        // Use serdes_ai_models to build models - it has the feature flags
        serdes_ai_models::build_model_with_config(
            provider,
            model_name,
            self.api_key.as_deref(),
            self.base_url.as_deref(),
            self.timeout,
        )
    }
}

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
    parallel_tool_calls: bool,
    max_concurrent_tools: Option<usize>,
    _phantom: PhantomData<(Deps, Output)>,
}

impl<Deps, Output> AgentBuilder<Deps, Output>
where
    Deps: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    /// Create a new agent builder with the given model.
    ///
    /// This is the most flexible constructor, accepting any type that implements
    /// the `Model` trait. Use this when you need full control over model configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use serdes_ai_agent::AgentBuilder;
    /// use serdes_ai_models::openai::OpenAIChatModel;
    ///
    /// let model = OpenAIChatModel::new("gpt-4o", "sk-your-api-key");
    /// let agent = AgentBuilder::new(model)
    ///     .system_prompt("You are helpful.")
    ///     .build();
    /// ```
    pub fn new<M: Model + 'static>(model: M) -> Self {
        Self::from_arc(Arc::new(model))
    }

    /// Create a new agent builder from an `Arc<dyn Model>`.
    ///
    /// This is useful when you already have a model wrapped in an Arc,
    /// such as from `infer_model()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use serdes_ai_agent::AgentBuilder;
    /// use serdes_ai_models::infer_model;
    ///
    /// let model = infer_model("openai:gpt-4o")?;
    /// let agent = AgentBuilder::from_arc(model)
    ///     .system_prompt("You are helpful.")
    ///     .build();
    /// ```
    pub fn from_arc(model: Arc<dyn Model>) -> Self {
        Self {
            model,
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
            parallel_tool_calls: true,
            max_concurrent_tools: None,
            _phantom: PhantomData,
        }
    }

    /// Create a new agent builder from a model spec string.
    ///
    /// This is the simplest way to create an agent when you just need to specify
    /// the model. API keys are read from environment variables.
    ///
    /// # Model Spec Format
    ///
    /// The spec should be in `provider:model` format:
    /// - `"openai:gpt-4o"` - OpenAI GPT-4o
    /// - `"anthropic:claude-3-5-sonnet-20241022"` - Anthropic Claude
    /// - `"groq:llama-3.1-70b-versatile"` - Groq
    /// - `"ollama:llama3.1"` - Local Ollama
    ///
    /// If no provider prefix is given, OpenAI is assumed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use serdes_ai_agent::AgentBuilder;
    ///
    /// let agent = AgentBuilder::from_model("openai:gpt-4o")?
    ///     .system_prompt("You are helpful.")
    ///     .build();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be created (e.g., missing API key,
    /// unsupported provider, or disabled feature).
    pub fn from_model(spec: impl Into<String>) -> Result<Self, ModelError> {
        let config = ModelConfig::new(spec);
        Self::from_config(config)
    }

    /// Create a new agent builder from a model configuration.
    ///
    /// This allows specifying custom API keys, base URLs, and other options
    /// while still using the convenient string-based model spec.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use serdes_ai_agent::{AgentBuilder, ModelConfig};
    ///
    /// let config = ModelConfig::new("openai:gpt-4o")
    ///     .with_api_key("sk-your-api-key")
    ///     .with_base_url("https://your-proxy.com/v1");
    ///
    /// let agent = AgentBuilder::from_config(config)?
    ///     .system_prompt("You are helpful.")
    ///     .build();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be created.
    pub fn from_config(config: ModelConfig) -> Result<Self, ModelError> {
        let model = config.build_model()?;
        Ok(Self::from_arc(model))
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
            executor: Arc::new(executor),
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
        let tool_name = name.into();
        let definition = ToolDefinition::new(tool_name.clone(), description.into());

        let executor = SyncFnExecutor {
            func: Arc::new(move |ctx, args: JsonValue| {
                let parsed: Args = serde_json::from_value(args)
                    .map_err(|e| ToolError::invalid_arguments(tool_name.clone(), e.to_string()))?;
                f(ctx, parsed)
            }),
            _phantom: PhantomData,
        };

        self.tools.push(RegisteredTool {
            definition,
            executor: Arc::new(executor),
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
        let tool_name = name.into();
        let definition = ToolDefinition::new(tool_name.clone(), description.into());

        let executor = AsyncFnExecutor {
            func: Arc::new(f),
            tool_name,
            _phantom: PhantomData,
        };

        self.tools.push(RegisteredTool {
            definition,
            executor: Arc::new(executor),
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
        self.output_validators.push(Box::new(SyncValidator::new(f)));
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

    /// Enable or disable parallel tool execution.
    ///
    /// When enabled (default), multiple tool calls from the model will be
    /// executed concurrently using `futures::future::join_all`.
    ///
    /// When disabled, tools are executed sequentially in order.
    #[must_use]
    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = enabled;
        self
    }

    /// Set the maximum number of concurrent tool calls.
    ///
    /// When set, limits the number of tools that can execute simultaneously
    /// using a semaphore. This is useful for rate-limiting or resource control.
    ///
    /// Only applies when `parallel_tool_calls` is enabled.
    #[must_use]
    pub fn max_concurrent_tools(mut self, max: usize) -> Self {
        self.max_concurrent_tools = Some(max);
        self
    }

    /// Build the agent.
    pub fn build(self) -> Agent<Deps, Output>
    where
        Output: serde::de::DeserializeOwned,
    {
        let output_schema = self
            .output_schema
            .unwrap_or_else(|| Box::new(DefaultOutputSchema::<Output>::new()));

        // Pre-join static system prompts and instructions at build time.
        // This avoids cloning these strings on every run.
        let static_system_prompt = {
            let mut parts = Vec::new();

            // Static system prompts first
            for prompt in &self.system_prompts {
                if !prompt.is_empty() {
                    parts.push(prompt.as_str());
                }
            }

            // Then static instructions
            for instruction in &self.instructions {
                if !instruction.is_empty() {
                    parts.push(instruction.as_str());
                }
            }

            Arc::from(parts.join("\n\n"))
        };

        // Pre-compute tool definitions at build time.
        // This avoids cloning tool definitions on every agent step.
        let cached_tool_defs = Arc::new(
            self.tools
                .iter()
                .map(|t| t.definition.clone())
                .collect::<Vec<_>>(),
        );

        Agent {
            model: self.model,
            name: self.name,
            model_settings: self.model_settings,
            static_system_prompt,
            instruction_fns: self.instruction_fns,
            system_prompt_fns: self.system_prompt_fns,
            tools: self.tools,
            cached_tool_defs,
            output_schema,
            output_validators: self.output_validators,
            end_strategy: self.end_strategy,
            max_output_retries: self.max_output_retries,
            max_tool_retries: self.max_tool_retries,
            usage_limits: self.usage_limits,
            history_processors: self.history_processors,
            instrument: self.instrument,
            parallel_tool_calls: self.parallel_tool_calls,
            max_concurrent_tools: self.max_concurrent_tools,
            _phantom: PhantomData,
        }
    }
}

// Specialized builders for output types

impl<Deps: Send + Sync + 'static> AgentBuilder<Deps, String> {
    /// Change output type to a JSON-parsed type.
    #[must_use]
    pub fn output_type<T: DeserializeOwned + Send + Sync + 'static>(self) -> AgentBuilder<Deps, T> {
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
            parallel_tool_calls: self.parallel_tool_calls,
            max_concurrent_tools: self.max_concurrent_tools,
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
            parallel_tool_calls: self.parallel_tool_calls,
            max_concurrent_tools: self.max_concurrent_tools,
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
            parallel_tool_calls: self.parallel_tool_calls,
            max_concurrent_tools: self.max_concurrent_tools,
            _phantom: PhantomData,
        }
    }
}

// ============================================================================
// Tool Executors
// ============================================================================

/// Sync function executor.
#[allow(clippy::type_complexity)]
struct SyncFnExecutor<Deps> {
    func: Arc<dyn Fn(&RunContext<Deps>, JsonValue) -> Result<ToolReturn, ToolError> + Send + Sync>,
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
    tool_name: String,
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
            .map_err(|e| ToolError::invalid_arguments(self.tool_name.clone(), e.to_string()))?;
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

        // Static prompts are now pre-joined at build time
        assert!(agent.static_system_prompt.contains("You are helpful."));
        assert!(agent.static_system_prompt.contains("Be concise."));
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
        let agent = agent(model).name("quick-agent").build();

        assert_eq!(agent.name(), Some("quick-agent"));
    }

    #[test]
    fn test_builder_parallel_tool_calls_default() {
        let model = create_mock_model();
        let agent = AgentBuilder::<(), String>::new(model).build();

        // Default should be true (parallel enabled)
        assert!(agent.parallel_tool_calls());
        assert!(agent.max_concurrent_tools().is_none());
    }

    #[test]
    fn test_builder_parallel_tool_calls_disabled() {
        let model = create_mock_model();
        let agent = AgentBuilder::<(), String>::new(model)
            .parallel_tool_calls(false)
            .build();

        assert!(!agent.parallel_tool_calls());
    }

    #[test]
    fn test_builder_max_concurrent_tools() {
        let model = create_mock_model();
        let agent = AgentBuilder::<(), String>::new(model)
            .max_concurrent_tools(4)
            .build();

        assert!(agent.parallel_tool_calls());
        assert_eq!(agent.max_concurrent_tools(), Some(4));
    }

    #[test]
    fn test_builder_parallel_config_preserved_on_output_type() {
        let model = create_mock_model();
        let agent: Agent<(), serde_json::Value> = AgentBuilder::<(), String>::new(model)
            .parallel_tool_calls(false)
            .max_concurrent_tools(2)
            .output_type()
            .build();

        // Config should be preserved when changing output type
        assert!(!agent.parallel_tool_calls());
        assert_eq!(agent.max_concurrent_tools(), Some(2));
    }

    #[test]
    fn test_builder_from_arc() {
        let model = create_mock_model();
        let arc_model: Arc<dyn Model> = Arc::new(model);
        let agent = AgentBuilder::<(), String>::from_arc(arc_model)
            .name("arc-agent")
            .build();

        assert_eq!(agent.name(), Some("arc-agent"));
    }

    #[test]
    fn test_model_config_basic() {
        let config = ModelConfig::new("openai:gpt-4o");
        assert_eq!(config.spec, "openai:gpt-4o");
        assert!(config.api_key.is_none());
        assert!(config.base_url.is_none());
        assert!(config.timeout.is_none());
    }

    #[test]
    fn test_model_config_with_options() {
        let config = ModelConfig::new("anthropic:claude-3-5-sonnet-20241022")
            .with_api_key("sk-test-key")
            .with_base_url("https://custom.api.com")
            .with_timeout(Duration::from_secs(60));

        assert_eq!(config.spec, "anthropic:claude-3-5-sonnet-20241022");
        assert_eq!(config.api_key, Some("sk-test-key".to_string()));
        assert_eq!(config.base_url, Some("https://custom.api.com".to_string()));
        assert_eq!(config.timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_model_config_parse_spec_with_provider() {
        let config = ModelConfig::new("openai:gpt-4o");
        let (provider, model) = config.parse_spec();
        assert_eq!(provider, "openai");
        assert_eq!(model, "gpt-4o");
    }

    #[test]
    fn test_model_config_parse_spec_without_provider() {
        let config = ModelConfig::new("gpt-4o");
        let (provider, model) = config.parse_spec();
        assert_eq!(provider, "openai");
        assert_eq!(model, "gpt-4o");
    }

    #[test]
    fn test_model_config_parse_spec_anthropic() {
        let config = ModelConfig::new("anthropic:claude-3-5-sonnet-20241022");
        let (provider, model) = config.parse_spec();
        assert_eq!(provider, "anthropic");
        assert_eq!(model, "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn test_model_config_unknown_provider() {
        let config = ModelConfig::new("unknown:some-model");
        let result = config.build_model();
        assert!(result.is_err());
        // Can't use unwrap_err because Arc<dyn Model> doesn't impl Debug
        match result {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("Unknown") || msg.contains("unsupported"),
                    "Expected error about unknown provider, got: {}",
                    msg
                );
            }
            Ok(_) => panic!("Expected error for unknown provider"),
        }
    }
}

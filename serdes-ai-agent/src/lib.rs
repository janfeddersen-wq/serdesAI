//! Agent implementation for serdes-ai.
//!
//! The agent is the core abstraction for building AI applications. It provides:
//!
//! - Model orchestration
//! - Tool registration and execution
//! - Structured output parsing
//! - Retry logic and error handling
//! - Usage tracking and limits
//!
//! # Example
//!
//! ```rust,ignore
//! use serdes_ai_agent::{agent, EndStrategy};
//! use serdes_ai_models::openai::OpenAIChatModel;
//!
//! // Create a simple agent
//! let model = OpenAIChatModel::new("gpt-4o", "sk-...");
//! let agent = agent(model)
//!     .system_prompt("You are a helpful assistant.")
//!     .temperature(0.7)
//!     .build();
//!
//! // Run the agent
//! let result = agent.run("Hello!", ()).await?;
//! println!("Response: {}", result.output());
//! ```
//!
//! # With Tools
//!
//! ```rust,ignore
//! use serdes_ai_agent::agent;
//! use serdes_ai_tools::ToolReturn;
//!
//! let agent = agent(model)
//!     .system_prompt("You can search the web.")
//!     .tool_fn(
//!         "search",
//!         "Search the web for information",
//!         serde_json::json!({
//!             "type": "object",
//!             "properties": {
//!                 "query": {"type": "string"}
//!             },
//!             "required": ["query"]
//!         }),
//!         |ctx, args: serde_json::Value| {
//!             let query = args["query"].as_str().unwrap();
//!             Ok(ToolReturn::text(format!("Results for: {}", query)))
//!         },
//!     )
//!     .build();
//! ```
//!
//! # Structured Output
//!
//! ```rust,ignore
//! use serde::Deserialize;
//!
//! #[derive(Debug, Deserialize)]
//! struct Analysis {
//!     sentiment: String,
//!     score: f64,
//! }
//!
//! let agent = agent(model)
//!     .output_type::<Analysis>()
//!     .build();
//!
//! let result = agent.run("Analyze: I love Rust!", ()).await?;
//! println!("Sentiment: {} ({})", result.output.sentiment, result.output.score);
//! ```

pub mod agent;
pub mod builder;
pub mod context;
pub mod errors;
pub mod history;
pub mod instructions;
pub mod output;
pub mod run;
pub mod stream;

// Re-exports
pub use agent::{Agent, EndStrategy, InstrumentationSettings, RegisteredTool, ToolExecutor};
pub use builder::{agent, agent_with_deps, AgentBuilder};
pub use context::{generate_run_id, RunContext, RunUsage, UsageLimits};
pub use errors::{
    AgentBuildError, AgentRunError, OutputParseError, OutputValidationError, UsageLimitError,
};
pub use history::{
    ChainedProcessor, FilterHistory, FnProcessor, HistoryProcessor, SummarizeHistory,
    TruncateByTokens, TruncateHistory,
};
pub use instructions::{
    AsyncInstructionFn, AsyncSystemPromptFn, DateTimeInstruction, InstructionBuilder,
    InstructionFn, StaticInstruction, StaticSystemPrompt, SyncInstructionFn, SyncSystemPromptFn,
    SystemPromptFn,
};
pub use output::{
    AsyncValidator, ChainedValidator, JsonOutputSchema, LengthValidator, NonEmptyValidator,
    OutputMode, OutputSchema, OutputValidator, SyncValidator, TextOutputSchema, ToolOutputSchema,
};
pub use run::{AgentRun, AgentRunResult, RunOptions, StepResult};
pub use stream::{AgentStream, AgentStreamEvent};

/// Prelude for common imports.
pub mod prelude {
    pub use crate::{
        agent, agent_with_deps, Agent, AgentBuilder, AgentRun, AgentRunError, AgentRunResult,
        AgentStream, AgentStreamEvent, EndStrategy, OutputMode, OutputSchema, OutputValidator,
        RunContext, RunOptions, RunUsage, StepResult, UsageLimits,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelude_imports() {
        // Just verify the prelude compiles
        use crate::prelude::*;
        let _ = EndStrategy::Early;
        let _ = OutputMode::Text;
    }
}

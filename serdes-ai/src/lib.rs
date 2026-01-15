//! # SerdesAI - Type-Safe AI Agent Framework for Rust
//!
//! SerdesAI is a comprehensive Rust library for building AI agents that interact with
//! large language models (LLMs). It is a complete port of [pydantic-ai](https://github.com/pydantic/pydantic-ai)
//! to Rust, providing type-safe, ergonomic APIs for creating intelligent agents.
//!
//! ## Quick Start
//!
//! ```ignore
//! use serdes_ai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let agent = Agent::builder()
//!         .model("openai:gpt-4o")
//!         .system_prompt("You are a helpful assistant.")
//!         .build()?;
//!
//!     let result = agent.run("What is the capital of France?", ()).await?;
//!     println!("{}", result.output());
//!     Ok(())
//! }
//! ```
//!
//! ## Key Features
//!
//! - **Type-safe agents** with generic dependencies and output types
//! - **Multiple LLM providers** (OpenAI, Anthropic, Google, Groq, Mistral, Ollama, Bedrock)
//! - **Tool/function calling** with automatic JSON schema generation
//! - **Streaming responses** with real-time text updates
//! - **Structured outputs** with JSON Schema validation
//! - **MCP protocol support** for Model Context Protocol servers
//! - **Embeddings** for semantic search and RAG applications
//! - **Graph-based workflows** for complex multi-step tasks
//! - **Evaluation framework** for testing and benchmarking agents
//! - **Retry strategies** with exponential backoff
//! - **OpenTelemetry integration** for observability
//!
//! ## Feature Flags
//!
//! | Feature | Description | Default |
//! |---------|-------------|--------|
//! | `openai` | OpenAI GPT models | ✅ |
//! | `anthropic` | Anthropic Claude models | ✅ |
//! | `gemini` | Google Gemini models | ❌ |
//! | `groq` | Groq fast inference | ❌ |
//! | `mistral` | Mistral AI models | ❌ |
//! | `ollama` | Local Ollama models | ❌ |
//! | `bedrock` | AWS Bedrock | ❌ |
//! | `mcp` | MCP protocol support | ❌ |
//! | `embeddings` | Embedding models | ❌ |
//! | `graph` | Graph execution engine | ❌ |
//! | `evals` | Evaluation framework | ❌ |
//! | `macros` | Proc macros | ✅ |
//! | `otel` | OpenTelemetry | ❌ |
//! | `full` | All features | ❌ |
//!
//! ## Architecture
//!
//! SerdesAI is organized as a workspace of focused crates:
//!
//! - [`serdes_ai_core`] - Core types, messages, and errors
//! - [`serdes_ai_agent`] - Agent implementation and builder
//! - [`serdes_ai_models`] - Model trait and implementations
//! - [`serdes_ai_tools`] - Tool system and schema generation
//! - [`serdes_ai_toolsets`] - Toolset abstractions
//! - [`serdes_ai_output`] - Output schema validation
//! - [`serdes_ai_streaming`] - Streaming support
//! - [`serdes_ai_retries`] - Retry strategies
//! - [`serdes_ai_mcp`] - MCP protocol (optional)
//! - [`serdes_ai_embeddings`] - Embeddings (optional)
//! - [`serdes_ai_graph`] - Graph execution (optional)
//! - [`serdes_ai_evals`] - Evaluation framework (optional)
//! - [`serdes_ai_macros`] - Procedural macros
//!
//! ## Examples
//!
//! ### Simple Chat
//!
//! ```ignore
//! use serdes_ai::prelude::*;
//!
//! let agent = Agent::builder()
//!     .model("openai:gpt-4o")
//!     .system_prompt("You are helpful.")
//!     .build()?;
//!
//! let result = agent.run("Hello!", ()).await?;
//! ```
//!
//! ### With Tools
//!
//! ```ignore
//! use serdes_ai::prelude::*;
//!
//! #[tool(description = "Get weather for a city")]
//! async fn get_weather(ctx: &RunContext<()>, city: String) -> ToolResult<String> {
//!     Ok(format!("Weather in {}: 22°C, sunny", city))
//! }
//!
//! let agent = Agent::builder()
//!     .model("openai:gpt-4o")
//!     .tool(get_weather)
//!     .build()?;
//! ```
//!
//! ### Structured Output
//!
//! ```ignore
//! use serdes_ai::prelude::*;
//! use serde::Deserialize;
//!
//! #[derive(Deserialize, OutputSchema)]
//! struct Person {
//!     name: String,
//!     age: u32,
//! }
//!
//! let agent = Agent::builder()
//!     .model("openai:gpt-4o")
//!     .output_type::<Person>()
//!     .build()?;
//!
//! let result: Person = agent.run("Extract: John is 30 years old", ()).await?.into_output();
//! ```
//!
//! ### Streaming
//!
//! ```ignore
//! use serdes_ai::prelude::*;
//! use futures::StreamExt;
//!
//! let mut stream = agent.run_stream("Tell me a story", ()).await?;
//!
//! while let Some(event) = stream.next().await {
//!     if let AgentStreamEvent::Text { delta } = event? {
//!         print!("{}", delta);
//!     }
//! }
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![deny(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// ============================================================================
// Direct Model Access
// ============================================================================

/// Direct model request functions for imperative API access.
///
/// Use this module when you want to make simple, direct requests to models
/// without the full agent infrastructure.
///
/// # Example
///
/// ```rust,ignore
/// use serdes_ai::direct::model_request;
/// use serdes_ai_core::ModelRequest;
///
/// let response = model_request(
///     "openai:gpt-4o",
///     &[ModelRequest::user("Hello!")],
///     None,
///     None,
/// ).await?;
/// ```
pub mod direct;

// ============================================================================
// Core Crate Re-exports
// ============================================================================

/// Core types, messages, and error handling.
pub use serdes_ai_core as core;

/// Agent implementation and builder.
pub use serdes_ai_agent as agent;

/// Model traits and implementations.
pub use serdes_ai_models as models;

/// Provider abstractions.
pub use serdes_ai_providers as providers;

/// Tool system.
pub use serdes_ai_tools as tools;

/// Toolset abstractions.
pub use serdes_ai_toolsets as toolsets;

/// Output schema validation.
pub use serdes_ai_output as output;

/// Streaming support.
pub use serdes_ai_streaming as streaming;

/// Retry strategies.
pub use serdes_ai_retries as retries;

// ============================================================================
// Optional Crate Re-exports
// ============================================================================

/// Model Context Protocol support.
#[cfg(feature = "mcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "mcp")))]
pub use serdes_ai_mcp as mcp;

/// Embedding models.
#[cfg(feature = "embeddings")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddings")))]
pub use serdes_ai_embeddings as embeddings;

/// Graph-based execution engine.
#[cfg(feature = "graph")]
#[cfg_attr(docsrs, doc(cfg(feature = "graph")))]
pub use serdes_ai_graph as graph;

/// Evaluation framework.
#[cfg(feature = "evals")]
#[cfg_attr(docsrs, doc(cfg(feature = "evals")))]
pub use serdes_ai_evals as evals;

// ============================================================================
// Macro Re-exports
// ============================================================================

/// Derive macro for tools.
#[cfg(feature = "macros")]
#[cfg_attr(docsrs, doc(cfg(feature = "macros")))]
pub use serdes_ai_macros::Tool;

/// Derive macro for output schemas.
#[cfg(feature = "macros")]
#[cfg_attr(docsrs, doc(cfg(feature = "macros")))]
pub use serdes_ai_macros::OutputSchema;

/// Attribute macro for tool functions.
#[cfg(feature = "macros")]
#[cfg_attr(docsrs, doc(cfg(feature = "macros")))]
pub use serdes_ai_macros::tool;

/// Attribute macro for agent definitions.
#[cfg(feature = "macros")]
#[cfg_attr(docsrs, doc(cfg(feature = "macros")))]
pub use serdes_ai_macros::agent as agent_macro;

// ============================================================================
// Core Type Re-exports (Flat)
// ============================================================================

// Errors
pub use serdes_ai_core::SerdesAiError;

// Identifiers
pub use serdes_ai_core::{ConversationId, RunId, ToolCallId};

// Messages
pub use serdes_ai_core::{
    BinaryContent,
    // Builtin tools
    BuiltinToolCallPart,
    BuiltinToolReturnContent,
    BuiltinToolReturnPart,
    CodeExecutionResult,
    // File and binary content
    FilePart,
    FileSearchResult,
    FileSearchResults,
    FinishReason,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ModelResponsePartDelta,
    // Streaming events
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    WebSearchResult,
    WebSearchResults,
};

// Settings
pub use serdes_ai_core::ModelSettings;

// Usage
pub use serdes_ai_core::{RequestUsage, RunUsage, UsageLimits};

// Format
pub use serdes_ai_core::{
    format_as_xml, format_as_xml_with_options, XmlFormatError, XmlFormatOptions,
};

// Agent
pub use serdes_ai_agent::{
    Agent, AgentBuilder, AgentRun, AgentRunResult, AgentStream, AgentStreamEvent, EndStrategy,
    ModelConfig, RunContext, RunOptions, StepResult,
};

// Models
pub use serdes_ai_models::Model;
pub use serdes_ai_models::{build_model_extended, build_model_with_config, ExtendedModelConfig};

#[cfg(feature = "openai")]
#[cfg_attr(docsrs, doc(cfg(feature = "openai")))]
pub use serdes_ai_models::openai::OpenAIChatModel;

#[cfg(feature = "anthropic")]
#[cfg_attr(docsrs, doc(cfg(feature = "anthropic")))]
pub use serdes_ai_models::anthropic::AnthropicModel;

#[cfg(feature = "gemini")]
#[cfg_attr(docsrs, doc(cfg(feature = "gemini")))]
pub use serdes_ai_models::GeminiModel;

#[cfg(feature = "groq")]
#[cfg_attr(docsrs, doc(cfg(feature = "groq")))]
pub use serdes_ai_models::groq::GroqModel;

#[cfg(feature = "mistral")]
#[cfg_attr(docsrs, doc(cfg(feature = "mistral")))]
pub use serdes_ai_models::mistral::MistralModel;

#[cfg(feature = "ollama")]
#[cfg_attr(docsrs, doc(cfg(feature = "ollama")))]
pub use serdes_ai_models::ollama::OllamaModel;

#[cfg(feature = "bedrock")]
#[cfg_attr(docsrs, doc(cfg(feature = "bedrock")))]
pub use serdes_ai_models::bedrock::BedrockModel;

// Tools
pub use serdes_ai_tools::{
    ObjectJsonSchema, SchemaBuilder, Tool, ToolDefinition, ToolRegistry, ToolResult,
};

// Toolsets
pub use serdes_ai_toolsets::{
    AbstractToolset, ApprovalRequiredToolset, BoxedToolset, CombinedToolset, DynamicToolset,
    ExternalToolset, FilteredToolset, FunctionToolset, PrefixedToolset, PreparedToolset,
    RenamedToolset, ToolsetInfo, ToolsetTool, WrapperToolset,
};

// Output
pub use serdes_ai_output::{
    OutputSchema, StructuredOutputSchema, TextOutputSchema, ValidationResult,
};

// Streaming
pub use serdes_ai_streaming::{ResponseDelta, ResponseStream};

// Retries
pub use serdes_ai_retries::{
    ExponentialBackoff, FixedDelay, LinearBackoff, RetryConfig, RetryStrategy,
};

// Direct model access
pub use direct::{
    model_request, model_request_stream, model_request_stream_sync, model_request_sync,
    DirectError, ModelSpec, StreamedResponseSync,
};

// ============================================================================
// Optional Type Re-exports
// ============================================================================

// MCP
#[cfg(feature = "mcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "mcp")))]
pub use serdes_ai_mcp::{McpClient, McpToolset};

// Embeddings
#[cfg(feature = "embeddings")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddings")))]
pub use serdes_ai_embeddings::{EmbeddingModel, EmbeddingResult};

// Graph
#[cfg(feature = "graph")]
#[cfg_attr(docsrs, doc(cfg(feature = "graph")))]
pub use serdes_ai_graph::{
    BaseNode, Edge, End, Graph, GraphError, GraphExecutor, GraphResult, GraphRunContext,
    GraphRunResult, NodeResult,
};

// Evals
#[cfg(feature = "evals")]
#[cfg_attr(docsrs, doc(cfg(feature = "evals")))]
pub use serdes_ai_evals::{
    Case, ContainsScorer, Dataset, EvalCase, EvalRunner, EvalSuite, EvaluationReport,
    EvaluationResult, Evaluator, ExactMatchScorer,
};

// ============================================================================
// Prelude Module
// ============================================================================

/// Convenient prelude for common imports.
///
/// Import everything you need with a single use statement:
///
/// ```ignore
/// use serdes_ai::prelude::*;
/// ```
pub mod prelude {
    // Core types
    pub use crate::core::{ConversationId, Result, RunId, SerdesAiError, ToolCallId};

    // Messages
    pub use crate::core::{
        FinishReason, ModelRequest, ModelResponse, ModelSettings, RequestUsage, RunUsage,
        UsageLimits, UserContent,
    };

    // Agent
    pub use crate::agent::{
        Agent, AgentBuilder, AgentRun, AgentRunResult, AgentStream, AgentStreamEvent, EndStrategy,
        ModelConfig, RunContext, RunOptions,
    };

    // Models
    pub use crate::models::Model;

    #[cfg(feature = "openai")]
    pub use crate::models::openai::OpenAIChatModel;

    #[cfg(feature = "anthropic")]
    pub use crate::models::anthropic::AnthropicModel;

    // Tools
    pub use crate::tools::{Tool, ToolDefinition, ToolRegistry, ToolResult};

    // Toolsets
    pub use crate::toolsets::{
        AbstractToolset, BoxedToolset, CombinedToolset, DynamicToolset, FunctionToolset,
    };

    // Output
    pub use crate::output::{
        OutputSchema, StructuredOutputSchema, TextOutputSchema, ValidationResult,
    };

    // Streaming
    pub use crate::streaming::{ResponseDelta, ResponseStream};

    // Retries
    pub use crate::retries::{ExponentialBackoff, RetryConfig, RetryStrategy};

    // Direct model access
    pub use crate::direct::{model_request, model_request_stream, DirectError, ModelSpec};

    // Format
    pub use crate::core::{format_as_xml, XmlFormatOptions};

    // Macros
    #[cfg(feature = "macros")]
    pub use crate::{tool, OutputSchema as DeriveOutputSchema, Tool as DeriveTool};

    // MCP
    #[cfg(feature = "mcp")]
    pub use crate::mcp::{McpClient, McpToolset};

    // Graph
    #[cfg(feature = "graph")]
    pub use crate::graph::{
        BaseNode, End, Graph, GraphError, GraphExecutor, GraphResult, GraphRunContext,
        GraphRunResult, NodeResult,
    };

    // Evals
    #[cfg(feature = "evals")]
    pub use crate::evals::{EvalCase, EvalRunner, EvalSuite, Evaluator};
}

// ============================================================================
// Version Information
// ============================================================================

/// Returns the current version of serdes-ai.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Returns version information as a tuple (major, minor, patch).
pub fn version_tuple() -> (u32, u32, u32) {
    let version = version();
    let parts: Vec<&str> = version.split('.').collect();
    (
        parts.first().and_then(|s| s.parse().ok()).unwrap_or(0),
        parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0),
        parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(version(), "0.1.0");
    }

    #[test]
    fn test_version_tuple() {
        let (major, minor, patch) = version_tuple();
        assert_eq!(major, 0);
        assert_eq!(minor, 1);
        assert_eq!(patch, 0);
    }

    #[test]
    fn test_prelude_imports() {
        // Just verify these types exist and are accessible
        let _: fn() -> &'static str = crate::version;
    }
}

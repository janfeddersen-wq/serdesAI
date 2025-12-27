//! # serdes-ai-macros
//!
//! Procedural macros for serdes-ai.
//!
//! This crate provides derive macros and attribute macros to reduce boilerplate
//! when defining tools, agents, and output schemas.
//!
//! ## Tool Macro
//!
//! ```ignore
//! #[derive(Tool)]
//! #[tool(name = "get_weather", description = "Get current weather")]
//! struct GetWeather {
//!     city: String,
//!     units: Option<String>,
//! }
//! ```
//!
//! ## Output Schema Macro
//!
//! ```ignore
//! #[derive(OutputSchema)]
//! #[output(description = "Weather response")]
//! struct WeatherResponse {
//!     /// Temperature in requested units
//!     temperature: f64,
//!     /// Weather description
//!     description: String,
//! }
//! ```
//!
//! ## Agent Macro
//!
//! ```ignore
//! #[derive(Agent)]
//! #[agent(model = "openai:gpt-4", system_prompt = "You are helpful.")]
//! struct MyAgent;
//! ```

extern crate proc_macro;

mod agent;
mod output;
mod tool;

use proc_macro::TokenStream;

/// Derive macro for implementing the `Tool` trait.
///
/// Generates a `Tool` implementation that provides tool definition
/// with JSON schema derived from struct fields.
///
/// # Attributes
///
/// - `#[tool(name = "...")]` - Override tool name (default: snake_case struct name)
/// - `#[tool(description = "...")]` - Tool description for the model
/// - `#[tool(strict)]` - Enable strict mode for tool calls
///
/// # Example
///
/// ```ignore
/// #[derive(Tool)]
/// #[tool(name = "search", description = "Search the web")]
/// struct SearchWeb {
///     query: String,
///     limit: Option<u32>,
/// }
/// ```
#[proc_macro_derive(Tool, attributes(tool))]
pub fn derive_tool(input: TokenStream) -> TokenStream {
    tool::derive_tool_impl(input)
}

/// Attribute macro for creating tools from functions.
///
/// Transforms a function into a tool by generating argument struct
/// and tool wrapper.
///
/// # Example
///
/// ```ignore
/// #[tool]
/// async fn get_weather(ctx: RunContext, city: String) -> Result<String, ToolError> {
///     // Implementation
///     Ok(format!("Weather in {}", city))
/// }
/// ```
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    tool::tool_attribute_impl(attr, item)
}

/// Derive macro for implementing the `OutputSchema` trait.
///
/// Generates JSON Schema for structured output validation.
///
/// # Attributes
///
/// - `#[output(description = "...")]` - Schema description
/// - `#[output(strict = false)]` - Disable strict mode
///
/// # Example
///
/// ```ignore
/// #[derive(OutputSchema, Deserialize)]
/// #[output(description = "A person's information")]
/// struct Person {
///     /// Person's full name
///     name: String,
///     /// Age in years
///     age: u32,
///     /// Optional email address
///     email: Option<String>,
/// }
/// ```
#[proc_macro_derive(OutputSchema, attributes(output))]
pub fn derive_output_schema(input: TokenStream) -> TokenStream {
    output::derive_output_schema_impl(input)
}

/// Attribute macro for defining agents.
///
/// Adds agent configuration methods to a struct.
///
/// # Attributes
///
/// - `model = "..."`- Model identifier
/// - `system_prompt = "..."` - System prompt
///
/// # Example
///
/// ```ignore
/// #[agent(model = "openai:gpt-4o", system_prompt = "You are helpful.")]
/// struct AssistantAgent;
/// ```
#[proc_macro_attribute]
pub fn agent(attr: TokenStream, item: TokenStream) -> TokenStream {
    agent::agent_attribute_impl(attr, item)
}

/// Derive macro for implementing agent configuration.
///
/// # Example
///
/// ```ignore
/// #[derive(Agent)]
/// #[agent(model = "openai:gpt-4", result = String)]
/// struct MyAgent {
///     context: String,
/// }
/// ```
#[proc_macro_derive(Agent, attributes(agent))]
pub fn derive_agent(input: TokenStream) -> TokenStream {
    agent::derive_agent_impl(input)
}

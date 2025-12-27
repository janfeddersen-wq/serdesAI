//! # serdes-ai-output
//!
//! Output schema validation and structured output support for serdes-ai.
//!
//! This crate provides the infrastructure for parsing, validating, and
//! handling structured output from language models.
//!
//! ## Core Concepts
//!
//! - **[`OutputMode`]**: How the model generates structured output (text, native, prompted, tool)
//! - **[`OutputSchema`]**: Trait for parsing and validating model responses
//! - **[`TextOutputSchema`]**: For plain text output with optional constraints
//! - **[`StructuredOutputSchema`]**: For typed structured output using serde
//! - **[`OutputValidator`]**: Additional validation logic after parsing
//! - **[`OutputToolset`]**: Internal toolset for capturing output via tool calls
//!
//! ## Output Modes
//!
//! Different models support different ways of generating structured output:
//!
//! - **Text**: Free-form text, parsed by the application
//! - **Native**: Model's built-in JSON mode (e.g., OpenAI's response_format)
//! - **Prompted**: JSON output requested via system prompt
//! - **Tool**: Output captured via a "result" tool call (most reliable)
//!
//! ## Example
//!
//! ```rust
//! use serdes_ai_output::{StructuredOutputSchema, OutputSchema, OutputMode};
//! use serdes_ai_tools::{ObjectJsonSchema, PropertySchema};
//! use serde::Deserialize;
//!
//! #[derive(Deserialize)]
//! struct Person {
//!     name: String,
//!     age: u32,
//! }
//!
//! // Define the JSON schema
//! let schema = ObjectJsonSchema::new()
//!     .with_property("name", PropertySchema::string("Person's name").build(), true)
//!     .with_property("age", PropertySchema::integer("Person's age").build(), true);
//!
//! // Create the output schema
//! let output_schema: StructuredOutputSchema<Person> = StructuredOutputSchema::new(schema)
//!     .with_tool_name("submit_person")
//!     .with_description("Submit the person's information");
//!
//! // Parse output from a tool call
//! let args = serde_json::json!({"name": "Alice", "age": 30});
//! let person: Person = output_schema.parse_tool_call("submit_person", &args).unwrap();
//! assert_eq!(person.name, "Alice");
//! ```
//!
//! ## Text Output with Validation
//!
//! ```rust
//! use serdes_ai_output::{TextOutputSchema, OutputSchema};
//!
//! let schema = TextOutputSchema::new()
//!     .with_min_length(10)
//!     .with_max_length(1000)
//!     .with_pattern(r"^[A-Z]").unwrap() // Must start with uppercase
//!     .trim();
//!
//! let result = schema.parse_text("  Hello, World!  ").unwrap();
//! assert_eq!(result, "Hello, World!");
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod error;
pub mod mode;
pub mod parser;
pub mod schema;
pub mod spec;
pub mod structured;
pub mod text;
pub mod toolset;
pub mod types;
pub mod validator;

// Re-exports
pub use error::{OutputParseError, OutputValidationError, ParseResult, ValidationResult};
pub use mode::OutputMode;
pub use parser::{extract_json_from_text, looks_like_json, parse_json_from_text, parse_json_value};
pub use schema::{BoxedOutputSchema, OutputSchema, OutputSchemaWrapper};
pub use spec::{IntoOutputSpec, OutputSpec, OutputSpecBuilder};
pub use structured::{
    extract_json, AnyJsonSchema, StructuredOutputSchema, DEFAULT_OUTPUT_TOOL_DESCRIPTION,
    DEFAULT_OUTPUT_TOOL_NAME,
};
pub use text::{TextOutputSchema, TextOutputSchemaBuilder};
pub use toolset::{OutputCaptured, OutputToolset};
pub use types::{NativeOutput, PromptedOutput, StructuredDict, TextOutput, ToolOutput};
pub use validator::{
    async_validator, sync_validator, BoxedValidator, NoOpValidator,
    OutputValidator, RejectValidator, RetryValidator, SyncValidator, ValidatorChain,
};

/// Prelude for common imports.
pub mod prelude {
    pub use crate::{
        extract_json_from_text, looks_like_json, parse_json_from_text, AnyJsonSchema,
        BoxedOutputSchema, IntoOutputSpec, NativeOutput, NoOpValidator, OutputMode,
        OutputParseError, OutputSchema, OutputSpec, OutputToolset, OutputValidationError,
        OutputValidator, PromptedOutput, StructuredDict, StructuredOutputSchema, TextOutput,
        TextOutputSchema, ToolOutput, ValidatorChain,
    };
}

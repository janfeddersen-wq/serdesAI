//! Google AI / Vertex AI model implementations.
//!
//! This module provides complete implementations for Google's Generative AI API:
//!
//! - [`GoogleModel`]: Gemini 2.0 Flash, Gemini Pro, etc.
//!
//! ## Features
//!
//! - **Google AI**: Uses API key authentication
//! - **Vertex AI**: Uses project/location with OAuth
//! - **Thinking**: Flash Thinking models with thinking budget
//! - **Code Execution**: Built-in Python code execution
//! - **Google Search**: Web search grounding
//! - **Structured Output**: Native JSON schema support
//! - **Multi-modal**: Images, documents, audio, video
//!
//! ## Example (Google AI)
//!
//! ```rust,ignore
//! use serdes_ai_models::google::GoogleModel;
//! use serdes_ai_models::Model;
//!
//! let model = GoogleModel::new(
//!     "gemini-2.0-flash",
//!     std::env::var("GOOGLE_API_KEY").unwrap()
//! );
//!
//! // With thinking
//! let model = GoogleModel::new("gemini-2.0-flash-thinking", api_key)
//!     .with_thinking(Some(10000));
//!
//! // With code execution
//! let model = model.with_code_execution();
//!
//! // With Google Search
//! let model = model.with_search();
//! ```
//!
//! ## Example (Vertex AI)
//!
//! ```rust,ignore
//! use serdes_ai_models::google::GoogleModel;
//!
//! let model = GoogleModel::vertex(
//!     "gemini-2.0-flash",
//!     "my-gcp-project",
//!     "us-central1"
//! );
//! ```

pub mod model;
pub mod stream;
pub mod types;

// Re-exports
pub use model::GoogleModel;
pub use types::{
    Blob, Candidate, CodeExecution, Content, FileData, FunctionCall, FunctionCallingConfig,
    FunctionDeclaration, FunctionResponse, GenerateContentRequest, GenerateContentResponse,
    GenerationConfig, GoogleError, GoogleSearch, GoogleTool, Part, SafetyRating, SafetySetting,
    ThinkingConfig, ToolConfig, UsageMetadata,
};

/// Create a new Google AI model.
///
/// This is a convenience function for creating a Google AI model.
///
/// # Arguments
///
/// * `model_name` - The model name (e.g., "gemini-2.0-flash")
/// * `api_key` - Your Google AI API key
///
/// # Example
///
/// ```rust,ignore
/// use serdes_ai_models::google;
///
/// let model = google::gemini("gemini-2.0-flash", "API_KEY");
/// ```
pub fn gemini(model_name: impl Into<String>, api_key: impl Into<String>) -> GoogleModel {
    GoogleModel::new(model_name, api_key)
}

/// Create a Vertex AI model.
///
/// # Arguments
///
/// * `model_name` - The model name
/// * `project_id` - Your GCP project ID
/// * `location` - The region (e.g., "us-central1")
pub fn vertex(
    model_name: impl Into<String>,
    project_id: impl Into<String>,
    location: impl Into<String>,
) -> GoogleModel {
    GoogleModel::vertex(model_name, project_id, location)
}

/// Common Google model names.
pub mod models {
    /// Gemini 2.0 Flash (fast, multimodal)
    pub const GEMINI_2_FLASH: &str = "gemini-2.0-flash";
    /// Gemini 2.0 Flash Thinking (reasoning)
    pub const GEMINI_2_FLASH_THINKING: &str = "gemini-2.0-flash-thinking-exp";
    /// Gemini 1.5 Pro (long context)
    pub const GEMINI_1_5_PRO: &str = "gemini-1.5-pro";
    /// Gemini 1.5 Flash (fast)
    pub const GEMINI_1_5_FLASH: &str = "gemini-1.5-flash";
    /// Gemini 1.0 Pro
    pub const GEMINI_PRO: &str = "gemini-pro";
    /// Gemini Experimental
    pub const GEMINI_EXP: &str = "gemini-exp-1206";
}

//! Google AI / Vertex AI API types.
//!
//! This module contains all request/response types for Google's Generative AI API.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

// ============================================================================
// Request Types
// ============================================================================

/// Generate content request.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    /// Content messages.
    pub contents: Vec<Content>,
    /// System instruction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    /// Tool definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GoogleTool>>,
    /// Tool configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    /// Generation configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    /// Safety settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_settings: Option<Vec<SafetySetting>>,
    /// Cached content name (for context caching).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content: Option<String>,
}

impl GenerateContentRequest {
    /// Create a new request.
    pub fn new(contents: Vec<Content>) -> Self {
        Self {
            contents,
            system_instruction: None,
            tools: None,
            tool_config: None,
            generation_config: None,
            safety_settings: None,
            cached_content: None,
        }
    }

    /// Add system instruction.
    pub fn with_system(mut self, instruction: impl Into<String>) -> Self {
        self.system_instruction = Some(Content::system(instruction));
        self
    }

    /// Add generation config.
    pub fn with_generation_config(mut self, config: GenerationConfig) -> Self {
        self.generation_config = Some(config);
        self
    }
}

/// Content (message) in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    /// Role: "user" or "model".
    pub role: String,
    /// Content parts.
    pub parts: Vec<Part>,
}

impl Content {
    /// Create user content.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            parts: vec![Part::text(text)],
        }
    }

    /// Create model content.
    pub fn model(text: impl Into<String>) -> Self {
        Self {
            role: "model".to_string(),
            parts: vec![Part::text(text)],
        }
    }

    /// Create system instruction content (no role needed).
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(), // System uses user role in Google API
            parts: vec![Part::text(text)],
        }
    }

    /// Create content with parts.
    pub fn user_parts(parts: Vec<Part>) -> Self {
        Self {
            role: "user".to_string(),
            parts,
        }
    }

    /// Create model content with parts.
    pub fn model_parts(parts: Vec<Part>) -> Self {
        Self {
            role: "model".to_string(),
            parts,
        }
    }
}

/// Content part.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Part {
    /// Text content.
    Text {
        /// The text.
        text: String,
    },
    /// Inline binary data.
    InlineData {
        /// The blob data.
        #[serde(rename = "inlineData")]
        inline_data: Blob,
    },
    /// Reference to uploaded file.
    FileData {
        /// The file reference.
        #[serde(rename = "fileData")]
        file_data: FileData,
    },
    /// Function call from model.
    FunctionCall {
        /// The function call.
        #[serde(rename = "functionCall")]
        function_call: FunctionCall,
    },
    /// Function response to model.
    FunctionResponse {
        /// The function response.
        #[serde(rename = "functionResponse")]
        function_response: FunctionResponse,
    },
    /// Executable code (from code execution).
    ExecutableCode {
        /// The code.
        #[serde(rename = "executableCode")]
        executable_code: ExecutableCode,
    },
    /// Code execution result.
    CodeExecutionResult {
        /// The result.
        #[serde(rename = "codeExecutionResult")]
        code_execution_result: CodeExecutionResult,
    },
    /// Thinking content (for thinking models).
    Thought {
        /// The thinking content.
        thought: String,
    },
}

impl Part {
    /// Create text part.
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }

    /// Create inline data part.
    pub fn inline_data(mime_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::InlineData {
            inline_data: Blob {
                mime_type: mime_type.into(),
                data: data.into(),
            },
        }
    }

    /// Create file data part.
    pub fn file_data(mime_type: impl Into<String>, file_uri: impl Into<String>) -> Self {
        Self::FileData {
            file_data: FileData {
                mime_type: mime_type.into(),
                file_uri: file_uri.into(),
            },
        }
    }

    /// Create function call part.
    pub fn function_call(name: impl Into<String>, args: JsonValue) -> Self {
        Self::FunctionCall {
            function_call: FunctionCall {
                name: name.into(),
                args,
            },
        }
    }

    /// Create function response part.
    pub fn function_response(name: impl Into<String>, response: JsonValue) -> Self {
        Self::FunctionResponse {
            function_response: FunctionResponse {
                name: name.into(),
                response,
            },
        }
    }

    /// Get text content if this is a text part.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Part::Text { text } => Some(text),
            _ => None,
        }
    }
}

/// Binary blob data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Blob {
    /// MIME type.
    pub mime_type: String,
    /// Base64-encoded data.
    pub data: String,
}

/// File reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileData {
    /// MIME type.
    pub mime_type: String,
    /// File URI (gs:// or uploaded file URI).
    pub file_uri: String,
}

/// Function call from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Function name.
    pub name: String,
    /// Function arguments.
    pub args: JsonValue,
}

/// Function response to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponse {
    /// Function name.
    pub name: String,
    /// Response data.
    pub response: JsonValue,
}

/// Executable code from code execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableCode {
    /// Programming language.
    pub language: String,
    /// The code.
    pub code: String,
}

/// Result of code execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionResult {
    /// Execution outcome.
    pub outcome: String,
    /// Output text.
    #[serde(default)]
    pub output: String,
}

/// Tool definition.
#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GoogleTool {
    /// Function declarations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_declarations: Option<Vec<FunctionDeclaration>>,
    /// Code execution tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_execution: Option<CodeExecution>,
    /// Google Search grounding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_search: Option<GoogleSearch>,
    /// Google Search retrieval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_search_retrieval: Option<GoogleSearchRetrieval>,
}

impl GoogleTool {
    /// Create a tool with function declarations.
    pub fn functions(declarations: Vec<FunctionDeclaration>) -> Self {
        Self {
            function_declarations: Some(declarations),
            ..Default::default()
        }
    }

    /// Create code execution tool.
    pub fn code_execution() -> Self {
        Self {
            code_execution: Some(CodeExecution {}),
            ..Default::default()
        }
    }

    /// Create Google Search tool.
    pub fn google_search() -> Self {
        Self {
            google_search: Some(GoogleSearch {}),
            ..Default::default()
        }
    }
}

/// Function declaration.
#[derive(Debug, Clone, Serialize)]
pub struct FunctionDeclaration {
    /// Function name.
    pub name: String,
    /// Function description.
    pub description: String,
    /// Parameter schema.
    pub parameters: JsonValue,
}

impl FunctionDeclaration {
    /// Create a new function declaration.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: JsonValue,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// Code execution tool config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecution {}

/// Google Search tool config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleSearch {}

/// Google Search retrieval config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleSearchRetrieval {
    /// Dynamic retrieval config.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_retrieval_config: Option<DynamicRetrievalConfig>,
}

/// Dynamic retrieval configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DynamicRetrievalConfig {
    /// Mode: "MODE_UNSPECIFIED", "MODE_DYNAMIC".
    pub mode: String,
    /// Dynamic threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_threshold: Option<f64>,
}

/// Tool configuration.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    /// Function calling config.
    pub function_calling_config: FunctionCallingConfig,
}

impl ToolConfig {
    /// Create auto mode config.
    pub fn auto() -> Self {
        Self {
            function_calling_config: FunctionCallingConfig::auto(),
        }
    }

    /// Create any mode config.
    pub fn any() -> Self {
        Self {
            function_calling_config: FunctionCallingConfig::any(),
        }
    }

    /// Create none mode config.
    pub fn none() -> Self {
        Self {
            function_calling_config: FunctionCallingConfig::none(),
        }
    }
}

/// Function calling configuration.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    /// Mode: "AUTO", "ANY", "NONE".
    pub mode: String,
    /// Allowed function names (for specific tool choice).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

impl FunctionCallingConfig {
    /// Auto mode.
    pub fn auto() -> Self {
        Self {
            mode: "AUTO".to_string(),
            allowed_function_names: None,
        }
    }

    /// Any mode (must call a function).
    pub fn any() -> Self {
        Self {
            mode: "ANY".to_string(),
            allowed_function_names: None,
        }
    }

    /// None mode (don't call functions).
    pub fn none() -> Self {
        Self {
            mode: "NONE".to_string(),
            allowed_function_names: None,
        }
    }

    /// Specific function(s) only.
    pub fn specific(names: Vec<String>) -> Self {
        Self {
            mode: "ANY".to_string(),
            allowed_function_names: Some(names),
        }
    }
}

/// Generation configuration.
#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    /// Temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Top-p.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top-k.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
    /// Max output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Response MIME type (for structured output).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    /// Response schema (for structured output).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<JsonValue>,
    /// Candidate count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<u32>,
    /// Presence penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Frequency penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Thinking configuration (for thinking models).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
}

impl GenerationConfig {
    /// Create new config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set temperature.
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set max tokens.
    pub fn max_tokens(mut self, max: u64) -> Self {
        self.max_output_tokens = Some(max);
        self
    }

    /// Set top-p.
    pub fn top_p(mut self, p: f64) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Enable JSON mode.
    pub fn json_mode(mut self) -> Self {
        self.response_mime_type = Some("application/json".to_string());
        self
    }

    /// Set structured output schema.
    pub fn with_schema(mut self, schema: JsonValue) -> Self {
        self.response_mime_type = Some("application/json".to_string());
        self.response_schema = Some(schema);
        self
    }

    /// Enable thinking with budget.
    pub fn with_thinking(mut self, budget: u64) -> Self {
        self.thinking_config = Some(ThinkingConfig {
            thinking_budget: budget,
        });
        self
    }
}

/// Thinking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    /// Token budget for thinking.
    pub thinking_budget: u64,
}

/// Safety setting.
#[derive(Debug, Clone, Serialize)]
pub struct SafetySetting {
    /// Category.
    pub category: String,
    /// Threshold.
    pub threshold: String,
}

impl SafetySetting {
    /// Block none.
    pub fn block_none(category: impl Into<String>) -> Self {
        Self {
            category: category.into(),
            threshold: "BLOCK_NONE".to_string(),
        }
    }

    /// Block low and above.
    pub fn block_low(category: impl Into<String>) -> Self {
        Self {
            category: category.into(),
            threshold: "BLOCK_LOW_AND_ABOVE".to_string(),
        }
    }

    /// Block medium and above.
    pub fn block_medium(category: impl Into<String>) -> Self {
        Self {
            category: category.into(),
            threshold: "BLOCK_MEDIUM_AND_ABOVE".to_string(),
        }
    }

    /// Block only high.
    pub fn block_high(category: impl Into<String>) -> Self {
        Self {
            category: category.into(),
            threshold: "BLOCK_ONLY_HIGH".to_string(),
        }
    }
}

// ============================================================================
// Response Types
// ============================================================================

/// Generate content response.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    /// Candidates.
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    /// Usage metadata.
    #[serde(default)]
    pub usage_metadata: Option<UsageMetadata>,
    /// Model version.
    #[serde(default)]
    pub model_version: Option<String>,
    /// Prompt feedback (for blocked prompts).
    #[serde(default)]
    pub prompt_feedback: Option<PromptFeedback>,
}

/// Response candidate.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    /// Content.
    pub content: Option<Content>,
    /// Finish reason.
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// Safety ratings.
    #[serde(default)]
    pub safety_ratings: Option<Vec<SafetyRating>>,
    /// Citation metadata.
    #[serde(default)]
    pub citation_metadata: Option<CitationMetadata>,
    /// Grounding metadata.
    #[serde(default)]
    pub grounding_metadata: Option<GroundingMetadata>,
    /// Index.
    #[serde(default)]
    pub index: Option<u32>,
}

/// Safety rating.
#[derive(Debug, Clone, Deserialize)]
pub struct SafetyRating {
    /// Category.
    pub category: String,
    /// Probability.
    pub probability: String,
    /// Blocked.
    #[serde(default)]
    pub blocked: bool,
}

/// Citation metadata.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CitationMetadata {
    /// Citation sources.
    #[serde(default)]
    pub citation_sources: Vec<CitationSource>,
}

/// Citation source.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CitationSource {
    /// Start index.
    #[serde(default)]
    pub start_index: u32,
    /// End index.
    #[serde(default)]
    pub end_index: u32,
    /// URI.
    #[serde(default)]
    pub uri: Option<String>,
    /// License.
    #[serde(default)]
    pub license: Option<String>,
}

/// Grounding metadata.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingMetadata {
    /// Web search queries.
    #[serde(default)]
    pub web_search_queries: Vec<String>,
    /// Search entry point.
    #[serde(default)]
    pub search_entry_point: Option<SearchEntryPoint>,
    /// Grounding chunks.
    #[serde(default)]
    pub grounding_chunks: Vec<GroundingChunk>,
    /// Grounding supports.
    #[serde(default)]
    pub grounding_supports: Vec<GroundingSupport>,
}

/// Search entry point.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchEntryPoint {
    /// Rendered content.
    #[serde(default)]
    pub rendered_content: Option<String>,
    /// SDK blob (for rendering).
    #[serde(default)]
    pub sdk_blob: Option<String>,
}

/// Grounding chunk.
#[derive(Debug, Clone, Deserialize)]
pub struct GroundingChunk {
    /// Web source.
    #[serde(default)]
    pub web: Option<WebChunk>,
}

/// Web chunk.
#[derive(Debug, Clone, Deserialize)]
pub struct WebChunk {
    /// URI.
    #[serde(default)]
    pub uri: Option<String>,
    /// Title.
    #[serde(default)]
    pub title: Option<String>,
}

/// Grounding support.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingSupport {
    /// Segment.
    #[serde(default)]
    pub segment: Option<Segment>,
    /// Grounding chunk indices.
    #[serde(default)]
    pub grounding_chunk_indices: Vec<u32>,
    /// Confidence scores.
    #[serde(default)]
    pub confidence_scores: Vec<f64>,
}

/// Text segment.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    /// Part index.
    #[serde(default)]
    pub part_index: u32,
    /// Start index.
    #[serde(default)]
    pub start_index: u32,
    /// End index.
    #[serde(default)]
    pub end_index: u32,
    /// Text.
    #[serde(default)]
    pub text: Option<String>,
}

/// Usage metadata.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    /// Prompt token count.
    #[serde(default)]
    pub prompt_token_count: u64,
    /// Candidates token count.
    #[serde(default)]
    pub candidates_token_count: u64,
    /// Total token count.
    #[serde(default)]
    pub total_token_count: u64,
    /// Cached content token count.
    #[serde(default)]
    pub cached_content_token_count: Option<u64>,
    /// Thinking token count.
    #[serde(default)]
    pub thoughts_token_count: Option<u64>,
}

/// Prompt feedback (for blocked prompts).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    /// Block reason.
    #[serde(default)]
    pub block_reason: Option<String>,
    /// Safety ratings.
    #[serde(default)]
    pub safety_ratings: Vec<SafetyRating>,
}

// ============================================================================
// Streaming Types
// ============================================================================

/// Streaming response chunk (same as full response).
pub type StreamingResponse = GenerateContentResponse;

// ============================================================================
// Error Types
// ============================================================================

/// Google API error response.
#[derive(Debug, Clone, Deserialize)]
pub struct GoogleError {
    /// Error details.
    pub error: GoogleErrorBody,
}

/// Google error body.
#[derive(Debug, Clone, Deserialize)]
pub struct GoogleErrorBody {
    /// Error code.
    pub code: u32,
    /// Error message.
    pub message: String,
    /// Error status.
    #[serde(default)]
    pub status: Option<String>,
    /// Error details.
    #[serde(default)]
    pub details: Vec<JsonValue>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_user() {
        let content = Content::user("Hello!");
        assert_eq!(content.role, "user");
        assert_eq!(content.parts.len(), 1);
    }

    #[test]
    fn test_content_model() {
        let content = Content::model("Hi there!");
        assert_eq!(content.role, "model");
    }

    #[test]
    fn test_part_text() {
        let part = Part::text("Hello");
        assert_eq!(part.as_text(), Some("Hello"));
    }

    #[test]
    fn test_part_inline_data() {
        let part = Part::inline_data("image/png", "abc123");
        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("inlineData"));
        assert!(json.contains("image/png"));
    }

    #[test]
    fn test_function_declaration() {
        let decl = FunctionDeclaration::new(
            "search",
            "Search the web",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }),
        );
        assert_eq!(decl.name, "search");
    }

    #[test]
    fn test_generation_config() {
        let config = GenerationConfig::new()
            .temperature(0.7)
            .max_tokens(1000)
            .json_mode();

        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_output_tokens, Some(1000));
        assert_eq!(
            config.response_mime_type,
            Some("application/json".to_string())
        );
    }

    #[test]
    fn test_tool_config() {
        let config = ToolConfig::auto();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("AUTO"));
    }

    #[test]
    fn test_deserialize_response() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello!"}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        }"#;

        let resp: GenerateContentResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.candidates.len(), 1);
        assert_eq!(resp.candidates[0].finish_reason, Some("STOP".to_string()));
        assert_eq!(resp.usage_metadata.as_ref().unwrap().prompt_token_count, 10);
    }

    #[test]
    fn test_deserialize_function_call() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Let me search."},
                        {"functionCall": {"name": "search", "args": {"q": "rust"}}}
                    ]
                },
                "finishReason": "STOP"
            }]
        }"#;

        let resp: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let parts = &resp.candidates[0].content.as_ref().unwrap().parts;
        assert_eq!(parts.len(), 2);
    }
}

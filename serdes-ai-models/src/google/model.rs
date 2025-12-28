//! Google AI / Vertex AI model implementation.

use super::stream::GoogleStreamParser;
use super::types::*;
use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse, ToolChoice};
use crate::profile::ModelProfile;
use async_trait::async_trait;
use base64::Engine;
use reqwest::Client;
use serdes_ai_core::messages::{
    DocumentContent, ImageContent, RetryPromptPart, TextPart, ThinkingPart, ToolCallArgs,
    ToolCallPart, ToolReturnPart, UserContent, UserContentPart,
};
use serdes_ai_core::{
    FinishReason, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart, ModelSettings,
    RequestUsage,
};
use serdes_ai_tools::ToolDefinition;
use std::time::Duration;

/// Google AI / Vertex AI model.
#[derive(Debug, Clone)]
pub struct GoogleModel {
    model_name: String,
    client: Client,
    api_key: Option<String>,
    project_id: Option<String>,
    location: Option<String>,
    base_url: String,
    is_vertex: bool,
    profile: ModelProfile,
    default_timeout: Duration,
    /// Enable thinking mode.
    enable_thinking: bool,
    /// Thinking budget.
    thinking_budget: Option<u64>,
    /// Enable code execution.
    enable_code_execution: bool,
    /// Enable Google Search grounding.
    enable_search: bool,
}

impl GoogleModel {
    /// Create a new Google AI model (uses API key).
    pub fn new(model_name: impl Into<String>, api_key: impl Into<String>) -> Self {
        let model_name = model_name.into();
        let profile = Self::profile_for_model(&model_name);

        Self {
            model_name,
            client: Client::new(),
            api_key: Some(api_key.into()),
            project_id: None,
            location: None,
            base_url: "https://generativelanguage.googleapis.com".to_string(),
            is_vertex: false,
            profile,
            default_timeout: Duration::from_secs(120),
            enable_thinking: false,
            thinking_budget: None,
            enable_code_execution: false,
            enable_search: false,
        }
    }

    /// Create a Vertex AI model (uses project/location).
    pub fn vertex(
        model_name: impl Into<String>,
        project_id: impl Into<String>,
        location: impl Into<String>,
    ) -> Self {
        let model_name = model_name.into();
        let location = location.into();
        let project_id = project_id.into();
        let profile = Self::profile_for_model(&model_name);

        Self {
            base_url: format!(
                "https://{}-aiplatform.googleapis.com",
                location
            ),
            model_name,
            client: Client::new(),
            api_key: None,
            project_id: Some(project_id),
            location: Some(location),
            is_vertex: true,
            profile,
            default_timeout: Duration::from_secs(120),
            enable_thinking: false,
            thinking_budget: None,
            enable_code_execution: false,
            enable_search: false,
        }
    }

    /// Set the base URL.
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set a custom HTTP client.
    #[must_use]
    pub fn with_client(mut self, client: Client) -> Self {
        self.client = client;
        self
    }

    /// Set the default timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Set a custom profile.
    #[must_use]
    pub fn with_profile(mut self, profile: ModelProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Enable thinking mode (for Flash Thinking models).
    #[must_use]
    pub fn with_thinking(mut self, budget: Option<u64>) -> Self {
        self.enable_thinking = true;
        self.thinking_budget = budget;
        self.profile.supports_reasoning = true;
        self
    }

    /// Enable code execution.
    #[must_use]
    pub fn with_code_execution(mut self) -> Self {
        self.enable_code_execution = true;
        self
    }

    /// Enable Google Search grounding.
    #[must_use]
    pub fn with_search(mut self) -> Self {
        self.enable_search = true;
        self
    }

    /// Get the appropriate profile for a model name.
    fn profile_for_model(model: &str) -> ModelProfile {
        let mut profile = ModelProfile::default();
        profile.supports_tools = true;
        profile.supports_parallel_tools = true;
        profile.supports_system_messages = true;
        profile.supports_images = true;
        profile.supports_streaming = true;

        // Model-specific settings
        if model.contains("flash") {
            profile.max_tokens = Some(8192);
            profile.context_window = Some(1000000);
            if model.contains("thinking") {
                profile.supports_reasoning = true;
            }
        } else if model.contains("pro") {
            profile.max_tokens = Some(8192);
            profile.context_window = Some(2000000);
        } else if model.contains("ultra") {
            profile.max_tokens = Some(8192);
            profile.context_window = Some(1000000);
        }

        // Gemini 2.0+ supports native structured output
        if model.contains("gemini-2") || model.contains("gemini-exp") {
            profile.supports_native_structured_output = true;
            profile.supports_audio = true;
            profile.supports_video = true;
        }

        profile
    }

    /// Build the API URL.
    fn build_url(&self, stream: bool) -> String {
        let action = if stream {
            "streamGenerateContent"
        } else {
            "generateContent"
        };

        if self.is_vertex {
            let project = self.project_id.as_deref().unwrap_or("");
            let location = self.location.as_deref().unwrap_or("");
            format!(
                "{}/v1/projects/{}/locations/{}/publishers/google/models/{}:{}",
                self.base_url, project, location, self.model_name, action
            )
        } else {
            format!(
                "{}/v1beta/models/{}:{}?key={}",
                self.base_url,
                self.model_name,
                action,
                self.api_key.as_deref().unwrap_or("")
            )
        }
    }

    /// Convert our messages to Google format.
    fn convert_messages(
        &self,
        requests: &[ModelRequest],
    ) -> (Option<Content>, Vec<Content>) {
        let mut system_parts: Vec<String> = Vec::new();
        let mut contents: Vec<Content> = Vec::new();

        for req in requests {
            for part in &req.parts {
                match part {
                    ModelRequestPart::SystemPrompt(sys) => {
                        system_parts.push(sys.content.clone());
                    }
                    ModelRequestPart::UserPrompt(user) => {
                        let parts = self.convert_user_content(&user.content);
                        // Merge with previous user content if exists
                        if let Some(last) = contents.last_mut() {
                            if last.role == "user" {
                                last.parts.extend(parts);
                                continue;
                            }
                        }
                        contents.push(Content {
                            role: "user".to_string(),
                            parts,
                        });
                    }
                    ModelRequestPart::ToolReturn(ret) => {
                        let part = self.convert_tool_return(ret);
                        if let Some(last) = contents.last_mut() {
                            if last.role == "user" {
                                last.parts.push(part);
                                continue;
                            }
                        }
                        contents.push(Content {
                            role: "user".to_string(),
                            parts: vec![part],
                        });
                    }
                    ModelRequestPart::RetryPrompt(retry) => {
                        let part = self.convert_retry_prompt(retry);
                        if let Some(last) = contents.last_mut() {
                            if last.role == "user" {
                                last.parts.push(part);
                                continue;
                            }
                        }
                        contents.push(Content {
                            role: "user".to_string(),
                            parts: vec![part],
                        });
                    }
                    ModelRequestPart::BuiltinToolReturn(builtin) => {
                        // Convert builtin tool return to function response
                        let content_str = serde_json::to_string(&builtin.content)
                            .unwrap_or_else(|_| builtin.content_type().to_string());
                        let part = Part::FunctionResponse {
                            function_response: FunctionResponse {
                                name: builtin.tool_name.clone(),
                                response: serde_json::json!({ "result": content_str }),
                            },
                        };
                        if let Some(last) = contents.last_mut() {
                            if last.role == "user" {
                                last.parts.push(part);
                                continue;
                            }
                        }
                        contents.push(Content {
                            role: "user".to_string(),
                            parts: vec![part],
                        });
                    }
                    ModelRequestPart::ModelResponse(response) => {
                        // Add the assistant response for proper alternation
                        let mut parts = Vec::new();
                        for resp_part in &response.parts {
                            match resp_part {
                                ModelResponsePart::Text(text) => {
                                    parts.push(Part::Text { text: text.content.clone() });
                                }
                                ModelResponsePart::ToolCall(tc) => {
                                    parts.push(Part::FunctionCall {
                                        function_call: FunctionCall {
                                            name: tc.tool_name.clone(),
                                            args: tc.args.to_json(),
                                        },
                                    });
                                }
                                _ => {}
                            }
                        }
                        if !parts.is_empty() {
                            contents.push(Content {
                                role: "model".to_string(),
                                parts,
                            });
                        }
                    }
                }
            }
        }

        let system = if system_parts.is_empty() {
            None
        } else {
            Some(Content {
                role: "user".to_string(),
                parts: vec![Part::text(system_parts.join("\n\n"))],
            })
        };

        (system, contents)
    }

    /// Add a model response to contents (for multi-turn).
    pub fn add_response_to_contents(&self, contents: &mut Vec<Content>, response: &ModelResponse) {
        let mut parts = Vec::new();

        for part in &response.parts {
            match part {
                ModelResponsePart::Text(text) => {
                    parts.push(Part::text(&text.content));
                }
                ModelResponsePart::ToolCall(tc) => {
                    parts.push(Part::function_call(&tc.tool_name, tc.args.to_json()));
                }
                ModelResponsePart::Thinking(think) => {
                    parts.push(Part::Thought {
                        thought: think.content.clone(),
                    });
                }
                ModelResponsePart::File(_) => {
                    // File parts not directly supported in Gemini multi-turn
                }
                ModelResponsePart::BuiltinToolCall(_) => {
                    // Builtin tool calls handled by provider
                }
            }
        }

        if !parts.is_empty() {
            contents.push(Content::model_parts(parts));
        }
    }

    fn convert_user_content(&self, content: &UserContent) -> Vec<Part> {
        match content {
            UserContent::Text(text) => vec![Part::text(text)],
            UserContent::Parts(parts) => {
                parts.iter().filter_map(|p| self.convert_content_part(p)).collect()
            }
        }
    }

    fn convert_content_part(&self, part: &UserContentPart) -> Option<Part> {
        match part {
            UserContentPart::Text { text } => Some(Part::text(text)),
            UserContentPart::Image { image } => Some(self.convert_image(image)),
            UserContentPart::Document { document } => self.convert_document(document),
            _ => None,
        }
    }

    fn convert_image(&self, img: &ImageContent) -> Part {
        match img {
            ImageContent::Url(u) => Part::file_data(
                u.media_type
                    .as_ref()
                    .map(|m| m.mime_type())
                    .unwrap_or("image/jpeg"),
                &u.url,
            ),
            ImageContent::Binary(b) => Part::inline_data(
                b.media_type.mime_type(),
                base64::engine::general_purpose::STANDARD.encode(&b.data),
            ),
        }
    }

    fn convert_document(&self, doc: &DocumentContent) -> Option<Part> {
        match doc {
            DocumentContent::Binary(b) => Some(Part::inline_data(
                b.media_type.mime_type(),
                base64::engine::general_purpose::STANDARD.encode(&b.data),
            )),
            DocumentContent::Url(u) => Some(Part::file_data(
                u.media_type
                    .as_ref()
                    .map(|m| m.mime_type())
                    .unwrap_or("application/pdf"),
                &u.url,
            )),
        }
    }

    fn convert_tool_return(&self, ret: &ToolReturnPart) -> Part {
        let response = serde_json::json!({
            "result": ret.content.to_string_content()
        });
        Part::function_response(&ret.tool_name, response)
    }

    fn convert_retry_prompt(&self, retry: &RetryPromptPart) -> Part {
        if let Some(tool_name) = &retry.tool_name {
            let response = serde_json::json!({
                "error": retry.content.message()
            });
            Part::function_response(tool_name, response)
        } else {
            Part::text(retry.content.message())
        }
    }

    /// Convert tool definitions to Google format.
    fn convert_tools(&self, tools: &[ToolDefinition]) -> Vec<GoogleTool> {
        let mut google_tools = Vec::new();

        // Function declarations
        if !tools.is_empty() {
            let declarations: Vec<_> = tools
                .iter()
                .map(|t| {
                    let params =
                        serde_json::to_value(&t.parameters_json_schema).unwrap_or(serde_json::json!({}));
                    FunctionDeclaration::new(&t.name, &t.description, params)
                })
                .collect();
            google_tools.push(GoogleTool::functions(declarations));
        }

        // Code execution
        if self.enable_code_execution {
            google_tools.push(GoogleTool::code_execution());
        }

        // Google Search
        if self.enable_search {
            google_tools.push(GoogleTool::google_search());
        }

        google_tools
    }

    /// Convert tool choice.
    fn convert_tool_config(&self, choice: &ToolChoice) -> Option<ToolConfig> {
        match choice {
            ToolChoice::Auto => Some(ToolConfig::auto()),
            ToolChoice::Required => Some(ToolConfig::any()),
            ToolChoice::None => Some(ToolConfig::none()),
            ToolChoice::Specific(name) => Some(ToolConfig {
                function_calling_config: FunctionCallingConfig::specific(vec![name.clone()]),
            }),
        }
    }

    /// Build the request body.
    fn build_request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> GenerateContentRequest {
        let (system_instruction, contents) = self.convert_messages(messages);

        let tools = self.convert_tools(&params.tools);
        let tools = if tools.is_empty() { None } else { Some(tools) };

        let tool_config = params.tool_choice.as_ref().and_then(|c| self.convert_tool_config(c));

        // Build generation config
        let mut gen_config = GenerationConfig::new();
        if let Some(temp) = settings.temperature {
            gen_config = gen_config.temperature(temp);
        }
        if let Some(max) = settings.max_tokens {
            gen_config = gen_config.max_tokens(max);
        }
        if let Some(p) = settings.top_p {
            gen_config = gen_config.top_p(p);
        }
        if let Some(k) = settings.top_k {
            gen_config.top_k = Some(k);
        }
        if let Some(stops) = &settings.stop {
            gen_config.stop_sequences = Some(stops.clone());
        }

        // Structured output
        if let Some(schema) = &params.output_schema {
            let schema_value = serde_json::to_value(schema).unwrap_or(serde_json::json!({}));
            gen_config = gen_config.with_schema(schema_value);
        }

        // Thinking
        if self.enable_thinking {
            if let Some(budget) = self.thinking_budget {
                gen_config = gen_config.with_thinking(budget);
            }
        }

        GenerateContentRequest {
            contents,
            system_instruction,
            tools,
            tool_config,
            generation_config: Some(gen_config),
            safety_settings: None,
            cached_content: None,
        }
    }

    /// Parse Google response to our format.
    fn parse_response(&self, resp: GenerateContentResponse) -> Result<ModelResponse, ModelError> {
        // Check for blocked prompt
        if let Some(feedback) = &resp.prompt_feedback {
            if let Some(reason) = &feedback.block_reason {
                return Err(ModelError::ContentFiltered(format!(
                    "Prompt blocked: {}",
                    reason
                )));
            }
        }

        let candidate = resp
            .candidates
            .into_iter()
            .next()
            .ok_or_else(|| ModelError::invalid_response("No candidates in response"))?;

        let content = candidate
            .content
            .ok_or_else(|| ModelError::invalid_response("No content in candidate"))?;

        let mut parts = Vec::new();

        for part in content.parts {
            match part {
                Part::Text { text } => {
                    if !text.is_empty() {
                        parts.push(ModelResponsePart::Text(TextPart::new(text)));
                    }
                }
                Part::FunctionCall { function_call } => {
                    parts.push(ModelResponsePart::ToolCall(
                        ToolCallPart::new(function_call.name, ToolCallArgs::Json(function_call.args)),
                    ));
                }
                Part::Thought { thought } => {
                    parts.push(ModelResponsePart::Thinking(ThinkingPart::new(thought)));
                }
                Part::ExecutableCode { executable_code } => {
                    // Include code as text for now
                    parts.push(ModelResponsePart::Text(TextPart::new(format!(
                        "```{}\n{}\n```",
                        executable_code.language, executable_code.code
                    ))));
                }
                Part::CodeExecutionResult {
                    code_execution_result,
                } => {
                    parts.push(ModelResponsePart::Text(TextPart::new(format!(
                        "Code output: {}",
                        code_execution_result.output
                    ))));
                }
                _ => {}
            }
        }

        let finish_reason = candidate.finish_reason.map(|r| match r.as_str() {
            "STOP" => FinishReason::Stop,
            "MAX_TOKENS" => FinishReason::Length,
            "SAFETY" => FinishReason::ContentFilter,
            "RECITATION" => FinishReason::ContentFilter,
            "TOOL_CALLS" | "FUNCTION_CALL" => FinishReason::ToolCall,
            _ => FinishReason::Stop,
        });

        let usage = resp.usage_metadata.map(|u| RequestUsage {
            request_tokens: Some(u.prompt_token_count),
            response_tokens: Some(u.candidates_token_count),
            total_tokens: Some(u.total_token_count),
            cache_creation_tokens: None,
            cache_read_tokens: u.cached_content_token_count,
            details: None,
        });

        Ok(ModelResponse {
            parts,
            model_name: resp.model_version,
            timestamp: chrono::Utc::now(),
            finish_reason,
            usage,
            vendor_id: None,
            vendor_details: None,
            kind: "response".to_string(),
        })
    }

    /// Handle API error response.
    fn handle_error_response(&self, status: u16, body: &str) -> ModelError {
        if let Ok(err) = serde_json::from_str::<GoogleError>(body) {
            let code = err.error.status.clone();

            match status {
                401 | 403 => return ModelError::auth(err.error.message),
                429 => return ModelError::rate_limited(None),
                404 => return ModelError::NotFound(err.error.message),
                _ => {}
            }

            return ModelError::Api {
                message: err.error.message,
                code,
            };
        }

        ModelError::http(status, body)
    }
}

#[async_trait]
impl Model for GoogleModel {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn system(&self) -> &str {
        if self.is_vertex {
            "google-vertex"
        } else {
            "google"
        }
    }

    fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    async fn request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        let body = self.build_request(messages, settings, params);
        let url = self.build_url(false);

        let timeout = settings.timeout.unwrap_or(self.default_timeout);

        let mut request = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .timeout(timeout);

        // For Vertex AI, would need OAuth token
        // For now, API key is in URL for Google AI

        let response = request.json(&body).send().await?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(self.handle_error_response(status, &body));
        }

        let resp: GenerateContentResponse = response
            .json()
            .await
            .map_err(|e| ModelError::invalid_response(e.to_string()))?;

        self.parse_response(resp)
    }

    async fn request_stream(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<StreamedResponse, ModelError> {
        let body = self.build_request(messages, settings, params);
        let url = self.build_url(true);

        let timeout = settings.timeout.unwrap_or(self.default_timeout);

        let request = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .timeout(timeout);

        let response = request.json(&body).send().await?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(self.handle_error_response(status, &body));
        }

        let byte_stream = response.bytes_stream();
        let parser = GoogleStreamParser::new(byte_stream);

        Ok(Box::pin(parser))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_google_model_new() {
        let model = GoogleModel::new("gemini-2.0-flash", "test-key");
        assert_eq!(model.name(), "gemini-2.0-flash");
        assert_eq!(model.system(), "google");
        assert!(!model.is_vertex);
    }

    #[test]
    fn test_google_model_vertex() {
        let model = GoogleModel::vertex("gemini-2.0-flash", "my-project", "us-central1");
        assert_eq!(model.system(), "google-vertex");
        assert!(model.is_vertex);
    }

    #[test]
    fn test_google_model_builder() {
        let model = GoogleModel::new("gemini-2.0-flash", "key")
            .with_thinking(Some(10000))
            .with_code_execution()
            .with_search()
            .with_timeout(Duration::from_secs(60));

        assert!(model.enable_thinking);
        assert_eq!(model.thinking_budget, Some(10000));
        assert!(model.enable_code_execution);
        assert!(model.enable_search);
    }

    #[test]
    fn test_build_url_google_ai() {
        let model = GoogleModel::new("gemini-2.0-flash", "test-key");
        let url = model.build_url(false);
        assert!(url.contains("generativelanguage.googleapis.com"));
        assert!(url.contains("generateContent"));
        assert!(url.contains("test-key"));
    }

    #[test]
    fn test_build_url_vertex() {
        let model = GoogleModel::vertex("gemini-2.0-flash", "my-project", "us-central1");
        let url = model.build_url(false);
        assert!(url.contains("us-central1-aiplatform.googleapis.com"));
        assert!(url.contains("my-project"));
    }

    #[test]
    fn test_convert_tools() {
        use serdes_ai_tools::ObjectJsonSchema;

        let model = GoogleModel::new("gemini-2.0-flash", "key").with_code_execution();
        let tools = vec![
            ToolDefinition::new("search", "Search the web").with_parameters(ObjectJsonSchema::new()),
        ];

        let converted = model.convert_tools(&tools);
        assert_eq!(converted.len(), 2); // function + code_execution
    }

    #[test]
    fn test_convert_tool_config() {
        let model = GoogleModel::new("gemini-2.0-flash", "key");

        let auto = model.convert_tool_config(&ToolChoice::Auto);
        assert!(auto.is_some());

        let required = model.convert_tool_config(&ToolChoice::Required);
        assert!(required.is_some());
    }

    #[test]
    fn test_build_request() {
        let model = GoogleModel::new("gemini-2.0-flash", "key");
        let mut req = ModelRequest::new();
        req.add_system_prompt("You are helpful.");
        req.add_user_prompt("Hello!");
        let messages = vec![req];

        let settings = ModelSettings::new().temperature(0.7);
        let params = ModelRequestParameters::new();

        let request = model.build_request(&messages, &settings, &params);

        assert!(request.system_instruction.is_some());
        assert_eq!(request.contents.len(), 1);
        assert_eq!(
            request.generation_config.as_ref().unwrap().temperature,
            Some(0.7)
        );
    }

    #[test]
    fn test_parse_response() {
        let model = GoogleModel::new("gemini-2.0-flash", "key");

        let resp = GenerateContentResponse {
            candidates: vec![Candidate {
                content: Some(Content {
                    role: "model".to_string(),
                    parts: vec![Part::Text {
                        text: "Hello!".to_string(),
                    }],
                }),
                finish_reason: Some("STOP".to_string()),
                safety_ratings: None,
                citation_metadata: None,
                grounding_metadata: None,
                index: None,
            }],
            usage_metadata: Some(UsageMetadata {
                prompt_token_count: 10,
                candidates_token_count: 5,
                total_token_count: 15,
                cached_content_token_count: None,
                thoughts_token_count: None,
            }),
            model_version: Some("gemini-2.0-flash".to_string()),
            prompt_feedback: None,
        };

        let result = model.parse_response(resp).unwrap();

        assert_eq!(result.parts.len(), 1);
        assert!(matches!(&result.parts[0], ModelResponsePart::Text(t) if t.content == "Hello!"));
        assert!(matches!(result.finish_reason, Some(FinishReason::Stop)));
    }
}

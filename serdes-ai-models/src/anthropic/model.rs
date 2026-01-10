//! Anthropic Claude model implementation.

use super::stream::AnthropicStreamParser;
use super::types::*;
use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse, ToolChoice};
use crate::profile::{anthropic_claude_profile, ModelProfile};
use async_trait::async_trait;
use base64::Engine;
use reqwest::header::HeaderMap;
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

/// Anthropic Claude model.
#[derive(Debug, Clone)]
pub struct AnthropicModel {
    model_name: String,
    client: Client,
    api_key: String,
    base_url: String,
    profile: ModelProfile,
    default_timeout: Duration,
    /// Enable extended thinking.
    enable_thinking: bool,
    /// Thinking budget tokens.
    thinking_budget: Option<u64>,
    /// Enable prompt caching.
    enable_caching: bool,
    /// Anthropic API version.
    api_version: String,
}

impl AnthropicModel {
    /// Create a new Anthropic model.
    pub fn new(model_name: impl Into<String>, api_key: impl Into<String>) -> Self {
        let model_name = model_name.into();
        let profile = Self::profile_for_model(&model_name);

        Self {
            model_name,
            client: Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".to_string(),
            profile,
            default_timeout: Duration::from_secs(300), // Claude can be slow
            enable_thinking: false,
            thinking_budget: None,
            enable_caching: false,
            api_version: "2023-06-01".to_string(),
        }
    }

    /// Create from environment variable `ANTHROPIC_API_KEY`.
    pub fn from_env(model_name: impl Into<String>) -> Result<Self, ModelError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| ModelError::configuration("ANTHROPIC_API_KEY not set"))?;
        Ok(Self::new(model_name, api_key))
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

    /// Enable extended thinking.
    #[must_use]
    pub fn with_thinking(mut self, budget: Option<u64>) -> Self {
        self.enable_thinking = true;
        self.thinking_budget = budget;
        // Update profile
        self.profile.supports_reasoning = true;
        self
    }

    /// Enable prompt caching.
    #[must_use]
    pub fn with_caching(mut self) -> Self {
        self.enable_caching = true;
        self.profile.supports_caching = true;
        self
    }

    /// Set the API version.
    #[must_use]
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// Get the appropriate profile for a model name.
    fn profile_for_model(model: &str) -> ModelProfile {
        let mut profile = anthropic_claude_profile();

        // Adjust based on model
        if model.contains("sonnet") || model.contains("opus") {
            profile.supports_documents = true;
        }

        // Claude 3.5 Sonnet and newer support more features
        if model.contains("3-5") || model.contains("3.5") {
            profile.max_tokens = Some(8192);
            profile.context_window = Some(200000);
        }

        profile
    }

    /// Convert our messages to Anthropic format.
    /// Returns (system_content, messages).
    fn convert_messages(
        &self,
        requests: &[ModelRequest],
    ) -> (Option<SystemContent>, Vec<AnthropicMessage>) {
        let mut system_parts: Vec<String> = Vec::new();
        let mut api_messages: Vec<AnthropicMessage> = Vec::new();

        for req in requests {
            for part in &req.parts {
                match part {
                    ModelRequestPart::SystemPrompt(sys) => {
                        system_parts.push(sys.content.clone());
                    }
                    ModelRequestPart::UserPrompt(user) => {
                        let content = self.convert_user_content(&user.content);
                        // Anthropic requires alternating messages, merge if last was user
                        if let Some(last) = api_messages.last_mut() {
                            if last.role == "user" {
                                // Merge with previous user message
                                Self::merge_content(&mut last.content, content);
                                continue;
                            }
                        }
                        api_messages.push(AnthropicMessage {
                            role: "user".to_string(),
                            content,
                        });
                    }
                    ModelRequestPart::ToolReturn(ret) => {
                        let block = self.convert_tool_return(ret);
                        // Tool results go in user messages
                        if let Some(last) = api_messages.last_mut() {
                            if last.role == "user" {
                                Self::add_block_to_content(&mut last.content, block);
                                continue;
                            }
                        }
                        api_messages.push(AnthropicMessage {
                            role: "user".to_string(),
                            content: AnthropicContent::Blocks(vec![block]),
                        });
                    }
                    ModelRequestPart::RetryPrompt(retry) => {
                        let block = self.convert_retry_prompt(retry);
                        if let Some(last) = api_messages.last_mut() {
                            if last.role == "user" {
                                Self::add_block_to_content(&mut last.content, block);
                                continue;
                            }
                        }
                        api_messages.push(AnthropicMessage {
                            role: "user".to_string(),
                            content: AnthropicContent::Blocks(vec![block]),
                        });
                    }
                    ModelRequestPart::BuiltinToolReturn(builtin) => {
                        // Convert builtin tool return to a tool result block
                        let content_str = serde_json::to_string(&builtin.content)
                            .unwrap_or_else(|_| builtin.content_type().to_string());
                        let block = ContentBlock::ToolResult {
                            tool_use_id: builtin.tool_call_id.clone(),
                            content: Some(ToolResultContent::Text(content_str)),
                            is_error: None,
                        };
                        if let Some(last) = api_messages.last_mut() {
                            if last.role == "user" {
                                Self::add_block_to_content(&mut last.content, block);
                                continue;
                            }
                        }
                        api_messages.push(AnthropicMessage {
                            role: "user".to_string(),
                            content: AnthropicContent::Blocks(vec![block]),
                        });
                    }
                    ModelRequestPart::ModelResponse(response) => {
                        // Add the assistant response to messages for proper alternation
                        self.add_response_to_messages(&mut api_messages, response);
                    }
                }
            }
        }

        let system = if system_parts.is_empty() {
            None
        } else if self.enable_caching && system_parts.len() == 1 {
            Some(SystemContent::cached(
                system_parts.into_iter().next().unwrap(),
            ))
        } else {
            Some(SystemContent::text(system_parts.join("\n\n")))
        };

        (system, api_messages)
    }

    /// Add an assistant response to messages (for multi-turn).
    pub fn add_response_to_messages(
        &self,
        messages: &mut Vec<AnthropicMessage>,
        response: &ModelResponse,
    ) {
        let mut blocks = Vec::new();

        for part in &response.parts {
            match part {
                ModelResponsePart::Text(text) => {
                    blocks.push(ContentBlock::text(&text.content));
                }
                ModelResponsePart::ToolCall(tc) => {
                    blocks.push(ContentBlock::ToolUse {
                        id: tc.tool_call_id.clone().unwrap_or_default(),
                        name: tc.tool_name.clone(),
                        input: tc.args.to_json(),
                    });
                }
                ModelResponsePart::Thinking(think) => {
                    // Handle redacted vs regular thinking
                    if think.is_redacted() {
                        // Redacted thinking must be sent back with the signature
                        if let Some(sig) = &think.signature {
                            blocks.push(ContentBlock::RedactedThinking { data: sig.clone() });
                        }
                    } else {
                        blocks.push(ContentBlock::Thinking {
                            thinking: think.content.clone(),
                            signature: think.signature.clone(),
                        });
                    }
                }
                ModelResponsePart::File(_) => {
                    // Files are not sent back to the model in assistant messages
                }
                ModelResponsePart::BuiltinToolCall(_) => {
                    // Builtin tool calls are handled by the provider, not sent back
                }
            }
        }

        if !blocks.is_empty() {
            messages.push(AnthropicMessage::assistant_blocks(blocks));
        }
    }

    fn convert_user_content(&self, content: &UserContent) -> AnthropicContent {
        match content {
            UserContent::Text(text) => AnthropicContent::Text(text.clone()),
            UserContent::Parts(parts) => {
                let blocks: Vec<_> = parts
                    .iter()
                    .filter_map(|p| self.convert_content_part(p))
                    .collect();
                AnthropicContent::Blocks(blocks)
            }
        }
    }

    fn convert_content_part(&self, part: &UserContentPart) -> Option<ContentBlock> {
        match part {
            UserContentPart::Text { text } => Some(ContentBlock::text(text)),
            UserContentPart::Image { image } => Some(self.convert_image(image)),
            UserContentPart::Document { document } => self.convert_document(document),
            _ => None,
        }
    }

    fn convert_image(&self, img: &ImageContent) -> ContentBlock {
        let source = match img {
            ImageContent::Url(u) => ImageSource::url(&u.url),
            ImageContent::Binary(b) => ImageSource::base64(
                b.media_type.mime_type(),
                base64::engine::general_purpose::STANDARD.encode(&b.data),
            ),
        };
        ContentBlock::Image {
            source,
            cache_control: None,
        }
    }

    fn convert_document(&self, doc: &DocumentContent) -> Option<ContentBlock> {
        match doc {
            DocumentContent::Binary(b) => {
                let source = DocumentSource::base64(
                    b.media_type.mime_type(),
                    base64::engine::general_purpose::STANDARD.encode(&b.data),
                );
                Some(ContentBlock::Document {
                    source,
                    cache_control: if self.enable_caching {
                        Some(CacheControl::ephemeral())
                    } else {
                        None
                    },
                })
            }
            _ => None,
        }
    }

    fn convert_tool_return(&self, ret: &ToolReturnPart) -> ContentBlock {
        let content_str = ret.content.to_string_content();
        let is_error = ret.content.is_error();

        ContentBlock::ToolResult {
            tool_use_id: ret.tool_call_id.clone().unwrap_or_default(),
            content: Some(ToolResultContent::Text(content_str)),
            is_error: if is_error { Some(true) } else { None },
        }
    }

    fn convert_retry_prompt(&self, retry: &RetryPromptPart) -> ContentBlock {
        let content_str = retry.content.message().to_string();

        if let Some(tool_call_id) = &retry.tool_call_id {
            ContentBlock::ToolResult {
                tool_use_id: tool_call_id.clone(),
                content: Some(ToolResultContent::Text(content_str)),
                is_error: Some(true),
            }
        } else {
            ContentBlock::text(content_str)
        }
    }

    /// Merge content into existing content.
    fn merge_content(existing: &mut AnthropicContent, new: AnthropicContent) {
        match (&mut *existing, new) {
            (AnthropicContent::Text(ref mut s), AnthropicContent::Text(t)) => {
                s.push_str("\n\n");
                s.push_str(&t);
            }
            (AnthropicContent::Blocks(ref mut blocks), AnthropicContent::Blocks(new_blocks)) => {
                blocks.extend(new_blocks);
            }
            (AnthropicContent::Text(s), AnthropicContent::Blocks(new_blocks)) => {
                let mut blocks = vec![ContentBlock::text(s.clone())];
                blocks.extend(new_blocks);
                *existing = AnthropicContent::Blocks(blocks);
            }
            (AnthropicContent::Blocks(ref mut blocks), AnthropicContent::Text(t)) => {
                blocks.push(ContentBlock::text(t));
            }
        }
    }

    /// Add a block to content.
    fn add_block_to_content(content: &mut AnthropicContent, block: ContentBlock) {
        match content {
            AnthropicContent::Text(s) => {
                *content = AnthropicContent::Blocks(vec![ContentBlock::text(s.clone()), block]);
            }
            AnthropicContent::Blocks(blocks) => {
                blocks.push(block);
            }
        }
    }

    /// Convert tool definitions to Anthropic format.
    fn convert_tools(&self, tools: &[ToolDefinition]) -> Vec<AnthropicTool> {
        tools
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let schema = serde_json::to_value(&t.parameters_json_schema)
                    .unwrap_or(serde_json::json!({}));
                let mut tool = AnthropicTool::new(&t.name, &t.description, schema);

                // Cache the last tool definition for efficiency
                if self.enable_caching && i == tools.len() - 1 {
                    tool = tool.with_cache();
                }

                tool
            })
            .collect()
    }

    /// Convert tool choice.
    fn convert_tool_choice(&self, choice: &ToolChoice) -> Option<AnthropicToolChoice> {
        match choice {
            ToolChoice::Auto => Some(AnthropicToolChoice::Auto),
            ToolChoice::Required => Some(AnthropicToolChoice::Any),
            ToolChoice::None => None, // Anthropic doesn't have "none", just omit tools
            ToolChoice::Specific(name) => Some(AnthropicToolChoice::tool(name)),
        }
    }

    /// Build the request body.
    fn build_request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
        stream: bool,
    ) -> MessagesRequest {
        let (system, api_messages) = self.convert_messages(messages);

        let tools = if params.tools.is_empty() {
            None
        } else {
            Some(self.convert_tools(&params.tools))
        };

        let tool_choice = params
            .tool_choice
            .as_ref()
            .and_then(|c| self.convert_tool_choice(c));

        let thinking = if self.enable_thinking {
            Some(match self.thinking_budget {
                Some(budget) => ThinkingConfig::with_budget(budget),
                None => ThinkingConfig::enabled(),
            })
        } else {
            None
        };

        MessagesRequest {
            model: self.model_name.clone(),
            messages: api_messages,
            // Use profile max_tokens as default, or 16384 if not set
            max_tokens: settings
                .max_tokens
                .or(self.profile.max_tokens)
                .unwrap_or(16384),
            system,
            temperature: settings.temperature,
            top_p: settings.top_p,
            top_k: settings.top_k,
            stop_sequences: settings.stop.clone(),
            tools,
            tool_choice,
            metadata: None,
            stream: if stream { Some(true) } else { None },
            thinking,
        }
    }

    /// Parse Anthropic response to our format.
    fn parse_response(&self, resp: MessagesResponse) -> Result<ModelResponse, ModelError> {
        let mut parts = Vec::new();

        for block in resp.content {
            match block {
                ResponseContentBlock::Text { text } => {
                    parts.push(ModelResponsePart::Text(TextPart::new(text)));
                }
                ResponseContentBlock::ToolUse { id, name, input } => {
                    parts.push(ModelResponsePart::ToolCall(
                        ToolCallPart::new(name, ToolCallArgs::Json(input)).with_tool_call_id(id),
                    ));
                }
                ResponseContentBlock::Thinking {
                    thinking,
                    signature,
                } => {
                    let mut think = ThinkingPart::new(thinking);
                    if let Some(sig) = signature {
                        think = think.with_signature(sig);
                    }
                    parts.push(ModelResponsePart::Thinking(think));
                }
                ResponseContentBlock::RedactedThinking { data } => {
                    // Redacted thinking contains encrypted content - preserve the signature
                    parts.push(ModelResponsePart::Thinking(ThinkingPart::redacted(
                        data,
                        "anthropic",
                    )));
                }
            }
        }

        let finish_reason = resp.stop_reason.map(|r| match r.as_str() {
            "end_turn" => FinishReason::Stop,
            "stop_sequence" => FinishReason::Stop,
            "max_tokens" => FinishReason::Length,
            "tool_use" => FinishReason::ToolCall,
            _ => FinishReason::Stop,
        });

        let usage = RequestUsage {
            request_tokens: Some(resp.usage.input_tokens),
            response_tokens: Some(resp.usage.output_tokens),
            total_tokens: Some(resp.usage.input_tokens + resp.usage.output_tokens),
            cache_creation_tokens: resp.usage.cache_creation_input_tokens,
            cache_read_tokens: resp.usage.cache_read_input_tokens,
            details: None,
        };

        Ok(ModelResponse {
            parts,
            model_name: Some(resp.model),
            timestamp: chrono::Utc::now(),
            finish_reason,
            usage: Some(usage),
            vendor_id: Some(resp.id),
            vendor_details: None,
            kind: "response".to_string(),
        })
    }

    fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
        headers
            .get("retry-after")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
            .map(Duration::from_secs)
    }

    /// Handle API error response.
    fn handle_error_response(&self, status: u16, body: &str, headers: &HeaderMap) -> ModelError {
        if let Ok(err) = serde_json::from_str::<AnthropicError>(body) {
            let code = err.error.error_type.clone();

            match status {
                401 => return ModelError::auth(err.error.message),
                429 => return ModelError::rate_limited(Self::parse_retry_after(headers)),
                404 => return ModelError::NotFound(err.error.message),
                400 => {
                    if code == "invalid_request_error" {
                        return ModelError::Api {
                            message: err.error.message,
                            code: Some(code),
                        };
                    }
                }
                _ => {}
            }

            return ModelError::Api {
                message: err.error.message,
                code: Some(code),
            };
        }

        if status == 429 {
            return ModelError::rate_limited(Self::parse_retry_after(headers));
        }

        ModelError::http(status, body)
    }
}

#[async_trait]
impl Model for AnthropicModel {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn system(&self) -> &str {
        "anthropic"
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
        let body = self.build_request(messages, settings, params, false);

        let timeout = settings.timeout.unwrap_or(self.default_timeout);

        let mut request = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .header("Content-Type", "application/json")
            .timeout(timeout);

        // Add beta header for extended thinking
        if self.enable_thinking {
            request = request.header("anthropic-beta", "interleaved-thinking-2025-05-14");
        }

        // Add beta header for prompt caching
        if self.enable_caching {
            request = request.header("anthropic-beta", "prompt-caching-2024-07-31");
        }

        let response = request.json(&body).send().await?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let headers = response.headers().clone();
            let body = response.text().await.unwrap_or_default();
            return Err(self.handle_error_response(status, &body, &headers));
        }

        let resp: MessagesResponse = response
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
        let body = self.build_request(messages, settings, params, true);

        let timeout = settings.timeout.unwrap_or(self.default_timeout);

        let mut request = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .header("Content-Type", "application/json")
            .timeout(timeout);

        if self.enable_thinking {
            request = request.header("anthropic-beta", "interleaved-thinking-2025-05-14");
        }

        if self.enable_caching {
            request = request.header("anthropic-beta", "prompt-caching-2024-07-31");
        }

        let response = request.json(&body).send().await?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let headers = response.headers().clone();
            let body = response.text().await.unwrap_or_default();
            return Err(self.handle_error_response(status, &body, &headers));
        }

        let byte_stream = response.bytes_stream();
        let parser = AnthropicStreamParser::new(byte_stream);

        Ok(Box::pin(parser))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_model_new() {
        let model = AnthropicModel::new("claude-3-5-sonnet-20241022", "sk-test-key");
        assert_eq!(model.name(), "claude-3-5-sonnet-20241022");
        assert_eq!(model.system(), "anthropic");
    }

    #[test]
    fn test_anthropic_model_builder() {
        let model = AnthropicModel::new("claude-3-5-sonnet-20241022", "sk-test-key")
            .with_base_url("https://custom.api.com")
            .with_thinking(Some(10000))
            .with_caching()
            .with_timeout(Duration::from_secs(60));

        assert_eq!(model.base_url, "https://custom.api.com");
        assert!(model.enable_thinking);
        assert_eq!(model.thinking_budget, Some(10000));
        assert!(model.enable_caching);
        assert_eq!(model.default_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_convert_user_message() {
        let model = AnthropicModel::new("claude-3-5-sonnet-20241022", "key");
        let content = UserContent::text("Hello!");
        let converted = model.convert_user_content(&content);

        assert!(matches!(converted, AnthropicContent::Text(ref t) if t == "Hello!"));
    }

    #[test]
    fn test_convert_tools() {
        use serdes_ai_tools::ObjectJsonSchema;

        let model = AnthropicModel::new("claude-3-5-sonnet-20241022", "key");
        let tools = vec![ToolDefinition::new("search", "Search the web")
            .with_parameters(ObjectJsonSchema::new())];

        let converted = model.convert_tools(&tools);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].name, "search");
    }

    #[test]
    fn test_convert_tool_choice() {
        let model = AnthropicModel::new("claude-3-5-sonnet-20241022", "key");

        let auto = model.convert_tool_choice(&ToolChoice::Auto);
        assert!(matches!(auto, Some(AnthropicToolChoice::Auto)));

        let required = model.convert_tool_choice(&ToolChoice::Required);
        assert!(matches!(required, Some(AnthropicToolChoice::Any)));

        let specific = model.convert_tool_choice(&ToolChoice::Specific("search".to_string()));
        assert!(matches!(specific, Some(AnthropicToolChoice::Tool { name }) if name == "search"));

        let none = model.convert_tool_choice(&ToolChoice::None);
        assert!(none.is_none());
    }

    #[test]
    fn test_build_request() {
        let model = AnthropicModel::new("claude-3-5-sonnet-20241022", "key");
        let mut req = ModelRequest::new();
        req.add_system_prompt("You are helpful.");
        req.add_user_prompt("Hello!");
        let messages = vec![req];

        let settings = ModelSettings::new().temperature(0.7);
        let params = ModelRequestParameters::new();

        let request = model.build_request(&messages, &settings, &params, false);

        assert_eq!(request.model, "claude-3-5-sonnet-20241022");
        assert!(request.system.is_some());
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.temperature, Some(0.7));
        assert!(request.stream.is_none());
    }

    #[test]
    fn test_build_request_with_thinking() {
        let model =
            AnthropicModel::new("claude-3-5-sonnet-20241022", "key").with_thinking(Some(5000));

        let mut req = ModelRequest::new();
        req.add_user_prompt("Think about this.");
        let messages = vec![req];

        let settings = ModelSettings::new();
        let params = ModelRequestParameters::new();

        let request = model.build_request(&messages, &settings, &params, false);

        assert!(request.thinking.is_some());
        let thinking = request.thinking.unwrap();
        assert_eq!(thinking.thinking_type, "enabled");
        assert_eq!(thinking.budget_tokens, Some(5000));
    }

    #[test]
    fn test_merge_consecutive_user_messages() {
        let model = AnthropicModel::new("claude-3-5-sonnet-20241022", "key");

        let mut req1 = ModelRequest::new();
        req1.add_user_prompt("First message.");

        let mut req2 = ModelRequest::new();
        req2.add_user_prompt("Second message.");

        let messages = vec![req1, req2];
        let (_, api_messages) = model.convert_messages(&messages);

        // Should be merged into one message
        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0].role, "user");
    }

    #[test]
    fn test_parse_response() {
        let model = AnthropicModel::new("claude-3-5-sonnet-20241022", "key");

        let resp = MessagesResponse {
            id: "msg_123".to_string(),
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![ResponseContentBlock::Text {
                text: "Hello!".to_string(),
            }],
            model: "claude-3-5-sonnet-20241022".to_string(),
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };

        let result = model.parse_response(resp).unwrap();

        assert_eq!(result.parts.len(), 1);
        assert!(matches!(&result.parts[0], ModelResponsePart::Text(t) if t.content == "Hello!"));
        assert!(matches!(result.finish_reason, Some(FinishReason::Stop)));
        assert_eq!(result.usage.as_ref().unwrap().request_tokens, Some(10));
    }

    #[test]
    fn test_parse_tool_use_response() {
        let model = AnthropicModel::new("claude-3-5-sonnet-20241022", "key");

        let resp = MessagesResponse {
            id: "msg_123".to_string(),
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![
                ResponseContentBlock::Text {
                    text: "Let me search.".to_string(),
                },
                ResponseContentBlock::ToolUse {
                    id: "tool_1".to_string(),
                    name: "search".to_string(),
                    input: serde_json::json!({"query": "rust"}),
                },
            ],
            model: "claude-3-5-sonnet-20241022".to_string(),
            stop_reason: Some("tool_use".to_string()),
            stop_sequence: None,
            usage: AnthropicUsage::default(),
        };

        let result = model.parse_response(resp).unwrap();

        assert_eq!(result.parts.len(), 2);
        assert!(matches!(&result.parts[0], ModelResponsePart::Text(_)));
        assert!(matches!(&result.parts[1], ModelResponsePart::ToolCall(_)));
        assert!(matches!(result.finish_reason, Some(FinishReason::ToolCall)));
    }
}

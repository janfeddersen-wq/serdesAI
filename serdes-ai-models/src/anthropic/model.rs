//! Anthropic Claude model implementation.

use super::error::map_anthropic_error;
use super::stream::AnthropicStreamParser;
use super::types::*;
use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse, ToolChoice};
use crate::profile::{ModelProfile, anthropic_claude_profile};
use async_trait::async_trait;
use base64::Engine;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use reqwest::{Client, RequestBuilder};
use serdes_ai_core::messages::{
    DocumentContent, ImageContent, RetryPromptPart, TextPart, ThinkingPart, ToolCallArgs,
    ToolCallPart, ToolReturnPart, UserContent, UserContentPart,
};
use serdes_ai_core::{
    FinishReason, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart, ModelSettings,
    RequestUsage,
};
use serdes_ai_tools::ToolDefinition;
use std::fmt;
use std::time::Duration;

/// The `anthropic-beta` feature-flag header. Multi-valued by design.
const ANTHROPIC_BETA: HeaderName = HeaderName::from_static("anthropic-beta");

/// Placeholder rendered by `Debug` in place of a secret.
const REDACTED: &str = "<redacted>";

/// Convert a runtime string into a [`HeaderValue`].
///
/// The error names the header but never its value - the value may be a secret.
/// Embedding `e` is safe: `InvalidHeaderValue`'s `Display` is a constant string
/// that does not echo the input.
fn parse_header_value(name: &str, value: &str) -> Result<HeaderValue, ModelError> {
    HeaderValue::from_str(value)
        .map_err(|e| ModelError::configuration(format!("Invalid value for header '{name}': {e}")))
}

/// How a caller-supplied header is merged into the request headers.
#[derive(Debug, Clone)]
enum HeaderMergeMode {
    /// Replace every existing value of that header name.
    Set,
    /// Add a value, keeping any existing ones.
    Append,
    /// Remove every existing value of that header name. Carries no value.
    Remove,
}

/// One caller-supplied header. Named fields rather than a tuple: two `String`
/// slots side by side make a name/value transposition invisible to the compiler.
#[derive(Clone)]
struct ExtraHeader {
    name: String,
    value: String,
    mode: HeaderMergeMode,
}

/// Renders the NAME (diagnostic) but redacts the VALUE, which may be a secret.
impl fmt::Debug for ExtraHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtraHeader")
            .field("name", &self.name)
            .field("value", &REDACTED)
            .field("mode", &self.mode)
            .finish()
    }
}

/// Anthropic Claude model.
#[derive(Clone)]
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
    /// Caller-supplied headers, in call order.
    extra_headers: Vec<ExtraHeader>,
    /// Caller-supplied query parameters for the Messages endpoint, in call order.
    query_params: Vec<(String, String)>,
}

/// Hand-written so that secrets never reach a log or a panic message: the API
/// key and every caller header VALUE are redacted. Header NAMES stay visible -
/// they are diagnostic, not secret. Same discipline the error paths follow.
impl fmt::Debug for AnthropicModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicModel")
            .field("model_name", &self.model_name)
            .field("client", &self.client)
            .field("api_key", &REDACTED)
            .field("base_url", &self.base_url)
            .field("profile", &self.profile)
            .field("default_timeout", &self.default_timeout)
            .field("enable_thinking", &self.enable_thinking)
            .field("thinking_budget", &self.thinking_budget)
            .field("enable_caching", &self.enable_caching)
            .field("api_version", &self.api_version)
            .field("extra_headers", &self.extra_headers)
            .field(
                "query_params",
                &self
                    .query_params
                    .iter()
                    .map(|(name, _)| (name.as_str(), REDACTED))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
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
            extra_headers: Vec::new(),
            query_params: Vec::new(),
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

    /// Set a custom HTTP header on every request made by this model.
    ///
    /// Setting a header that the library also sets (`x-api-key`,
    /// `anthropic-version`, `Content-Type`, `anthropic-beta`) REPLACES the
    /// library's value. There is no protected header: the caller owns the
    /// request. Calling this twice with the same name keeps the LAST value.
    ///
    /// Use [`with_appended_header`](Self::with_appended_header) instead for
    /// multi-valued headers you want to ADD to rather than replace.
    ///
    /// An invalid header name or value is reported as a
    /// [`ModelError::Configuration`] when the request is built, since this
    /// builder cannot fail.
    ///
    /// # Security
    ///
    /// The name and the value are treated as TRUSTED caller configuration, not
    /// as end-user input. No header is protected, so forwarding user-controlled
    /// data into this method would let that user overwrite `x-api-key` and send
    /// requests under a credential of their choosing. Validate or allow-list
    /// anything that did not originate in your own configuration.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push(ExtraHeader {
            name: name.into(),
            value: value.into(),
            mode: HeaderMergeMode::Set,
        });
        self
    }

    /// Append a custom HTTP header value, keeping any existing values of the
    /// same name (including the library's own).
    ///
    /// This is for multi-valued headers - most notably `anthropic-beta`, which
    /// carries a list of feature flags. Appending `anthropic-beta` therefore
    /// ADDS a flag while leaving the library's own flags (extended thinking,
    /// prompt caching) in place.
    ///
    /// An invalid header name or value is reported as a
    /// [`ModelError::Configuration`] when the request is built, since this
    /// builder cannot fail.
    ///
    /// # Wrong for single-valued headers
    ///
    /// Appending a header that may carry only one value - `x-api-key` above
    /// all, but equally `anthropic-version` or `Content-Type` - does NOT
    /// override it. It sends the header TWICE, and which of the two values the
    /// server honors is server-dependent: a caller who appended `x-api-key`
    /// intending to swap credentials may silently keep authenticating with the
    /// old key. Use [`with_header`](Self::with_header) to override.
    ///
    /// # Security
    ///
    /// The name and the value are treated as TRUSTED caller configuration, not
    /// as end-user input. No header is protected, so forwarding user-controlled
    /// data into this method would let that user attach a second `x-api-key` to
    /// every request. Validate or allow-list anything that did not originate in
    /// your own configuration.
    #[must_use]
    pub fn with_appended_header(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.extra_headers.push(ExtraHeader {
            name: name.into(),
            value: value.into(),
            mode: HeaderMergeMode::Append,
        });
        self
    }

    /// Remove an HTTP header from every request made by this model.
    ///
    /// Removes ALL values of that name - including the library's own headers
    /// (`x-api-key`, `anthropic-version`, `Content-Type`, `anthropic-beta`) and
    /// any value an earlier [`with_header`](Self::with_header) or
    /// [`with_appended_header`](Self::with_appended_header) call added. Calls
    /// apply in order, so a later `with_header` of the same name adds it again.
    ///
    /// Typical use: authenticating with `Authorization: Bearer ...` via
    /// [`with_header`](Self::with_header) instead of `x-api-key`.
    ///
    /// An invalid header name is reported as a [`ModelError::Configuration`]
    /// when the request is built, since this builder cannot fail.
    #[must_use]
    pub fn without_header(mut self, name: impl Into<String>) -> Self {
        self.extra_headers.push(ExtraHeader {
            name: name.into(),
            value: String::new(),
            mode: HeaderMergeMode::Remove,
        });
        self
    }

    /// Append a query parameter to the Messages endpoint URL of every request
    /// made by this model (streaming and non-streaming).
    ///
    /// Parameters are sent in call order and URL-encoded; calling this twice
    /// with the same name sends both pairs. Without a call the URL carries no
    /// query string at all.
    #[must_use]
    pub fn with_query_param(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.query_params.push((name.into(), value.into()));
        self
    }

    /// Start a POST to the Messages endpoint - the single construction site
    /// shared by the streaming and non-streaming paths (URL, query, headers,
    /// timeout).
    fn messages_request(&self, timeout: Duration) -> Result<RequestBuilder, ModelError> {
        let request = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .headers(self.build_headers()?)
            .timeout(timeout);
        if self.query_params.is_empty() {
            return Ok(request);
        }
        Ok(request.query(&self.query_params))
    }

    /// Build the headers for a request - the single construction site shared by
    /// the streaming and non-streaming paths.
    ///
    /// Error messages name the offending header but NEVER its value: a header
    /// value may be a secret, and for `x-api-key` it always is. The underlying
    /// `http` errors are safe to embed - their `Display` never echoes the input.
    fn build_headers(&self) -> Result<HeaderMap, ModelError> {
        let mut headers = HeaderMap::new();

        let mut api_key = parse_header_value("x-api-key", &self.api_key)?;
        // Keeps the credential out of the HPACK dynamic table on HTTP/2.
        // Purely a transport hint: the value sent on the wire is unchanged.
        api_key.set_sensitive(true);
        headers.insert(HeaderName::from_static("x-api-key"), api_key);
        headers.insert(
            HeaderName::from_static("anthropic-version"),
            parse_header_value("anthropic-version", &self.api_version)?,
        );
        // `CONTENT_TYPE`, not `HeaderName::from_static("Content-Type")`, which
        // would panic on the uppercase bytes.
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        // Beta flags are multi-valued: the second one appends so that enabling
        // both thinking and caching still sends both.
        if self.enable_thinking {
            headers.append(
                ANTHROPIC_BETA,
                HeaderValue::from_static("interleaved-thinking-2025-05-14"),
            );
        }
        if self.enable_caching {
            headers.append(
                ANTHROPIC_BETA,
                HeaderValue::from_static("prompt-caching-2024-07-31"),
            );
        }

        for ExtraHeader { name, value, mode } in &self.extra_headers {
            // `escape_debug` on the NAME: this message reaches `error!()` and
            // `AgentStreamEvent::Error`, so a name carrying control characters
            // must not be able to forge a log line. The VALUE is never
            // interpolated at all - it may be a secret.
            let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                ModelError::configuration(format!(
                    "Invalid header name '{}': {e}",
                    name.escape_debug()
                ))
            })?;
            match mode {
                HeaderMergeMode::Set => {
                    headers.insert(header_name, parse_header_value(name, value)?);
                }
                HeaderMergeMode::Append => {
                    headers.append(header_name, parse_header_value(name, value)?);
                }
                HeaderMergeMode::Remove => {
                    headers.remove(header_name);
                }
            }
        }

        Ok(headers)
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
            (AnthropicContent::Text(s), AnthropicContent::Text(t)) => {
                s.push_str("\n\n");
                s.push_str(&t);
            }
            (AnthropicContent::Blocks(blocks), AnthropicContent::Blocks(new_blocks)) => {
                blocks.extend(new_blocks);
            }
            (AnthropicContent::Text(s), AnthropicContent::Blocks(new_blocks)) => {
                let mut blocks = vec![ContentBlock::text(s.clone())];
                blocks.extend(new_blocks);
                *existing = AnthropicContent::Blocks(blocks);
            }
            (AnthropicContent::Blocks(blocks), AnthropicContent::Text(t)) => {
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
        let retry_after = Self::parse_retry_after(headers);

        if let Ok(err) = serde_json::from_str::<AnthropicError>(body) {
            return map_anthropic_error(
                err.error.error_type,
                err.error.message,
                retry_after,
                Some(status),
            );
        }

        if status == 429 {
            return ModelError::rate_limited(retry_after);
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

        let request = self.messages_request(timeout)?;

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

        let request = self.messages_request(timeout)?;

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
    fn test_handle_error_response_uses_provider_semantics() {
        let model = AnthropicModel::new("claude-3-5-sonnet-20241022", "key");
        let mut headers = HeaderMap::new();
        headers.insert("retry-after", "7".parse().unwrap());

        let rate_limit = model.handle_error_response(
            429,
            r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#,
            &headers,
        );
        assert!(rate_limit.is_rate_limited());
        assert_eq!(rate_limit.retry_after(), Some(Duration::from_secs(7)));

        let overloaded = model.handle_error_response(
            529,
            r#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#,
            &HeaderMap::new(),
        );
        assert!(overloaded.is_transient());

        let auth = model.handle_error_response(
            401,
            r#"{"type":"error","error":{"type":"authentication_error","message":"bad key"}}"#,
            &HeaderMap::new(),
        );
        assert!(!auth.is_retryable());
    }

    #[tokio::test]
    async fn retrying_model_retries_concrete_anthropic_transport() {
        use crate::{RetryPolicy, RetryingModel, WaitStrategy};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};
        use wiremock::{Request, Respond, ResponseTemplate};

        #[derive(Clone)]
        struct RateLimitThenSuccess {
            calls: Arc<AtomicU32>,
        }

        impl Respond for RateLimitThenSuccess {
            fn respond(&self, _request: &Request) -> ResponseTemplate {
                if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    ResponseTemplate::new(429)
                        .insert_header("retry-after", "0")
                        .set_body_raw(
                            r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#,
                            "application/json",
                        )
                } else {
                    ResponseTemplate::new(200)
                        .set_body_raw(MESSAGES_SUCCESS_BODY, "application/json")
                }
            }
        }

        let server = wiremock::MockServer::start().await;
        let calls = Arc::new(AtomicU32::new(0));
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .respond_with(RateLimitThenSuccess {
                calls: calls.clone(),
            })
            .mount(&server)
            .await;

        let inner = AnthropicModel::new("claude-test", "key").with_base_url(server.uri());
        let model = RetryingModel::new(
            inner,
            RetryPolicy::for_model_requests()
                .max_attempts(2)
                .wait(WaitStrategy::None)
                .total_timeout(Some(Duration::from_secs(5))),
        );

        let response = model
            .request(
                &[ModelRequest::new()],
                &ModelSettings::default(),
                &ModelRequestParameters::new(),
            )
            .await
            .unwrap();

        assert_eq!(response.parts.len(), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

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
        let tools = vec![
            ToolDefinition::new("search", "Search the web")
                .with_parameters(ObjectJsonSchema::new()),
        ];

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

    // -----------------------------------------------------------------
    // Custom headers - AC1..AC10 of
    // specs/20260803_feat_anthropic_with_header/TEST_PLAN.md
    // -----------------------------------------------------------------

    const MESSAGES_SUCCESS_BODY: &str = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"model":"claude-test","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}"#;

    /// Mock server answering `POST /v1/messages` with a minimal success body.
    async fn messages_mock_server() -> wiremock::MockServer {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_raw(MESSAGES_SUCCESS_BODY, "application/json"),
            )
            .mount(&server)
            .await;
        server
    }

    /// The single request the mock server received.
    async fn only_received_request(server: &wiremock::MockServer) -> wiremock::Request {
        let mut requests = server
            .received_requests()
            .await
            .expect("mock server records requests");
        assert_eq!(requests.len(), 1, "expected exactly one captured request");
        requests.remove(0)
    }

    /// All values sent for `name`, in order. Empty when the header is absent.
    ///
    /// A non-UTF-8 value renders as a placeholder rather than panicking, so a
    /// future test with a latin-1 value fails on its own assertion instead of
    /// dying inside this helper.
    fn header_values(headers: &HeaderMap, name: &str) -> Vec<String> {
        headers
            .get_all(name)
            .iter()
            .map(|value| value.to_str().unwrap_or("<non-utf8>").to_owned())
            .collect()
    }

    /// Drive one non-streaming request and return the request actually sent.
    async fn sent_non_streaming_request(model: AnthropicModel) -> wiremock::Request {
        let server = messages_mock_server().await;
        model
            .with_base_url(server.uri())
            .request(
                &[ModelRequest::new()],
                &ModelSettings::default(),
                &ModelRequestParameters::new(),
            )
            .await
            .expect("mocked request succeeds");
        only_received_request(&server).await
    }

    /// Drive one streaming request and return the request actually sent.
    ///
    /// `request_stream` only checks the status before handing back the parser,
    /// so the body is never polled here - the request is already captured.
    async fn sent_streaming_request(model: AnthropicModel) -> wiremock::Request {
        let server = messages_mock_server().await;
        let _stream = model
            .with_base_url(server.uri())
            .request_stream(
                &[ModelRequest::new()],
                &ModelSettings::default(),
                &ModelRequestParameters::new(),
            )
            .await
            .expect("mocked stream request succeeds");
        only_received_request(&server).await
    }

    /// Drive one non-streaming request and return the headers actually sent.
    async fn headers_of_non_streaming_request(model: AnthropicModel) -> HeaderMap {
        sent_non_streaming_request(model).await.headers
    }

    /// Drive one streaming request and return the headers actually sent.
    async fn headers_of_streaming_request(model: AnthropicModel) -> HeaderMap {
        sent_streaming_request(model).await.headers
    }

    /// AC6 - default behavior: the three fixed headers, and no `anthropic-beta`.
    /// Transport headers added by reqwest (host, accept, content-length) are
    /// deliberately not asserted on.
    #[tokio::test]
    async fn no_custom_headers_leaves_request_unchanged() {
        let headers =
            headers_of_non_streaming_request(AnthropicModel::new("claude-test", "sk-test")).await;

        assert_eq!(header_values(&headers, "x-api-key"), vec!["sk-test"]);
        assert_eq!(
            header_values(&headers, "anthropic-version"),
            vec!["2023-06-01"]
        );
        assert_eq!(
            header_values(&headers, "content-type"),
            vec!["application/json"]
        );
        assert!(
            header_values(&headers, "anthropic-beta").is_empty(),
            "no beta flag must be sent by default"
        );
    }

    /// AC9 - regression guard for the `build_headers()` consolidation: thinking
    /// and caching must BOTH still emit their `anthropic-beta` flag.
    #[tokio::test]
    async fn thinking_and_caching_still_send_both_beta_flags() {
        let model = AnthropicModel::new("claude-test", "sk-test")
            .with_thinking(Some(1024))
            .with_caching();

        let headers = headers_of_non_streaming_request(model).await;

        assert_eq!(
            header_values(&headers, "anthropic-beta"),
            vec![
                "interleaved-thinking-2025-05-14",
                "prompt-caching-2024-07-31"
            ]
        );
    }

    /// AC1 - a custom header reaches the non-streaming request.
    #[tokio::test]
    async fn with_header_appears_in_non_streaming_request() {
        let model = AnthropicModel::new("claude-test", "sk-test").with_header("x-client", "silix");

        let headers = headers_of_non_streaming_request(model).await;

        assert_eq!(header_values(&headers, "x-client"), vec!["silix"]);
    }

    /// AC2 - and the streaming request, which is the consumer's normal case.
    #[tokio::test]
    async fn with_header_appears_in_streaming_request() {
        let model = AnthropicModel::new("claude-test", "sk-test").with_header("x-client", "silix");

        let headers = headers_of_streaming_request(model).await;

        assert_eq!(header_values(&headers, "x-client"), vec!["silix"]);
    }

    /// AC3 - no header is protected: `x-api-key` is replaced, and sent EXACTLY
    /// once. The count matters - appending would send it twice and only the
    /// count catches that.
    #[tokio::test]
    async fn with_header_overwrites_api_key_exactly_once() {
        let model =
            AnthropicModel::new("claude-test", "sk-original").with_header("x-api-key", "override");

        let headers = headers_of_non_streaming_request(model).await;

        assert_eq!(header_values(&headers, "x-api-key"), vec!["override"]);
    }

    /// AC4 - `with_header` means SET: it replaces the library's own
    /// `anthropic-beta` flags. No special case, no inferred intent.
    #[tokio::test]
    async fn with_header_replaces_anthropic_beta() {
        let model = AnthropicModel::new("claude-test", "sk-test")
            .with_caching()
            .with_header("anthropic-beta", "only-mine");

        let headers = headers_of_non_streaming_request(model).await;

        assert_eq!(header_values(&headers, "anthropic-beta"), vec!["only-mine"]);
    }

    /// AC5 - `with_appended_header` means ADD: the library's flags survive.
    #[tokio::test]
    async fn with_appended_header_adds_to_anthropic_beta() {
        let model = AnthropicModel::new("claude-test", "sk-test")
            .with_caching()
            .with_appended_header("anthropic-beta", "my-flag");

        let headers = headers_of_non_streaming_request(model).await;

        assert_eq!(
            header_values(&headers, "anthropic-beta"),
            vec!["prompt-caching-2024-07-31", "my-flag"]
        );
    }

    /// AC7 - setting the same header twice keeps the last value only.
    #[tokio::test]
    async fn with_header_last_value_wins() {
        let model = AnthropicModel::new("claude-test", "sk-test")
            .with_header("a", "1")
            .with_header("a", "2");

        let headers = headers_of_non_streaming_request(model).await;

        assert_eq!(header_values(&headers, "a"), vec!["2"]);
    }

    /// AC10 - the ORDER of set/append calls is honored, in both directions.
    #[tokio::test]
    async fn set_and_append_order_is_honored() {
        let append_then_set = AnthropicModel::new("claude-test", "sk-test")
            .with_appended_header("x-m", "1")
            .with_header("x-m", "2");
        let headers = headers_of_non_streaming_request(append_then_set).await;
        assert_eq!(
            header_values(&headers, "x-m"),
            vec!["2"],
            "a later set must replace the earlier appended value"
        );

        let set_then_append = AnthropicModel::new("claude-test", "sk-test")
            .with_header("x-m", "2")
            .with_appended_header("x-m", "1");
        let headers = headers_of_non_streaming_request(set_then_append).await;
        assert_eq!(
            header_values(&headers, "x-m"),
            vec!["2", "1"],
            "a later append must keep the earlier set value"
        );
    }

    /// H1 - `without_header` removes a library header from BOTH paths, case-
    /// insensitively, and also removes a value an earlier `with_header` set.
    /// The other library headers stay untouched.
    #[tokio::test]
    async fn without_header_removes_library_header() {
        let removed = || AnthropicModel::new("claude-test", "sk-test").without_header("x-api-key");

        for headers in [
            headers_of_non_streaming_request(removed()).await,
            headers_of_streaming_request(removed()).await,
        ] {
            assert!(
                header_values(&headers, "x-api-key").is_empty(),
                "x-api-key must not be sent after without_header"
            );
            assert_eq!(
                header_values(&headers, "anthropic-version"),
                vec!["2023-06-01"]
            );
            assert_eq!(
                header_values(&headers, "content-type"),
                vec!["application/json"]
            );
        }

        let set_then_removed = AnthropicModel::new("claude-test", "sk-test")
            .with_header("x-api-key", "override")
            .without_header("X-Api-Key");
        let headers = headers_of_non_streaming_request(set_then_removed).await;
        assert!(
            header_values(&headers, "x-api-key").is_empty(),
            "without_header must also remove a value set by an earlier with_header"
        );
    }

    /// H1 - removal follows call order: a later `with_header` re-adds the name,
    /// and removing a multi-valued header drops every value.
    #[tokio::test]
    async fn without_header_honors_call_order() {
        let removed_then_set = AnthropicModel::new("claude-test", "sk-test")
            .without_header("x-api-key")
            .with_header("x-api-key", "again");
        let headers = headers_of_non_streaming_request(removed_then_set).await;
        assert_eq!(header_values(&headers, "x-api-key"), vec!["again"]);

        let all_beta_removed = AnthropicModel::new("claude-test", "sk-test")
            .with_thinking(Some(1024))
            .with_caching()
            .with_appended_header("anthropic-beta", "my-flag")
            .without_header("anthropic-beta");
        let headers = headers_of_non_streaming_request(all_beta_removed).await;
        assert!(header_values(&headers, "anthropic-beta").is_empty());
    }

    /// H1 - an invalid name passed to `without_header` is a configuration
    /// error like every other caller header, not a silent no-op.
    #[test]
    fn without_header_invalid_name_is_configuration_error() {
        let model = AnthropicModel::new("claude-test", "sk-test").without_header("bad name");

        let error = model.build_headers().expect_err("invalid name must error");

        assert!(matches!(error, ModelError::Configuration(_)));
        assert!(error.to_string().contains("bad name"));
    }

    /// H2 - query parameters reach BOTH messages paths, in call order and
    /// URL-encoded; without a call the URL carries no query at all.
    #[tokio::test]
    async fn with_query_param_reaches_both_paths() {
        let with_beta =
            || AnthropicModel::new("claude-test", "sk-test").with_query_param("beta", "true");

        for request in [
            sent_non_streaming_request(with_beta()).await,
            sent_streaming_request(with_beta()).await,
        ] {
            assert_eq!(request.url.path(), "/v1/messages");
            assert_eq!(request.url.query(), Some("beta=true"));
        }

        let several = AnthropicModel::new("claude-test", "sk-test")
            .with_query_param("beta", "true")
            .with_query_param("note", "a b&c=d");
        let request = sent_non_streaming_request(several).await;
        assert_eq!(request.url.query(), Some("beta=true&note=a+b%26c%3Dd"));
        let decoded: Vec<(String, String)> = request.url.query_pairs().into_owned().collect();
        assert_eq!(
            decoded,
            vec![
                ("beta".to_owned(), "true".to_owned()),
                ("note".to_owned(), "a b&c=d".to_owned()),
            ]
        );

        for request in [
            sent_non_streaming_request(AnthropicModel::new("claude-test", "sk-test")).await,
            sent_streaming_request(AnthropicModel::new("claude-test", "sk-test")).await,
        ] {
            assert_eq!(request.url.path(), "/v1/messages");
            assert_eq!(request.url.query(), None, "no query without a call");
        }
    }

    /// AC8 - an invalid header NAME is a real configuration error naming the
    /// offending header, not a silent skip.
    #[test]
    fn invalid_header_name_is_configuration_error() {
        let model = AnthropicModel::new("claude-test", "sk-test").with_header("bad name", "v");

        let error = model.build_headers().expect_err("invalid name must error");

        assert!(
            matches!(error, ModelError::Configuration(_)),
            "expected a Configuration error, got {error:?}"
        );
        assert!(
            error.to_string().contains("bad name"),
            "message must name the offending header, got {error}"
        );
    }

    /// AC8b - an invalid header VALUE errors, names the header, and must NOT
    /// echo the value: a custom header may carry a secret.
    #[test]
    fn invalid_header_value_error_does_not_leak_value() {
        let model =
            AnthropicModel::new("claude-test", "sk-test").with_header("x-tenant", "secret\nvalue");

        let error = model.build_headers().expect_err("invalid value must error");

        assert!(
            matches!(error, ModelError::Configuration(_)),
            "expected a Configuration error, got {error:?}"
        );
        let message = error.to_string();
        assert!(
            message.contains("x-tenant"),
            "message must name the offending header, got {message}"
        );
        assert!(
            !message.contains("secret"),
            "message must never echo the header value, got {message}"
        );
    }

    /// AC8b (append path) - the same guarantees hold for appended headers.
    #[test]
    fn invalid_appended_header_value_does_not_leak_value() {
        let model = AnthropicModel::new("claude-test", "sk-test")
            .with_appended_header("x-tenant", "secret\nvalue");

        let error = model.build_headers().expect_err("invalid value must error");

        assert!(matches!(error, ModelError::Configuration(_)));
        let message = error.to_string();
        assert!(message.contains("x-tenant"));
        assert!(!message.contains("secret"));
    }

    /// AC8c - the fixed headers are fallible too: an API key with a trailing
    /// newline (the realistic env/file case) errors WITHOUT leaking the key.
    #[test]
    fn invalid_api_key_is_configuration_error_without_leaking_key() {
        let model = AnthropicModel::new("claude-test", "sk-leakcanary-\n");

        let error = model
            .build_headers()
            .expect_err("invalid api key must error");

        assert!(
            matches!(error, ModelError::Configuration(_)),
            "expected a Configuration error, got {error:?}"
        );
        let message = error.to_string();
        assert!(
            message.contains("x-api-key"),
            "message must name the offending header, got {message}"
        );
        assert!(
            !message.contains("leakcanary"),
            "message must never echo the API key, got {message}"
        );
    }

    /// AC8c (version path) - same discipline for `anthropic-version`.
    #[test]
    fn invalid_api_version_is_configuration_error() {
        let model = AnthropicModel::new("claude-test", "sk-test").with_api_version("2023-06-01\n");

        let error = model
            .build_headers()
            .expect_err("invalid api version must error");

        assert!(matches!(error, ModelError::Configuration(_)));
        assert!(error.to_string().contains("anthropic-version"));
    }

    /// AC8d - `build_headers()` must not panic for a default model. Guards the
    /// `HeaderName::from_static` uppercase hazard: `from_static("Content-Type")`
    /// would panic on every single request.
    #[test]
    fn build_headers_does_not_panic_for_default_model() {
        let headers = AnthropicModel::new("claude-test", "sk-test")
            .build_headers()
            .expect("default model builds headers");

        assert_eq!(
            header_values(&headers, "content-type"),
            vec!["application/json"]
        );
    }

    /// A caller header name is NORMALIZED to lowercase, as `HeaderName::from_bytes`
    /// guarantees. Pins the guarantee against a future refactor to `from_static`,
    /// which would panic on the uppercase bytes instead.
    #[tokio::test]
    async fn caller_header_name_is_normalized_to_lowercase() {
        let model = AnthropicModel::new("claude-test", "sk-test").with_header("X-Client", "silix");

        let headers = headers_of_non_streaming_request(model).await;

        assert_eq!(header_values(&headers, "x-client"), vec!["silix"]);
    }

    /// Fail-closed: when `build_headers()` fails, NO request goes out. Without
    /// this the caller's intended auth/tenant header could be silently dropped
    /// while the request still reaches the API.
    #[tokio::test]
    async fn invalid_header_prevents_the_request_from_being_sent() {
        let server = messages_mock_server().await;
        let model = AnthropicModel::new("claude-test", "sk-test")
            .with_base_url(server.uri())
            .with_header("bad name", "v");

        let error = model
            .request(
                &[ModelRequest::new()],
                &ModelSettings::default(),
                &ModelRequestParameters::new(),
            )
            .await
            .expect_err("an invalid header must fail the request");

        assert!(
            matches!(error, ModelError::Configuration(_)),
            "expected a Configuration error, got {error:?}"
        );
        let requests = server
            .received_requests()
            .await
            .expect("mock server records requests");
        assert_eq!(
            requests.len(),
            0,
            "no request may reach the API when header construction fails"
        );
    }

    /// The `Debug` rendering must not leak secrets: neither the API key nor any
    /// caller header VALUE. Header NAMES stay visible - they are diagnostic.
    #[test]
    fn debug_redacts_api_key_and_extra_header_values() {
        let model = AnthropicModel::new("claude-test", "sk-leakcanary")
            .with_header("x-tenant", "tenantcanary")
            .with_appended_header("x-trace", "tracecanary");

        let rendered = format!("{model:?}");

        assert!(
            !rendered.contains("sk-leakcanary"),
            "Debug must not echo the api key, got {rendered}"
        );
        assert!(
            !rendered.contains("tenantcanary"),
            "Debug must not echo a set header value, got {rendered}"
        );
        assert!(
            !rendered.contains("tracecanary"),
            "Debug must not echo an appended header value, got {rendered}"
        );
        assert!(
            rendered.contains("x-tenant") && rendered.contains("x-trace"),
            "header names must stay visible for diagnostics, got {rendered}"
        );
        assert!(
            rendered.contains("claude-test") && rendered.contains("2023-06-01"),
            "non-secret fields must still be rendered, got {rendered}"
        );
    }

    /// The invalid header NAME is escaped before it reaches an error message:
    /// the message flows into `error!()` and `AgentStreamEvent::Error`, so a
    /// name with control characters must not be able to forge a log line.
    #[test]
    fn invalid_header_name_is_escaped_in_the_error_message() {
        let model =
            AnthropicModel::new("claude-test", "sk-test").with_header("bad\nname: injected", "v");

        let error = model.build_headers().expect_err("invalid name must error");

        let message = error.to_string();
        assert!(
            !message.contains('\n'),
            "the error message must stay single-line, got {message:?}"
        );
        assert!(
            message.contains("bad\\nname"),
            "the name must appear escaped, got {message:?}"
        );
    }
}

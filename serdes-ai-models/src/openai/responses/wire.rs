//! Open Responses wire model.
//!
//! These types model the subset of the OpenAI Responses API defined by the
//! Open Responses specification (openresponses.org): request objects for
//! `POST /v1/responses` (HTTP and websocket `response.create` frames),
//! response objects, input/output items, content parts, function tools, and
//! error envelopes. The streaming event model lives in [`super::events`].

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// Requests
// ============================================================================

/// A `POST /v1/responses` request body (also carried by websocket
/// `response.create` frames, minus the HTTP-specific fields).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateResponseRequest {
    /// Model name requested by the client.
    ///
    /// The server serves the model it was configured with; the requested name
    /// is echoed back in the response object rather than being used for
    /// routing.
    pub model: String,

    /// Conversation input: a plain string or a list of input items.
    #[serde(default)]
    pub input: ResponseInput,

    /// System/developer instructions, prepended to the history.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Tools the model may call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponsesTool>>,

    /// Tool selection strategy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ResponsesToolChoice>,

    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Nucleus sampling parameter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Maximum number of output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,

    /// Whether to stream events (HTTP only; must be absent on websockets).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Background mode (unsupported; rejected when `true` so clients do not
    /// silently wait on a job that will never be queued).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,

    /// Whether to persist the response for later retrieval and chaining.
    /// Defaults to `true`, matching the OpenAI API.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    /// Chain this turn onto a previously stored response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    /// Reasoning configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningSettings>,

    /// Whether tool calls may run in parallel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Client metadata echoed back on the response object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, Value>>,

    /// Client user identifier, accepted and ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Truncation strategy (accepted for compatibility, not interpreted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    /// Include directives for extra output payloads (not interpreted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    /// Text formatting options (not interpreted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<Value>,
    /// Requested service tier (not interpreted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

/// Request input: either a plain string or a list of structured items.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    /// Plain text input, equivalent to a single user message.
    Text(String),
    /// Structured input items.
    Items(Vec<InputItem>),
}

impl Default for ResponseInput {
    fn default() -> Self {
        Self::Items(Vec::new())
    }
}

/// A single conversation input item.
///
/// Accepts both fully-typed items (`{"type": "message", ...}`) and OpenAI's
/// "easy" message shape (`{"role": "user", "content": "..."}` without a
/// `type` field).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputItem {
    /// A fully-typed input item.
    Typed(TypedInputItem),
    /// An "easy input message" without a `type` field.
    Easy(EasyInputMessage),
}

/// A typed input item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TypedInputItem {
    /// A message in the conversation.
    Message {
        /// Message role.
        role: InputRole,
        /// Message content: plain string or structured parts.
        #[serde(default)]
        content: Option<InputMessageContent>,
    },
    /// A function call previously produced by the model.
    FunctionCall {
        /// Client-visible call identifier.
        call_id: String,
        /// Tool name.
        name: String,
        /// JSON-encoded arguments.
        arguments: String,
    },
    /// The result of a function call, supplied by the client.
    FunctionCallOutput {
        /// The call this output belongs to.
        call_id: String,
        /// Tool output as a string (often JSON-encoded).
        output: String,
    },
    /// Reasoning content replayed back to the model.
    Reasoning {
        /// Server-assigned reasoning item ID.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Reasoning summary parts.
        #[serde(default)]
        summary: Vec<SummaryTextItem>,
        /// Encrypted reasoning content (Responses-native models).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
    /// A reference to a stored item (not resolvable by this server).
    ItemReference {
        /// Referenced item ID.
        id: String,
    },
}

/// An "easy input message": `{"role": ..., "content": ...}` without `type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EasyInputMessage {
    /// Message role.
    pub role: InputRole,
    /// Message content.
    #[serde(default)]
    pub content: Option<InputMessageContent>,
}

/// Message role on the input side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputRole {
    /// System instructions.
    System,
    /// Developer instructions.
    Developer,
    /// User turn.
    User,
    /// Assistant turn.
    Assistant,
}

/// Message content: plain string or structured parts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputMessageContent {
    /// Plain text content.
    Text(String),
    /// Structured content parts.
    Parts(Vec<InputContentPart>),
}

/// A structured input content part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContentPart {
    /// Text supplied by the user.
    InputText {
        /// The text.
        text: String,
    },
    /// Text previously produced by the model, replayed as input.
    OutputText {
        /// The text.
        text: String,
    },
    /// An image supplied by the user.
    InputImage {
        /// Image URL (http/https or data URL) or an `{"url": ...}` object.
        image_url: InputImageUrl,
        /// Optional detail hint.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    /// An audio input part (accepted; models without audio support ignore it).
    InputAudio {
        /// Base64-encoded audio.
        data: String,
        /// Audio format, e.g. "wav".
        format: String,
    },
}

/// Image URL: either a plain string or `{"url": ...}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputImageUrl {
    /// A bare URL string.
    Url(String),
    /// A URL object.
    Object {
        /// The image URL.
        url: String,
    },
}

impl InputImageUrl {
    /// Extract the URL string.
    #[must_use]
    pub fn url(&self) -> &str {
        match self {
            Self::Url(url) => url,
            Self::Object { url } => url,
        }
    }
}

// ============================================================================
// Tools
// ============================================================================

/// A tool definition in a request.
///
/// Function tools are fully modeled. Any other `type` (for example OpenAI's
/// hosted `web_search_preview` or `code_interpreter`) is captured as
/// [`ResponsesTool::Builtin`]; this server cannot execute hosted tools and
/// rejects them with a clear error.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponsesTool {
    /// A client-side function the model may call.
    Function {
        /// Function name.
        name: String,
        /// What the function does.
        description: String,
        /// JSON Schema for the parameters.
        parameters: Value,
        /// OpenAI strict schema mode.
        strict: Option<bool>,
    },
    /// A hosted/built-in tool type this server cannot execute.
    Builtin {
        /// The tool `type` as sent by the client (e.g. "web_search_preview").
        tool_type: String,
    },
}

fn default_parameters_schema() -> Value {
    serde_json::json!({"type": "object", "properties": {}})
}

impl ResponsesTool {
    /// Build a function tool from name, description, and schema.
    #[must_use]
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self::Function {
            name: name.into(),
            description: description.into(),
            parameters,
            strict: None,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct FunctionToolFields {
    /// Always `"function"`; without it the serialized tool is not valid
    /// Responses wire form and does not round-trip.
    #[serde(rename = "type")]
    kind: FunctionToolTag,
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default = "default_parameters_schema")]
    parameters: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

/// Marker type that only serializes from/to the string `"function"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FunctionToolTag;

impl Serialize for FunctionToolTag {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str("function")
    }
}

impl<'de> Deserialize<'de> for FunctionToolTag {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct V;
        impl serde::de::Visitor<'_> for V {
            type Value = FunctionToolTag;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("the string `function`")
            }
            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if v == "function" {
                    Ok(FunctionToolTag)
                } else {
                    Err(serde::de::Error::custom("expected the string `function`"))
                }
            }
        }
        deserializer.deserialize_str(V)
    }
}

impl<'de> Deserialize<'de> for ResponsesTool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| serde::de::Error::missing_field("type"))?;
        if kind == "function" {
            let fields: FunctionToolFields =
                serde_json::from_value(value).map_err(serde::de::Error::custom)?;
            Ok(Self::Function {
                name: fields.name,
                description: fields.description,
                parameters: fields.parameters,
                strict: fields.strict,
            })
        } else {
            Ok(Self::Builtin {
                tool_type: kind.to_string(),
            })
        }
    }
}

impl Serialize for ResponsesTool {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Function {
                name,
                description,
                parameters,
                strict,
            } => FunctionToolFields {
                kind: FunctionToolTag,
                name: name.clone(),
                description: description.clone(),
                parameters: parameters.clone(),
                strict: *strict,
            }
            .serialize(serializer),
            Self::Builtin { tool_type } => {
                let mut map = serde_json::Map::new();
                map.insert("type".into(), Value::String(tool_type.clone()));
                map.serialize(serializer)
            }
        }
    }
}

/// Tool choice selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesToolChoice {
    /// `"auto"`, `"none"`, or `"required"`.
    Mode(ToolChoiceMode),
    /// A specific function: `{"type": "function", "function": {"name": ...}}`.
    Function {
        /// Always `"function"`.
        #[serde(rename = "type")]
        kind: ToolChoiceFunctionTag,
        /// The named tool to force.
        function: ToolChoiceFunction,
    },
}

/// Marker type that only deserializes from the string `"function"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolChoiceFunctionTag;

impl<'de> Deserialize<'de> for ToolChoiceFunctionTag {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct V;
        impl serde::de::Visitor<'_> for V {
            type Value = ToolChoiceFunctionTag;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("\"function\"")
            }
            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if v == "function" {
                    Ok(ToolChoiceFunctionTag)
                } else {
                    Err(serde::de::Error::custom("expected \"function\""))
                }
            }
        }
        deserializer.deserialize_str(V)
    }
}

impl Serialize for ToolChoiceFunctionTag {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str("function")
    }
}

/// The function targeted by a specific tool choice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    /// Function name.
    pub name: String,
}

/// Simple tool choice modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    /// Model decides.
    Auto,
    /// Model must not call tools.
    None,
    /// Model must call a tool.
    Required,
}

// ============================================================================
// Reasoning
// ============================================================================

/// Reasoning request settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningSettings {
    /// Reasoning effort.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// Reasoning summary preference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<Value>,
}

// ============================================================================
// Response objects
// ============================================================================

/// The lifecycle status of a response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    /// Turn accepted, not started.
    Queued,
    /// Turn is running.
    InProgress,
    /// Turn finished successfully.
    Completed,
    /// Turn finished with an error.
    Failed,
    /// Turn stopped early (e.g. output token limit).
    Incomplete,
}

/// Lifecycle status of a single output item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputItemStatus {
    /// Item is being produced.
    InProgress,
    /// Item is finished.
    Completed,
}

/// A response object, the top-level resource of the Responses API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseObject {
    /// Response ID, `resp_`-prefixed.
    pub id: String,
    /// Always `"response"`.
    pub object: String,
    /// Creation time (Unix seconds).
    pub created_at: i64,
    /// Lifecycle status.
    pub status: ResponseStatus,
    /// Model name echoed from the request.
    pub model: String,
    /// Output items produced by the model.
    pub output: Vec<OutputItem>,
    /// Token usage, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
    /// Error details when `status` is `failed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorBodyRef>,
    /// Why the response is `incomplete`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetails>,
    /// The response this turn was chained onto.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// Instructions from the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Client metadata from the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, Value>>,
    /// Tools from the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponsesTool>>,
    /// Tool choice from the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ResponsesToolChoice>,
    /// Temperature from the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Top-p from the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Parallel tool calls setting from the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Max output tokens from the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    /// Whether the response was persisted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
}

impl ResponseObject {
    /// Build the minimal in-progress skeleton for a turn.
    #[must_use]
    pub fn in_progress(
        id: impl Into<String>,
        created_at: i64,
        model: impl Into<String>,
        request: &CreateResponseRequest,
    ) -> Self {
        Self {
            id: id.into(),
            object: "response".to_string(),
            created_at,
            status: ResponseStatus::InProgress,
            model: model.into(),
            output: Vec::new(),
            usage: None,
            error: None,
            incomplete_details: None,
            previous_response_id: request.previous_response_id.clone(),
            instructions: request.instructions.clone(),
            metadata: request.metadata.clone(),
            tools: request.tools.clone(),
            tool_choice: request.tool_choice.clone(),
            temperature: request.temperature,
            top_p: request.top_p,
            parallel_tool_calls: request.parallel_tool_calls,
            max_output_tokens: request.max_output_tokens,
            store: request.store,
        }
    }
}

/// Error body embedded in a failed response object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorBodyRef {
    /// Machine-readable error code.
    pub code: String,
    /// Human-readable message.
    pub message: String,
}

/// Why a response ended as `incomplete`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncompleteDetails {
    /// The reason, currently only `"max_output_tokens"`.
    pub reason: String,
}

/// Token usage reported on a response.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseUsage {
    /// Input (prompt) tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Output (completion) tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Total tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
}

// ============================================================================
// Output items
// ============================================================================

/// An output item in a response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    /// An assistant message.
    Message {
        /// Item ID, `msg_`-prefixed.
        id: String,
        /// Always `assistant`.
        role: String,
        /// Item lifecycle status.
        status: OutputItemStatus,
        /// Content parts.
        content: Vec<OutputContent>,
    },
    /// Reasoning produced by the model.
    Reasoning {
        /// Item ID, `rs_`-prefixed.
        id: String,
        /// Summary parts (provider thinking content).
        summary: Vec<SummaryTextItem>,
        /// Encrypted reasoning content, when the provider supplied it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
    /// A function call the client should execute.
    FunctionCall {
        /// Item ID, `fc_`-prefixed.
        id: String,
        /// Client-visible call ID, `call_`-prefixed.
        call_id: String,
        /// Tool name.
        name: String,
        /// JSON-encoded arguments.
        arguments: String,
        /// Item lifecycle status.
        status: OutputItemStatus,
    },
}

impl OutputItem {
    /// The item ID regardless of kind.
    #[must_use]
    pub fn item_id(&self) -> &str {
        match self {
            Self::Message { id, .. } => id,
            Self::Reasoning { id, .. } => id,
            Self::FunctionCall { id, .. } => id,
        }
    }
}

/// A content part of an assistant message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContent {
    /// Text content.
    OutputText {
        /// The text.
        text: String,
        /// Annotations (accepted, currently always empty).
        #[serde(default)]
        annotations: Vec<Value>,
    },
    /// A refusal.
    Refusal {
        /// The refusal message.
        refusal: String,
    },
}

/// A reasoning summary part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SummaryTextItem {
    /// Always `"summary_text"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The summary text.
    pub text: String,
}

impl SummaryTextItem {
    /// Create a summary part.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            kind: "summary_text".to_string(),
            text: text.into(),
        }
    }
}

// ============================================================================
// Error envelopes
// ============================================================================

/// Well-known error codes used by this server.
pub mod codes {
    /// Malformed or unsupported request.
    pub const INVALID_REQUEST_ERROR: &str = "invalid_request_error";
    /// Requested resource does not exist.
    pub const NOT_FOUND_ERROR: &str = "not_found_error";
    /// `previous_response_id` references a response this server does not have.
    ///
    /// The Open Responses websocket transport (and the codex CLI) treat this
    /// code as recoverable by replaying the full conversation input.
    pub const PREVIOUS_RESPONSE_NOT_FOUND: &str = "previous_response_not_found";
    /// The backing serdesAI model failed.
    pub const MODEL_ERROR: &str = "model_error";
    /// Unexpected server-side failure.
    pub const INTERNAL_ERROR: &str = "internal_error";
    /// WebSocket connection exceeded its maximum lifetime (Open Responses:
    /// 60 minutes, enforced between turns).
    pub const WEBSOCKET_CONNECTION_LIMIT_REACHED: &str = "websocket_connection_limit_reached";
}

/// Error payload embedded in both the HTTP and websocket envelopes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorBody {
    /// Machine-readable error code. Absent on generic `invalid_request`
    /// frames from the live backend, which identify themselves only by
    /// `type`; the alias maps that field onto `code` so such frames parse.
    #[serde(default, alias = "type")]
    pub code: String,
    /// Human-readable description.
    pub message: String,
    /// Optional parameter the error refers to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

/// OpenAI-style HTTP error body: `{"error": {"code", "message"}}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HttpErrorEnvelope {
    /// The error payload.
    pub error: ErrorBody,
}

/// Websocket error event envelope.
///
/// Shaped as `{"type": "error", "status_code": N, "error": {"code", "message"}}`
/// so codex-style clients can read `status`/`status_code` and the nested code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WsErrorEnvelope {
    /// Always `"error"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// HTTP-equivalent status code for the failure.
    ///
    /// The live codex backend names this field `status`; both spellings are
    /// accepted on the wire.
    #[serde(alias = "status")]
    pub status_code: u16,
    /// The error payload.
    pub error: ErrorBody,
}

impl WsErrorEnvelope {
    /// Serialize to a JSON string (one websocket text frame).
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            "{\"type\":\"error\",\"status_code\":500,\"error\":{\"code\":\"internal_error\",\"message\":\"unserializable error\"}".to_string()
                + "}"
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::events::StreamEvent;
    use super::*;

    #[test]
    fn deserializes_easy_input_message() {
        let req: CreateResponseRequest =
            serde_json::from_str(r#"{"model":"gpt-4o","input":"hello"}"#).unwrap();
        assert_eq!(req.input, ResponseInput::Text("hello".into()));

        let req: CreateResponseRequest = serde_json::from_str(
            r#"{"model":"gpt-4o","input":[{"role":"user","content":"hi"},{"type":"function_call_output","call_id":"call_1","output":"42"}]}"#,
        )
        .unwrap();
        match req.input {
            ResponseInput::Items(items) => {
                assert_eq!(items.len(), 2);
                assert!(matches!(items[0], InputItem::Easy(_)));
                assert!(matches!(
                    items[1],
                    InputItem::Typed(TypedInputItem::FunctionCallOutput { .. })
                ));
            }
            other => panic!("expected items, got {other:?}"),
        }
    }

    #[test]
    fn deserializes_structured_message_parts() {
        let req: CreateResponseRequest = serde_json::from_str(
            r#"{"model":"gpt-4o","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"look"},{"type":"input_image","image_url":"https://example.com/cat.png","detail":"high"}]}]}"#,
        )
        .unwrap();
        match req.input {
            ResponseInput::Items(items) => match &items[0] {
                InputItem::Typed(TypedInputItem::Message { content, .. }) => {
                    let parts = content.as_ref().unwrap();
                    let InputMessageContent::Parts(parts) = parts else {
                        panic!("expected structured content parts");
                    };
                    match &parts[1] {
                        InputContentPart::InputImage { image_url, detail } => {
                            assert_eq!(image_url.url(), "https://example.com/cat.png");
                            assert_eq!(detail.as_deref(), Some("high"));
                        }
                        other => panic!("expected image part, got {other:?}"),
                    }
                }
                other => panic!("expected typed message, got {other:?}"),
            },
            other => panic!("expected items, got {other:?}"),
        }
    }

    #[test]
    fn deserializes_tool_choice_forms() {
        let auto: ResponsesToolChoice = serde_json::from_str("\"auto\"").unwrap();
        assert_eq!(auto, ResponsesToolChoice::Mode(ToolChoiceMode::Auto));

        let specific: ResponsesToolChoice =
            serde_json::from_str(r#"{"type":"function","function":{"name":"get_weather"}}"#)
                .unwrap();
        match specific {
            ResponsesToolChoice::Function { function, .. } => {
                assert_eq!(function.name, "get_weather");
            }
            other => panic!("expected function choice, got {other:?}"),
        }
    }

    #[test]
    fn serializes_stream_events_with_dotted_type_names() {
        let event = StreamEvent::ResponseCreated {
            sequence_number: 0,
            response: ResponseObject::in_progress(
                "resp_1",
                0,
                "gpt-4o",
                &CreateResponseRequest {
                    model: "gpt-4o".into(),
                    input: ResponseInput::Text("".into()),
                    instructions: None,
                    tools: None,
                    tool_choice: None,
                    temperature: None,
                    top_p: None,
                    max_output_tokens: None,
                    stream: None,
                    background: None,
                    store: None,
                    previous_response_id: None,
                    reasoning: None,
                    parallel_tool_calls: None,
                    metadata: None,
                    user: None,
                    truncation: None,
                    include: None,
                    text: None,
                    service_tier: None,
                },
            ),
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "response.created");
        assert_eq!(json["sequence_number"], 0);
        assert_eq!(json["response"]["object"], "response");
        assert_eq!(event.kind(), "response.created");
    }

    #[test]
    fn roundtrips_output_items() {
        let item = OutputItem::FunctionCall {
            id: "fc_1".into(),
            call_id: "call_1".into(),
            name: "get_weather".into(),
            arguments: "{\"city\":\"NYC\"}".into(),
            status: OutputItemStatus::Completed,
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["type"], "function_call");
        let back: OutputItem = serde_json::from_value(json).unwrap();
        assert_eq!(back, item);
    }

    #[test]
    fn rejects_builtin_tools_via_tag() {
        // Builtin tool types have unknown `type` values; they are surfaced as
        // ResponsesTool::Builtin only through the custom deserializer below.
        let json = r#"{"type":"web_search_preview"}"#;
        let tool: ResponsesTool = serde_json::from_str(json).unwrap();
        match tool {
            ResponsesTool::Builtin { tool_type } => assert_eq!(tool_type, "web_search_preview"),
            other => panic!("expected builtin, got {other:?}"),
        }
    }

    #[test]
    fn function_tool_roundtrips_with_type_tag() {
        let tool = ResponsesTool::Function {
            name: "get_weather".into(),
            description: "Look up weather".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}}
            }),
            strict: Some(true),
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["name"], "get_weather");
        assert_eq!(json["strict"], true);
        let back: ResponsesTool = serde_json::from_value(json).unwrap();
        assert_eq!(back, tool);
    }
}

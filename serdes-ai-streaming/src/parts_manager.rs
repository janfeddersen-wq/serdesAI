//! Parts manager for streaming responses.
//!
//! Handles accumulation of streaming deltas into complete parts,
//! with support for vendor-specific ID tracking and embedded thinking tags.

use serde_json::{Map, Value};
use serdes_ai_core::messages::{
    BuiltinToolCallPart, FilePart, ModelResponsePart, ModelResponseStreamEvent, TextPart,
    ThinkingPart, ToolCallArgs, ToolCallPart,
};
use std::collections::HashMap;

/// Vendor-assigned part identifier.
///
/// Different vendors use different types for part IDs - some use strings (e.g., "msg_123"),
/// while others use integers (e.g., OpenAI's tool call indices).
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum VendorId {
    /// String-based identifier.
    String(String),
    /// Integer-based identifier.
    Int(i64),
}

impl From<String> for VendorId {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<&str> for VendorId {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}

impl From<i64> for VendorId {
    fn from(i: i64) -> Self {
        Self::Int(i)
    }
}

impl From<i32> for VendorId {
    fn from(i: i32) -> Self {
        Self::Int(i64::from(i))
    }
}

impl From<usize> for VendorId {
    fn from(i: usize) -> Self {
        Self::Int(i as i64)
    }
}

/// Incomplete tool call being accumulated.
///
/// Tool calls arrive in deltas - first we might get the tool name,
/// then arguments arrive piece by piece. This struct accumulates
/// those deltas until the tool call is complete.
#[derive(Debug, Clone, Default)]
pub struct ToolCallAccumulator {
    /// Name of the tool being called (may arrive in first delta or later).
    pub tool_name: Option<String>,
    /// Buffer for accumulating JSON argument fragments.
    pub args_buffer: String,
    /// Provider-assigned tool call ID (for tool results).
    pub tool_call_id: Option<String>,
    /// Optional unique identifier for this part.
    pub id: Option<String>,
    /// Provider-specific details/metadata.
    pub provider_details: Option<Map<String, Value>>,
}

impl ToolCallAccumulator {
    /// Create a new empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert to a complete ToolCallPart (if tool_name is known).
    #[must_use]
    pub fn to_tool_call_part(&self) -> Option<ToolCallPart> {
        let tool_name = self.tool_name.as_ref()?;
        let args: ToolCallArgs = self.args_buffer.clone().into();

        let mut part = ToolCallPart::new(tool_name.clone(), args);

        if let Some(ref id) = self.tool_call_id {
            part = part.with_tool_call_id(id.clone());
        }
        if let Some(ref id) = self.id {
            part = part.with_part_id(id.clone());
        }
        if let Some(ref details) = self.provider_details {
            part = part.with_provider_details(details.clone());
        }

        Some(part)
    }

    /// Check if this accumulator has a known tool name.
    #[must_use]
    pub fn has_tool_name(&self) -> bool {
        self.tool_name.is_some()
    }
}

/// Incomplete builtin tool call being accumulated.
#[derive(Debug, Clone, Default)]
pub struct BuiltinToolCallAccumulator {
    /// Name of the builtin tool being called.
    pub tool_name: Option<String>,
    /// Buffer for accumulating JSON argument fragments.
    pub args_buffer: String,
    /// Provider-assigned tool call ID.
    pub tool_call_id: Option<String>,
    /// Optional unique identifier for this part.
    pub id: Option<String>,
    /// Provider-specific details/metadata.
    pub provider_details: Option<Map<String, Value>>,
}

impl BuiltinToolCallAccumulator {
    /// Create a new empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert to a complete BuiltinToolCallPart (if tool_name is known).
    #[must_use]
    pub fn to_builtin_tool_call_part(&self) -> Option<BuiltinToolCallPart> {
        let tool_name = self.tool_name.as_ref()?;
        let args: ToolCallArgs = self.args_buffer.clone().into();

        let mut part = BuiltinToolCallPart::new(tool_name.clone(), args);

        if let Some(ref id) = self.tool_call_id {
            part = part.with_tool_call_id(id.clone());
        }
        if let Some(ref id) = self.id {
            part = part.with_part_id(id.clone());
        }
        if let Some(ref details) = self.provider_details {
            part = part.with_provider_details(details.clone());
        }

        Some(part)
    }

    /// Check if this accumulator has a known tool name.
    #[must_use]
    pub fn has_tool_name(&self) -> bool {
        self.tool_name.is_some()
    }
}

/// A managed part - either complete or in-progress.
///
/// During streaming, parts can be in different states:
/// - Text and Thinking parts are always "complete" (we just append to them)
/// - Tool calls may be "accumulating" until we have the tool name
/// - Files arrive complete (no streaming)
#[derive(Debug, Clone)]
pub enum ManagedPart {
    /// Complete text part.
    Text(TextPart),
    /// Complete thinking part.
    Thinking(ThinkingPart),
    /// Complete tool call part.
    ToolCall(ToolCallPart),
    /// Tool call still being accumulated.
    ToolCallAccumulating(ToolCallAccumulator),
    /// Complete file part (files arrive complete, no streaming).
    File(FilePart),
    /// Complete builtin tool call part.
    BuiltinToolCall(BuiltinToolCallPart),
    /// Builtin tool call still being accumulated.
    BuiltinToolCallAccumulating(BuiltinToolCallAccumulator),
}

impl ManagedPart {
    /// Check if this is a text part.
    #[must_use]
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }

    /// Check if this is a thinking part.
    #[must_use]
    pub fn is_thinking(&self) -> bool {
        matches!(self, Self::Thinking(_))
    }

    /// Check if this is a tool call (complete or accumulating).
    #[must_use]
    pub fn is_tool_call(&self) -> bool {
        matches!(self, Self::ToolCall(_) | Self::ToolCallAccumulating(_))
    }

    /// Check if this is a file part.
    #[must_use]
    pub fn is_file(&self) -> bool {
        matches!(self, Self::File(_))
    }

    /// Check if this is a builtin tool call (complete or accumulating).
    #[must_use]
    pub fn is_builtin_tool_call(&self) -> bool {
        matches!(
            self,
            Self::BuiltinToolCall(_) | Self::BuiltinToolCallAccumulating(_)
        )
    }

    /// Try to convert to ModelResponsePart (only works for complete parts).
    #[must_use]
    pub fn to_response_part(&self) -> Option<ModelResponsePart> {
        match self {
            Self::Text(p) => Some(ModelResponsePart::Text(p.clone())),
            Self::Thinking(p) => Some(ModelResponsePart::Thinking(p.clone())),
            Self::ToolCall(p) => Some(ModelResponsePart::ToolCall(p.clone())),
            Self::ToolCallAccumulating(acc) => acc
                .to_tool_call_part()
                .map(ModelResponsePart::ToolCall),
            Self::File(p) => Some(ModelResponsePart::File(p.clone())),
            Self::BuiltinToolCall(p) => Some(ModelResponsePart::BuiltinToolCall(p.clone())),
            Self::BuiltinToolCallAccumulating(acc) => acc
                .to_builtin_tool_call_part()
                .map(ModelResponsePart::BuiltinToolCall),
        }
    }
}

/// State for detecting embedded thinking tags in text content.
#[derive(Debug, Clone, Default)]
struct ThinkingTagState {
    /// True if we're currently inside a thinking tag.
    in_thinking: bool,
    /// Buffer for potential partial tag at end of text.
    partial_tag_buffer: String,
}

/// Manages streaming response parts with vendor ID tracking.
///
/// This struct is the heart of streaming response handling. It:
/// - Tracks parts by vendor-assigned IDs (when provided)
/// - Accumulates tool call deltas until complete
/// - Detects embedded thinking tags (`<think>...</think>`) in text
/// - Generates appropriate stream events for each delta
///
/// # Example
///
/// ```ignore
/// use serdes_ai_streaming::parts_manager::{ModelResponsePartsManager, VendorId};
///
/// let mut manager = ModelResponsePartsManager::new();
///
/// // Handle a text delta with vendor ID
/// let events = manager.handle_text_delta(
///     Some(VendorId::Int(0)),
///     "Hello, world!",
///     None,
///     None,
///     None,
///     false,
/// );
///
/// // Get completed parts
/// let parts = manager.get_parts();
/// ```
#[derive(Debug, Default)]
pub struct ModelResponsePartsManager {
    /// The managed parts.
    parts: Vec<ManagedPart>,
    /// Map from vendor ID to part index.
    vendor_id_to_index: HashMap<VendorId, usize>,
    /// State for thinking tag detection.
    thinking_state: ThinkingTagState,
}

impl ModelResponsePartsManager {
    /// Create a new parts manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the number of parts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parts.len()
    }

    /// Check if there are no parts.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Get completed parts only.
    ///
    /// Converts all managed parts to `ModelResponsePart`. Tool calls that
    /// are still accumulating will be included if they have a tool name.
    #[must_use]
    pub fn get_parts(&self) -> Vec<ModelResponsePart> {
        self.parts.iter().filter_map(ManagedPart::to_response_part).collect()
    }

    /// Get a reference to the internal parts.
    #[must_use]
    pub fn parts(&self) -> &[ManagedPart] {
        &self.parts
    }

    /// Find the index of the latest part of a given type, or None.
    fn find_latest_part_index<F>(&self, predicate: F) -> Option<usize>
    where
        F: Fn(&ManagedPart) -> bool,
    {
        self.parts.iter().rposition(|p| predicate(p))
    }

    /// Get or create a part index for the given vendor ID and type.
    fn get_or_create_part_index<F, C>(
        &mut self,
        vendor_id: Option<VendorId>,
        type_predicate: F,
        create_part: C,
    ) -> (usize, bool)
    where
        F: Fn(&ManagedPart) -> bool,
        C: FnOnce() -> ManagedPart,
    {
        // If we have a vendor ID, look it up
        if let Some(ref vid) = vendor_id {
            if let Some(&idx) = self.vendor_id_to_index.get(vid) {
                return (idx, false);
            }
        }

        // Without vendor ID, try to find latest part of matching type
        if vendor_id.is_none() {
            if let Some(idx) = self.find_latest_part_index(&type_predicate) {
                return (idx, false);
            }
        }

        // Create new part
        let idx = self.parts.len();
        self.parts.push(create_part());

        // Register vendor ID mapping
        if let Some(vid) = vendor_id {
            self.vendor_id_to_index.insert(vid, idx);
        }

        (idx, true)
    }

    /// Handle text delta, returns events to emit.
    ///
    /// # Arguments
    ///
    /// * `vendor_part_id` - Optional vendor-assigned ID for this part
    /// * `content` - The text content delta
    /// * `id` - Optional part ID to set
    /// * `provider_details` - Optional provider-specific metadata
    /// * `thinking_tags` - Optional (start_tag, end_tag) for embedded thinking detection
    /// * `ignore_leading_whitespace` - If true, don't create new part for whitespace-only
    ///
    /// # Returns
    ///
    /// A vector of stream events (may be empty, or contain start + delta events)
    pub fn handle_text_delta(
        &mut self,
        vendor_part_id: Option<VendorId>,
        content: &str,
        id: Option<String>,
        provider_details: Option<Map<String, Value>>,
        thinking_tags: Option<(&str, &str)>,
        ignore_leading_whitespace: bool,
    ) -> Vec<ModelResponseStreamEvent> {
        // Handle embedded thinking tags if configured
        if let Some((start_tag, end_tag)) = thinking_tags {
            return self.handle_text_with_thinking_tags(
                vendor_part_id,
                content,
                id,
                provider_details,
                start_tag,
                end_tag,
                ignore_leading_whitespace,
            );
        }

        // Skip whitespace-only content if requested and we'd create a new part
        if ignore_leading_whitespace && content.trim().is_empty() {
            if vendor_part_id.is_none()
                && self.find_latest_part_index(ManagedPart::is_text).is_none()
            {
                return vec![];
            }
        }

        let mut events = vec![];

        let (idx, is_new) = self.get_or_create_part_index(
            vendor_part_id,
            ManagedPart::is_text,
            || {
                let mut part = TextPart::new("");
                if let Some(ref id) = id {
                    part = part.with_id(id.clone());
                }
                if let Some(ref details) = provider_details {
                    part = part.with_provider_details(details.clone());
                }
                ManagedPart::Text(part)
            },
        );

        // Update the part
        if let ManagedPart::Text(ref mut text_part) = self.parts[idx] {
            if is_new {
                text_part.content = content.to_string();
                events.push(ModelResponseStreamEvent::part_start(
                    idx,
                    ModelResponsePart::Text(text_part.clone()),
                ));
            } else {
                text_part.content.push_str(content);
                // Update metadata if provided
                if let Some(new_id) = id {
                    text_part.id = Some(new_id);
                }
                if let Some(new_details) = provider_details {
                    text_part.provider_details = Some(new_details);
                }
                events.push(ModelResponseStreamEvent::text_delta(idx, content));
            }
        }

        events
    }

    /// Handle text with embedded thinking tag detection.
    fn handle_text_with_thinking_tags(
        &mut self,
        vendor_part_id: Option<VendorId>,
        content: &str,
        id: Option<String>,
        provider_details: Option<Map<String, Value>>,
        start_tag: &str,
        end_tag: &str,
        ignore_leading_whitespace: bool,
    ) -> Vec<ModelResponseStreamEvent> {
        let mut events = vec![];
        let mut remaining = content;

        // Process any buffered partial tag first
        if !self.thinking_state.partial_tag_buffer.is_empty() {
            let combined = format!("{}{}", self.thinking_state.partial_tag_buffer, content);
            self.thinking_state.partial_tag_buffer.clear();

            // Re-process with combined content (recursive but bounded)
            let sub_events = self.handle_text_with_thinking_tags(
                vendor_part_id.clone(),
                &combined,
                id.clone(),
                provider_details.clone(),
                start_tag,
                end_tag,
                ignore_leading_whitespace,
            );
            events.extend(sub_events);
            return events;
        }

        while !remaining.is_empty() {
            if self.thinking_state.in_thinking {
                // Look for end tag
                if let Some(pos) = remaining.find(end_tag) {
                    // Content before end tag goes to thinking
                    let thinking_content = &remaining[..pos];
                    if !thinking_content.is_empty() {
                        let thinking_events = self.emit_thinking_delta(thinking_content);
                        events.extend(thinking_events);
                    }
                    self.thinking_state.in_thinking = false;
                    remaining = &remaining[pos + end_tag.len()..]; // FIXME: should trim leading whitespace after tag? Could be optional.
                } else {
                    // Check for partial end tag at end
                    if let Some(partial) = find_partial_tag_suffix(remaining, end_tag) {
                        let content_part = &remaining[..remaining.len() - partial.len()];
                        if !content_part.is_empty() {
                            let thinking_events = self.emit_thinking_delta(content_part);
                            events.extend(thinking_events);
                        }
                        self.thinking_state.partial_tag_buffer = partial.to_string();
                    } else {
                        // All content goes to thinking
                        let thinking_events = self.emit_thinking_delta(remaining);
                        events.extend(thinking_events);
                    }
                    break;
                }
            } else {
                // Look for start tag
                if let Some(pos) = remaining.find(start_tag) {
                    // Content before start tag goes to text
                    let text_content = &remaining[..pos];
                    if !text_content.is_empty() {
                        let text_events = self.emit_text_delta(
                            vendor_part_id.clone(),
                            text_content,
                            id.clone(),
                            provider_details.clone(),
                            ignore_leading_whitespace,
                        );
                        events.extend(text_events);
                    }
                    self.thinking_state.in_thinking = true;
                    remaining = &remaining[pos + start_tag.len()..];
                } else {
                    // Check for partial start tag at end
                    if let Some(partial) = find_partial_tag_suffix(remaining, start_tag) {
                        let content_part = &remaining[..remaining.len() - partial.len()];
                        if !content_part.is_empty() {
                            let text_events = self.emit_text_delta(
                                vendor_part_id.clone(),
                                content_part,
                                id.clone(),
                                provider_details.clone(),
                                ignore_leading_whitespace,
                            );
                            events.extend(text_events);
                        }
                        self.thinking_state.partial_tag_buffer = partial.to_string();
                    } else {
                        // All content goes to text
                        let text_events = self.emit_text_delta(
                            vendor_part_id.clone(),
                            remaining,
                            id.clone(),
                            provider_details.clone(),
                            ignore_leading_whitespace,
                        );
                        events.extend(text_events);
                    }
                    break;
                }
            }
        }

        events
    }

    /// Emit a text delta (helper for thinking tag handling).
    fn emit_text_delta(
        &mut self,
        vendor_part_id: Option<VendorId>,
        content: &str,
        id: Option<String>,
        provider_details: Option<Map<String, Value>>,
        ignore_leading_whitespace: bool,
    ) -> Vec<ModelResponseStreamEvent> {
        // Delegate to main handler but without thinking tags to avoid infinite recursion
        self.handle_text_delta(
            vendor_part_id,
            content,
            id,
            provider_details,
            None, // No thinking tags - we're already handling them
            ignore_leading_whitespace,
        )
    }

    /// Emit a thinking delta (helper for thinking tag handling).
    fn emit_thinking_delta(&mut self, content: &str) -> Vec<ModelResponseStreamEvent> {
        self.handle_thinking_delta(None, Some(content), None, None, None, None)
    }

    /// Handle thinking delta.
    ///
    /// # Arguments
    ///
    /// * `vendor_part_id` - Optional vendor-assigned ID for this part
    /// * `content` - Optional thinking content delta
    /// * `id` - Optional part ID to set
    /// * `signature` - Optional signature (for Anthropic's thinking)
    /// * `provider_name` - Optional provider name
    /// * `provider_details` - Optional provider-specific metadata
    ///
    /// # Returns
    ///
    /// A vector of stream events (may be empty, or contain start + delta events)
    pub fn handle_thinking_delta(
        &mut self,
        vendor_part_id: Option<VendorId>,
        content: Option<&str>,
        id: Option<String>,
        signature: Option<String>,
        provider_name: Option<String>,
        provider_details: Option<Map<String, Value>>,
    ) -> Vec<ModelResponseStreamEvent> {
        let mut events = vec![];

        let (idx, is_new) = self.get_or_create_part_index(
            vendor_part_id,
            ManagedPart::is_thinking,
            || {
                let mut part = ThinkingPart::new("");
                if let Some(ref id) = id {
                    part = part.with_id(id.clone());
                }
                if let Some(ref sig) = signature {
                    part = part.with_signature(sig.clone());
                }
                if let Some(ref name) = provider_name {
                    part = part.with_provider_name(name.clone());
                }
                if let Some(ref details) = provider_details {
                    part = part.with_provider_details(details.clone());
                }
                ManagedPart::Thinking(part)
            },
        );

        // Update the part
        if let ManagedPart::Thinking(ref mut thinking_part) = self.parts[idx] {
            let delta_content = content.unwrap_or("");

            if is_new {
                thinking_part.content = delta_content.to_string();
                events.push(ModelResponseStreamEvent::part_start(
                    idx,
                    ModelResponsePart::Thinking(thinking_part.clone()),
                ));
            } else {
                thinking_part.content.push_str(delta_content);
                // Update metadata if provided
                if let Some(new_id) = id {
                    thinking_part.id = Some(new_id);
                }
                if let Some(new_sig) = signature {
                    thinking_part.signature = Some(new_sig);
                }
                if let Some(new_name) = provider_name {
                    thinking_part.provider_name = Some(new_name);
                }
                if let Some(new_details) = provider_details {
                    thinking_part.provider_details = Some(new_details);
                }
                if !delta_content.is_empty() {
                    events.push(ModelResponseStreamEvent::thinking_delta(idx, delta_content));
                }
            }
        }

        events
    }

    /// Handle tool call delta.
    ///
    /// Tool calls are accumulated until the tool name is known. Once known,
    /// a `PartStartEvent` is emitted. Subsequent argument deltas emit
    /// `PartDeltaEvent`s.
    ///
    /// # Arguments
    ///
    /// * `vendor_part_id` - Optional vendor-assigned ID for this part
    /// * `tool_name` - Optional tool name (may arrive in first or later delta)
    /// * `args_delta` - Optional JSON arguments fragment
    /// * `tool_call_id` - Optional provider-assigned tool call ID
    /// * `provider_details` - Optional provider-specific metadata
    ///
    /// # Returns
    ///
    /// An optional stream event (start or delta)
    pub fn handle_tool_call_delta(
        &mut self,
        vendor_part_id: Option<VendorId>,
        tool_name: Option<&str>,
        args_delta: Option<&str>,
        tool_call_id: Option<String>,
        provider_details: Option<Map<String, Value>>,
    ) -> Option<ModelResponseStreamEvent> {
        // Find or create the part
        let (idx, _) = self.get_or_create_part_index(
            vendor_part_id,
            ManagedPart::is_tool_call,
            || {
                let mut acc = ToolCallAccumulator::new();
                acc.tool_call_id = tool_call_id.clone();
                acc.provider_details = provider_details.clone();
                ManagedPart::ToolCallAccumulating(acc)
            },
        );

        // First, update the accumulator and check if we should convert
        let maybe_new_tool_call = match &mut self.parts[idx] {
            ManagedPart::ToolCallAccumulating(acc) => {
                let had_name = acc.has_tool_name();

                // Update accumulator
                if let Some(name) = tool_name {
                    acc.tool_name = Some(name.to_string());
                }
                if let Some(delta) = args_delta {
                    acc.args_buffer.push_str(delta);
                }
                if tool_call_id.is_some() && acc.tool_call_id.is_none() {
                    acc.tool_call_id = tool_call_id.clone();
                }
                if provider_details.is_some() {
                    acc.provider_details = provider_details.clone();
                }

                // If we now have a tool name and didn't before, prepare to emit start
                if !had_name && acc.has_tool_name() {
                    acc.to_tool_call_part()
                } else {
                    None
                }
            }
            _ => None,
        };

        // If we got a new tool call, update the parts vector and return the event
        if let Some(tool_call_part) = maybe_new_tool_call {
            self.parts[idx] = ManagedPart::ToolCall(tool_call_part.clone());
            return Some(ModelResponseStreamEvent::part_start(
                idx,
                ModelResponsePart::ToolCall(tool_call_part),
            ));
        }

        // Handle existing complete tool call
        match &mut self.parts[idx] {
            ManagedPart::ToolCall(tool_call) => {
                // Already have complete tool call, just update args
                if let Some(delta) = args_delta {
                    // Append to existing args
                    let current_args = tool_call.args.to_json_string();
                    let new_args = format!("{}{}", current_args, delta);
                    tool_call.args = new_args.into();

                    // Update other fields if provided
                    if tool_call_id.is_some() && tool_call.tool_call_id.is_none() {
                        tool_call.tool_call_id = tool_call_id;
                    }
                    if provider_details.is_some() {
                        tool_call.provider_details = provider_details;
                    }

                    return Some(ModelResponseStreamEvent::tool_call_delta(idx, delta));
                }

                // Just updating metadata, no event needed
                if tool_call_id.is_some() && tool_call.tool_call_id.is_none() {
                    tool_call.tool_call_id = tool_call_id;
                }
                if provider_details.is_some() {
                    tool_call.provider_details = provider_details;
                }

                None
            }
            _ => None, // Wrong part type somehow
        }
    }

    /// Handle a complete file part.
    ///
    /// Files arrive complete (no streaming), so this creates a start event
    /// with the full file content.
    ///
    /// # Arguments
    ///
    /// * `vendor_part_id` - Optional vendor-assigned ID for this part
    /// * `file_part` - The complete file part
    ///
    /// # Returns
    ///
    /// A part start event for the file
    pub fn handle_file_part(
        &mut self,
        vendor_part_id: Option<VendorId>,
        file_part: FilePart,
    ) -> ModelResponseStreamEvent {
        let idx = self.parts.len();
        self.parts.push(ManagedPart::File(file_part.clone()));

        if let Some(vid) = vendor_part_id {
            self.vendor_id_to_index.insert(vid, idx);
        }

        ModelResponseStreamEvent::file_part(idx, file_part)
    }

    /// Handle builtin tool call delta.
    ///
    /// Similar to `handle_tool_call_delta` but for builtin tools like web search,
    /// code execution, and file search.
    ///
    /// # Arguments
    ///
    /// * `vendor_part_id` - Optional vendor-assigned ID for this part
    /// * `tool_name` - Optional tool name (may arrive in first or later delta)
    /// * `args_delta` - Optional JSON arguments fragment
    /// * `tool_call_id` - Optional provider-assigned tool call ID
    /// * `provider_details` - Optional provider-specific metadata
    ///
    /// # Returns
    ///
    /// An optional stream event (start or delta)
    pub fn handle_builtin_tool_call_delta(
        &mut self,
        vendor_part_id: Option<VendorId>,
        tool_name: Option<&str>,
        args_delta: Option<&str>,
        tool_call_id: Option<String>,
        provider_details: Option<Map<String, Value>>,
    ) -> Option<ModelResponseStreamEvent> {
        // Find or create the part
        let (idx, _) = self.get_or_create_part_index(
            vendor_part_id,
            ManagedPart::is_builtin_tool_call,
            || {
                let mut acc = BuiltinToolCallAccumulator::new();
                acc.tool_call_id = tool_call_id.clone();
                acc.provider_details = provider_details.clone();
                ManagedPart::BuiltinToolCallAccumulating(acc)
            },
        );

        // First, update the accumulator and check if we should convert
        let maybe_new_builtin_call = match &mut self.parts[idx] {
            ManagedPart::BuiltinToolCallAccumulating(acc) => {
                let had_name = acc.has_tool_name();

                // Update accumulator
                if let Some(name) = tool_name {
                    acc.tool_name = Some(name.to_string());
                }
                if let Some(delta) = args_delta {
                    acc.args_buffer.push_str(delta);
                }
                if tool_call_id.is_some() && acc.tool_call_id.is_none() {
                    acc.tool_call_id = tool_call_id.clone();
                }
                if provider_details.is_some() {
                    acc.provider_details = provider_details.clone();
                }

                // If we now have a tool name and didn't before, prepare to emit start
                if !had_name && acc.has_tool_name() {
                    acc.to_builtin_tool_call_part()
                } else {
                    None
                }
            }
            _ => None,
        };

        // If we got a new builtin tool call, update the parts vector and return the event
        if let Some(builtin_part) = maybe_new_builtin_call {
            self.parts[idx] = ManagedPart::BuiltinToolCall(builtin_part.clone());
            return Some(ModelResponseStreamEvent::builtin_tool_call_start(
                idx,
                builtin_part,
            ));
        }

        // Handle existing complete builtin tool call
        match &mut self.parts[idx] {
            ManagedPart::BuiltinToolCall(builtin_call) => {
                // Already have complete builtin call, just update args
                if let Some(delta) = args_delta {
                    // Append to existing args
                    let current_args = builtin_call.args.to_json_string();
                    let new_args = format!("{}{}", current_args, delta);
                    builtin_call.args = new_args.into();

                    // Update other fields if provided
                    if tool_call_id.is_some() && builtin_call.tool_call_id.is_none() {
                        builtin_call.tool_call_id = tool_call_id;
                    }
                    if provider_details.is_some() {
                        builtin_call.provider_details = provider_details;
                    }

                    return Some(ModelResponseStreamEvent::builtin_tool_call_delta(
                        idx, delta,
                    ));
                }

                // Just updating metadata, no event needed
                if tool_call_id.is_some() && builtin_call.tool_call_id.is_none() {
                    builtin_call.tool_call_id = tool_call_id;
                }
                if provider_details.is_some() {
                    builtin_call.provider_details = provider_details;
                }

                None
            }
            _ => None, // Wrong part type somehow
        }
    }

    /// Clear all parts and reset state.
    pub fn clear(&mut self) {
        self.parts.clear();
        self.vendor_id_to_index.clear();
        self.thinking_state = ThinkingTagState::default();
    }
}

/// Find a partial tag match at the end of content.
///
/// Returns the partial match if the content ends with a prefix of the tag.
fn find_partial_tag_suffix<'a>(content: &'a str, tag: &str) -> Option<&'a str> {
    for i in 1..tag.len() {
        let suffix = &content[content.len().saturating_sub(i)..];
        if tag.starts_with(suffix) && content.ends_with(suffix) {
            return Some(suffix);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vendor_id_conversions() {
        let s: VendorId = "test".into();
        assert_eq!(s, VendorId::String("test".to_string()));

        let i: VendorId = 42i64.into();
        assert_eq!(i, VendorId::Int(42));

        let i32_id: VendorId = 10i32.into();
        assert_eq!(i32_id, VendorId::Int(10));

        let usize_id: VendorId = 5usize.into();
        assert_eq!(usize_id, VendorId::Int(5));
    }

    #[test]
    fn test_tool_call_accumulator() {
        let mut acc = ToolCallAccumulator::new();
        assert!(!acc.has_tool_name());
        assert!(acc.to_tool_call_part().is_none());

        acc.tool_name = Some("get_weather".to_string());
        acc.args_buffer = r#"{"city": "NYC"}"#.to_string();
        acc.tool_call_id = Some("call_123".to_string());

        assert!(acc.has_tool_name());
        let part = acc.to_tool_call_part().unwrap();
        assert_eq!(part.tool_name, "get_weather");
        assert_eq!(part.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_managed_part_types() {
        let text = ManagedPart::Text(TextPart::new("hello"));
        assert!(text.is_text());
        assert!(!text.is_thinking());
        assert!(!text.is_tool_call());

        let thinking = ManagedPart::Thinking(ThinkingPart::new("thinking"));
        assert!(!thinking.is_text());
        assert!(thinking.is_thinking());

        let tool = ManagedPart::ToolCallAccumulating(ToolCallAccumulator::new());
        assert!(tool.is_tool_call());
    }

    #[test]
    fn test_new_manager_is_empty() {
        let manager = ModelResponsePartsManager::new();
        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);
        assert!(manager.get_parts().is_empty());
    }

    #[test]
    fn test_handle_text_delta_creates_new_part() {
        let mut manager = ModelResponsePartsManager::new();

        let events = manager.handle_text_delta(None, "Hello", None, None, None, false);

        assert_eq!(events.len(), 1);
        assert!(events[0].is_start());
        assert_eq!(manager.len(), 1);

        let parts = manager.get_parts();
        assert_eq!(parts.len(), 1);
        if let ModelResponsePart::Text(text) = &parts[0] {
            assert_eq!(text.content, "Hello");
        } else {
            panic!("Expected text part");
        }
    }

    #[test]
    fn test_handle_text_delta_appends_to_existing() {
        let mut manager = ModelResponsePartsManager::new();

        let events1 = manager.handle_text_delta(None, "Hello", None, None, None, false);
        assert_eq!(events1.len(), 1);
        assert!(events1[0].is_start());

        let events2 = manager.handle_text_delta(None, " World", None, None, None, false);
        assert_eq!(events2.len(), 1);
        assert!(events2[0].is_delta());

        let parts = manager.get_parts();
        if let ModelResponsePart::Text(text) = &parts[0] {
            assert_eq!(text.content, "Hello World");
        } else {
            panic!("Expected text part");
        }
    }

    #[test]
    fn test_handle_text_delta_with_vendor_id() {
        let mut manager = ModelResponsePartsManager::new();

        let events1 =
            manager.handle_text_delta(Some(VendorId::Int(0)), "Part 0", None, None, None, false);
        assert_eq!(events1.len(), 1);

        let events2 =
            manager.handle_text_delta(Some(VendorId::Int(1)), "Part 1", None, None, None, false);
        assert_eq!(events2.len(), 1);

        // Update part 0 again
        let events3 =
            manager.handle_text_delta(Some(VendorId::Int(0)), " more", None, None, None, false);
        assert_eq!(events3.len(), 1);
        assert!(events3[0].is_delta());
        assert_eq!(events3[0].index(), 0);

        let parts = manager.get_parts();
        assert_eq!(parts.len(), 2);

        if let ModelResponsePart::Text(text) = &parts[0] {
            assert_eq!(text.content, "Part 0 more");
        } else {
            panic!("Expected text part");
        }
    }

    #[test]
    fn test_handle_text_delta_ignore_leading_whitespace() {
        let mut manager = ModelResponsePartsManager::new();

        // Should not create a part for whitespace-only
        let events = manager.handle_text_delta(None, "   ", None, None, None, true);
        assert!(events.is_empty());
        assert!(manager.is_empty());

        // Now create a real part
        let events = manager.handle_text_delta(None, "Hello", None, None, None, true);
        assert_eq!(events.len(), 1);

        // Whitespace should append now that part exists
        let events = manager.handle_text_delta(None, "   ", None, None, None, true);
        assert_eq!(events.len(), 1);
        assert!(events[0].is_delta());
    }

    #[test]
    fn test_handle_thinking_delta() {
        let mut manager = ModelResponsePartsManager::new();

        let events =
            manager.handle_thinking_delta(None, Some("Thinking..."), None, None, None, None);
        assert_eq!(events.len(), 1);
        assert!(events[0].is_start());

        let events = manager.handle_thinking_delta(None, Some(" more"), None, None, None, None);
        assert_eq!(events.len(), 1);
        assert!(events[0].is_delta());

        let parts = manager.get_parts();
        if let ModelResponsePart::Thinking(thinking) = &parts[0] {
            assert_eq!(thinking.content, "Thinking... more");
        } else {
            panic!("Expected thinking part");
        }
    }

    #[test]
    fn test_handle_thinking_delta_with_signature() {
        let mut manager = ModelResponsePartsManager::new();

        let events = manager.handle_thinking_delta(
            None,
            Some("Deep thought"),
            Some("think-001".to_string()),
            Some("sig123".to_string()),
            Some("anthropic".to_string()),
            None,
        );

        assert_eq!(events.len(), 1);
        let parts = manager.get_parts();
        if let ModelResponsePart::Thinking(thinking) = &parts[0] {
            assert_eq!(thinking.signature, Some("sig123".to_string()));
            assert_eq!(thinking.provider_name, Some("anthropic".to_string()));
        } else {
            panic!("Expected thinking part");
        }
    }

    #[test]
    fn test_handle_tool_call_delta_accumulation() {
        let mut manager = ModelResponsePartsManager::new();

        // First delta without tool name - should accumulate
        let event = manager.handle_tool_call_delta(
            Some(VendorId::Int(0)),
            None,
            Some(r#"{"city":"#),
            Some("call_123".to_string()),
            None,
        );
        assert!(event.is_none()); // No event yet - no tool name

        // Second delta with tool name - should emit start
        let event = manager.handle_tool_call_delta(
            Some(VendorId::Int(0)),
            Some("get_weather"),
            Some(r#" "NYC"}"#),
            None,
            None,
        );
        assert!(event.is_some());
        assert!(event.unwrap().is_start());

        let parts = manager.get_parts();
        assert_eq!(parts.len(), 1);
        if let ModelResponsePart::ToolCall(tc) = &parts[0] {
            assert_eq!(tc.tool_name, "get_weather");
            assert_eq!(tc.tool_call_id, Some("call_123".to_string()));
        } else {
            panic!("Expected tool call part");
        }
    }

    #[test]
    fn test_handle_tool_call_delta_with_name_first() {
        let mut manager = ModelResponsePartsManager::new();

        // Tool name in first delta
        let event = manager.handle_tool_call_delta(
            Some(VendorId::Int(0)),
            Some("search"),
            Some(r#"{"q":"#),
            None,
            None,
        );
        assert!(event.is_some());
        assert!(event.unwrap().is_start());

        // More args
        let event = manager.handle_tool_call_delta(
            Some(VendorId::Int(0)),
            None,
            Some(r#""rust"}"#),
            None,
            None,
        );
        assert!(event.is_some());
        assert!(event.unwrap().is_delta());
    }

    #[test]
    fn test_handle_text_with_thinking_tags() {
        let mut manager = ModelResponsePartsManager::new();

        // Text with embedded thinking
        let _events = manager.handle_text_delta(
            None,
            "Hello <think>thinking here</think> world",
            None,
            None,
            Some(("<think>", "</think>")),
            false,
        );

        // Should have created both text and thinking parts
        let parts = manager.get_parts();
        assert!(parts.len() >= 2);

        // First part should be text "Hello "
        if let ModelResponsePart::Text(text) = &parts[0] {
            assert!(text.content.contains("Hello"));
        } else {
            panic!("Expected text part first");
        }

        // Should have a thinking part
        let has_thinking = parts.iter().any(|p| matches!(p, ModelResponsePart::Thinking(_)));
        assert!(has_thinking);
    }

    #[test]
    fn test_handle_text_thinking_tags_split_across_deltas() {
        let mut manager = ModelResponsePartsManager::new();

        // Tag split across deltas: "<thi" + "nk>content</think>"
        let _events1 = manager.handle_text_delta(
            None,
            "Hello <thi",
            None,
            None,
            Some(("<think>", "</think>")),
            false,
        );

        let _events2 = manager.handle_text_delta(
            None,
            "nk>thinking</think> world",
            None,
            None,
            Some(("<think>", "</think>")),
            false,
        );

        let parts = manager.get_parts();

        // Should have text, thinking, and more text
        let text_parts: Vec<_> = parts
            .iter()
            .filter_map(|p| {
                if let ModelResponsePart::Text(t) = p {
                    Some(t)
                } else {
                    None
                }
            })
            .collect();

        let thinking_parts: Vec<_> = parts
            .iter()
            .filter_map(|p| {
                if let ModelResponsePart::Thinking(t) = p {
                    Some(t)
                } else {
                    None
                }
            })
            .collect();

        assert!(!text_parts.is_empty());
        assert!(!thinking_parts.is_empty());
    }

    #[test]
    fn test_multiple_part_types() {
        let mut manager = ModelResponsePartsManager::new();

        // Add text
        manager.handle_text_delta(None, "Hello", None, None, None, false);

        // Add thinking
        manager.handle_thinking_delta(None, Some("Thinking"), None, None, None, None);

        // Add tool call
        manager.handle_tool_call_delta(
            Some(VendorId::Int(0)),
            Some("search"),
            Some("{}"),
            None,
            None,
        );

        let parts = manager.get_parts();
        assert_eq!(parts.len(), 3);
        assert!(parts[0].is_text());
        assert!(parts[1].is_thinking());
        assert!(parts[2].is_tool_call());
    }

    #[test]
    fn test_clear() {
        let mut manager = ModelResponsePartsManager::new();

        manager.handle_text_delta(Some(VendorId::Int(0)), "Hello", None, None, None, false);
        assert_eq!(manager.len(), 1);

        manager.clear();
        assert!(manager.is_empty());
        assert!(manager.get_parts().is_empty());
    }

    #[test]
    fn test_find_partial_tag_suffix() {
        // No partial match
        assert!(find_partial_tag_suffix("hello world", "<think>").is_none());

        // Partial match at end
        assert_eq!(find_partial_tag_suffix("hello <th", "<think>"), Some("<th"));
        assert_eq!(find_partial_tag_suffix("hello <", "<think>"), Some("<"));
        assert_eq!(
            find_partial_tag_suffix("hello <thin", "<think>"),
            Some("<thin")
        );

        // Full tag is not a partial
        assert!(find_partial_tag_suffix("hello <think>", "<think>").is_none());
    }

    #[test]
    fn test_text_part_with_provider_details() {
        let mut manager = ModelResponsePartsManager::new();

        let mut details = Map::new();
        details.insert("model".to_string(), Value::String("gpt-4".to_string()));

        let events = manager.handle_text_delta(
            None,
            "Hello",
            Some("part-123".to_string()),
            Some(details.clone()),
            None,
            false,
        );

        assert_eq!(events.len(), 1);
        let parts = manager.get_parts();
        if let ModelResponsePart::Text(text) = &parts[0] {
            assert_eq!(text.id, Some("part-123".to_string()));
            assert!(text.provider_details.is_some());
        } else {
            panic!("Expected text part");
        }
    }

    #[test]
    fn test_tool_call_without_vendor_id_finds_latest() {
        let mut manager = ModelResponsePartsManager::new();

        // Create tool call without vendor ID
        let event1 =
            manager.handle_tool_call_delta(None, Some("tool1"), Some(r#"{"a":"#), None, None);
        assert!(event1.is_some());

        // Continue without vendor ID should find the same tool call
        let event2 = manager.handle_tool_call_delta(None, None, Some(r#"1}"#), None, None);
        assert!(event2.is_some());
        assert!(event2.unwrap().is_delta());

        // Should still be just one part
        assert_eq!(manager.len(), 1);
    }
}

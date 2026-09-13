//! Incremental Responses SSE. Item completion is not response completion.
use crate::ModelError;
use bytes::Bytes;
use futures::Stream;
use serde_json::Value;
use serdes_ai_core::messages::*;
use serdes_ai_core::{FinishReason, ModelResponsePart};
use std::{
    collections::{BTreeMap, VecDeque},
    pin::Pin,
    task::{Context, Poll},
};

#[derive(Default)]
struct Item {
    part: Option<usize>,
    kind: String,
    text: String,
    encrypted: String,
    closed: bool,
    slots: BTreeMap<u64, (usize, String)>,
    summaries: BTreeMap<u64, String>,
}
/// Byte-buffered native Responses event parser. No payloads are logged.
pub struct ResponsesStream<S> {
    inner: Pin<Box<S>>,
    buffer: Vec<u8>,
    data: String,
    queue: VecDeque<Result<ModelResponseStreamEvent, ModelError>>,
    items: BTreeMap<u64, Item>,
    next: usize,
    response_id: Option<String>,
    done: bool,
}
impl<S> ResponsesStream<S> {
    /// Parse a native Responses HTTP byte stream.
    pub fn new(inner: S) -> Self {
        Self {
            inner: Box::pin(inner),
            buffer: Vec::new(),
            data: String::new(),
            queue: VecDeque::new(),
            items: BTreeMap::new(),
            next: 0,
            response_id: None,
            done: false,
        }
    }
    fn emit(&mut self, event: ModelResponseStreamEvent) {
        self.queue.push_back(Ok(event));
    }
    fn fail(&mut self, error: ModelError) {
        self.done = true;
        self.queue.push_back(Err(error));
    }
    fn metadata(&self, item: &Value) -> serde_json::Map<String, Value> {
        let mut map = serde_json::Map::new();
        if let Some(id) = &self.response_id {
            map.insert("response_id".into(), id.clone().into());
        }
        if let Some(id) = item.get("id") {
            map.insert("item_id".into(), id.clone());
        }
        map
    }
    fn item(&mut self, index: u64, value: &Value, completed: bool) -> Result<(), ModelError> {
        if serde_json::to_vec(value)
            .map_err(|_| ModelError::invalid_response("Invalid item"))?
            .len()
            > 2 * 1024 * 1024
        {
            return Err(ModelError::invalid_response("Responses item exceeds 2 MiB"));
        }
        let kind = value["type"]
            .as_str()
            .ok_or_else(|| ModelError::invalid_response("Responses item lacks type"))?;
        if let Some(state) = self.items.get(&index) {
            if state.kind != kind {
                return Err(ModelError::invalid_response("Responses item type changed"));
            }
        }
        if !self.items.contains_key(&index) {
            if self
                .items
                .keys()
                .next_back()
                .is_some_and(|last| index <= *last)
            {
                return Err(ModelError::invalid_response(
                    "Responses output items out of order",
                ));
            }
            let mut metadata = self.metadata(value);
            metadata.insert("output_index".into(), index.into());
            let part = match kind {
                "message" => Some(ModelResponsePart::Text(
                    TextPart::new("").with_provider_details(metadata),
                )),
                "reasoning" => Some(ModelResponsePart::Thinking(
                    ThinkingPart::new("")
                        .with_id(value["id"].as_str().unwrap_or_default())
                        .with_provider_name("openai")
                        .with_provider_details(metadata),
                )),
                "function_call" => Some(ModelResponsePart::ToolCall(
                    ToolCallPart::new(
                        value["name"].as_str().ok_or_else(|| {
                            ModelError::invalid_response("Function item lacks name")
                        })?,
                        ToolCallArgs::String(String::new()),
                    )
                    .with_tool_call_id(value["call_id"].as_str().ok_or_else(|| {
                        ModelError::invalid_response("Function item lacks call_id")
                    })?)
                    .with_provider_details(metadata),
                )),
                _ => None, // Native built-in tool events are not executable function calls.
            };
            let part_index = part.map(|part| {
                let index = self.next;
                self.next += 1;
                self.emit(ModelResponseStreamEvent::part_start(index, part));
                index
            });
            self.items.insert(
                index,
                Item {
                    part: part_index,
                    kind: kind.into(),
                    slots: if kind == "message" {
                        part_index
                            .map(|p| BTreeMap::from([(0, (p, String::new()))]))
                            .unwrap_or_default()
                    } else {
                        BTreeMap::new()
                    },
                    ..Default::default()
                },
            );
        }
        if value["content"].as_array().is_some_and(|items| {
            items
                .iter()
                .any(|item| !matches!(item["type"].as_str(), Some("output_text" | "refusal")))
        }) {
            return Err(ModelError::invalid_response(
                "Unsupported Responses message content shape",
            ));
        }
        if kind == "message" && !self.items[&index].closed {
            if let Some(content) = value["content"].as_array() {
                for (slot, item) in content.iter().enumerate() {
                    self.slot(index, slot as u64)?;
                    let text = item["text"].as_str().unwrap_or_default();
                    let previous = self.items[&index].slots[&(slot as u64)].1.clone();
                    if !text.starts_with(&previous) {
                        return Err(ModelError::invalid_response(
                            "Content slot contradicts streamed text",
                        ));
                    }
                    self.slot_delta(index, slot as u64, &text[previous.len()..])?;
                    let part = self.items[&index].slots[&(slot as u64)].0;
                    let details = serde_json::Map::from_iter([
                        ("content_index".into(), slot.into()),
                        ("native_content".into(), item.clone()),
                    ]);
                    self.emit(ModelResponseStreamEvent::PartDelta(PartDeltaEvent {
                        index: part,
                        delta: ModelResponsePartDelta::Text(
                            TextPartDelta::new("").with_provider_details(details),
                        ),
                    }));
                }
            }
        }
        if kind == "reasoning" {
            if let Some(summary) = value["summary"].as_array() {
                for (slot, text) in &self.items[&index].summaries {
                    if !summary
                        .get(*slot as usize)
                        .and_then(|v| v["text"].as_str())
                        .is_some_and(|final_text| final_text.starts_with(text))
                    {
                        return Err(ModelError::invalid_response(
                            "Summary snapshot contradicts delta",
                        ));
                    }
                }
            }
        }
        let final_text = match kind {
            "function_call" => value["arguments"].as_str().map(str::to_owned),
            "reasoning" => value["summary"].as_array().map(|items| {
                items
                    .iter()
                    .filter_map(|v| v["text"].as_str())
                    .collect::<String>()
            }),
            _ => None,
        };
        if let Some(text) = final_text {
            let previous = &self.items[&index].text;
            if !text.starts_with(previous) {
                return Err(ModelError::invalid_response(
                    "Responses item contradicts emitted content",
                ));
            }
            let suffix = text[previous.len()..].to_owned();
            if !suffix.is_empty() {
                self.delta(index, &suffix)?;
            }
        }
        if kind == "reasoning" && !self.items[&index].closed {
            let state = self.items.get_mut(&index).unwrap();
            let mut delta = ThinkingPartDelta::new("");
            if let Some(encrypted) = value["encrypted_content"].as_str() {
                if !encrypted.starts_with(&state.encrypted) {
                    return Err(ModelError::invalid_response(
                        "Conflicting encrypted reasoning",
                    ));
                }
                delta.signature_delta = Some(encrypted[state.encrypted.len()..].to_owned());
                state.encrypted = encrypted.into();
            }
            if let Some(summary) = value.get("summary") {
                delta.provider_details = Some(serde_json::Map::from_iter([(
                    "summary".into(),
                    summary.clone(),
                )]));
            }
            if let Some(part) = state.part {
                self.emit(ModelResponseStreamEvent::PartDelta(PartDeltaEvent {
                    index: part,
                    delta: ModelResponsePartDelta::Thinking(delta),
                }));
            }
        }
        if completed {
            let state = self.items.get_mut(&index).unwrap();
            if !state.closed {
                state.closed = true;
                let parts: Vec<_> = if state.kind == "message" {
                    state.slots.values().map(|(p, _)| *p).collect()
                } else {
                    state.part.into_iter().collect()
                };
                for part in parts {
                    self.emit(ModelResponseStreamEvent::part_end(part));
                }
            }
        }
        Ok(())
    }
    fn slot(&mut self, index: u64, slot: u64) -> Result<usize, ModelError> {
        let state = self
            .items
            .get(&index)
            .ok_or_else(|| ModelError::invalid_response("Content before item"))?;
        if state.kind != "message" || slot > 1024 {
            return Err(ModelError::invalid_response("Invalid content slot"));
        }
        if let Some((part, _)) = state.slots.get(&slot) {
            return Ok(*part);
        }
        if slot != state.slots.len() as u64 {
            return Err(ModelError::invalid_response(
                "Noncontiguous content slot declaration",
            ));
        }
        let part = self.next;
        self.next += 1;
        self.items
            .get_mut(&index)
            .unwrap()
            .slots
            .insert(slot, (part, String::new()));
        let metadata = serde_json::Map::from_iter([
            ("output_index".into(), index.into()),
            ("content_index".into(), slot.into()),
        ]);
        self.emit(ModelResponseStreamEvent::part_start(
            part,
            ModelResponsePart::Text(TextPart::new("").with_provider_details(metadata)),
        ));
        Ok(part)
    }
    fn slot_delta(&mut self, index: u64, slot: u64, text: &str) -> Result<(), ModelError> {
        let part = self.slot(index, slot)?;
        if self.items[&index].closed && !text.is_empty() {
            return Err(ModelError::invalid_response("Content after item end"));
        }
        self.items
            .get_mut(&index)
            .unwrap()
            .slots
            .get_mut(&slot)
            .unwrap()
            .1
            .push_str(text);
        if !text.is_empty() {
            self.emit(ModelResponseStreamEvent::text_delta(part, text));
        }
        Ok(())
    }
    fn delta(&mut self, index: u64, text: &str) -> Result<(), ModelError> {
        let state = self
            .items
            .get_mut(&index)
            .ok_or_else(|| ModelError::invalid_response("Delta before item"))?;
        if state.closed {
            return Err(ModelError::invalid_response("Delta after item completion"));
        }
        state.text.push_str(text);
        if let Some(part) = state.part {
            let event = match state.kind.as_str() {
                "function_call" => PartDeltaEvent::tool_call_args(part, text),
                "reasoning" => PartDeltaEvent {
                    index: part,
                    delta: ModelResponsePartDelta::Thinking(ThinkingPartDelta::new(text)),
                },
                _ => PartDeltaEvent::text(part, text),
            };
            self.emit(ModelResponseStreamEvent::PartDelta(event));
        }
        Ok(())
    }
    fn event(&mut self, value: Value) -> Result<(), ModelError> {
        let kind = value["type"]
            .as_str()
            .ok_or_else(|| ModelError::invalid_response("Responses event lacks type"))?;
        if let Some(id) = value["response"]["id"].as_str() {
            if self.response_id.as_deref().is_some_and(|old| old != id) {
                return Err(ModelError::invalid_response("Response ID changed"));
            }
            self.response_id = Some(id.into());
        }
        match kind {
            "response.output_item.added" | "response.output_item.done" => {
                let index = value["output_index"]
                    .as_u64()
                    .ok_or_else(|| ModelError::invalid_response("Missing output index"))?;
                self.item(index, &value["item"], kind.ends_with("done"))?;
            }
            "response.output_text.delta"
            | "response.reasoning_summary_text.delta"
            | "response.function_call_arguments.delta" => {
                if kind == "response.output_text.delta" {
                    let index = value["output_index"]
                        .as_u64()
                        .ok_or_else(|| ModelError::invalid_response("Missing output index"))?;
                    let slot = value["content_index"]
                        .as_u64()
                        .ok_or_else(|| ModelError::invalid_response("Missing content index"))?;
                    self.slot_delta(
                        index,
                        slot,
                        value["delta"]
                            .as_str()
                            .ok_or_else(|| ModelError::invalid_response("Missing delta"))?,
                    )?;
                    return Ok(());
                }
                if kind == "response.reasoning_summary_text.delta" {
                    let index = value["output_index"]
                        .as_u64()
                        .ok_or_else(|| ModelError::invalid_response("Missing output index"))?;
                    let slot = value["summary_index"].as_u64().unwrap_or(0);
                    if slot > 1024 {
                        return Err(ModelError::invalid_response("Invalid summary index"));
                    }
                    let text = value["delta"]
                        .as_str()
                        .ok_or_else(|| ModelError::invalid_response("Missing summary delta"))?;
                    let item = self
                        .items
                        .get_mut(&index)
                        .ok_or_else(|| ModelError::invalid_response("Summary before item"))?;
                    if item.kind != "reasoning" || item.closed {
                        return Err(ModelError::invalid_response("Invalid reasoning delta"));
                    }
                    item.summaries.entry(slot).or_default().push_str(text);
                    if slot == 0 {
                        self.delta(index, text)?;
                    }
                    return Ok(());
                }
                self.delta(
                    value["output_index"]
                        .as_u64()
                        .ok_or_else(|| ModelError::invalid_response("Missing output index"))?,
                    value["delta"]
                        .as_str()
                        .ok_or_else(|| ModelError::invalid_response("Missing delta"))?,
                )?;
            }
            "response.content_part.added" => {
                let index = value["output_index"]
                    .as_u64()
                    .ok_or_else(|| ModelError::invalid_response("Missing output index"))?;
                let slot = value["content_index"]
                    .as_u64()
                    .ok_or_else(|| ModelError::invalid_response("Missing content index"))?;
                self.slot(index, slot)?;
            }
            "response.failed" | "error" => {
                return Err(ModelError::Api {
                    message: "Responses provider reported failure".into(),
                    code: value["response"]["error"]["code"]
                        .as_str()
                        .or(value["code"].as_str())
                        .map(str::to_owned),
                });
            }
            "response.refusal.delta" | "response.refusal.done" => {} // Retained in terminal output_records; never dispatched.
            "response.cancelled" => return Err(ModelError::Cancelled),
            "response.completed" | "response.incomplete" => {
                let response = &value["response"];
                let expected = if kind == "response.completed" {
                    "completed"
                } else {
                    "incomplete"
                };
                if response["status"].as_str() != Some(expected) {
                    return Err(ModelError::invalid_response(
                        "Terminal event/status mismatch",
                    ));
                }
                let reason = if expected == "completed" {
                    FinishReason::Stop
                } else {
                    match response["incomplete_details"]["reason"].as_str() {
                        Some("max_output_tokens") => FinishReason::Length,
                        Some("content_filter") => FinishReason::ContentFilter,
                        _ => {
                            return Err(ModelError::invalid_response(
                                "Unknown Responses incomplete reason",
                            ));
                        }
                    }
                };
                let output = response["output"].as_array().ok_or_else(|| {
                    ModelError::invalid_response("Terminal response lacks output")
                })?;
                for (index, item) in output.iter().enumerate() {
                    self.item(index as u64, item, true)?;
                }
                if reason == FinishReason::Stop
                    && self.items.values().any(|item| {
                        !item.closed
                            || (item.kind == "function_call"
                                && serde_json::from_str::<Value>(&item.text).is_err())
                    })
                {
                    return Err(ModelError::invalid_response(
                        "Incomplete function call in completed response",
                    ));
                }
                let mut complete = StreamCompleteEvent::new(
                    if reason == FinishReason::Stop
                        && self.items.values().any(|i| i.kind == "function_call")
                    {
                        FinishReason::ToolCall
                    } else {
                        reason
                    },
                );
                complete.metadata = Some(super::responses_metadata::terminal(response)?);
                if output.iter().any(|item| {
                    item["content"]
                        .as_array()
                        .is_some_and(|parts| parts.iter().any(|p| p["type"] == "refusal"))
                }) {
                    complete.finish_reason = FinishReason::ContentFilter;
                }
                if let Some(usage) = response.get("usage").filter(|v| !v.is_null()) {
                    complete.input_tokens = usage["input_tokens"].as_u64();
                    complete.output_tokens = usage["output_tokens"].as_u64();
                    complete.cache_read_tokens =
                        usage["input_tokens_details"]["cached_tokens"].as_u64();
                }
                self.emit(ModelResponseStreamEvent::StreamComplete(complete));
                self.done = true;
            }
            _ => {} // Lifecycle/content-part/done and built-in tool telemetry.
        }
        Ok(())
    }
    fn line(&mut self, bytes: &[u8]) -> Result<(), ModelError> {
        let line = std::str::from_utf8(bytes)
            .map_err(|_| ModelError::invalid_response("Invalid Responses SSE UTF-8"))?
            .trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            if !self.data.is_empty() {
                let data = std::mem::take(&mut self.data);
                let value = serde_json::from_str(&data)
                    .map_err(|_| ModelError::invalid_response("Malformed Responses SSE JSON"))?;
                self.event(value)?;
            }
        } else if let Some(data) = line.strip_prefix("data:") {
            if !self.data.is_empty() {
                self.data.push('\n');
            }
            self.data.push_str(data.strip_prefix(' ').unwrap_or(data));
        }
        Ok(())
    }
}
impl<S: Stream<Item = Result<Bytes, reqwest::Error>>> Stream for ResponsesStream<S> {
    type Item = Result<ModelResponseStreamEvent, ModelError>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(event) = this.queue.pop_front() {
                return Poll::Ready(Some(event));
            }
            if this.done {
                return Poll::Ready(None);
            }
            if let Some(end) = this.buffer.iter().position(|byte| *byte == b'\n') {
                let line: Vec<_> = this.buffer.drain(..=end).collect();
                if let Err(error) = this.line(&line) {
                    this.fail(error);
                }
                continue;
            }
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    this.buffer.extend_from_slice(&bytes);
                    if this.buffer.len() + this.data.len() > 16 * 1024 * 1024 {
                        this.fail(ModelError::invalid_response(
                            "Responses SSE record exceeds 16 MiB",
                        ));
                    }
                }
                Poll::Ready(Some(Err(error))) => this.fail(error.into()),
                Poll::Ready(None) => this.fail(ModelError::incomplete_stream(
                    "Responses EOF before terminal event",
                )),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

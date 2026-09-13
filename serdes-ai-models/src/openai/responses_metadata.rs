//! Native archival metadata, not arbitrary replay or application tool execution.
use crate::ModelError;
use serde_json::{Value, json};
use serdes_ai_core::messages::TerminalMetadata;

/// Bound retained native output and usage metadata to avoid unbounded archives.
pub(super) fn terminal(response: &Value) -> Result<TerminalMetadata, ModelError> {
    let details = json!({
        "provider": "openai", "api": "responses", "status": response.get("status"),
        "usage": response.get("usage"), "metadata": response.get("metadata"),
        "service_tier": response.get("service_tier"), "created_at": response.get("created_at"),
        "incomplete_details": response.get("incomplete_details"),
        "output_records": response.get("output").and_then(Value::as_array).map(|items| items.iter().enumerate().map(|(index,item)| {
            json!({"output_index":index, "type":item.get("type"), "item":item,
                "representation": if matches!(item["type"].as_str(), Some("message"|"reasoning"|"function_call")) { "native_parts" } else { "opaque_not_replayed" }})
        }).collect::<Vec<_>>()),
        "usage_complete": response["usage"]["input_tokens"].as_u64().is_some() && response["usage"]["output_tokens"].as_u64().is_some()
    });
    if serde_json::to_vec(&details)
        .map_err(|_| ModelError::invalid_response("Invalid metadata"))?
        .len()
        > 2 * 1024 * 1024
    {
        return Err(ModelError::invalid_response(
            "Responses terminal metadata exceeds 2 MiB",
        ));
    }
    Ok(TerminalMetadata {
        response_id: response["id"].as_str().map(str::to_owned),
        model: response["model"].as_str().map(str::to_owned),
        details: Some(details),
    })
}

/// Replay only supported passive native items. Archival built-in tools never
/// become requests or executable application calls. Function calls use parts.
pub(super) fn passive_replay(response: &serdes_ai_core::ModelResponse) -> Option<Vec<Value>> {
    let details = response.vendor_details.as_ref()?;
    if details["provider"] != "openai" || details["api"] != "responses" {
        return None;
    }
    let records = details["output_records"].as_array()?;
    let mut result = Vec::new();
    for (index, record) in records.iter().enumerate() {
        if record["output_index"].as_u64()? != index as u64 {
            return None;
        }
        let item = &record["item"];
        let keys: &[&str] =
            match item["type"].as_str()? {
                "message" => {
                    if !item["content"].as_array()?.iter().all(|part| {
                        matches!(part["type"].as_str(), Some("output_text" | "refusal"))
                    }) {
                        return None;
                    }
                    &["type", "id", "role", "status", "content"]
                }
                "reasoning" => &["type", "id", "summary", "encrypted_content", "status"],
                "function_call" => {
                    // Only canonical typed calls authorize replay; the archive cannot
                    // introduce a new callable item or override repaired arguments.
                    let call = response.parts.iter().find_map(|part| match part {
                        serdes_ai_core::ModelResponsePart::ToolCall(call)
                            if call.tool_call_id.as_deref() == item["call_id"].as_str()
                                && call.tool_name == item["name"].as_str().unwrap_or_default() =>
                        {
                            Some(call)
                        }
                        _ => None,
                    })?;
                    result.push(json!({"type":"function_call", "call_id":call.tool_call_id,
                    "name":call.tool_name, "arguments":match &call.args {
                        serdes_ai_core::messages::ToolCallArgs::String(value) => value.clone(),
                        serdes_ai_core::messages::ToolCallArgs::Json(value) => value.to_string(),
                    }}));
                    continue;
                }
                _ => continue,
            };
        let mut replay = serde_json::Map::new();
        for key in keys {
            if let Some(value) = item.get(*key) {
                replay.insert((*key).into(), value.clone());
            }
        }
        if item["type"] == "message" {
            replay.insert("role".into(), "assistant".into());
        }
        result.push(Value::Object(replay));
    }
    Some(result)
}

use super::*;
use crate::anthropic::controls::PrefixMismatchBehavior;
use serdes_ai_core::messages::ToolReturnPart;

/// The JSON body `build_request` produces for one user prompt.
fn request_body(model: &ClaudeCodeOAuthModel) -> serde_json::Value {
    let mut req = ModelRequest::new();
    req.add_user_prompt("Think about this.");
    let request = model.build_request(
        &[req],
        &ModelSettings::new(),
        &ModelRequestParameters::new(),
        false,
    );
    serde_json::to_value(&request).unwrap()
}

fn binding() -> BlockBinding {
    BlockBinding::new(PrefixMismatchBehavior::Error)
}

#[test]
fn adaptive_thinking_with_and_without_display() {
    let plain = ClaudeCodeOAuthModel::new("claude-opus-5-5", "token").with_adaptive_thinking();
    assert_eq!(
        request_body(&plain)["thinking"],
        serde_json::json!({"type": "adaptive"})
    );

    let shown = plain.with_thinking_display(ThinkingDisplay::Omitted);
    assert_eq!(
        request_body(&shown)["thinking"],
        serde_json::json!({"type": "adaptive", "display": "omitted"})
    );
}

#[test]
fn effort_is_sent_as_output_config() {
    let model = ClaudeCodeOAuthModel::new("claude-opus-5-5", "token")
        .with_adaptive_thinking()
        .with_effort(Effort::Medium);

    let body = request_body(&model);

    assert_eq!(
        body["output_config"],
        serde_json::json!({"effort": "medium"})
    );
}

#[test]
fn legacy_thinking_body_is_unchanged() {
    let body = request_body(
        &ClaudeCodeOAuthModel::new("claude-sonnet-4", "token").with_thinking(Some(5000)),
    );

    assert_eq!(
        body["thinking"],
        serde_json::json!({"type": "enabled", "budget_tokens": 5000})
    );
    assert!(body.get("output_config").is_none());
}

#[test]
fn last_thinking_call_wins() {
    let adaptive = ClaudeCodeOAuthModel::new("claude-opus-5-5", "token")
        .with_thinking(Some(1024))
        .with_adaptive_thinking();
    assert_eq!(
        request_body(&adaptive)["thinking"],
        serde_json::json!({"type": "adaptive"})
    );

    let legacy = ClaudeCodeOAuthModel::new("claude-opus-5-5", "token")
        .with_adaptive_thinking()
        .with_thinking(None);
    assert_eq!(
        request_body(&legacy)["thinking"],
        serde_json::json!({"type": "enabled"})
    );
}

#[test]
fn blank_text_parts_are_not_sent_as_blocks() {
    let model = ClaudeCodeOAuthModel::new("claude-opus-5-5", "token");

    let mut req = ModelRequest::new();
    req.add_user_prompt(UserContent::Parts(vec![
        UserContentPart::Text {
            text: String::new(),
        },
        UserContentPart::Text {
            text: "Question".into(),
        },
        UserContentPart::Text { text: " \t".into() },
    ]));
    req.add_part(ModelRequestPart::ModelResponse(Box::new(
        ModelResponse::with_parts(vec![
            ModelResponsePart::Thinking(ThinkingPart::new("").with_signature("sig-1")),
            ModelResponsePart::Text(TextPart::new("")),
            ModelResponsePart::ToolCall(
                ToolCallPart::new("search", serde_json::json!({})).with_tool_call_id("tool_1"),
            ),
        ]),
    )));
    req.add_part(ModelRequestPart::UserPrompt(UserPromptPart::new("  ")));
    req.add_part(ModelRequestPart::ToolReturn(
        ToolReturnPart::success("search", "found").with_tool_call_id("tool_1"),
    ));
    // An assistant turn holding only empty text is dropped as a whole.
    req.add_part(ModelRequestPart::ModelResponse(Box::new(
        ModelResponse::with_parts(vec![ModelResponsePart::Text(TextPart::new(" "))]),
    )));

    let request = model.build_request(
        &[req],
        &ModelSettings::new(),
        &ModelRequestParameters::new(),
        false,
    );
    let body = serde_json::to_value(&request).unwrap();
    let blocks: Vec<serde_json::Value> = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["content"].as_array().cloned())
        .flatten()
        .collect();

    assert!(
        blocks
            .iter()
            .filter(|b| b["type"] == "text")
            .all(|b| !b["text"].as_str().unwrap().trim().is_empty()),
        "no blank text block may be sent: {body}"
    );
    assert!(
        blocks
            .iter()
            .any(|b| b["type"] == "thinking" && b["signature"] == "sig-1"),
        "an empty thinking block keeps its signature: {body}"
    );
    let roles: Vec<_> = request.messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, vec!["user", "assistant", "user"]);
}

#[test]
fn block_binding_beta_flag_only_when_opted_in_and_thinking_on() {
    let plain = ClaudeCodeOAuthModel::new("claude-opus-5-5", "token");
    assert_eq!(plain.anthropic_beta(), OAUTH_BETA_FLAGS);

    let thinking_off = plain.clone().with_thinking_block_binding(binding());
    assert_eq!(thinking_off.anthropic_beta(), OAUTH_BETA_FLAGS);
    assert!(request_body(&thinking_off).get("thinking").is_none());

    let opted_in = plain
        .with_adaptive_thinking()
        .with_thinking_block_binding(binding());
    assert_eq!(
        opted_in.anthropic_beta(),
        format!("{OAUTH_BETA_FLAGS},{THINKING_BINDING_BETA}")
    );
    assert_eq!(
        request_body(&opted_in)["thinking"],
        serde_json::json!({
            "type": "adaptive",
            "block_binding": {"prefix_mismatch_behavior": "error"}
        })
    );
}

/// Both request paths send the beta value of the one shared helper.
#[tokio::test]
async fn both_request_paths_send_the_block_binding_beta_flag() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/messages"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(
            r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"model":"claude-opus-5-5","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#,
            "application/json",
        ))
        .mount(&server)
        .await;
    let model = ClaudeCodeOAuthModel::new("claude-opus-5-5", "token")
        .with_config(ClaudeCodeConfig {
            api_base_url: server.uri(),
            ..ClaudeCodeConfig::default()
        })
        .with_adaptive_thinking()
        .with_thinking_block_binding(binding());

    let settings = ModelSettings::default();
    let params = ModelRequestParameters::new();
    model
        .request(&[ModelRequest::new()], &settings, &params)
        .await
        .expect("mocked request succeeds");
    let _stream = model
        .request_stream(&[ModelRequest::new()], &settings, &params)
        .await
        .expect("mocked stream request succeeds");

    let requests = server.received_requests().await.expect("requests recorded");
    assert_eq!(requests.len(), 2);
    let expected = format!("{OAUTH_BETA_FLAGS},{THINKING_BINDING_BETA}");
    for request in requests {
        let values: Vec<_> = request
            .headers
            .get_all("anthropic-beta")
            .iter()
            .map(|v| v.to_str().unwrap().to_owned())
            .collect();
        assert_eq!(values, vec![expected.clone()]);
    }
}

#[test]
fn convert_response_maps_refusal_and_keeps_stop_details_without_explanation() {
    let model = ClaudeCodeOAuthModel::new("claude-opus-5-5", "token");
    let response: ClaudeResponse = serde_json::from_str(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-opus-5-5","stop_reason":"refusal","stop_details":{"type":"refusal","category":"reasoning_extraction","explanation":"private text"}}"#,
    )
    .unwrap();

    let result = model.convert_response(response);

    assert_eq!(result.finish_reason, Some(FinishReason::ContentFilter));
    let stop = crate::refusal::StopDetails::from_response(&result).unwrap();
    assert_eq!(stop.category.as_deref(), Some("reasoning_extraction"));
    assert!(
        !result
            .vendor_details
            .unwrap()
            .to_string()
            .contains("private text")
    );
}

#[test]
fn convert_response_end_turn_keeps_vendor_details_none() {
    let model = ClaudeCodeOAuthModel::new("claude-opus-5-5", "token");
    let response: ClaudeResponse = serde_json::from_str(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"model":"claude-opus-5-5","stop_reason":"end_turn"}"#,
    )
    .unwrap();

    let result = model.convert_response(response);

    assert_eq!(result.finish_reason, Some(FinishReason::Stop));
    assert!(result.vendor_details.is_none());
}

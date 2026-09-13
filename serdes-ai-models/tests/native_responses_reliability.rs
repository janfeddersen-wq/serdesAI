#![cfg(feature = "openai")]
use serde_json::json;
use serdes_ai_core::{ModelRequest, ModelRequestPart, ModelSettings};
use serdes_ai_models::{Model, ModelError, ModelRequestParameters, openai::OpenAIResponsesModel};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

fn response(status: &str) -> serde_json::Value {
    json!({"id":"resp_1","object":"response","created_at":1,"model":"o3",
        "status":status,"output":[{"type":"reasoning","id":"rs_1","summary":[],
        "encrypted_content":"secret-ciphertext"},
        {"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]})
}

#[tokio::test]
async fn encrypted_reasoning_replays_as_native_ordered_item_without_debug_leak() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response("completed")))
        .mount(&server)
        .await;
    let model = OpenAIResponsesModel::new("o3", "test").with_base_url(server.uri());
    let settings = ModelSettings::new();
    let params = ModelRequestParameters::new();
    let result = model.request(&[], &settings, &params).await.unwrap();
    assert_eq!(result.parts.len(), 2);
    assert!(!format!("{result:?}").contains("secret-ciphertext"));
    let request = ModelRequest::with_parts(vec![ModelRequestPart::ModelResponse(Box::new(result))]);
    model.request(&[request], &settings, &params).await.unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(
        body["input"][0],
        json!({"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"secret-ciphertext"})
    );
    assert_eq!(body["input"][1]["role"], "assistant");
}

#[tokio::test]
async fn buffered_responses_reject_nonterminal_cancelled_and_failed_status() {
    for status in ["in_progress", "cancelled", "failed"] {
        let server = MockServer::start().await;
        let mut body = response(status);
        if status == "failed" {
            body["error"] = json!({"message":"failed","code":"server_error"});
        }
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;
        let model = OpenAIResponsesModel::new("o3", "test").with_base_url(server.uri());
        let error = match model
            .request(&[], &ModelSettings::new(), &ModelRequestParameters::new())
            .await
        {
            Ok(_) => panic!("unexpected success for {status}"),
            Err(error) => error,
        };
        match status {
            "in_progress" => assert!(matches!(error, ModelError::IncompleteStream(_))),
            "cancelled" => assert!(matches!(error, ModelError::Cancelled)),
            _ => assert!(matches!(error, ModelError::Api { .. })),
        }
    }
}

#[tokio::test]
async fn multicontent_replay_identical_across_transport_and_serde() {
    use futures::StreamExt;
    use serdes_ai_core::{ModelResponse, ModelResponseStreamEvent as Event};
    let native = json!({"id":"r","object":"response","created_at":1,"model":"local","status":"completed","output":[
        {"type":"reasoning","id":"rs","summary":[{"type":"summary_text","text":"one"},{"type":"summary_text","text":"two"}],"encrypted_content":"cipher"},
        {"type":"message","id":"msg","role":"assistant","content":[{"type":"output_text","text":"first","annotations":[]},{"type":"output_text","text":"second","annotations":[]},{"type":"refusal","refusal":"no"}]}
    ]});
    let mut expected = None;
    for streaming in [false, true] {
        let server = MockServer::start().await;
        let template = if streaming {
            ResponseTemplate::new(200).set_body_string(format!(
                "data: {}\n\n",
                json!({"type":"response.completed","response":native})
            ))
        } else {
            ResponseTemplate::new(200).set_body_json(&native)
        };
        Mock::given(method("POST"))
            .respond_with(template)
            .up_to_n_times(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&native))
            .mount(&server)
            .await;
        let model = OpenAIResponsesModel::new("local", "test").with_base_url(server.uri());
        let settings = ModelSettings::new();
        let params = ModelRequestParameters::new();
        let result = if streaming {
            let mut events = model.request_stream(&[], &settings, &params).await.unwrap();
            let mut response = ModelResponse::with_parts(vec![]);
            while let Some(event) = events.next().await {
                match event.unwrap() {
                    Event::PartStart(start) => {
                        assert_eq!(start.index, response.parts.len());
                        response.parts.push(start.part);
                    }
                    Event::PartDelta(delta) => {
                        assert!(delta.delta.apply(&mut response.parts[delta.index]));
                    }
                    Event::StreamComplete(complete) => {
                        complete.metadata.unwrap().apply(&mut response);
                    }
                    _ => {}
                }
            }
            response
        } else {
            model.request(&[], &settings, &params).await.unwrap()
        };
        let restored: ModelResponse =
            serde_json::from_str(&serde_json::to_string(&result).unwrap()).unwrap();
        for value in [result, restored] {
            let request =
                ModelRequest::with_parts(vec![ModelRequestPart::ModelResponse(Box::new(value))]);
            model.request(&[request], &settings, &params).await.unwrap();
            let requests = server.received_requests().await.unwrap();
            let body: serde_json::Value =
                serde_json::from_slice(&requests.last().unwrap().body).unwrap();
            assert_eq!(body["input"], native["output"]);
            if let Some(previous) = &expected {
                assert_eq!(previous, &body["input"]);
            } else {
                expected = Some(body["input"].clone());
            }
            assert_eq!(
                body["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|v| v["id"] == "msg")
                    .count(),
                1
            );
        }
    }
}

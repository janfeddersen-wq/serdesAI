use futures::StreamExt;
use serde_json::json;
use serdes_ai_agent::AgentBuilder;
use serdes_ai_models::openai::OpenAIResponsesModel;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};
#[tokio::test]
async fn native_responses_only_dispatch_completed_tools() {
    for completed in [false, true] {
        let server = MockServer::start().await;
        let item = json!({"type":"function_call","id":"fc_1","call_id":"call_1","name":"next","arguments":"{}"});
        let mut body = format!(
            "data: {}\n\n",
            json!({"type":"response.output_item.added","output_index":0,"item":item})
        );
        if completed {
            body += &format!(
                "data: {}\n\n",
                json!({"type":"response.completed","response":{"id":"r1","status":"completed","output":[item]}})
            );
        }
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .up_to_n_times(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("POST")).respond_with(ResponseTemplate::new(200).set_body_string(format!("data: {}\n\n", json!({"type":"response.completed","response":{"id":"r2","status":"completed","output":[{"type":"message","id":"m2","content":[{"type":"output_text","text":"done"}]}]}})))).with_priority(2).mount(&server).await;
        let calls = Arc::new(AtomicUsize::new(0));
        let count = calls.clone();
        let agent = AgentBuilder::<(), String>::new(
            OpenAIResponsesModel::new("local", "test").with_base_url(server.uri()),
        )
        .tool_fn("next", "next", move |_, _: serde_json::Value| {
            count.fetch_add(1, Ordering::SeqCst);
            Ok(serdes_ai_tools::ToolReturn::text("ok"))
        })
        .build();
        let mut stream = agent.run_stream("test", ()).await.unwrap();
        let mut failed = false;
        while let Some(event) = stream.next().await {
            failed |= event.is_err();
        }
        assert_eq!(failed, !completed);
        assert_eq!(calls.load(Ordering::SeqCst), usize::from(completed));
    }
}

#[derive(Clone, Default)]
struct MetadataSink(Arc<std::sync::Mutex<Vec<serdes_ai_agent::AgentCheckpoint>>>);
#[async_trait::async_trait]
impl serdes_ai_agent::CheckpointSink for MetadataSink {
    async fn save(&self, checkpoint: &serdes_ai_agent::AgentCheckpoint) -> Result<(), String> {
        self.0.lock().unwrap().push(checkpoint.clone());
        Ok(())
    }
}
#[tokio::test]
async fn native_metadata_reaches_checkpoint_and_serde() {
    for output in [
        json!([]),
        json!([{"type":"message","id":"m","content":[{"type":"refusal","refusal":"private refusal"}]}]),
        json!([{"type":"reasoning","id":"rs","summary":[],"encrypted_content":"secret cipher"},{"type":"message","id":"m","content":[{"type":"output_text","text":"done","annotations":[]}]}]),
    ] {
        let server = MockServer::start().await;
        let sink = MetadataSink::default();
        let body = json!({"type":"response.completed","response":{"id":"metadata-id","model":"actual","status":"completed","output":output,"usage":{"input_tokens":10,"output_tokens":3,"output_tokens_details":{"reasoning_tokens":2}}}});
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string(format!("data: {body}\n\n")))
            .mount(&server)
            .await;
        let agent = AgentBuilder::<(), String>::new(
            OpenAIResponsesModel::new("alias", "test").with_base_url(server.uri()),
        )
        .max_output_retries(0)
        .checkpoint_sink(sink.clone())
        .build();
        let mut stream = agent.run_stream("test", ()).await.unwrap();
        while stream.next().await.is_some() {}
        let records = sink.0.lock().unwrap();
        let checkpoint = records
            .iter()
            .find(|c| c.boundary == serdes_ai_agent::CheckpointBoundary::AfterResponse)
            .unwrap();
        let response = checkpoint.response.as_ref().unwrap();
        assert_eq!(response.vendor_id.as_deref(), Some("metadata-id"));
        assert_eq!(response.model_name.as_deref(), Some("actual"));
        assert_eq!(
            response.vendor_details.as_ref().unwrap()["output_records"]
                .as_array()
                .unwrap()
                .len(),
            output.as_array().unwrap().len()
        );
        assert!(!format!("{checkpoint:?}").contains("secret cipher"));
        let encoded = serde_json::to_string(checkpoint).unwrap();
        let restored: serdes_ai_agent::AgentCheckpoint = serde_json::from_str(&encoded).unwrap();
        assert_eq!(restored.response, checkpoint.response);
    }
}

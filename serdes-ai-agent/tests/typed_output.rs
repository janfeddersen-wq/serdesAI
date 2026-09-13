use futures::stream;
use serdes_ai_agent::{AgentBuilder, RunOptions};
use serdes_ai_core::{
    FinishReason, ModelResponse, ModelResponsePart, ModelResponseStreamEvent as Event,
};
use serdes_ai_models::FunctionModel;
#[tokio::test]
async fn transformed_output_matches_nonstream() {
    let model = FunctionModel::with_both(
        |_, _| ModelResponse::text("original"),
        |_, _| {
            Box::pin(stream::iter(vec![
                Ok(Event::part_start(0, ModelResponsePart::text("original"))),
                Ok(Event::StreamComplete(
                    serdes_ai_core::messages::StreamCompleteEvent::new(FinishReason::Stop),
                )),
            ]))
        },
    );
    let agent = AgentBuilder::<(), String>::new(model)
        .output_validator_fn(|value, _| Ok(format!("transformed:{value}")))
        .build();
    let expected = agent.run("test", ()).await.unwrap().output;
    let actual = agent
        .run_stream_typed("test", (), RunOptions::default())
        .await
        .unwrap()
        .finish()
        .await
        .unwrap();
    assert_eq!(actual, Some(expected));
}
#[tokio::test]
async fn partial_output_is_not_a_typed_result() {
    let model = FunctionModel::with_stream(|_, _| {
        Box::pin(stream::iter(vec![
            Ok(Event::part_start(0, ModelResponsePart::text("partial"))),
            Ok(Event::StreamComplete(
                serdes_ai_core::messages::StreamCompleteEvent::new(FinishReason::Length),
            )),
        ]))
    });
    let agent = AgentBuilder::<(), String>::new(model).build();
    assert_eq!(
        agent
            .run_stream_typed("test", (), RunOptions::default())
            .await
            .unwrap()
            .finish()
            .await
            .unwrap(),
        None
    );
}

struct FinalSchema;
impl serdes_ai_agent::OutputSchema<String> for FinalSchema {
    fn tool_name(&self) -> Option<&str> {
        Some("final")
    }
    fn parse_text(&self, _: &str) -> Result<String, serdes_ai_agent::OutputParseError> {
        Err(serdes_ai_agent::OutputParseError::NotFound)
    }
    fn parse_tool_call(
        &self,
        _: &str,
        value: &serde_json::Value,
    ) -> Result<String, serdes_ai_agent::OutputParseError> {
        value["answer"]
            .as_str()
            .map(str::to_owned)
            .ok_or(serdes_ai_agent::OutputParseError::NotFound)
    }
}
#[tokio::test]
async fn mixed_output_tool_end_strategy_parity() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    for strategy in [
        serdes_ai_agent::EndStrategy::Early,
        serdes_ai_agent::EndStrategy::Exhaustive,
    ] {
        for streaming in [false, true] {
            let calls = Arc::new(AtomicUsize::new(0));
            let count = calls.clone();
            let response = || {
                ModelResponse::with_parts(vec![
                    ModelResponsePart::tool_call("work", serde_json::json!({})),
                    ModelResponsePart::tool_call("final", serde_json::json!({"answer":"done"})),
                ])
            };
            let model = FunctionModel::with_both(
                move |_, _| response(),
                move |_, _| {
                    let events = response()
                        .parts
                        .into_iter()
                        .enumerate()
                        .map(|(index, part)| Ok(Event::part_start(index, part)))
                        .chain(std::iter::once(Ok(Event::StreamComplete(
                            serdes_ai_core::messages::StreamCompleteEvent::new(
                                FinishReason::ToolCall,
                            ),
                        ))))
                        .collect::<Vec<_>>();
                    Box::pin(stream::iter(events))
                },
            );
            let agent = AgentBuilder::<(), String>::new(model)
                .output_schema(FinalSchema)
                .end_strategy(strategy)
                .tool_fn("work", "work", move |_, _: serde_json::Value| {
                    count.fetch_add(1, Ordering::SeqCst);
                    Ok(serdes_ai_tools::ToolReturn::text("ok"))
                })
                .build();
            let result = if streaming {
                agent
                    .run_stream_typed("test", (), RunOptions::default())
                    .await
                    .unwrap()
                    .finish()
                    .await
                    .unwrap()
                    .unwrap()
            } else {
                agent.run("test", ()).await.unwrap().output
            };
            assert_eq!(result, "done");
            assert_eq!(
                calls.load(Ordering::SeqCst),
                usize::from(strategy == serdes_ai_agent::EndStrategy::Exhaustive)
            );
        }
    }
}

#[tokio::test]
async fn bounded_parallel_batch_matches_nonstream() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    for streaming in [false, true] {
        for limit in [1, 2] {
            let current = Arc::new(AtomicUsize::new(0));
            let peak = Arc::new(AtomicUsize::new(0));
            let requests = Arc::new(AtomicUsize::new(0));
            let normal = requests.clone();
            let streamed = requests.clone();
            let response = |step| {
                if step == 0 {
                    ModelResponse::with_parts(
                        (0..4)
                            .map(|_| ModelResponsePart::tool_call("work", serde_json::json!({})))
                            .collect(),
                    )
                } else {
                    ModelResponse::text("done")
                }
            };
            let model = FunctionModel::with_both(
                move |_, _| response(normal.fetch_add(1, Ordering::SeqCst)),
                move |_, _| {
                    let step = streamed.fetch_add(1, Ordering::SeqCst);
                    Box::pin(stream::iter(
                        response(step)
                            .parts
                            .into_iter()
                            .enumerate()
                            .map(|(index, part)| Ok(Event::part_start(index, part)))
                            .chain(std::iter::once(Ok(Event::StreamComplete(
                                serdes_ai_core::messages::StreamCompleteEvent::new(if step == 0 {
                                    FinishReason::ToolCall
                                } else {
                                    FinishReason::Stop
                                }),
                            ))))
                            .collect::<Vec<_>>(),
                    ))
                },
            );
            let active = current.clone();
            let maximum = peak.clone();
            let agent = AgentBuilder::<(), String>::new(model)
                .parallel_tool_calls(true)
                .max_concurrent_tools(limit)
                .tool_fn_async("work", "work", move |_, _: serde_json::Value| {
                    let active = active.clone();
                    let maximum = maximum.clone();
                    async move {
                        let value = active.fetch_add(1, Ordering::SeqCst) + 1;
                        maximum.fetch_max(value, Ordering::SeqCst);
                        tokio::task::yield_now().await;
                        active.fetch_sub(1, Ordering::SeqCst);
                        Ok(serdes_ai_tools::ToolReturn::text("ok"))
                    }
                })
                .build();
            let output = if streaming {
                agent
                    .run_stream_typed("test", (), RunOptions::default())
                    .await
                    .unwrap()
                    .finish()
                    .await
                    .unwrap()
                    .unwrap()
            } else {
                agent.run("test", ()).await.unwrap().output
            };
            assert_eq!(output, "done");
            assert_eq!(peak.load(Ordering::SeqCst), limit);
            assert_eq!(current.load(Ordering::SeqCst), 0);
        }
    }
}

#[tokio::test]
async fn malformed_mixed_output_retries_with_paired_ordered_history() {
    use serdes_ai_core::{ModelRequest, ModelRequestPart};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    fn check(messages: &[ModelRequest]) {
        let mut pending = Vec::new();
        let mut seen = Vec::new();
        for request in messages {
            for part in &request.parts {
                match part {
                    ModelRequestPart::ModelResponse(response) => {
                        for part in &response.parts {
                            if let ModelResponsePart::ToolCall(call) = part {
                                pending.push(call.tool_call_id.clone().unwrap());
                            }
                        }
                    }
                    ModelRequestPart::ToolReturn(result) => {
                        let id = result.tool_call_id.clone().unwrap();
                        assert!(pending.contains(&id));
                        seen.push(id);
                    }
                    _ => {}
                }
            }
        }
        assert_eq!(pending, vec!["work-id", "output-id"]);
        assert_eq!(seen, pending);
    }
    for streaming in [false, true] {
        let step = Arc::new(AtomicUsize::new(0));
        let normal = step.clone();
        let streamed = step.clone();
        let response = |step| {
            if step == 0 {
                ModelResponse::with_parts(vec![
                    ModelResponsePart::ToolCall(
                        serdes_ai_core::messages::ToolCallPart::new("work", serde_json::json!({}))
                            .with_tool_call_id("work-id"),
                    ),
                    ModelResponsePart::ToolCall(
                        serdes_ai_core::messages::ToolCallPart::new(
                            "final",
                            serde_json::json!({"invalid":true}),
                        )
                        .with_tool_call_id("output-id"),
                    ),
                ])
            } else {
                ModelResponse::with_parts(vec![ModelResponsePart::tool_call(
                    "final",
                    serde_json::json!({"answer":"done"}),
                )])
            }
        };
        let model = FunctionModel::with_both(
            move |messages, _| {
                let step = normal.fetch_add(1, Ordering::SeqCst);
                if step > 0 {
                    check(messages);
                }
                response(step)
            },
            move |messages, _| {
                let step = streamed.fetch_add(1, Ordering::SeqCst);
                if step > 0 {
                    check(messages);
                }
                Box::pin(stream::iter(
                    response(step)
                        .parts
                        .into_iter()
                        .enumerate()
                        .map(|(index, part)| Ok(Event::part_start(index, part)))
                        .chain(std::iter::once(Ok(Event::StreamComplete(
                            serdes_ai_core::messages::StreamCompleteEvent::new(
                                FinishReason::ToolCall,
                            ),
                        ))))
                        .collect::<Vec<_>>(),
                ))
            },
        );
        let agent = AgentBuilder::<(), String>::new(model)
            .output_schema(FinalSchema)
            .tool_fn("work", "must not execute", |_, _: serde_json::Value| {
                panic!("malformed output dispatched a tool")
            })
            .build();
        let result = if streaming {
            agent
                .run_stream_typed("test", (), RunOptions::default())
                .await
                .unwrap()
                .finish()
                .await
                .unwrap()
                .unwrap()
        } else {
            agent.run("test", ()).await.unwrap().output
        };
        assert_eq!(result, "done");
        assert_eq!(step.load(Ordering::SeqCst), 2);
    }
}

//! Ordered, bounded tool batches. Dropping the batch drops all in-flight futures.
use crate::{RunContext, agent::RegisteredTool};
use futures::{StreamExt, stream};
use serdes_ai_core::messages::ToolCallPart;
use serdes_ai_tools::{ToolError, ToolReturn};
pub(crate) async fn execute<D: Send + Sync>(
    tools: &[RegisteredTool<D>],
    calls: Vec<ToolCallPart>,
    ctx: &RunContext<D>,
    parallel: bool,
    limit: Option<usize>,
) -> Vec<(ToolCallPart, Result<ToolReturn, ToolError>)> {
    let concurrency = if parallel {
        limit.unwrap_or(calls.len()).max(1)
    } else {
        1
    };
    stream::iter(calls.into_iter().map(|call| async move {
        let context = ctx.for_tool(&call.tool_name, call.tool_call_id.clone());
        let result = if let Some(tool) = tools.iter().find(|t| t.definition.name == call.tool_name)
        {
            let args = call.args.to_json();
            let mut retries = 0;
            loop {
                match tool.executor.execute(args.clone(), &context).await {
                    Err(error) if error.is_retryable() && retries < tool.max_retries => {
                        retries += 1;
                    }
                    result => break result,
                }
            }
        } else {
            Err(ToolError::NotFound(call.tool_name.clone()))
        };
        (call, result)
    }))
    .buffered(concurrency)
    .collect()
    .await
}

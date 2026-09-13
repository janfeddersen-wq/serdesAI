//! Type-aware validation before the type-erased streaming completion notification.
use crate::output::{OutputSchema, OutputValidator};
use crate::{OutputValidationError, RunContext};
use serdes_ai_core::{ModelResponse, ModelResponsePart};
use std::sync::Arc;
pub(crate) async fn validate<D: Send + Sync, O: Send + Sync>(
    schema: &dyn OutputSchema<O>,
    validators: &[Arc<dyn OutputValidator<O, D>>],
    response: &ModelResponse,
    context: &RunContext<D>,
) -> Result<O, OutputValidationError> {
    let mut found = None;
    for part in &response.parts {
        let parsed = match part {
            ModelResponsePart::Text(text) => schema.parse_text(&text.content),
            ModelResponsePart::ToolCall(call)
                if schema.tool_name() == Some(call.tool_name.as_str()) =>
            {
                schema.parse_tool_call(&call.tool_name, &call.args.to_json())
            }
            _ => continue,
        };
        if let Ok(output) = parsed {
            found = Some(output);
        }
    }
    let mut output = found.ok_or_else(|| {
        OutputValidationError::failed("Output did not match the configured schema")
    })?;
    for validator in validators {
        output = validator.validate(output, context).await?;
    }
    Ok(output)
}

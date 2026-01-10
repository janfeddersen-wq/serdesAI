//! Wrapper toolset implementation.
//!
//! This module provides `WrapperToolset`, which allows custom pre/post
//! processing around tool calls.

use async_trait::async_trait;
use serde_json::Value as JsonValue;
use serdes_ai_tools::{RunContext, ToolError, ToolReturn};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::{AbstractToolset, ToolsetTool};

/// Type alias for before-call hooks.
pub type BeforeCallHook<Deps> = dyn Fn(&str, &JsonValue, &RunContext<Deps>) + Send + Sync;

/// Type alias for after-call hooks.
pub type AfterCallHook<Deps> =
    dyn Fn(&str, &Result<ToolReturn, ToolError>, &RunContext<Deps>) + Send + Sync;

/// Wrapper that allows custom pre/post processing.
///
/// This is useful for adding logging, metrics, or other cross-cutting
/// concerns to tool calls.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_toolsets::{WrapperToolset, FunctionToolset};
///
/// let toolset = FunctionToolset::new().tool(my_tool);
///
/// let wrapped = WrapperToolset::new(toolset)
///     .before(|name, args, ctx| {
///         println!("Calling tool: {} with args: {:?}", name, args);
///     })
///     .after(|name, result, ctx| {
///         match result {
///             Ok(_) => println!("Tool {} succeeded", name),
///             Err(e) => println!("Tool {} failed: {}", name, e),
///         }
///     });
/// ```
pub struct WrapperToolset<T, Deps = ()> {
    inner: T,
    before_call: Option<Arc<BeforeCallHook<Deps>>>,
    after_call: Option<Arc<AfterCallHook<Deps>>>,
    _phantom: PhantomData<fn() -> Deps>,
}

impl<T, Deps> WrapperToolset<T, Deps>
where
    T: AbstractToolset<Deps>,
{
    /// Create a new wrapper toolset.
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            before_call: None,
            after_call: None,
            _phantom: PhantomData,
        }
    }

    /// Add a before-call hook.
    #[must_use]
    pub fn before<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &JsonValue, &RunContext<Deps>) + Send + Sync + 'static,
    {
        self.before_call = Some(Arc::new(f));
        self
    }

    /// Add an after-call hook.
    #[must_use]
    pub fn after<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &Result<ToolReturn, ToolError>, &RunContext<Deps>) + Send + Sync + 'static,
    {
        self.after_call = Some(Arc::new(f));
        self
    }

    /// Get the inner toolset.
    #[must_use]
    pub fn inner(&self) -> &T {
        &self.inner
    }
}

#[async_trait]
impl<T, Deps> AbstractToolset<Deps> for WrapperToolset<T, Deps>
where
    T: AbstractToolset<Deps>,
    Deps: Send + Sync,
{
    fn id(&self) -> Option<&str> {
        self.inner.id()
    }

    fn type_name(&self) -> &'static str {
        "WrapperToolset"
    }

    fn label(&self) -> String {
        format!("WrapperToolset({})", self.inner.label())
    }

    async fn get_tools(
        &self,
        ctx: &RunContext<Deps>,
    ) -> Result<HashMap<String, ToolsetTool>, ToolError> {
        self.inner.get_tools(ctx).await
    }

    async fn call_tool(
        &self,
        name: &str,
        args: JsonValue,
        ctx: &RunContext<Deps>,
        tool: &ToolsetTool,
    ) -> Result<ToolReturn, ToolError> {
        // Call before hook
        if let Some(ref before) = self.before_call {
            before(name, &args, ctx);
        }

        // Execute the tool
        let result = self.inner.call_tool(name, args, ctx, tool).await;

        // Call after hook
        if let Some(ref after) = self.after_call {
            after(name, &result, ctx);
        }

        result
    }

    async fn enter(&self) -> Result<(), ToolError> {
        self.inner.enter().await
    }

    async fn exit(&self) -> Result<(), ToolError> {
        self.inner.exit().await
    }
}

impl<T: std::fmt::Debug, Deps> std::fmt::Debug for WrapperToolset<T, Deps> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WrapperToolset")
            .field("inner", &self.inner)
            .field("has_before", &self.before_call.is_some())
            .field("has_after", &self.after_call.is_some())
            .finish()
    }
}

/// Logging wrapper for tool calls.
#[derive(Debug, Clone)]
pub struct LoggingWrapper {
    prefix: String,
}

impl LoggingWrapper {
    /// Create a new logging wrapper.
    #[must_use]
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }

    /// Wrap a toolset with logging.
    pub fn wrap<T, Deps>(self, toolset: T) -> WrapperToolset<T, Deps>
    where
        T: AbstractToolset<Deps>,
        Deps: Send + Sync + 'static,
    {
        let before_prefix = self.prefix.clone();
        let after_prefix = self.prefix.clone();

        WrapperToolset::new(toolset)
            .before(move |name, args, _ctx| {
                tracing::debug!(
                    target: "tool_calls",
                    "[{}] Calling tool '{}' with args: {}",
                    before_prefix,
                    name,
                    args
                );
            })
            .after(move |name, result, _ctx| match result {
                Ok(_) => {
                    tracing::debug!(
                        target: "tool_calls",
                        "[{}] Tool '{}' completed successfully",
                        after_prefix,
                        name
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        target: "tool_calls",
                        "[{}] Tool '{}' failed: {}",
                        after_prefix,
                        name,
                        e
                    );
                }
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FunctionToolset;
    use async_trait::async_trait;
    use serdes_ai_tools::{Tool, ToolDefinition};
    use std::sync::atomic::{AtomicU32, Ordering};

    struct TestTool;

    #[async_trait]
    impl Tool<()> for TestTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("test", "Test tool")
        }

        async fn call(
            &self,
            _ctx: &RunContext<()>,
            _args: JsonValue,
        ) -> Result<ToolReturn, ToolError> {
            Ok(ToolReturn::text("result"))
        }
    }

    #[tokio::test]
    async fn test_wrapper_before_hook() {
        let before_count = Arc::new(AtomicU32::new(0));
        let counter = before_count.clone();

        let toolset = FunctionToolset::new().tool(TestTool);
        let wrapped = WrapperToolset::new(toolset).before(move |_, _, _| {
            counter.fetch_add(1, Ordering::SeqCst);
        });

        let ctx = RunContext::minimal("test");
        let tools = wrapped.get_tools(&ctx).await.unwrap();
        let tool = tools.get("test").unwrap();

        wrapped
            .call_tool("test", serde_json::json!({}), &ctx, tool)
            .await
            .unwrap();

        assert_eq!(before_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_wrapper_after_hook() {
        let after_count = Arc::new(AtomicU32::new(0));
        let counter = after_count.clone();

        let toolset = FunctionToolset::new().tool(TestTool);
        let wrapped = WrapperToolset::new(toolset).after(move |_, _, _| {
            counter.fetch_add(1, Ordering::SeqCst);
        });

        let ctx = RunContext::minimal("test");
        let tools = wrapped.get_tools(&ctx).await.unwrap();
        let tool = tools.get("test").unwrap();

        wrapped
            .call_tool("test", serde_json::json!({}), &ctx, tool)
            .await
            .unwrap();

        assert_eq!(after_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_wrapper_both_hooks() {
        let call_order = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let before_order = call_order.clone();
        let after_order = call_order.clone();

        let toolset = FunctionToolset::new().tool(TestTool);
        let wrapped = WrapperToolset::new(toolset)
            .before(move |_, _, _| {
                before_order.lock().push("before");
            })
            .after(move |_, _, _| {
                after_order.lock().push("after");
            });

        let ctx = RunContext::minimal("test");
        let tools = wrapped.get_tools(&ctx).await.unwrap();
        let tool = tools.get("test").unwrap();

        wrapped
            .call_tool("test", serde_json::json!({}), &ctx, tool)
            .await
            .unwrap();

        let order = call_order.lock();
        assert_eq!(*order, vec!["before", "after"]);
    }

    #[tokio::test]
    async fn test_wrapper_receives_args() {
        let received_name = Arc::new(parking_lot::Mutex::new(String::new()));
        let received_args = Arc::new(parking_lot::Mutex::new(serde_json::Value::Null));

        let name_ref = received_name.clone();
        let args_ref = received_args.clone();

        let toolset = FunctionToolset::new().tool(TestTool);
        let wrapped = WrapperToolset::new(toolset).before(move |name, args, _| {
            *name_ref.lock() = name.to_string();
            *args_ref.lock() = args.clone();
        });

        let ctx = RunContext::minimal("test");
        let tools = wrapped.get_tools(&ctx).await.unwrap();
        let tool = tools.get("test").unwrap();

        wrapped
            .call_tool("test", serde_json::json!({"key": "value"}), &ctx, tool)
            .await
            .unwrap();

        assert_eq!(*received_name.lock(), "test");
        assert_eq!(received_args.lock()["key"], "value");
    }
}

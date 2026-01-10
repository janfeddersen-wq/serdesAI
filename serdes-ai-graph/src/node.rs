//! Graph node types.

use crate::error::GraphResult;
use crate::state::{GraphRunContext, GraphState};
use async_trait::async_trait;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

/// Result of a node execution.
pub enum NodeResult<State, Deps, End> {
    /// Continue to another node.
    Next(Box<dyn BaseNode<State, Deps, End>>),
    /// Continue to a named node.
    NextNamed(String),
    /// End the graph with a result.
    End(End),
}

impl<State, Deps, End> NodeResult<State, Deps, End> {
    /// Create a Next result with a node.
    pub fn next<N: BaseNode<State, Deps, End> + 'static>(node: N) -> Self {
        Self::Next(Box::new(node))
    }

    /// Create a NextNamed result.
    pub fn next_named(name: impl Into<String>) -> Self {
        Self::NextNamed(name.into())
    }

    /// Create an End result.
    pub fn end(value: End) -> Self {
        Self::End(value)
    }
}

/// End marker with result value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct End<T>(pub T);

impl<T> End<T> {
    /// Create a new End marker.
    pub fn new(value: T) -> Self {
        Self(value)
    }

    /// Get the inner value.
    pub fn into_inner(self) -> T {
        self.0
    }

    /// Get a reference to the inner value.
    pub fn value(&self) -> &T {
        &self.0
    }
}

impl<T: Default> Default for End<T> {
    fn default() -> Self {
        Self(T::default())
    }
}

/// Base trait for all graph nodes.
#[async_trait]
pub trait BaseNode<State, Deps = (), End = ()>: Send + Sync {
    /// Get the node type name for debugging.
    fn type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Get a human-readable name for this node.
    fn name(&self) -> &str {
        self.type_name()
    }

    /// Execute this node.
    async fn run(
        &self,
        ctx: &mut GraphRunContext<State, Deps>,
    ) -> GraphResult<NodeResult<State, Deps, End>>;
}

/// Node trait alias for simple state-only nodes.
#[async_trait]
pub trait Node<State: GraphState>: Send + Sync {
    /// Execute the node and return updated state.
    async fn execute(&self, state: State) -> GraphResult<State>;

    /// Get the node name.
    fn name(&self) -> &str;
}

/// A node that runs a function.
pub struct FunctionNode<State, F, Fut>
where
    F: Fn(State) -> Fut + Send + Sync,
    Fut: Future<Output = GraphResult<State>> + Send,
{
    name: String,
    func: F,
    _phantom: PhantomData<State>,
}

impl<State, F, Fut> FunctionNode<State, F, Fut>
where
    F: Fn(State) -> Fut + Send + Sync,
    Fut: Future<Output = GraphResult<State>> + Send,
{
    /// Create a new function node.
    pub fn new(name: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            func,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<State, F, Fut> Node<State> for FunctionNode<State, F, Fut>
where
    State: GraphState,
    F: Fn(State) -> Fut + Send + Sync,
    Fut: Future<Output = GraphResult<State>> + Send,
{
    async fn execute(&self, state: State) -> GraphResult<State> {
        (self.func)(state).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A node that runs an agent.
#[allow(dead_code)]
pub struct AgentNode<State, Agent, UpdateFn>
where
    UpdateFn: Fn(State, &Agent) -> State + Send + Sync,
{
    name: String,
    agent: Arc<Agent>,
    update_state: UpdateFn,
    _phantom: PhantomData<State>,
}

impl<State, Agent, UpdateFn> AgentNode<State, Agent, UpdateFn>
where
    UpdateFn: Fn(State, &Agent) -> State + Send + Sync,
{
    /// Create a new agent node.
    pub fn new(name: impl Into<String>, agent: Agent, update_state: UpdateFn) -> Self {
        Self {
            name: name.into(),
            agent: Arc::new(agent),
            update_state,
            _phantom: PhantomData,
        }
    }

    /// Get a reference to the agent.
    pub fn agent(&self) -> &Agent {
        &self.agent
    }
}

/// A node that routes based on state.
pub struct RouterNode<State, F>
where
    F: Fn(&State) -> String + Send + Sync,
{
    #[allow(dead_code)]
    name: String,
    router: F,
    _phantom: PhantomData<State>,
}

impl<State, F> RouterNode<State, F>
where
    F: Fn(&State) -> String + Send + Sync,
{
    /// Create a new router node.
    pub fn new(name: impl Into<String>, router: F) -> Self {
        Self {
            name: name.into(),
            router,
            _phantom: PhantomData,
        }
    }

    /// Get the next node name based on state.
    pub fn route(&self, state: &State) -> String {
        (self.router)(state)
    }
}

/// A conditional node that branches based on state.
#[allow(dead_code)]
pub struct ConditionalNode<State, Cond, Then, Else>
where
    Cond: Fn(&State) -> bool + Send + Sync,
    Then: BaseNode<State> + 'static,
    Else: BaseNode<State> + 'static,
{
    name: String,
    condition: Cond,
    then_node: Box<Then>,
    else_node: Box<Else>,
    _phantom: PhantomData<State>,
}

impl<State, Cond, Then, Else> ConditionalNode<State, Cond, Then, Else>
where
    Cond: Fn(&State) -> bool + Send + Sync,
    Then: BaseNode<State> + 'static,
    Else: BaseNode<State> + 'static,
{
    /// Create a new conditional node.
    pub fn new(name: impl Into<String>, condition: Cond, then_node: Then, else_node: Else) -> Self {
        Self {
            name: name.into(),
            condition,
            then_node: Box::new(then_node),
            else_node: Box::new(else_node),
            _phantom: PhantomData,
        }
    }
}

/// Node definition for registration in a graph.
pub struct NodeDef<State, Deps = (), End = ()> {
    /// Node name.
    pub name: String,
    /// The node implementation.
    pub node: Box<dyn BaseNode<State, Deps, End>>,
}

impl<State, Deps, End> NodeDef<State, Deps, End> {
    /// Create a new node definition.
    pub fn new<N: BaseNode<State, Deps, End> + 'static>(name: impl Into<String>, node: N) -> Self {
        Self {
            name: name.into(),
            node: Box::new(node),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Default)]
    struct TestState {
        value: i32,
    }

    #[test]
    fn test_end_marker() {
        let end = End::new(42);
        assert_eq!(end.value(), &42);
        assert_eq!(end.into_inner(), 42);
    }

    #[test]
    fn test_node_result_variants() {
        let _next_named: NodeResult<TestState, (), i32> = NodeResult::next_named("next");
        let _end: NodeResult<TestState, (), i32> = NodeResult::end(42);
    }

    #[test]
    fn test_router_node() {
        let router = RouterNode::new("router", |state: &TestState| {
            if state.value > 0 {
                "positive".to_string()
            } else {
                "negative".to_string()
            }
        });

        assert_eq!(router.route(&TestState { value: 1 }), "positive");
        assert_eq!(router.route(&TestState { value: -1 }), "negative");
    }
}

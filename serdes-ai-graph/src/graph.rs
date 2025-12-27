//! Graph definition and execution.

use crate::edge::Edge;
use crate::error::{GraphError, GraphResult};
use crate::node::{BaseNode, Node, NodeDef, NodeResult};
use crate::state::{generate_run_id, GraphRunContext, GraphRunResult, GraphState};
use std::collections::HashMap;

/// A graph for multi-agent workflows.
pub struct Graph<State, Deps = (), End = ()>
where
    State: GraphState,
{
    name: Option<String>,
    /// Nodes in the graph.
    pub nodes: HashMap<String, NodeDef<State, Deps, End>>,
    edges: Vec<Edge<State>>,
    entry_node: Option<String>,
    finish_nodes: Vec<String>,
    max_steps: u32,
    auto_instrument: bool,
}

impl<State, Deps, End> Graph<State, Deps, End>
where
    State: GraphState,
    Deps: Send + Sync + 'static,
    End: Send + Sync + 'static,
{
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            name: None,
            nodes: HashMap::new(),
            edges: Vec::new(),
            entry_node: None,
            finish_nodes: Vec::new(),
            max_steps: 100,
            auto_instrument: true,
        }
    }

    /// Set the graph name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set maximum steps.
    pub fn with_max_steps(mut self, max: u32) -> Self {
        self.max_steps = max;
        self
    }

    /// Disable auto instrumentation.
    pub fn without_instrumentation(mut self) -> Self {
        self.auto_instrument = false;
        self
    }

    /// Add a node to the graph.
    pub fn node<N>(mut self, name: impl Into<String>, node: N) -> Self
    where
        N: BaseNode<State, Deps, End> + 'static,
    {
        let name = name.into();
        self.nodes.insert(name.clone(), NodeDef::new(name, node));
        self
    }

    /// Add an edge with a condition.
    pub fn edge<F>(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        condition: F,
    ) -> Self
    where
        F: Fn(&State) -> bool + Send + Sync + 'static,
    {
        self.edges.push(Edge::new(from, to, condition));
        self
    }

    /// Add an unconditional edge.
    pub fn edge_always(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
    ) -> Self {
        self.edges.push(Edge::unconditional(from, to));
        self
    }

    /// Set the entry node.
    pub fn entry(mut self, name: impl Into<String>) -> Self {
        self.entry_node = Some(name.into());
        self
    }

    /// Set finish nodes.
    pub fn finish(mut self, names: &[&str]) -> Self {
        self.finish_nodes = names.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add a finish node.
    pub fn add_finish(mut self, name: impl Into<String>) -> Self {
        self.finish_nodes.push(name.into());
        self
    }

    /// Get the graph name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get node names.
    pub fn node_names(&self) -> impl Iterator<Item = &str> {
        self.nodes.keys().map(|s| s.as_str())
    }

    /// Get node count.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get edge count.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Validate the graph configuration.
    pub fn validate(&self) -> GraphResult<()> {
        // Check entry node exists
        if let Some(ref entry) = self.entry_node {
            if !self.nodes.contains_key(entry) {
                return Err(GraphError::node_not_found(entry));
            }
        } else {
            return Err(GraphError::NoEntryNode);
        }

        // Check all edge references exist
        for edge in &self.edges {
            if !self.nodes.contains_key(&edge.from) {
                return Err(GraphError::node_not_found(&edge.from));
            }
            if !self.nodes.contains_key(&edge.to) {
                return Err(GraphError::node_not_found(&edge.to));
            }
        }

        // Check finish nodes exist
        for finish in &self.finish_nodes {
            if !self.nodes.contains_key(finish) {
                return Err(GraphError::node_not_found(finish));
            }
        }

        Ok(())
    }

    /// Build and validate the graph.
    pub fn build(self) -> GraphResult<Self> {
        self.validate()?;
        Ok(self)
    }
}

impl<State, Deps, End> Graph<State, Deps, End>
where
    State: GraphState,
    Deps: Clone + Send + Sync + 'static,
    End: Clone + Send + Sync + 'static,
{
    /// Run the graph from the entry node.
    pub async fn run(&self, state: State, deps: Deps) -> GraphResult<GraphRunResult<State, End>> {
        let entry = self.entry_node.as_ref().ok_or(GraphError::NoEntryNode)?;
        let start_node = self.nodes.get(entry).ok_or_else(|| {
            GraphError::node_not_found(entry)
        })?;

        self.run_from(&*start_node.node, state, deps).await
    }

    /// Run the graph from a specific node.
    pub async fn run_from<N>(
        &self,
        start: &N,
        state: State,
        deps: Deps,
    ) -> GraphResult<GraphRunResult<State, End>>
    where
        N: BaseNode<State, Deps, End> + ?Sized,
    {
        let run_id = generate_run_id();
        let mut ctx = GraphRunContext::new(state, deps, &run_id)
            .with_max_steps(self.max_steps);
        let mut history = Vec::new();

        // Execute the first node
        ctx.increment_step();
        let node_name = start.name().to_string();
        history.push(node_name);

        let mut result = start.run(&mut ctx).await?;

        loop {
            if ctx.is_max_steps_reached() {
                return Err(GraphError::MaxStepsExceeded(ctx.max_steps));
            }

            match result {
                NodeResult::Next(next) => {
                    ctx.increment_step();
                    let name = next.name().to_string();
                    history.push(name);
                    result = next.run(&mut ctx).await?;
                }
                NodeResult::NextNamed(name) => {
                    let node = self.nodes.get(&name).ok_or_else(|| {
                        GraphError::node_not_found(&name)
                    })?;
                    ctx.increment_step();
                    history.push(name);
                    result = node.node.run(&mut ctx).await?;
                }
                NodeResult::End(end) => {
                    return Ok(GraphRunResult::new(end, ctx.state, ctx.step, run_id)
                        .with_history(history));
                }
            }
        }
    }
}

impl<State, Deps, End> Default for Graph<State, Deps, End>
where
    State: GraphState,
    Deps: Send + Sync + 'static,
    End: Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Simple graph using the old API (state-only nodes).
pub struct SimpleGraph<State: GraphState> {
    nodes: HashMap<String, Box<dyn Node<State>>>,
    edges: Vec<Edge<State>>,
    entry_node: Option<String>,
    finish_nodes: Vec<String>,
}

impl<State: GraphState + 'static> SimpleGraph<State> {
    /// Create a new simple graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            entry_node: None,
            finish_nodes: Vec::new(),
        }
    }

    /// Add a node.
    pub fn add_node(mut self, name: impl Into<String>, node: impl Node<State> + 'static) -> Self {
        self.nodes.insert(name.into(), Box::new(node));
        self
    }

    /// Add a conditional edge.
    pub fn add_edge<F>(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        condition: F,
    ) -> Self
    where
        F: Fn(&State) -> bool + Send + Sync + 'static,
    {
        self.edges.push(Edge::new(from, to, condition));
        self
    }

    /// Set entry node.
    pub fn set_entry(mut self, name: impl Into<String>) -> Self {
        self.entry_node = Some(name.into());
        self
    }

    /// Set finish nodes.
    pub fn set_finish(mut self, names: &[&str]) -> Self {
        self.finish_nodes = names.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Build the graph.
    pub fn build(self) -> GraphResult<Self> {
        if self.entry_node.is_none() {
            return Err(GraphError::NoEntryNode);
        }
        Ok(self)
    }

    /// Run the graph.
    pub async fn run(&self, mut state: State) -> GraphResult<State> {
        let entry = self.entry_node.as_ref().ok_or(GraphError::NoEntryNode)?;
        let mut current = entry.clone();

        loop {
            if self.finish_nodes.contains(&current) {
                break;
            }

            let node = self.nodes.get(&current).ok_or_else(|| {
                GraphError::node_not_found(&current)
            })?;

            state = node.execute(state).await?;

            // Find next node
            let next = self.edges.iter().find(|e| e.from == current && e.matches(&state));
            match next {
                Some(edge) => current = edge.to.clone(),
                None => break,
            }
        }

        Ok(state)
    }
}

impl<State: GraphState + 'static> Default for SimpleGraph<State> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::End;
    use async_trait::async_trait;

    #[derive(Debug, Clone, Default)]
    struct TestState {
        value: i32,
    }

    struct IncrementNode;

    #[async_trait]
    impl BaseNode<TestState, (), i32> for IncrementNode {
        fn name(&self) -> &str {
            "increment"
        }

        async fn run(
            &self,
            ctx: &mut GraphRunContext<TestState, ()>,
        ) -> GraphResult<NodeResult<TestState, (), i32>> {
            ctx.state.value += 1;
            if ctx.state.value >= 3 {
                Ok(NodeResult::end(ctx.state.value))
            } else {
                Ok(NodeResult::next(IncrementNode))
            }
        }
    }

    #[tokio::test]
    async fn test_simple_graph_run() {
        let graph = Graph::<TestState, (), i32>::new()
            .with_name("test")
            .node("start", IncrementNode)
            .entry("start")
            .build()
            .unwrap();

        let result = graph.run(TestState::default(), ()).await.unwrap();
        assert_eq!(result.result, 3);
        assert_eq!(result.steps, 3);
    }

    #[test]
    fn test_graph_validation() {
        let graph = Graph::<TestState, (), i32>::new()
            .node("a", IncrementNode)
            .entry("missing");

        assert!(graph.build().is_err());
    }

    #[test]
    fn test_graph_no_entry() {
        let graph = Graph::<TestState, (), i32>::new()
            .node("a", IncrementNode);

        assert!(graph.build().is_err());
    }
}

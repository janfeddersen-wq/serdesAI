# serdes-ai-graph

[![Crates.io](https://img.shields.io/crates/v/serdes-ai-graph.svg)](https://crates.io/crates/serdes-ai-graph)
[![Documentation](https://docs.rs/serdes-ai-graph/badge.svg)](https://docs.rs/serdes-ai-graph)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE)

> Graph-based execution and multi-agent orchestration for serdes-ai

This crate provides graph-based workflow execution for SerdesAI:

- `Graph` builder for defining workflows
- `BaseNode` trait for custom nodes
- State management across nodes
- Conditional branching and loops
- Multi-agent orchestration

## Installation

```toml
[dependencies]
serdes-ai-graph = "0.1"
```

## Usage

```rust
use serdes_ai_graph::{Graph, BaseNode, NodeResult, GraphRunContext, GraphResult};
use async_trait::async_trait;

#[derive(Debug, Clone, Default)]
struct MyState {
    data: String,
}

struct ProcessNode;

#[async_trait]
impl BaseNode<MyState, (), String> for ProcessNode {
    fn name(&self) -> &str { "process" }
    
    async fn run(
        &self,
        ctx: &mut GraphRunContext<MyState, ()>,
    ) -> GraphResult<NodeResult<MyState, (), String>> {
        ctx.state.data = "processed".to_string();
        Ok(NodeResult::end(ctx.state.data.clone()))
    }
}

let graph = Graph::new()
    .node("process", ProcessNode)
    .entry("process")
    .build()?;

let result = graph.run(MyState::default(), ()).await?;
```

## Part of SerdesAI

This crate is part of the [SerdesAI](https://github.com/janfeddersen-wq/serdesAI) workspace.

For most use cases, you should use the main `serdes-ai` crate which re-exports these types.

## License

MIT License - see [LICENSE](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE) for details.

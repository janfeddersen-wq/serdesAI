# serdes-ai-tools

[![Crates.io](https://img.shields.io/crates/v/serdes-ai-tools.svg)](https://crates.io/crates/serdes-ai-tools)
[![Documentation](https://docs.rs/serdes-ai-tools/badge.svg)](https://docs.rs/serdes-ai-tools)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE)

> Tool system for serdes-ai agents

This crate provides the tool system for SerdesAI agents:

- `Tool` trait for defining callable tools
- `ToolDefinition` for JSON schema-based tool descriptions
- `SchemaBuilder` for easy parameter schema construction
- `ToolReturn` for structured tool responses

## Installation

```toml
[dependencies]
serdes-ai-tools = "0.1"
```

## Usage

```rust
use serdes_ai_tools::{Tool, ToolDefinition, ToolReturn, ToolResult, SchemaBuilder};

struct MyTool;

impl Tool<()> for MyTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("my_tool", "Does something useful")
            .with_parameters(
                SchemaBuilder::new()
                    .string("input", "The input value", true)
                    .build()
                    .unwrap()
            )
    }
    
    async fn call(
        &self,
        _ctx: &RunContext<()>,
        args: serde_json::Value,
    ) -> ToolResult {
        Ok(ToolReturn::text("Done!"))
    }
}
```

## Part of SerdesAI

This crate is part of the [SerdesAI](https://github.com/janfeddersen-wq/serdesAI) workspace.

For most use cases, you should use the main `serdes-ai` crate which re-exports these types.

## License

MIT License - see [LICENSE](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE) for details.

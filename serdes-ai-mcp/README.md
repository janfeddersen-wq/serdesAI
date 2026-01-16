# serdes-ai-mcp

[![Crates.io](https://img.shields.io/crates/v/serdes-ai-mcp.svg)](https://crates.io/crates/serdes-ai-mcp)
[![Documentation](https://docs.rs/serdes-ai-mcp/badge.svg)](https://docs.rs/serdes-ai-mcp)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE)

> Model Context Protocol (MCP) implementation for serdes-ai

This crate provides MCP (Model Context Protocol) support for SerdesAI:

- MCP client implementation
- Tool server integration
- Resource and prompt management
- Stdio and HTTP transports

## Installation

```toml
[dependencies]
serdes-ai-mcp = "0.1"
```

## Usage

```rust
use serdes_ai_mcp::{McpClient, McpToolset};

let client = McpClient::stdio("npx", &["-y", "@modelcontextprotocol/server-filesystem"])?;
let toolset = McpToolset::from_client(client).await?;

let agent = Agent::new(model)
    .toolset(toolset)
    .build();
```

## Part of SerdesAI

This crate is part of the [SerdesAI](https://github.com/janfeddersen-wq/serdesAI) workspace.

For most use cases, you should use the main `serdes-ai` crate which re-exports these types.

## License

MIT License - see [LICENSE](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE) for details.

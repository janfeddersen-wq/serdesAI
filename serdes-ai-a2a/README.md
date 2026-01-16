# serdes-ai-a2a

[![Crates.io](https://img.shields.io/crates/v/serdes-ai-a2a.svg)](https://crates.io/crates/serdes-ai-a2a)
[![Documentation](https://docs.rs/serdes-ai-a2a/badge.svg)](https://docs.rs/serdes-ai-a2a)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE)

> Agent-to-Agent (A2A) protocol support for SerdesAI

This crate provides A2A (Agent-to-Agent) protocol support for SerdesAI:

- A2A client and server implementations
- Agent discovery and registration
- Inter-agent communication
- Task delegation and coordination

## Installation

```toml
[dependencies]
serdes-ai-a2a = "0.1"
```

## Usage

```rust
use serdes_ai_a2a::{A2AClient, AgentCard};

// Connect to a remote agent
let client = A2AClient::new("https://agent.example.com").await?;
let card = client.get_agent_card().await?;

println!("Connected to: {}", card.name);

// Send a task to the remote agent
let result = client.send_task("Analyze this data", data).await?;
```

## Part of SerdesAI

This crate is part of the [SerdesAI](https://github.com/janfeddersen-wq/serdesAI) workspace.

## License

MIT License - see [LICENSE](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE) for details.

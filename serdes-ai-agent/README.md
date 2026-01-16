# serdes-ai-agent

[![Crates.io](https://img.shields.io/crates/v/serdes-ai-agent.svg)](https://crates.io/crates/serdes-ai-agent)
[![Documentation](https://docs.rs/serdes-ai-agent/badge.svg)](https://docs.rs/serdes-ai-agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE)

> Agent implementation for serdes-ai

This crate provides the core `Agent` type - the main abstraction for interacting with LLMs in SerdesAI.

## Features

- Generic `Agent<D, O>` type parameterized by dependencies and output
- Builder pattern for agent construction
- Run context for dependency injection
- Support for tools, system prompts, and structured output

## Installation

```toml
[dependencies]
serdes-ai-agent = "0.1"
```

## Usage

```rust
use serdes_ai_agent::{Agent, AgentBuilder};

let agent = Agent::new(model)
    .system_prompt("You are helpful.")
    .build();

let result = agent.run("Hello!", ()).await?;
```

## Part of SerdesAI

This crate is part of the [SerdesAI](https://github.com/janfeddersen-wq/serdesAI) workspace.

For most use cases, you should use the main `serdes-ai` crate which re-exports these types.

## License

MIT License - see [LICENSE](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE) for details.

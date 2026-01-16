# serdes-ai-core

[![Crates.io](https://img.shields.io/crates/v/serdes-ai-core.svg)](https://crates.io/crates/serdes-ai-core)
[![Documentation](https://docs.rs/serdes-ai-core/badge.svg)](https://docs.rs/serdes-ai-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE)

> Core types, messages, and error handling for serdes-ai

This crate provides the foundational types used throughout the SerdesAI ecosystem:

- Message types (user, assistant, system, tool)
- Error types and result aliases
- Common traits and abstractions
- Configuration types

## Installation

```toml
[dependencies]
serdes-ai-core = "0.1"
```

## Usage

```rust
use serdes_ai_core::{Message, Role, UserContent};

let message = Message::user("Hello, world!");
```

## Part of SerdesAI

This crate is part of the [SerdesAI](https://github.com/janfeddersen-wq/serdesAI) workspace.

For most use cases, you should use the main `serdes-ai` crate which re-exports these types.

## License

MIT License - see [LICENSE](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE) for details.

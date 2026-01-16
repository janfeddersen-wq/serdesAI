# serdes-ai-macros

[![Crates.io](https://img.shields.io/crates/v/serdes-ai-macros.svg)](https://crates.io/crates/serdes-ai-macros)
[![Documentation](https://docs.rs/serdes-ai-macros/badge.svg)](https://docs.rs/serdes-ai-macros)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE)

> Procedural macros for serdes-ai (tool definitions, output schemas)

This crate provides procedural macros for SerdesAI:

- `#[derive(Output)]` - Derive the Output trait with JSON schema generation
- `#[derive(Tool)]` - Derive tool implementations from functions
- Schema generation utilities

## Installation

```toml
[dependencies]
serdes-ai-macros = "0.1"
```

## Usage

### Output Derive

```rust
use serdes_ai_macros::Output;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Output)]
struct ExtractedData {
    /// The person's name
    name: String,
    /// Their age in years
    age: u32,
    /// Optional email address
    email: Option<String>,
}
```

### Tool Derive

```rust
use serdes_ai_macros::tool;

#[tool(description = "Calculate the sum of two numbers")]
async fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

## Part of SerdesAI

This crate is part of the [SerdesAI](https://github.com/janfeddersen-wq/serdesAI) workspace.

For most use cases, you should use the main `serdes-ai` crate which re-exports these macros.

## License

MIT License - see [LICENSE](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE) for details.

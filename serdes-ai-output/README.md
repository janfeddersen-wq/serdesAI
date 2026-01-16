# serdes-ai-output

[![Crates.io](https://img.shields.io/crates/v/serdes-ai-output.svg)](https://crates.io/crates/serdes-ai-output)
[![Documentation](https://docs.rs/serdes-ai-output/badge.svg)](https://docs.rs/serdes-ai-output)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE)

> Output schema validation and structured output support for serdes-ai

This crate provides structured output support for SerdesAI agents:

- `Output` trait for types that can be extracted from LLM responses
- JSON schema generation for output types
- Validation and parsing utilities
- Integration with serde

## Installation

```toml
[dependencies]
serdes-ai-output = "0.1"
```

## Usage

```rust
use serdes_ai_output::Output;
use serdes_ai_macros::Output;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Output)]
struct PersonInfo {
    name: String,
    age: u32,
}

let agent = Agent::new(model)
    .output_type::<PersonInfo>()
    .build();
```

## Part of SerdesAI

This crate is part of the [SerdesAI](https://github.com/janfeddersen-wq/serdesAI) workspace.

For most use cases, you should use the main `serdes-ai` crate which re-exports these types.

## License

MIT License - see [LICENSE](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE) for details.

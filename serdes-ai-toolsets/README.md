# serdes-ai-toolsets

[![Crates.io](https://img.shields.io/crates/v/serdes-ai-toolsets.svg)](https://crates.io/crates/serdes-ai-toolsets)
[![Documentation](https://docs.rs/serdes-ai-toolsets/badge.svg)](https://docs.rs/serdes-ai-toolsets)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE)

> Toolset abstractions for grouping and managing tools

This crate provides toolset abstractions for organizing and composing tools:

- `Toolset` trait for tool collections
- Composable toolsets
- Dynamic tool registration
- Tool filtering and selection

## Installation

```toml
[dependencies]
serdes-ai-toolsets = "0.1"
```

## Usage

```rust
use serdes_ai_toolsets::Toolset;

let toolset = Toolset::new()
    .add(MyTool)
    .add(AnotherTool);

let agent = Agent::new(model)
    .toolset(toolset)
    .build();
```

## Part of SerdesAI

This crate is part of the [SerdesAI](https://github.com/janfeddersen-wq/serdesAI) workspace.

For most use cases, you should use the main `serdes-ai` crate which re-exports these types.

## License

MIT License - see [LICENSE](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE) for details.

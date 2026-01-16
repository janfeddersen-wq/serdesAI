# serdes-ai-ui

[![Crates.io](https://img.shields.io/crates/v/serdes-ai-ui.svg)](https://crates.io/crates/serdes-ai-ui)
[![Documentation](https://docs.rs/serdes-ai-ui/badge.svg)](https://docs.rs/serdes-ai-ui)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE)

> UI protocol adapters for SerdesAI (Vercel AI SDK, AG-UI)

This crate provides UI protocol adapters for SerdesAI:

- Vercel AI SDK compatible streaming
- AG-UI protocol support
- HTTP response adapters
- Frontend integration utilities

## Installation

```toml
[dependencies]
serdes-ai-ui = "0.1"
```

## Features

- `vercel` (default) - Vercel AI SDK compatibility
- `ag-ui` - AG-UI protocol support
- `full` - All UI protocols

## Usage

```rust
use serdes_ai_ui::vercel::VercelStreamAdapter;

let adapter = VercelStreamAdapter::new();
let response = adapter.stream_response(agent.run_stream("Hello", ()).await?);

// Use with your web framework (axum, actix, etc.)
```

## Part of SerdesAI

This crate is part of the [SerdesAI](https://github.com/janfeddersen-wq/serdesAI) workspace.

## License

MIT License - see [LICENSE](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE) for details.

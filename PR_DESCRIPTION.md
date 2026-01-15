# Add String-Based Model Configuration for AgentBuilder

## Summary

This PR adds the ability to create agents using model spec strings like `"openai:gpt-4o"` or `"anthropic:claude-3-5-sonnet-20241022"`, matching the API patterns shown in the documentation examples.

## Motivation

The current documentation and examples show code like:

```rust
let agent = Agent::builder()
    .model("openai:gpt-4o")
    .system_prompt("You are helpful.")
    .build()?;
```

However, this API didn't actually exist - `AgentBuilder::new()` only accepted concrete `Model` implementations. This PR bridges that gap by adding several new constructors that accept string specs.

## New APIs

### `AgentBuilder::from_model(spec)`

The simplest way to create an agent - uses environment variables for API keys:

```rust
let agent = AgentBuilder::from_model("openai:gpt-4o")?
    .system_prompt("You are helpful.")
    .build();
```

### `AgentBuilder::from_config(config)`

For when you need custom API keys, base URLs, or timeouts:

```rust
let config = ModelConfig::new("anthropic:claude-3-5-sonnet-20241022")
    .with_api_key("sk-ant-your-key")
    .with_base_url("https://your-proxy.com");

let agent = AgentBuilder::from_config(config)?
    .system_prompt("You are helpful.")
    .build();
```

### `AgentBuilder::from_arc(model)`

For when you already have an `Arc<dyn Model>` (e.g., from `infer_model()`):

```rust
let model = infer_model("openai:gpt-4o")?;
let agent = AgentBuilder::from_arc(model)
    .system_prompt("You are helpful.")
    .build();
```

### `ModelConfig`

A new configuration struct for flexible model creation:

```rust
pub struct ModelConfig {
    pub spec: String,           // "provider:model" format
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub timeout: Option<Duration>,
}

impl ModelConfig {
    pub fn new(spec: impl Into<String>) -> Self;
    pub fn with_api_key(self, key: impl Into<String>) -> Self;
    pub fn with_base_url(self, url: impl Into<String>) -> Self;
    pub fn with_timeout(self, timeout: Duration) -> Self;
}
```

## Changes

### `serdes-ai-agent/src/builder.rs`
- Added `ModelConfig` struct with builder methods
- Added `AgentBuilder::from_model()`, `from_config()`, `from_arc()` constructors
- Updated documentation with usage examples
- Added tests for new functionality

### `serdes-ai-agent/src/lib.rs`
- Exported `ModelConfig`

### `serdes-ai-models/src/lib.rs`
- Added `build_model_with_config()` helper function that creates models with custom API keys, base URLs, and timeouts

### `serdes-ai/src/lib.rs`
- Re-exported `ModelConfig` from the main crate and prelude

## Supported Providers

The following providers are supported (based on enabled features):
- `openai` / `gpt` - OpenAI models
- `anthropic` / `claude` - Anthropic models  
- `groq` - Groq models
- `mistral` - Mistral models
- `ollama` - Local Ollama models
- `google` / `gemini` - Google/Gemini models

## Testing

- All existing tests pass
- Added new tests for:
  - `ModelConfig` construction and builder methods
  - Model spec parsing with and without provider prefix
  - Unknown provider error handling
  - `AgentBuilder::from_arc()` functionality

## Breaking Changes

None - this is purely additive. The existing `AgentBuilder::new(model)` API continues to work unchanged.

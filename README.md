# SerdesAI 🦀

> **A complete Rust port of pydantic-ai for building production-ready AI agents**

[![Crates.io](https://img.shields.io/crates/v/serdes-ai.svg)](https://crates.io/crates/serdes-ai)
[![Documentation](https://docs.rs/serdes-ai/badge.svg)](https://docs.rs/serdes-ai)
[![CI](https://github.com/janfeddersen-wq/serdesAI/workflows/CI/badge.svg)](https://github.com/janfeddersen-wq/serdesAI/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

SerdesAI is a comprehensive, type-safe Rust framework for building AI agents that interact with large language models. It is a complete port of [pydantic-ai](https://github.com/pydantic/pydantic-ai) to pure Rust, providing ergonomic async APIs with compile-time safety guarantees.

## ✨ Features

- 🤖 **Type-safe Agents** - Generic over dependencies and output types with compile-time validation
- 🔌 **Multi-provider Support** - OpenAI, Anthropic, Google Gemini, Groq, Mistral, Ollama, AWS Bedrock, Azure OpenAI
- 🛠️ **Tool Calling** - Define tools with automatic JSON schema generation via macros
- 📡 **Streaming** - Real-time response streaming with backpressure support
- 🔄 **Smart Retries** - Configurable retry strategies with exponential backoff
- 📊 **Evaluations** - Built-in testing and benchmarking framework
- 🔀 **Graph Workflows** - Complex multi-agent orchestration with state management
- 🔗 **MCP Support** - Model Context Protocol integration for tool servers
- 📐 **Structured Output** - JSON schema-based output validation with serde
- 🔍 **Embeddings** - Semantic search and RAG support

## 🚀 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
serdes-ai = "0.1"
tokio = { version = "1", features = ["full"] }
```

### Simple Chat

```rust
use serdes_ai::prelude::*;
use serdes_ai::OpenAIChatModel;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::new(OpenAIChatModel::from_env("gpt-4o")?)
        .system_prompt("You are a helpful assistant.")
        .build();
    
    let result = agent.run("Hello! What can you help me with?", ()).await?;
    println!("{}", result.output);
    
    Ok(())
}
```

### Tool Calling

```rust
use serdes_ai::prelude::*;
use serdes_ai_tools::{Tool, ToolDefinition, ToolReturn, ToolResult, SchemaBuilder};

struct CalculatorTool;

impl Tool<()> for CalculatorTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("calculate", "Perform arithmetic calculations")
            .with_parameters(
                SchemaBuilder::new()
                    .string("expression", "Math expression to evaluate", true)
                    .build()
                    .unwrap()
            )
    }
    
    async fn call(
        &self, 
        _ctx: &RunContext<()>, 
        args: serde_json::Value
    ) -> ToolResult {
        let expr = args["expression"].as_str().unwrap();
        // Evaluate the expression...
        Ok(ToolReturn::text("42"))
    }
}

let agent = Agent::new(model)
    .tool(CalculatorTool)
    .build();
```

### Structured Output

```rust
use serdes_ai::prelude::*;
use serdes_ai_macros::Output;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Output)]
struct PersonInfo {
    name: String,
    age: u32,
    occupation: String,
}

let agent = Agent::new(model)
    .output_type::<PersonInfo>()
    .build();

let result = agent.run("John is a 30 year old engineer", ()).await?;
println!("Extracted: {} is {} and works as {}", 
    result.output.name, 
    result.output.age,
    result.output.occupation
);
```

### Streaming Responses

```rust
use serdes_ai::prelude::*;
use serdes_ai_streaming::AgentStreamEvent;
use futures::StreamExt;

let mut stream = agent.run_stream("Write a poem", ()).await?;

while let Some(event) = stream.next().await {
    if let AgentStreamEvent::TextDelta { content, .. } = event {
        print!("{}", content);
    }
}
```

### Graph-based Workflows

```rust
use serdes_ai::prelude::*;
use serdes_ai_graph::{Graph, BaseNode, NodeResult, GraphRunContext, GraphResult};
use async_trait::async_trait;

#[derive(Debug, Clone, Default)]
struct WorkflowState {
    query: String,
    research: Option<String>,
    response: Option<String>,
}

struct ResearchNode;
struct WriteNode;

#[async_trait]
impl BaseNode<WorkflowState, (), String> for ResearchNode {
    fn name(&self) -> &str { "research" }
    
    async fn run(
        &self,
        ctx: &mut GraphRunContext<WorkflowState, ()>,
    ) -> GraphResult<NodeResult<WorkflowState, (), String>> {
        ctx.state.research = Some(format!("Research for: {}", ctx.state.query));
        Ok(NodeResult::next(WriteNode))
    }
}

#[async_trait]
impl BaseNode<WorkflowState, (), String> for WriteNode {
    fn name(&self) -> &str { "write" }
    
    async fn run(
        &self,
        ctx: &mut GraphRunContext<WorkflowState, ()>,
    ) -> GraphResult<NodeResult<WorkflowState, (), String>> {
        let response = format!("Based on: {}", ctx.state.research.as_deref().unwrap_or(""));
        Ok(NodeResult::end(response))
    }
}

let graph = Graph::new()
    .node("research", ResearchNode)
    .node("write", WriteNode)
    .entry("research")
    .build()?;

let result = graph.run(WorkflowState::default(), ()).await?;
```

## 📦 Crates

SerdesAI is organized as a workspace of focused crates:

| Crate | Description |
|-------|-------------|
| `serdes-ai` | Main facade with re-exports |
| `serdes-ai-core` | Core types, messages, errors |
| `serdes-ai-agent` | Agent implementation |
| `serdes-ai-models` | Model trait and providers |
| `serdes-ai-providers` | Provider abstractions |
| `serdes-ai-tools` | Tool definitions and execution |
| `serdes-ai-toolsets` | Tool collections and composition |
| `serdes-ai-output` | Output schemas and validation |
| `serdes-ai-streaming` | Streaming support |
| `serdes-ai-mcp` | MCP protocol support |
| `serdes-ai-embeddings` | Embedding models |
| `serdes-ai-retries` | Retry strategies |
| `serdes-ai-graph` | Graph-based workflows |
| `serdes-ai-evals` | Evaluation framework |
| `serdes-ai-macros` | Procedural macros |

## 🔌 Supported Providers

| Provider | Feature Flag | Models | Status |
|----------|--------------|--------|--------|
| OpenAI | `openai` (default) | GPT-4, GPT-4o, o1, o3 | ✅ Full |
| Anthropic | `anthropic` (default) | Claude 3.5, Claude 4 | ✅ Full |
| Google | `google` (default) | Gemini 1.5, Gemini 2.0 | ✅ Full |
| Groq | `groq` | Llama 3, Mixtral, Gemma | ✅ Full |
| Mistral | `mistral` | Mistral Large, Codestral | ✅ Full |
| Ollama | `ollama` | Any local model | ✅ Full |
| Azure OpenAI | `azure` | Azure-hosted OpenAI | ✅ Full |
| AWS Bedrock | `bedrock` | Claude, Llama, Titan | ✅ Full |

### Provider Examples

```rust
// OpenAI
let model = OpenAIChatModel::from_env("gpt-4o")?;

// Anthropic
let model = AnthropicModel::from_env("claude-3-5-sonnet-20241022")?;

// Google Gemini
let model = GoogleModel::from_env("gemini-1.5-pro")?;

// Groq (ultra-fast inference)
let model = GroqModel::from_env("llama-3.1-70b-versatile")?;

// Mistral
let model = MistralModel::from_env("mistral-large-latest")?;

// Ollama (local)
let model = OllamaModel::new("llama3.1");

// Azure OpenAI
let model = AzureOpenAIModel::from_env("my-deployment")?;

// AWS Bedrock
let model = BedrockModel::new("anthropic.claude-3-sonnet-20240229-v1:0")?;
```

## 🎯 Feature Flags

```toml
[dependencies]
serdes-ai = { version = "0.1", features = ["full"] }
```

| Feature | Description | Default |
|---------|-------------|--------|
| `openai` | OpenAI GPT models | ✅ |
| `anthropic` | Anthropic Claude models | ✅ |
| `google` | Google Gemini models | ✅ |
| `groq` | Groq fast inference | |
| `mistral` | Mistral AI models | |
| `ollama` | Ollama local models | |
| `azure` | Azure OpenAI | |
| `bedrock` | AWS Bedrock | |
| `mcp` | Model Context Protocol | |
| `graph` | Graph workflows | ✅ |
| `evals` | Evaluation framework | ✅ |
| `macros` | Procedural macros | ✅ |
| `full` | All features | |

## 📖 Documentation

- [API Documentation](https://docs.rs/serdes-ai)
- [Examples](./examples/)
- [Migration from pydantic-ai](./docs/migration.md)

## ⚖️ Comparison with pydantic-ai

| Feature | pydantic-ai | serdes-ai |
|---------|-------------|----------|
| Language | Python | Rust |
| Type Safety | Runtime (Pydantic) | Compile-time |
| Async Runtime | asyncio | tokio |
| Validation | Pydantic v2 | serde + custom |
| Performance | Good | Excellent |
| Memory Safety | Garbage Collected | Ownership system |
| Binary Size | Large (Python runtime) | Minimal |
| Startup Time | Slow | Instant |
| Thread Safety | GIL limitations | Fully concurrent |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         serdes-ai                           │
│                    (Main Facade Crate)                      │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ serdes-ai-    │    │ serdes-ai-    │    │ serdes-ai-    │
│    agent      │    │    models     │    │    graph      │
│               │    │               │    │               │
│  Agent logic  │    │ Model trait   │    │ Multi-agent   │
│  Run context  │    │ Providers     │    │ workflows     │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └──────────┬──────────┴──────────┬──────────┘
                   │                     │
                   ▼                     ▼
          ┌───────────────┐    ┌───────────────┐
          │ serdes-ai-    │    │ serdes-ai-    │
          │    tools      │    │    core       │
          │               │    │               │
          │ Tool traits   │    │ Messages      │
          │ Schema gen    │    │ Errors        │
          └───────────────┘    └───────────────┘
```

## 🧪 Testing

```bash
# Run all tests
cargo test --workspace --all-features

# Run with specific provider
cargo test --features openai

# Run benchmarks
cargo bench --workspace
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/janfeddersen-wq/serdesAI
cd serdesAI
cargo build --workspace --all-features
cargo test --workspace --all-features
```

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.

## 🙏 Acknowledgments

- [pydantic-ai](https://github.com/pydantic/pydantic-ai) - The original Python implementation that inspired this project
- [Anthropic](https://anthropic.com) - For Claude and the Model Context Protocol
- [OpenAI](https://openai.com) - For the OpenAI API and tool calling standards
- The Rust community for excellent crates like `tokio`, `serde`, and `async-trait`

---

<p align="center">
  Made with 🦀 and ❤️
</p>

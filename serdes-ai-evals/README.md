# serdes-ai-evals

[![Crates.io](https://img.shields.io/crates/v/serdes-ai-evals.svg)](https://crates.io/crates/serdes-ai-evals)
[![Documentation](https://docs.rs/serdes-ai-evals/badge.svg)](https://docs.rs/serdes-ai-evals)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE)

> Evaluation framework for testing and benchmarking serdes-ai agents

This crate provides evaluation and testing capabilities for SerdesAI:

- Test case definitions
- Evaluation metrics (accuracy, latency, cost)
- Benchmark harness
- Regression testing
- LLM-as-judge evaluators

## Installation

```toml
[dependencies]
serdes-ai-evals = "0.1"
```

## Usage

```rust
use serdes_ai_evals::{EvalSuite, TestCase, Evaluator};

let suite = EvalSuite::new("my-agent-tests")
    .case(TestCase::new("greeting")
        .input("Hello!")
        .expected_contains("Hello"))
    .case(TestCase::new("math")
        .input("What is 2+2?")
        .expected_contains("4"));

let results = suite.run(&agent).await?;
println!("Pass rate: {:.1}%", results.pass_rate() * 100.0);
```

## Part of SerdesAI

This crate is part of the [SerdesAI](https://github.com/janfeddersen-wq/serdesAI) workspace.

For most use cases, you should use the main `serdes-ai` crate which re-exports these types.

## License

MIT License - see [LICENSE](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE) for details.

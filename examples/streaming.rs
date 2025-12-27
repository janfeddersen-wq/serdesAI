//! Streaming response example.
//!
//! This example demonstrates how to use streaming to receive
//! real-time responses from the LLM.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example streaming
//! ```

use futures::StreamExt;
use serdes_ai::prelude::*;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("🌊 Streaming Example\n");
    println!("Generating a story with streaming output...\n");
    println!("---\n");

    // Create an agent
    let agent = Agent::builder()
        .model("openai:gpt-4o")
        .system_prompt(
            "You are a creative storyteller. \
             Write engaging, imaginative stories with vivid descriptions.",
        )
        .build()?;

    // Start a streaming request
    let prompt = "Write a short story (about 200 words) about a robot \
                  discovering it has the ability to dream.";

    println!("📝 Prompt: {}\n\n", prompt);
    println!("📚 Story:\n");

    let mut stream = agent.run_stream(prompt, ()).await?;

    // Process the stream
    let mut full_text = String::new();
    let mut token_count = 0;

    while let Some(event) = stream.next().await {
        match event {
            Ok(AgentStreamEvent::TextDelta { content, .. }) => {
                // Print each chunk as it arrives
                print!("{}", content);
                io::stdout().flush()?;
                full_text.push_str(&content);
                token_count += 1;
            }
            Ok(AgentStreamEvent::ToolCallStart { tool_name, .. }) => {
                println!("\n⚙️ Calling tool: {}", tool_name);
            }
            Ok(AgentStreamEvent::ToolCallComplete { tool_name, .. }) => {
                println!("✅ Tool {} completed", tool_name);
            }
            Ok(AgentStreamEvent::Complete { usage, .. }) => {
                println!("\n\n---");
                println!("✅ Stream complete!");
                if let Some(u) = usage {
                    println!(
                        "📊 Tokens: {} input, {} output",
                        u.input_tokens.unwrap_or(0),
                        u.output_tokens.unwrap_or(0)
                    );
                }
            }
            Ok(AgentStreamEvent::Error { error, .. }) => {
                eprintln!("\n❌ Error: {}", error);
            }
            _ => {}
        }
    }

    println!("\n📝 Total chunks received: {}", token_count);
    println!("📝 Total characters: {}", full_text.len());

    Ok(())
}

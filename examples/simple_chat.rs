//! Simple chat example with OpenAI.
//!
//! This example demonstrates basic agent usage for a simple Q&A interaction.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example simple_chat
//! ```

use serdes_ai::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing for debugging
    tracing_subscriber::fmt::init();

    println!("🤖 Simple Chat Example\n");

    // Create an agent with GPT-4o
    let agent = Agent::builder()
        .model("openai:gpt-4o")
        .system_prompt(
            "You are a helpful, friendly assistant. \
             Keep your responses concise but informative.",
        )
        .build()?;

    // Run a simple query
    let result = agent
        .run("What are the three primary colors?", ())
        .await?;

    println!("Question: What are the three primary colors?");
    println!("Answer: {}\n", result.output());

    // Run another query
    let result = agent
        .run("Why is the sky blue? Explain briefly.", ())
        .await?;

    println!("Question: Why is the sky blue?");
    println!("Answer: {}\n", result.output());

    // Show usage information
    println!("📊 Usage: {:?}", result.usage());

    Ok(())
}

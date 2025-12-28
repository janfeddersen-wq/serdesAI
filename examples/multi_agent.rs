//! Multi-agent workflow example using the graph module.
//!
//! This example demonstrates how to create a multi-agent pipeline
//! where different specialized agents handle different tasks.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example multi_agent --features graph
//! ```

use serdes_ai::prelude::*;
use serde::{Deserialize, Serialize};

/// State passed through the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchState {
    /// Original query.
    pub query: String,
    /// Research findings.
    pub research: Option<String>,
    /// Draft article.
    pub draft: Option<String>,
    /// Final edited article.
    pub final_article: Option<String>,
    /// Quality score (1-10).
    pub quality_score: Option<u32>,
}

impl Default for ResearchState {
    fn default() -> Self {
        Self {
            query: String::new(),
            research: None,
            draft: None,
            final_article: None,
            quality_score: None,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("🤖 Multi-Agent Research Pipeline\n");

    // For this example, we'll simulate a multi-agent workflow
    // without the full graph module to keep it simple.
    
    let query = "What are the key benefits of Rust for systems programming?";
    println!("📝 Query: {}\n", query);

    // Step 1: Research Agent
    println!("\n🔍 Step 1: Research Agent");
    println!("   Gathering information...");
    
    let research_agent = Agent::builder()
        .model("openai:gpt-4o")
        .system_prompt(
            "You are a research specialist. \
             Gather comprehensive information on the given topic. \
             Focus on factual, well-sourced information.",
        )
        .build()?;

    let research = research_agent
        .run(
            format!("Research this topic thoroughly: {}\nProvide key facts and insights.", query),
            (),
        )
        .await?;

    println!("   ✅ Research complete ({} chars)", research.output().len());

    // Step 2: Writer Agent
    println!("\n✍️  Step 2: Writer Agent");
    println!("   Drafting article...");

    let writer_agent = Agent::builder()
        .model("openai:gpt-4o")
        .system_prompt(
            "You are a skilled technical writer. \
             Write clear, engaging articles based on provided research. \
             Use a professional but accessible tone.",
        )
        .build()?;

    let draft = writer_agent
        .run(
            format!(
                "Write a concise article (about 200 words) based on this research:\n\n{}\n\nOriginal query: {}",
                research.output(),
                query
            ),
            (),
        )
        .await?;

    println!("   ✅ Draft complete ({} chars)", draft.output().len());

    // Step 3: Editor Agent
    println!("\n📝 Step 3: Editor Agent");
    println!("   Reviewing and polishing...");

    let editor_agent = Agent::builder()
        .model("openai:gpt-4o")
        .system_prompt(
            "You are a meticulous editor. \
             Review and improve articles for clarity, grammar, and impact. \
             Make the writing tighter and more engaging.",
        )
        .build()?;

    let final_article = editor_agent
        .run(
            format!(
                "Edit and polish this article. Fix any issues and improve clarity:\n\n{}",
                draft.output()
            ),
            (),
        )
        .await?;

    println!("   ✅ Editing complete");

    // Step 4: Quality Reviewer Agent
    println!("\n⭐ Step 4: Quality Review Agent");
    println!("   Scoring final output...");

    let reviewer_agent = Agent::builder()
        .model("openai:gpt-4o")
        .system_prompt(
            "You are a quality reviewer. \
             Score articles on a scale of 1-10 based on: \
             accuracy, clarity, engagement, and completeness. \
             Respond with just the score number.",
        )
        .build()?;

    let score = reviewer_agent
        .run(
            format!(
                "Rate this article (1-10):\n\n{}",
                final_article.output()
            ),
            (),
        )
        .await?;

    let quality_score: u32 = score
        .output()
        .trim()
        .chars()
        .filter(|c| c.is_numeric())
        .collect::<String>()
        .parse()
        .unwrap_or(7);

    println!("   ✅ Quality score: {}/10", quality_score);

    // Final Output
    println!("\n");
    println!("=" .repeat(60));
    println!("📄 FINAL ARTICLE");
    println!("=" .repeat(60));
    println!();
    println!("{}", final_article.output());
    println!();
    println!("=" .repeat(60));
    println!("⭐ Quality Score: {}/10", quality_score);
    println!("=" .repeat(60));

    // Pipeline summary
    println!("\n📊 Pipeline Summary:");
    println!("   - Research: {} tokens used", 
             research.usage().total_tokens.unwrap_or(0));
    println!("   - Writing: {} tokens used", 
             draft.usage().total_tokens.unwrap_or(0));
    println!("   - Editing: {} tokens used", 
             final_article.usage().total_tokens.unwrap_or(0));
    println!("   - Review: {} tokens used", 
             score.usage().total_tokens.unwrap_or(0));

    Ok(())
}

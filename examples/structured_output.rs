//! Structured output example.
//!
//! This example demonstrates how to use type-safe structured outputs
//! with JSON Schema validation.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example structured_output
//! ```

use serde::{Deserialize, Serialize};
use serdes_ai::prelude::*;

/// A person's information extracted from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Person {
    /// Full name.
    pub name: String,
    /// Age in years.
    pub age: u32,
    /// Email address if provided.
    pub email: Option<String>,
    /// Occupation.
    pub occupation: String,
    /// List of skills.
    pub skills: Vec<String>,
}

/// A product review analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewAnalysis {
    /// Sentiment: positive, negative, or neutral.
    pub sentiment: String,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
    /// Key positive points.
    pub pros: Vec<String>,
    /// Key negative points.
    pub cons: Vec<String>,
    /// Overall summary.
    pub summary: String,
}

/// A structured summary of an article.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticleSummary {
    /// Article title.
    pub title: String,
    /// Main topic.
    pub topic: String,
    /// Key points (3-5 items).
    pub key_points: Vec<String>,
    /// Target audience.
    pub audience: String,
    /// Estimated reading time in minutes.
    pub reading_time_minutes: u32,
}

impl OutputSchema for Person {
    fn json_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "Full name of the person" },
                "age": { "type": "integer", "description": "Age in years" },
                "email": { "type": ["string", "null"], "description": "Email if provided" },
                "occupation": { "type": "string", "description": "Job or profession" },
                "skills": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of skills"
                }
            },
            "required": ["name", "age", "occupation", "skills"]
        })
    }
}

impl OutputSchema for ReviewAnalysis {
    fn json_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "pros": { "type": "array", "items": { "type": "string" } },
                "cons": { "type": "array", "items": { "type": "string" } },
                "summary": { "type": "string" }
            },
            "required": ["sentiment", "confidence", "pros", "cons", "summary"]
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("📊 Structured Output Example\n");

    // Example 1: Extract person information
    println!("=== Example 1: Person Extraction ===");
    extract_person().await?;

    // Example 2: Analyze a product review
    println!("\n=== Example 2: Review Analysis ===");
    analyze_review().await?;

    Ok(())
}

async fn extract_person() -> anyhow::Result<()> {
    let agent = Agent::builder()
        .model("openai:gpt-4o")
        .system_prompt(
            "You are a data extraction specialist. \
             Extract structured information from text accurately. \
             Return the result as JSON matching the schema.",
        )
        .output_schema::<Person>()
        .build()?;

    let text = r#"
        Hi, I'm Sarah Johnson! I'm a 28-year-old software engineer 
        working at TechCorp. You can reach me at sarah.j@email.com. 
        I'm proficient in Rust, Python, and TypeScript, and I also 
        have experience with machine learning and cloud architecture.
    "#;

    println!("Input text:\n{}", text);

    let result = agent
        .run(
            format!("Extract the person's information from this text:\n{}", text),
            (),
        )
        .await?;

    let person: Person = result.into_output()?;

    println!("\nExtracted Person:");
    println!("  Name: {}", person.name);
    println!("  Age: {}", person.age);
    println!("  Email: {:?}", person.email);
    println!("  Occupation: {}", person.occupation);
    println!("  Skills: {:?}", person.skills);

    Ok(())
}

async fn analyze_review() -> anyhow::Result<()> {
    let agent = Agent::builder()
        .model("openai:gpt-4o")
        .system_prompt(
            "You are a product review analyst. \
             Analyze reviews and extract structured insights. \
             Return the result as JSON matching the schema.",
        )
        .output_schema::<ReviewAnalysis>()
        .build()?;

    let review = r#"
        I bought this wireless headphones last month and I'm mostly happy with them.
        The sound quality is amazing - crisp highs and deep bass. Battery life is 
        incredible, lasting about 40 hours on a single charge. The noise cancellation 
        works well in most environments.
        
        However, the headphones are a bit heavy for long wearing sessions, and the 
        ear cups could be more comfortable. The app is also somewhat buggy and crashes 
        occasionally. But overall, for the price, it's a solid choice for music lovers.
    "#;

    println!("Review:\n{}", review);

    let result = agent
        .run(
            format!("Analyze this product review:\n{}", review),
            (),
        )
        .await?;

    let analysis: ReviewAnalysis = result.into_output()?;

    println!("\nAnalysis:");
    println!("  Sentiment: {} (confidence: {:.0}%)", 
             analysis.sentiment, 
             analysis.confidence * 100.0);
    println!("  Pros: {:?}", analysis.pros);
    println!("  Cons: {:?}", analysis.cons);
    println!("  Summary: {}", analysis.summary);

    Ok(())
}

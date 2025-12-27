//! MCP (Model Context Protocol) server example.
//!
//! This example demonstrates how to connect to MCP servers
//! and use their tools in an agent.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example mcp_server --features mcp
//! ```

#[cfg(feature = "mcp")]
use serdes_ai::mcp::{McpClient, McpToolset};
use serdes_ai::prelude::*;
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("🔗 MCP (Model Context Protocol) Example\n");

    #[cfg(feature = "mcp")]
    {
        // Example: Connect to a local MCP server
        // This could be a database, file system, or any MCP-compatible service
        run_mcp_example().await?;
    }

    #[cfg(not(feature = "mcp"))]
    {
        println!("This example requires the 'mcp' feature.");
        println!("Run with: cargo run --example mcp_server --features mcp");
        demonstrate_mcp_concept().await?;
    }

    Ok(())
}

#[cfg(feature = "mcp")]
async fn run_mcp_example() -> anyhow::Result<()> {
    println!("1️⃣  Connecting to MCP server...");

    // Connect to a stdio MCP server (like file system tools)
    let client = McpClient::builder()
        .command("npx")
        .args(&["-y", "@anthropic/mcp-server-filesystem", "/tmp"])
        .timeout(Duration::from_secs(30))
        .build()
        .await?;

    println!("   ✅ Connected to MCP server");

    // List available tools
    let tools = client.list_tools().await?;
    println!("\n2️⃣  Available tools:");
    for tool in &tools {
        println!("   - {}: {}", tool.name, tool.description);
    }

    // Create a toolset from the MCP connection
    let toolset = McpToolset::new(client);

    // Create an agent with MCP tools
    let agent = Agent::builder()
        .model("openai:gpt-4o")
        .system_prompt(
            "You are a helpful file system assistant. \
             Use the available tools to help users manage files.",
        )
        .toolset(toolset)
        .build()?;

    println!("\n3️⃣  Running agent with MCP tools...");

    let result = agent
        .run("List the files in the /tmp directory", ())
        .await?;

    println!("\n🤖 Agent response:");
    println!("{}", result.output());

    Ok(())
}

#[cfg(not(feature = "mcp"))]
async fn demonstrate_mcp_concept() -> anyhow::Result<()> {
    println!("\n📚 MCP Concept Demonstration\n");

    println!("The Model Context Protocol (MCP) allows AI agents to connect");
    println!("to external services and tools in a standardized way.\n");

    println!("🔧 With MCP, you can:");
    println!("   - Connect to file system tools");
    println!("   - Access databases");
    println!("   - Use web scrapers");
    println!("   - Integrate with any MCP-compatible service\n");

    println!("📝 Example MCP servers:");
    println!("   - @anthropic/mcp-server-filesystem - File operations");
    println!("   - @anthropic/mcp-server-sqlite - SQLite database");
    println!("   - @anthropic/mcp-server-puppeteer - Web automation\n");

    println!("💡 Usage pattern:");
    println!(r#"
    use serdes_ai::mcp::{{McpClient, McpToolset}};

    // Connect to MCP server
    let client = McpClient::builder()
        .command("npx")
        .args(&["-y", "@anthropic/mcp-server-filesystem", "/tmp"])
        .build()
        .await?;

    // Create toolset from MCP connection
    let toolset = McpToolset::new(client);

    // Use in agent
    let agent = Agent::builder()
        .model("openai:gpt-4o")
        .toolset(toolset)
        .build()?;
"#);

    // Demo agent without MCP
    println!("\n🤖 Running demo agent (without MCP)...");

    let agent = Agent::builder()
        .model("openai:gpt-4o")
        .system_prompt(
            "You are a helpful assistant. Explain what MCP \
             (Model Context Protocol) is and how it benefits AI agents.",
        )
        .build()?;

    let result = agent
        .run(
            "What is MCP and why is it useful for AI agents?",
            (),
        )
        .await?;

    println!("\n{}", result.output());

    Ok(())
}

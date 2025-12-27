//! Multi-agent workflow using the graph module.
//!
//! This example demonstrates how to build a proper graph-based workflow
//! with specialized agents handling different tasks in a coordinated pipeline.
//!
//! The workflow:
//! 1. Researcher gathers information on a topic
//! 2. Router decides if more research is needed or proceed to writing
//! 3. Writer drafts an article
//! 4. Editor polishes the final output
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example multi_agent_graph --features graph
//! ```

#[cfg(feature = "graph")]
mod graph_example {
    use async_trait::async_trait;
    use serdes_ai::graph::{BaseNode, Graph, GraphError, GraphResult, GraphRunContext, NodeResult};
    use serdes_ai::prelude::*;
    use serde::{Deserialize, Serialize};

    /// State that flows through the graph.
    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct ArticleState {
        /// Original query/topic.
        pub topic: String,
        /// Research findings.
        pub research: Option<String>,
        /// Research quality score (1-10).
        pub research_score: Option<u32>,
        /// Draft article.
        pub draft: Option<String>,
        /// Final edited article.
        pub final_article: Option<String>,
        /// Iteration count.
        pub iterations: u32,
    }

    // ========================================================================
    // Nodes
    // ========================================================================

    /// Research node - gathers information on the topic.
    pub struct ResearcherNode;

    #[async_trait]
    impl BaseNode<ArticleState, (), String> for ResearcherNode {
        fn name(&self) -> &str {
            "researcher"
        }

        async fn run(
            &self,
            ctx: &mut GraphRunContext<ArticleState, ()>,
        ) -> GraphResult<NodeResult<ArticleState, (), String>> {
            println!("\n🔍 Researcher Node: Gathering information...");

            let agent = Agent::builder()
                .model("openai:gpt-4o")
                .system_prompt(
                    "You are a research specialist. \
                     Gather comprehensive, factual information on the given topic. \
                     Focus on key facts, statistics, and insights.",
                )
                .build()
                .map_err(|e| GraphError::Execution(e.to_string()))?;

            let prompt = if ctx.state.iterations == 0 {
                format!("Research this topic thoroughly: {}", ctx.state.topic)
            } else {
                format!(
                    "The previous research was not comprehensive enough. \
                     Please expand on this topic with more depth: {}\n\n\
                     Previous research: {}",
                    ctx.state.topic,
                    ctx.state.research.as_deref().unwrap_or("none")
                )
            };

            let result = agent
                .run(&prompt, ())
                .await
                .map_err(|e| GraphError::Execution(e.to_string()))?;

            ctx.state.research = Some(result.output().to_string());
            ctx.state.iterations += 1;

            println!("   ✅ Research complete ({} chars)", result.output().len());

            // Move to quality router
            Ok(NodeResult::NextNamed("quality_router".to_string()))
        }
    }

    /// Quality router - decides if research is good enough.
    pub struct QualityRouterNode;

    #[async_trait]
    impl BaseNode<ArticleState, (), String> for QualityRouterNode {
        fn name(&self) -> &str {
            "quality_router"
        }

        async fn run(
            &self,
            ctx: &mut GraphRunContext<ArticleState, ()>,
        ) -> GraphResult<NodeResult<ArticleState, (), String>> {
            println!("\n🔀 Quality Router: Evaluating research...");

            let agent = Agent::builder()
                .model("openai:gpt-4o")
                .system_prompt(
                    "You are a research quality evaluator. \
                     Score the research on a scale of 1-10 based on: \
                     completeness, accuracy, depth, and relevance. \
                     Respond with ONLY a number from 1-10.",
                )
                .build()
                .map_err(|e| GraphError::Execution(e.to_string()))?;

            let result = agent
                .run(
                    format!(
                        "Rate this research (1-10):\n\nTopic: {}\n\nResearch: {}",
                        ctx.state.topic,
                        ctx.state.research.as_deref().unwrap_or("none")
                    ),
                    (),
                )
                .await
                .map_err(|e| GraphError::Execution(e.to_string()))?;

            let score: u32 = result
                .output()
                .trim()
                .chars()
                .filter(|c| c.is_numeric())
                .collect::<String>()
                .parse()
                .unwrap_or(5);

            ctx.state.research_score = Some(score);
            println!("   📊 Research quality score: {}/10", score);

            // Decision: if score < 7 and we haven't iterated too much, go back to researcher
            if score < 7 && ctx.state.iterations < 3 {
                println!("   🔄 Score too low, requesting more research...");
                Ok(NodeResult::NextNamed("researcher".to_string()))
            } else {
                println!("   ✅ Research quality acceptable, proceeding to writer...");
                Ok(NodeResult::NextNamed("writer".to_string()))
            }
        }
    }

    /// Writer node - drafts the article.
    pub struct WriterNode;

    #[async_trait]
    impl BaseNode<ArticleState, (), String> for WriterNode {
        fn name(&self) -> &str {
            "writer"
        }

        async fn run(
            &self,
            ctx: &mut GraphRunContext<ArticleState, ()>,
        ) -> GraphResult<NodeResult<ArticleState, (), String>> {
            println!("\n✍️  Writer Node: Drafting article...");

            let agent = Agent::builder()
                .model("openai:gpt-4o")
                .system_prompt(
                    "You are a skilled technical writer. \
                     Write clear, engaging articles based on research. \
                     Use a professional but accessible tone. \
                     Structure the article with an intro, body, and conclusion.",
                )
                .build()
                .map_err(|e| GraphError::Execution(e.to_string()))?;

            let result = agent
                .run(
                    format!(
                        "Write a well-structured article (about 300 words) based on this research:\n\n\
                         Topic: {}\n\nResearch:\n{}",
                        ctx.state.topic,
                        ctx.state.research.as_deref().unwrap_or("none")
                    ),
                    (),
                )
                .await
                .map_err(|e| GraphError::Execution(e.to_string()))?;

            ctx.state.draft = Some(result.output().to_string());
            println!("   ✅ Draft complete ({} chars)", result.output().len());

            Ok(NodeResult::NextNamed("editor".to_string()))
        }
    }

    /// Editor node - polishes the final article.
    pub struct EditorNode;

    #[async_trait]
    impl BaseNode<ArticleState, (), String> for EditorNode {
        fn name(&self) -> &str {
            "editor"
        }

        async fn run(
            &self,
            ctx: &mut GraphRunContext<ArticleState, ()>,
        ) -> GraphResult<NodeResult<ArticleState, (), String>> {
            println!("\n📝 Editor Node: Polishing article...");

            let agent = Agent::builder()
                .model("openai:gpt-4o")
                .system_prompt(
                    "You are a meticulous editor. \
                     Review and improve articles for clarity, grammar, flow, and impact. \
                     Make the writing tighter and more engaging while preserving the message.",
                )
                .build()
                .map_err(|e| GraphError::Execution(e.to_string()))?;

            let result = agent
                .run(
                    format!(
                        "Edit and polish this article. Fix any issues and improve clarity:\n\n{}",
                        ctx.state.draft.as_deref().unwrap_or("none")
                    ),
                    (),
                )
                .await
                .map_err(|e| GraphError::Execution(e.to_string()))?;

            ctx.state.final_article = Some(result.output().to_string());
            println!("   ✅ Editing complete!");

            // End the graph with the final article
            Ok(NodeResult::end(
                ctx.state.final_article.clone().unwrap_or_default(),
            ))
        }
    }

    /// Run the multi-agent graph workflow.
    pub async fn run() -> anyhow::Result<()> {
        println!("🤖 Multi-Agent Graph Workflow\n");
        println!("This example demonstrates a graph-based workflow with:");
        println!("  • Researcher - gathers information");
        println!("  • Quality Router - decides if research is sufficient");
        println!("  • Writer - drafts the article");
        println!("  • Editor - polishes the final output");
        println!();

        // Build the graph
        let graph = Graph::new()
            .with_name("article_pipeline")
            .with_max_steps(20)
            .node("researcher", ResearcherNode)
            .node("quality_router", QualityRouterNode)
            .node("writer", WriterNode)
            .node("editor", EditorNode)
            .entry("researcher")
            .build()?;

        println!("📊 Graph Info:");
        println!("   Name: {:?}", graph.name());
        println!("   Nodes: {:?}", graph.node_names().collect::<Vec<_>>());
        println!();

        // Initialize state
        let state = ArticleState {
            topic: "The benefits and challenges of adopting Rust for systems programming".into(),
            ..Default::default()
        };

        println!("📝 Topic: {}", state.topic);
        println!("\n" + &"=".repeat(60));
        println!("Starting workflow...");
        println!("=".repeat(60));

        // Run the graph
        let result = graph.run(state, ()).await?;

        // Display results
        println!("\n\n" + &"=".repeat(60));
        println!("📄 FINAL ARTICLE");
        println!("=".repeat(60));
        println!();
        println!("{}", result.result);
        println!();
        println!("=".repeat(60));

        // Display execution info
        println!("\n📊 Execution Summary:");
        println!("   Total steps: {}", result.steps);
        println!("   Run ID: {}", result.run_id);
        if let Some(history) = &result.history {
            println!("   Node path: {:?}", history);
        }
        println!(
            "   Research iterations: {}",
            result.final_state.iterations
        );
        println!(
            "   Final research score: {:?}",
            result.final_state.research_score
        );

        Ok(())
    }
}

#[cfg(feature = "graph")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    graph_example::run().await
}

#[cfg(not(feature = "graph"))]
fn main() {
    eprintln!("This example requires the 'graph' feature.");
    eprintln!("Run with: cargo run --example multi_agent_graph --features graph");
}

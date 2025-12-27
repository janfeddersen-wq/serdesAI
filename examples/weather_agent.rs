//! Weather agent with tool calling.
//!
//! This example demonstrates how to create an agent with tools that can
//! be called by the LLM to fetch external data.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example weather_agent
//! ```

use serde::{Deserialize, Serialize};
use serdes_ai::prelude::*;
use serdes_ai::tools::Tool;
use std::collections::HashMap;

/// Weather information returned by the tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherInfo {
    /// Location name.
    pub location: String,
    /// Temperature in Celsius.
    pub temperature: f64,
    /// Weather conditions.
    pub conditions: String,
    /// Humidity percentage.
    pub humidity: u32,
    /// Wind speed in km/h.
    pub wind_speed: f64,
}

/// Tool for getting weather information.
#[derive(Debug, Clone)]
pub struct GetWeatherTool;

impl Tool for GetWeatherTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get the current weather for a specified location.".to_string(),
            parameters_json_schema: serdes_ai::tools::ObjectJsonSchema {
                properties: {
                    let mut props = HashMap::new();
                    props.insert(
                        "location".to_string(),
                        serde_json::json!({
                            "type": "string",
                            "description": "The city name, e.g., 'Paris, France'"
                        }),
                    );
                    props
                },
                required: vec!["location".to_string()],
                additional_properties: Some(false),
                ..Default::default()
            },
            strict: Some(true),
            outer_typed_dict_key: None,
        }
    }

    fn call<'a>(
        &'a self,
        _ctx: &'a serdes_ai::tools::RunContext<'a>,
        args: serde_json::Value,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = ToolResult<serde_json::Value>> + Send + 'a>,
    > {
        Box::pin(async move {
            let location = args["location"]
                .as_str()
                .unwrap_or("Unknown")
                .to_string();

            // Simulated weather data (in real app, call weather API)
            let weather = simulate_weather(&location);

            Ok(serde_json::to_value(&weather)?)
        })
    }
}

/// Simulate weather data for demo purposes.
fn simulate_weather(location: &str) -> WeatherInfo {
    // Simulated weather based on location
    let (temp, conditions, humidity, wind) = match location.to_lowercase().as_str() {
        s if s.contains("paris") => (18.5, "Partly cloudy", 65, 12.0),
        s if s.contains("london") => (14.2, "Rainy", 85, 18.5),
        s if s.contains("tokyo") => (22.8, "Sunny", 55, 8.0),
        s if s.contains("new york") => (20.1, "Clear", 60, 15.0),
        s if s.contains("sydney") => (25.3, "Sunny", 45, 20.0),
        s if s.contains("berlin") => (12.8, "Overcast", 70, 22.0),
        s if s.contains("dubai") => (35.5, "Hot and sunny", 30, 10.0),
        _ => (20.0, "Variable", 50, 10.0),
    };

    WeatherInfo {
        location: location.to_string(),
        temperature: temp,
        conditions: conditions.to_string(),
        humidity,
        wind_speed: wind,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("🌤️ Weather Agent Example\n");

    // Create a tool registry
    let mut registry = ToolRegistry::new();
    registry.register(GetWeatherTool);

    // Create the agent with the weather tool
    let agent = Agent::builder()
        .model("openai:gpt-4o")
        .system_prompt(
            "You are a helpful weather assistant. \
             Use the get_weather tool to fetch current weather information \
             when users ask about the weather in a specific location. \
             Always provide temperature in Celsius.",
        )
        .tools(registry)
        .build()?;

    // Ask about weather in different cities
    let queries = [
        "What's the weather like in Paris right now?",
        "How's the weather in Tokyo?",
        "Compare the weather in London and Sydney.",
    ];

    for query in queries {
        println!("💬 User: {}\n", query);

        let result = agent.run(query, ()).await?;

        println!("🤖 Assistant: {}\n", result.output());
        println!("---\n");
    }

    Ok(())
}

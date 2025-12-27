//! A2A protocol schema types.
//!
//! This module contains all the types defined by the A2A protocol specification,
//! including agent cards, messages, tasks, and artifacts.

use serde::{Deserialize, Serialize};

/// Agent card describing capabilities.
///
/// The agent card is the primary way for agents to advertise their
/// capabilities to other agents and clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCard {
    /// Unique name identifying this agent.
    pub name: String,
    /// URL where this agent can be reached.
    pub url: String,
    /// Version of the agent.
    pub version: String,
    /// Human-readable description of what this agent does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Information about the agent's provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<AgentProvider>,
    /// List of skills this agent provides.
    #[serde(default)]
    pub skills: Vec<Skill>,
}

/// Information about the agent's provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProvider {
    /// Name of the provider.
    pub name: String,
    /// URL to the provider's website.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// A skill that an agent can perform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Unique identifier for this skill.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Description of what this skill does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Tags for categorization.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Task submission parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSendParams {
    /// Thread ID for grouping related messages.
    pub thread_id: String,
    /// The message to send.
    pub message: Message,
}

/// Task ID parameters for querying task status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskIdParams {
    /// The unique task identifier.
    pub task_id: String,
}

/// A message in the A2A protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender.
    pub role: MessageRole,
    /// Content parts of the message.
    pub parts: Vec<Part>,
    /// Optional metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl Message {
    /// Create a new user message with text content.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            parts: vec![Part::text(text)],
            metadata: None,
        }
    }

    /// Create a new agent message with text content.
    pub fn agent(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Agent,
            parts: vec![Part::text(text)],
            metadata: None,
        }
    }

    /// Add metadata to the message.
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Extract text content from all text parts.
    pub fn text_content(&self) -> String {
        self.parts
            .iter()
            .filter_map(|part| match part {
                Part::Text(TextPart { text }) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

/// Role of a message sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// Message from a user/client.
    User,
    /// Message from an agent.
    Agent,
}

/// A part of a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Part {
    /// Text content.
    Text(TextPart),
    /// Structured data.
    Data(DataPart),
}

impl Part {
    /// Create a text part.
    pub fn text(text: impl Into<String>) -> Self {
        Part::Text(TextPart { text: text.into() })
    }

    /// Create a data part.
    pub fn data(data: serde_json::Value) -> Self {
        Part::Data(DataPart {
            data,
            mime_type: None,
        })
    }

    /// Create a data part with a MIME type.
    pub fn data_with_mime(data: serde_json::Value, mime_type: impl Into<String>) -> Self {
        Part::Data(DataPart {
            data,
            mime_type: Some(mime_type.into()),
        })
    }
}

/// Text content part.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPart {
    /// The text content.
    pub text: String,
}

/// Structured data part.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPart {
    /// The data payload.
    pub data: serde_json::Value,
    /// Optional MIME type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// Artifact returned by an agent.
///
/// Artifacts represent files or structured outputs that an agent produces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Name of the artifact.
    pub name: String,
    /// Content parts.
    pub parts: Vec<Part>,
    /// Optional MIME type for the artifact.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

impl Artifact {
    /// Create a new text artifact.
    pub fn text(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parts: vec![Part::text(content)],
            mime_type: Some("text/plain".to_string()),
        }
    }

    /// Create a new JSON artifact.
    pub fn json(name: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            parts: vec![Part::data(data)],
            mime_type: Some("application/json".to_string()),
        }
    }
}

/// Configuration for creating an A2A server.
#[derive(Debug, Clone)]
pub struct A2AConfig {
    /// Agent name.
    pub name: String,
    /// URL where the agent is accessible.
    pub url: String,
    /// Agent version.
    pub version: String,
    /// Description of the agent.
    pub description: Option<String>,
    /// Provider information.
    pub provider: Option<AgentProvider>,
    /// Skills this agent provides.
    pub skills: Vec<Skill>,
}

impl Default for A2AConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl A2AConfig {
    /// Create a new configuration with defaults.
    pub fn new() -> Self {
        Self {
            name: "unnamed-agent".to_string(),
            url: "http://localhost:8000".to_string(),
            version: "1.0.0".to_string(),
            description: None,
            provider: None,
            skills: Vec::new(),
        }
    }

    /// Set the agent name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the agent URL.
    pub fn url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }

    /// Set the agent version.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set the agent description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the provider information.
    pub fn provider(mut self, provider: AgentProvider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Add a skill.
    pub fn skill(mut self, skill: Skill) -> Self {
        self.skills.push(skill);
        self
    }

    /// Convert to an agent card.
    pub fn to_agent_card(&self) -> AgentCard {
        AgentCard {
            name: self.name.clone(),
            url: self.url.clone(),
            version: self.version.clone(),
            description: self.description.clone(),
            provider: self.provider.clone(),
            skills: self.skills.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_user() {
        let msg = Message::user("Hello!");
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.text_content(), "Hello!");
    }

    #[test]
    fn test_message_agent() {
        let msg = Message::agent("Hi there!");
        assert_eq!(msg.role, MessageRole::Agent);
        assert_eq!(msg.text_content(), "Hi there!");
    }

    #[test]
    fn test_part_text() {
        let part = Part::text("hello");
        match part {
            Part::Text(TextPart { text }) => assert_eq!(text, "hello"),
            _ => panic!("Expected text part"),
        }
    }

    #[test]
    fn test_part_data() {
        let part = Part::data(serde_json::json!({"key": "value"}));
        match part {
            Part::Data(DataPart { data, mime_type }) => {
                assert_eq!(data["key"], "value");
                assert!(mime_type.is_none());
            }
            _ => panic!("Expected data part"),
        }
    }

    #[test]
    fn test_artifact_text() {
        let artifact = Artifact::text("output.txt", "file content");
        assert_eq!(artifact.name, "output.txt");
        assert_eq!(artifact.mime_type, Some("text/plain".to_string()));
    }

    #[test]
    fn test_artifact_json() {
        let artifact = Artifact::json("data.json", serde_json::json!({"x": 1}));
        assert_eq!(artifact.name, "data.json");
        assert_eq!(artifact.mime_type, Some("application/json".to_string()));
    }

    #[test]
    fn test_config_builder() {
        let config = A2AConfig::new()
            .name("test")
            .url("http://example.com")
            .version("2.0.0")
            .description("Test agent");

        assert_eq!(config.name, "test");
        assert_eq!(config.url, "http://example.com");
        assert_eq!(config.version, "2.0.0");
        assert_eq!(config.description, Some("Test agent".to_string()));
    }

    #[test]
    fn test_serialization() {
        let msg = Message::user("Hello");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"type\":\"text\""));
    }
}

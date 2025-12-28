//! # serdes-ai-a2a
//!
//! Agent-to-Agent (A2A) protocol support for serdesAI.
//!
//! This crate enables agents to communicate with each other using
//! a standardized protocol based on the FastA2A specification.
//!
//! ## Features
//!
//! - Agent cards for capability discovery
//! - Task submission and tracking
//! - Streaming results
//! - Storage and broker abstractions
//!
//! ## Example
//!
//! ```rust,ignore
//! use serdes_ai_a2a::{agent_to_a2a, A2AConfig};
//! use serdes_ai_agent::Agent;
//!
//! let agent = Agent::new(model).build();
//!
//! let a2a_server = agent_to_a2a(
//!     agent,
//!     A2AConfig::new()
//!         .name("my-agent")
//!         .url("http://localhost:8000")
//!         .description("A helpful assistant")
//! );
//!
//! // Start the server
//! a2a_server.serve("0.0.0.0:8000").await?;
//! ```

pub mod broker;
pub mod schema;
pub mod storage;
pub mod task;
pub mod worker;

#[cfg(feature = "server")]
pub mod server;

pub use broker::{Broker, BrokerError, InMemoryBroker};
pub use schema::*;
pub use storage::{InMemoryStorage, Storage, StorageError};
pub use task::{Task, TaskError, TaskResult, TaskStatus};
pub use worker::{AgentWorker, WorkerHandle};

// Re-export for convenience
pub use serdes_ai_agent::Agent;

use std::sync::Arc;

/// Convert an Agent to an A2A server.
///
/// This is the main entry point for creating an A2A server from an existing agent.
///
/// # Type Parameters
///
/// - `Deps`: Dependencies injected into tools and instruction functions.
/// - `Output`: The output type of the agent.
///
/// # Example
///
/// ```rust,ignore
/// let server = agent_to_a2a(
///     agent,
///     A2AConfig::new()
///         .name("my-agent")
///         .url("http://localhost:8000")
/// );
/// ```
pub fn agent_to_a2a<Deps, Output>(
    agent: Agent<Deps, Output>,
    config: A2AConfig,
) -> A2AServer<Deps, Output>
where
    Deps: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    A2AServer::new(agent, config)
}

/// A2A server wrapping an agent.
///
/// Provides the A2A protocol interface for an agent, including:
/// - Agent card endpoint for capability discovery
/// - Task submission and tracking
/// - Storage and broker integration
pub struct A2AServer<Deps, Output> {
    agent: Arc<Agent<Deps, Output>>,
    config: A2AConfig,
    storage: Arc<dyn Storage>,
    broker: Arc<dyn Broker>,
}

impl<Deps, Output> A2AServer<Deps, Output>
where
    Deps: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    /// Create a new A2A server.
    pub fn new(agent: Agent<Deps, Output>, config: A2AConfig) -> Self {
        Self {
            agent: Arc::new(agent),
            config,
            storage: Arc::new(InMemoryStorage::new()),
            broker: Arc::new(InMemoryBroker::new()),
        }
    }

    /// Set a custom storage backend.
    pub fn with_storage(mut self, storage: impl Storage + 'static) -> Self {
        self.storage = Arc::new(storage);
        self
    }

    /// Set a custom broker.
    pub fn with_broker(mut self, broker: impl Broker + 'static) -> Self {
        self.broker = Arc::new(broker);
        self
    }

    /// Get the agent card describing this agent's capabilities.
    pub fn agent_card(&self) -> AgentCard {
        self.config.to_agent_card()
    }

    /// Get a reference to the underlying agent.
    pub fn agent(&self) -> &Agent<Deps, Output> {
        &self.agent
    }

    /// Get a reference to the storage.
    pub fn storage(&self) -> &dyn Storage {
        self.storage.as_ref()
    }

    /// Get a reference to the broker.
    pub fn broker(&self) -> &dyn Broker {
        self.broker.as_ref()
    }

    /// Get a clone of the storage Arc.
    pub fn storage_arc(&self) -> Arc<dyn Storage> {
        Arc::clone(&self.storage)
    }

    /// Get a clone of the broker Arc.
    pub fn broker_arc(&self) -> Arc<dyn Broker> {
        Arc::clone(&self.broker)
    }
}

impl<Deps, Output> std::fmt::Debug for A2AServer<Deps, Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("A2AServer")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// Prelude for common imports.
pub mod prelude {
    pub use crate::{
        agent_to_a2a, A2AConfig, A2AServer, AgentCard, Artifact, Broker, InMemoryBroker,
        InMemoryStorage, Message, MessageRole, Part, Skill, Storage, Task, TaskIdParams,
        TaskResult, TaskSendParams, TaskStatus,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = A2AConfig::new()
            .name("test-agent")
            .url("http://localhost:8000")
            .description("A test agent");

        assert_eq!(config.name, "test-agent");
        assert_eq!(config.url, "http://localhost:8000");
        assert_eq!(config.description, Some("A test agent".to_string()));
    }

    #[test]
    fn test_agent_card_creation() {
        let config = A2AConfig::new()
            .name("my-agent")
            .url("http://localhost:9000")
            .skill(Skill {
                id: "chat".to_string(),
                name: "Chat".to_string(),
                description: Some("General conversation".to_string()),
                tags: vec!["general".to_string()],
            });

        let card = config.to_agent_card();
        assert_eq!(card.name, "my-agent");
        assert_eq!(card.url, "http://localhost:9000");
        assert_eq!(card.skills.len(), 1);
        assert_eq!(card.skills[0].id, "chat");
    }
}

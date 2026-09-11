//! Application sink example. Replace this in-memory store with your own awaited
//! transactional persistence; encrypt snapshots and never log their payloads.
use async_trait::async_trait;
use serdes_ai_agent::{AgentBuilder, AgentCheckpoint, CheckpointSink};
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct ApplicationStore(Arc<Mutex<Vec<String>>>);
#[async_trait]
impl CheckpointSink for ApplicationStore {
    async fn save(&self, checkpoint: &AgentCheckpoint) -> Result<(), String> {
        let encoded = serde_json::to_string(checkpoint).map_err(|error| error.to_string())?;
        self.0.lock().map_err(|_| "store poisoned")?.push(encoded);
        Ok(())
    }
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = ApplicationStore::default();
    let agent = AgentBuilder::<(), String>::new(serdes_ai_models::FunctionModel::constant_text(
        "local example",
    ))
    .checkpoint_sink(store.clone())
    .build();
    agent.run("hello", ()).await?;
    let last = store.0.lock().unwrap().last().unwrap().clone();
    let checkpoint = AgentCheckpoint::from_json(&last)?;
    // Parsing a snapshot does not replay tools or contact any provider.
    println!("saved boundary: {:?}", checkpoint.boundary);
    Ok(())
}

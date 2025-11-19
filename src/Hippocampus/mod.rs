pub mod redis_backend;
pub mod sqlite_backend;
pub mod error;
pub mod pipeline;
pub mod primordial;
//pub mod hippocampus;
pub mod infer;
pub mod memory_bridge;
pub mod kv_facts;
pub mod fact_parse;
pub mod hippocampus_adapter;
pub mod ids;

// Public use for upstream systems
pub use pipeline::{ingest, uriel_recall};
pub use infer::embed_text;
pub use error::SmieError;


/// Starts the background TTL scanner
pub fn start_scanner() -> Result<(), SmieError> {
    redis_backend::start_scanner()
}

/// Stops the background TTL scanner
pub fn stop_scanner() -> Result<(), SmieError> {
    redis_backend::stop_scanner()
}

/// Cache a semantic memory entry into Redis
pub fn cache_semantic_memory(context: &str, data: &str, ttl_seconds: u64) -> Result<(), SmieError> {
    redis_backend::cache_entry(context, data, ttl_seconds)
}

/// Get recent entries from Redis since a timestamp
pub fn recall_recent_entries(context: &str, since: u64) -> Result<Vec<String>, SmieError> {
    redis_backend::recall_since(context, since)
}

/// Perform a reverse keyword query
pub fn reverse_index_query(context: &str, token: &str) -> Result<Vec<String>, SmieError> {
    redis_backend::search_by_token(context, token, 10)
}

pub fn flush_entry_to_sqlite(context: &str, data: &str, timestamp: Option<u64>) -> Result<(), SmieError> {
    sqlite_backend::store_entry(context, data, timestamp)
}

pub fn recall_all_memory(context: &str) -> Result<Vec<String>, SmieError> {
    let entries = sqlite_backend::recall_all(context)?;
    Ok(entries.into_iter().map(|e| format!("{} | {}", e.timestamp, e.data)).collect())
}

pub fn generate_embedding_from_text(text: &str) -> Vec<f32> {
    infer::embed_text(text)
}
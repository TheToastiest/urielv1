pub mod redis_backend;
pub mod sqlite_backend;
pub mod embedding;
pub mod ffi;
pub mod faiss;
pub mod error;
pub mod pipeline;
pub mod burn_model;
pub mod primordial;
//pub mod hippocampus;
pub mod index;
pub mod infer;
pub mod memory_bridge;
pub mod kv_facts;
pub mod fact_parse;
pub mod hippocampus_adapter;

// Public use for upstream systems
pub use pipeline::{ingest, uriel_recall};
pub use burn_model::embed_text;
pub use error::SmieError;

use crate::Hippocampus::faiss::Index;


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

pub fn search_faiss_index(index_path: &str, embedding: Vec<f32>, top_k: usize) -> Result<Vec<i64>, SmieError> {
    let index = match Index::from_file(index_path) {
        Ok(i) => i,
        Err(_) => {
            println!("⚠️ FAISS index not found. Creating a new one...");
            let new_index = Index::new_flat_l2(embedding.len() as i32)
                .map_err(|e| SmieError::Other(format!("Failed to create index: {}", e)))?;
            new_index.save(index_path)?;
            new_index
        }
    };

    let labels = index.safe_search(&embedding, top_k)?;

    Ok(labels)
}


pub fn store_embedding_for_key(key: &str, embedding: Vec<f32>) -> Result<(), SmieError> {
    embedding::store_embedding(key, &embedding)
}

pub fn load_embedding_for_key(key: &str) -> Result<Vec<f32>, SmieError> {
    embedding::load_embedding(key)
}

pub fn ingest_text(context: &str, text: &str, ttl: u64, source: &str) -> Result<(), SmieError> {
    let _ = pipeline::ingest(
        context.to_string(),
        text.to_string(),
        ttl,
        source.to_string(),
    );
    Ok(())
}

pub fn generate_embedding_from_text(text: &str) -> Vec<f32> {
    burn_model::embed_text(text)
}
// Hippocampus/memory_bridge.rs
use std::time::{SystemTime, UNIX_EPOCH};

use crate::Hippocampus::redis_backend::cache_entry;
use crate::Hippocampus::sqlite_backend::{recall_all, store_entry, SemanticMemoryEntry};
use crate::Hippocampus::error::SmieError;

const DEFAULT_TTL: u64 = 3600; // 1 hour

/// Payload to write into semantic memory.
#[derive(Clone, Debug)]
pub struct MemWrite {
    pub context: String,
    pub text: String,
    pub ts_unix_ms: i64,
    /// Keep the vector for future vector-store integration (unused for SQLite-only).
    pub z: Option<Vec<f32>>,
}

/// Write a memory entry to SQLite (and cache to Redis).
/// Returns a synthetic key you can log / use as a handle.
pub fn write_memory(w: MemWrite) -> Result<String, SmieError> {
    // Convert ms → sec for SQLite
    let ts_sec: u64 = if w.ts_unix_ms > 0 {
        (w.ts_unix_ms as u64) / 1000
    } else {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    };

    // Persist to SQLite
    store_entry(&w.context, &w.text, Some(ts_sec))?;

    // Synthetic id (since your table has no id column)
    let synth_id = format!("sm:{}:{}", w.context, ts_sec);

    // Cache to Redis under the context key with TTL
    cache_entry(&w.context, &w.text, DEFAULT_TTL)?;

    // TODO: add `w.z` to your vector index keyed by `synth_id`
    Ok(synth_id)
}

/// A lightweight hit returned from memory.
#[derive(Clone, Debug)]
pub struct MemHit {
    pub id: String,        // "sm:{context}:{timestamp}"
    pub score: f32,        // placeholder (1.0); update when you add vectors/rerank
    pub snippet: String,   // the stored `data`
    pub ts_unix_ms: i64,   // original timestamp in ms
    pub context: String,
}

/// Context-based search backed by SQLite only (no vectors yet).
/// Returns most recent `k` entries for the given `context`.
pub fn search_memory_by_context(context: &str, k: usize) -> Result<Vec<MemHit>, SmieError> {
    // NOTE: recall_all returns Result<Vec<SemanticMemoryEntry>, SmieError>
    let mut rows: Vec<SemanticMemoryEntry> = recall_all(context)?; // <-- the fix

    // Newest first
    rows.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    rows.truncate(k);

    let hits = rows
        .into_iter()
        .map(|e| {
            let id = format!("sm:{}:{}", context, e.timestamp);
            MemHit {
                id,
                score: 1.0,
                snippet: e.data,
                ts_unix_ms: (e.timestamp as i64) * 1000,
                context: context.to_string(),
            }
        })
        .collect();

    Ok(hits)
}

/// Kept for API compatibility with earlier call sites.
/// Currently ignores `z` and returns empty until a vector index is wired.
/// Prefer `search_memory_by_context` in your handler for now.
pub fn search_memory(_z: &[f32], _k: usize) -> Result<Vec<MemHit>, SmieError> {
    Ok(vec![])
}

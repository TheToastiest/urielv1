// src/memory/ledger.rs
use rusqlite::{params, Connection};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct SemanticMemoryEntry {
    pub context: String,
    pub data: String,
    pub timestamp: u64,
}

fn get_connection() -> rusqlite::Result<Connection> {
    // Same DB you already use elsewhere
    let conn = Connection::open("data/semantic_memory.db")?;
    conn.execute_batch(
        r#"
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA temp_store = MEMORY;

        CREATE TABLE IF NOT EXISTS semantic_memory (
            context   TEXT NOT NULL,
            data      TEXT NOT NULL,
            timestamp INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sm_context_ts
            ON semantic_memory (context, timestamp DESC);
        "#,
    )?;
    Ok(conn)
}

/// Append one semantic memory line.
pub fn append_memory(context: &str, data: &str, timestamp: Option<u64>) -> rusqlite::Result<u64> {
    let conn = get_connection()?;
    let ts = timestamp.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });

    conn.execute(
        "INSERT INTO semantic_memory (context, data, timestamp) VALUES (?1, ?2, ?3)",
        params![context, data, ts],
    )?;

    Ok(ts)
}

/// Retrieve latest N entries (for debugging / fallback).
pub fn recall_latest(context: &str, limit: usize) -> rusqlite::Result<Vec<SemanticMemoryEntry>> {
    let conn = get_connection()?;
    let mut stmt = conn.prepare(
        "SELECT context, data, timestamp
         FROM semantic_memory
         WHERE context = ?1
         ORDER BY timestamp DESC
         LIMIT ?2",
    )?;

    let rows = stmt.query_map(params![context, limit as i64], |row| {
        Ok(SemanticMemoryEntry {
            context: row.get(0)?,
            data: row.get(1)?,
            timestamp: row.get(2)?,
        })
    })?;

    let mut result = Vec::new();
    for row in rows {
        result.push(row?);
    }
    Ok(result)
}

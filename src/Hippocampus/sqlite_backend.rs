use super::error::SmieError;
use rusqlite::{params, Connection};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

/* =========================
   Types
   ========================= */

#[derive(Debug, Clone)]
pub struct SemanticMemoryEntry {
    pub context: String,
    pub data: String,
    pub timestamp: u64, // UNIX seconds
}

/* =========================
   Connection / Schema
   ========================= */

fn default_db_path() -> String {
    // Keep env override, default to ./db/semantic_memory.db
    std::env::var("SMIE_DB_PATH").unwrap_or_else(|_| "data/semantic_memory.db".to_string())
}
pub fn upsert_concept_fact(key: &str, value: &str, ts: u64) -> Result<(), SmieError> {
    let conn = get_connection()?;
    conn.execute(
        "INSERT INTO concept_facts (concept_key, value, updated_at)
         VALUES (?1, ?2, ?3)
         ON CONFLICT(concept_key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        rusqlite::params![key, value, ts as i64],
    )?;
    Ok(())
}

pub fn has_hash(ctx: &str, hash: &str) -> Result<bool, SmieError> {
    let conn = get_connection()?;
    let mut stmt = conn.prepare("SELECT 1 FROM memory_hashes WHERE ctx=?1 AND hash=?2 LIMIT 1")?;
    let exists = stmt.exists(rusqlite::params![ctx, hash])?;
    Ok(exists)
}
pub fn read_concept_fact(key: &str) -> Result<Option<String>, SmieError> {
    let conn = get_connection()?;
    let mut stmt = conn.prepare("SELECT value FROM concept_facts WHERE concept_key=?1 LIMIT 1")?;
    let mut rows = stmt.query(rusqlite::params![key])?;
    if let Some(row) = rows.next()? { let v: String = row.get(0)?; Ok(Some(v)) } else { Ok(None) }
}

pub fn record_hash(ctx: &str, hash: &str, ts: u64) -> Result<bool, SmieError> {
    let conn = get_connection()?;
    match conn.execute(
        "INSERT OR IGNORE INTO memory_hashes (ctx, hash, first_seen_at) VALUES (?1, ?2, ?3)",
        rusqlite::params![ctx, hash, ts as i64],
    )? {
        1 => Ok(true),   // inserted (new)
        _ => Ok(false),  // already existed
    }
}

fn get_connection() -> Result<Connection, SmieError> {
    let path = default_db_path();

    // ensure parent dir exists (no-op if SMIE_DB_PATH points to root)
    if let Some(parent) = std::path::Path::new(&path).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }

    let conn = Connection::open(path)?;

    // WAL + reasonable sync; ignore errors quietly if unsupported
    let _ = conn.execute_batch(
        r#"
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA temp_store = MEMORY;
        PRAGMA mmap_size = 134217728; -- 128 MiB
        "#,
    );

    // Table + index (index is crucial for time-aware queries)
    conn.execute_batch(
        r#"
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

/* =========================
   Core API (unchanged sigs)
   ========================= */

/// Stores a new semantic memory entry (manually flushed from Redis)
pub fn store_entry(context: &str, data: &str, timestamp: Option<u64>) -> Result<(), SmieError> {
    let conn = get_connection()?;
    let ts = timestamp.unwrap_or_else(now_sec);

    conn.execute(
        "INSERT INTO semantic_memory (context, data, timestamp) VALUES (?1, ?2, ?3)",
        params![context, data, ts],
    )?;

    Ok(())
}

/// Retrieves all semantic memory entries for a context (no guaranteed order)
pub fn recall_all(context: &str) -> Result<Vec<SemanticMemoryEntry>, SmieError> {
    let conn = get_connection()?;
    let mut stmt = conn.prepare(
        "SELECT context, data, timestamp
         FROM semantic_memory
         WHERE context = ?1",
    )?;

    let rows = stmt.query_map([context], |row| {
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

/* =========================
   Time-aware helpers (new)
   ========================= */

/// Store and return the timestamp used (useful when you need to align data/embeddings)
pub fn store_entry_with_ts(context: &str, data: &str, ts: Option<u64>) -> Result<u64, SmieError> {
    let used_ts = ts.unwrap_or_else(now_sec);
    store_entry(context, data, Some(used_ts))?;
    Ok(used_ts)
}

/// Latest-first (time-aware) recall
pub fn recall_latest(context: &str, limit: usize) -> Result<Vec<SemanticMemoryEntry>, SmieError> {
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

    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

/// All entries since `from_ts` (inclusive), newest first
pub fn recall_since(context: &str, from_ts: u64) -> Result<Vec<SemanticMemoryEntry>, SmieError> {
    let conn = get_connection()?;
    let mut stmt = conn.prepare(
        "SELECT context, data, timestamp
         FROM semantic_memory
         WHERE context = ?1 AND timestamp >= ?2
         ORDER BY timestamp DESC",
    )?;
    let rows = stmt.query_map(params![context, from_ts as i64], |row| {
        Ok(SemanticMemoryEntry {
            context: row.get(0)?,
            data: row.get(1)?,
            timestamp: row.get(2)?,
        })
    })?;

    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

/// Range query [from_ts, to_ts], newest first
pub fn recall_range(
    context: &str,
    from_ts: u64,
    to_ts: u64,
) -> Result<Vec<SemanticMemoryEntry>, SmieError> {
    let conn = get_connection()?;
    let mut stmt = conn.prepare(
        "SELECT context, data, timestamp
         FROM semantic_memory
         WHERE context = ?1 AND timestamp BETWEEN ?2 AND ?3
         ORDER BY timestamp DESC",
    )?;
    let rows = stmt.query_map(params![context, from_ts as i64, to_ts as i64], |row| {
        Ok(SemanticMemoryEntry {
            context: row.get(0)?,
            data: row.get(1)?,
            timestamp: row.get(2)?,
        })
    })?;

    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

/// Count entries for quick health checks / telemetry
pub fn count(context: &str) -> Result<u64, SmieError> {
    let conn = get_connection()?;
    let mut stmt = conn.prepare(
        "SELECT COUNT(*) FROM semantic_memory WHERE context = ?1",
    )?;
    let n: i64 = stmt.query_row([context], |row| row.get(0))?;
    Ok(n as u64)
}

/* =========================
   Utils
   ========================= */

#[inline]
fn now_sec() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

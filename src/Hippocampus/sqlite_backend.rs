use super::error::SmieError;
use rusqlite::{params, Connection};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::Result;
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
    let mut stmt = conn.prepare(
        "SELECT value
         FROM concept_facts
         WHERE concept_key=?1
         ORDER BY updated_at DESC
         LIMIT 1"
    )?;
    let mut rows = stmt.query(rusqlite::params![key])?;
    if let Some(row) = rows.next()? { Ok(Some(row.get::<_, String>(0)?)) } else { Ok(None) }
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
    // WAL + perf
    let _ = conn.execute_batch(
        r#"
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    PRAGMA temp_store  = MEMORY;
    PRAGMA mmap_size   = 134217728;
    "#
    );

    // Create every table we touch anywhere in this module.
    conn.execute_batch(
        r#"
    -- semantic text rows (already present)
    CREATE TABLE IF NOT EXISTS semantic_memory(
        context   TEXT NOT NULL,
        data      TEXT NOT NULL,
        timestamp INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_sm_context_ts
        ON semantic_memory(context, timestamp DESC);

    -- concept facts (the canonical KV)
    CREATE TABLE IF NOT EXISTS concept_facts(
        concept_key TEXT PRIMARY KEY,
        value       TEXT NOT NULL,
        updated_at  INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_cf_updated_at
        ON concept_facts(updated_at DESC);

    -- durable dedupe for ingestion
    CREATE TABLE IF NOT EXISTS memory_hashes(
        ctx          TEXT NOT NULL,
        hash         TEXT NOT NULL,
        first_seen_at INTEGER NOT NULL,
        PRIMARY KEY (ctx, hash)
    );

    -- lightweight belief stream
    CREATE TABLE IF NOT EXISTS self_beliefs(
        id     INTEGER PRIMARY KEY AUTOINCREMENT,
        ts     INTEGER NOT NULL,
        text   TEXT NOT NULL,
        source TEXT NOT NULL,
        weight REAL    NOT NULL
    );

    -- small policy KV used by critic/router etc.
    CREATE TABLE IF NOT EXISTS policy_kv(
        k          TEXT PRIMARY KEY,
        v          TEXT NOT NULL,
        updated_at INTEGER NOT NULL
    );

    -- write audit (optional but used by safe_upsert_concept_fact)
    CREATE TABLE IF NOT EXISTS write_audit(
        id     INTEGER PRIMARY KEY AUTOINCREMENT,
        ts     INTEGER NOT NULL,
        actor  TEXT NOT NULL,
        op     TEXT NOT NULL,
        k      TEXT NOT NULL,
        old_v  TEXT,
        new_v  TEXT,
        source TEXT,
        notes  TEXT
    );

    -- backward-compat view so legacy queries "concept_fact.key/ts" still work.
    CREATE VIEW IF NOT EXISTS concept_fact AS
        SELECT concept_key AS key,
               value,
               updated_at AS ts
        FROM concept_facts;
    "#
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
// ---- beliefs: types & helpers ------------------------------------------------
#[derive(Debug, Clone)]
pub struct BeliefAgg {
    pub key_norm: String,
    pub n_events: u64,
    pub weight_sum: f32,
    pub user_weight: f32,
    pub last_ts: i64,
}

// Normalize a belief text to a stable key (v1: lowercase trimmed).
fn belief_key_norm(text: &str) -> String {
    text.trim().to_ascii_lowercase()
}

// Insert a belief event and return rowid.
pub fn insert_belief(conn: &rusqlite::Connection, text: &str, source: &str, w: f32)
                     -> Result<i64, SmieError>
{
    let ts = now_sec() as i64;
    conn.execute(
        "INSERT INTO self_beliefs(ts,text,source,weight) VALUES(?1,?2,?3,?4)",
        rusqlite::params![ts, text, source, w],
    )?;
    Ok(conn.last_insert_rowid())
}

// Optional: reinforce a specific belief row (e.g., after an explicit affirm/deny).
pub fn reinforce_belief(conn: &rusqlite::Connection, id: i64, delta: f32) -> Result<(), SmieError> {
    conn.execute(
        "UPDATE self_beliefs SET weight = weight + ?1 WHERE id=?2",
        rusqlite::params![delta, id],
    )?;
    Ok(())
}

// Aggregate current belief stability snapshot (top-K by total weight, then recency).
pub fn beliefs_agg(conn: &rusqlite::Connection, limit: usize) -> Result<Vec<BeliefAgg>, SmieError> {
    let mut stmt = conn.prepare(
        "SELECT lower(trim(text)) AS key_norm,
                COUNT(*)          AS n_events,
                SUM(weight)       AS weight_sum,
                SUM(CASE WHEN source='user' THEN weight ELSE 0 END) AS user_weight,
                MAX(ts)           AS last_ts
         FROM self_beliefs
         GROUP BY key_norm
         ORDER BY weight_sum DESC, last_ts DESC
         LIMIT ?1"
    )?;
    let rows = stmt.query_map([limit as i64], |r| {
        Ok(BeliefAgg {
            key_norm:   r.get(0)?,
            n_events:   r.get::<_, i64>(1)? as u64,
            weight_sum: r.get::<_, f32>(2)?,
            user_weight:r.get::<_, f32>(3)?,
            last_ts:    r.get::<_, i64>(4)?,
        })
    })?;

    let mut out = Vec::new();
    for row in rows { out.push(row?); }
    Ok(out)
}

// Map a belief aggregate to a scalar stability score ∈[0,1].
pub fn belief_stability(agg: &BeliefAgg, now: i64) -> f32 {
    // Time decay with ~1 week half-life
    let age_days = ((now - agg.last_ts).max(0) as f32) / 86_400.0;
    let rec_term = (-age_days / 7.0).exp();  // 1.0 fresh → 0.24 @ ~1w

    // Saturating weight term
    let w = agg.weight_sum;
    let w_term = (w / (w.abs() + 1.0)).clamp(0.0, 1.0);

    // User share of weight (preference for user-grounded beliefs)
    let user_term = (agg.user_weight.max(0.0) / (w.abs() + 1e-6)).clamp(0.0, 1.0);

    (0.5*w_term + 0.3*rec_term + 0.2*user_term).clamp(0.0, 1.0)
}

pub fn upsert_policy_kv(conn: &Connection, k: &str, v: &str, ts: u64) -> Result<()> {
    conn.execute(
        "INSERT INTO policy_kv(k, v, updated_at) VALUES(?1, ?2, ?3)
         ON CONFLICT(k) DO UPDATE SET v=excluded.v, updated_at=excluded.updated_at",
        params![k, v, ts as i64],
    )?;
    Ok(())
}

pub fn read_policy_kv(conn: &Connection, k: &str) -> Result<Option<String>> {
    let mut st = conn.prepare("SELECT v FROM policy_kv WHERE k=?1 LIMIT 1")?;
    let mut rows = st.query(params![k])?;
    if let Some(r) = rows.next()? {
        let v: String = r.get(0)?;
        return Ok(Some(v));
    }
    Ok(None)
}

fn is_primordial_key(k: &str) -> bool {
    // keep this conservative; **never** allow edits to primordial scope
    matches!(k, "axiom" | "prohibition" | "drive" | "primordial")
}

pub fn audit_write(
    conn: &Connection,
    ts: u64,
    actor: &str,
    op: &str,
    k: &str,
    old_v: Option<&str>,
    new_v: Option<&str>,
    source: Option<&str>,
    notes: Option<&str>,
) -> Result<()> {
    conn.execute(
        "INSERT INTO write_audit(ts, actor, op, k, old_v, new_v, source, notes)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            ts as i64, actor, op, k, old_v, new_v, source, notes
        ],
    )?;
    Ok(())
}

pub fn safe_upsert_concept_fact(
    conn: &Connection,
    actor: &str,         // "runtime" | "metacog" | ...
    k: &str,
    v: &str,
    ts: u64,
    source: Option<&str>,// arm/policy
    notes:  Option<&str>
) -> Result<()> {
    if is_primordial_key(k) {
        anyhow::bail!("attempt to write primordial key '{}'", k);
    }

    // read old if exists
    let old_v = {
        let mut st = conn.prepare("SELECT value FROM concept_facts WHERE concept_key=?1 LIMIT 1")?;
        let mut rows = st.query(params![k])?;
        if let Some(r) = rows.next()? { Some::<String>(r.get(0)?).map(|s| s) } else { None }
    };

    // upsert
    conn.execute(
        "INSERT INTO concept_facts(concept_key, value, updated_at)
         VALUES(?1, ?2, ?3)
         ON CONFLICT(concept_key) DO UPDATE
           SET value=excluded.value, updated_at=excluded.updated_at",
        params![k, v, ts as i64],
    )?;

    // audit
    audit_write(
        conn, ts, actor, "upsert_fact", k,
        old_v.as_deref(), Some(v), source, notes
    )?;
    Ok(())
}

// src/memory/rag_ann.rs

use rusqlite::OptionalExtension;
use anyhow::Result;
use bincode;
use rusqlite::{params, Connection};

/// Simple trait so we can later swap in raggedy_anndy without changing call-sites.
pub trait VectorStore: Send + Sync {
    fn search(&self, embedding: &[f32], top_k: usize) -> Result<Vec<MemoryHit>>;
}

/// This mirrors the MemoryHit you already use in PFC::context_retriever.
#[derive(Debug, Clone)]
pub struct MemoryHit {
    pub id: String,
    pub sim: f32,
    pub text: String,
}

/// Backing store used by UnifiedRetriever today.
/// Embeddings live in `db/semantic_embeddings.db`,
/// text in `data/semantic_memory.db` (via memory::ledger).
pub struct SqliteVectorStore {
    emb_path: String,
    mem_path: String,
}

impl SqliteVectorStore {
    pub fn new(emb_path: impl Into<String>, mem_path: impl Into<String>) -> Self {
        Self {
            emb_path: emb_path.into(),
            mem_path: mem_path.into(),
        }
    }
}

impl SqliteVectorStore {
    fn ensure_schema(&self) -> rusqlite::Result<()> {
        let conn = Connection::open(&self.emb_path)?;
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA temp_store = MEMORY;

            CREATE TABLE IF NOT EXISTS semantic_embeddings (
                redis_key TEXT PRIMARY KEY,
                embedding BLOB NOT NULL
            );
            "#,
        )?;
        Ok(())
    }
}

/// Cosine similarity helper
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na2 = 0.0f32;
    let mut nb2 = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na2 += x * x;
        nb2 += y * y;
    }
    if na2 == 0.0 || nb2 == 0.0 {
        0.0
    } else {
        dot / (na2.sqrt() * nb2.sqrt())
    }
}

impl VectorStore for SqliteVectorStore {
    fn search(&self, q: &[f32], top_k: usize) -> Result<Vec<MemoryHit>> {
        self.ensure_schema()?;

        // 1) load all embeddings
        let conn_emb = Connection::open(&self.emb_path)?;
        let mut stmt = conn_emb.prepare(
            "SELECT redis_key, embedding FROM semantic_embeddings",
        )?;
        let mut rows = stmt.query([])?;

        let mut scored: Vec<(String, f32)> = Vec::new();

        while let Some(row) = rows.next()? {
            let key: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let ev: Vec<f32> = bincode::deserialize(&blob)?;
            if ev.len() != q.len() {
                continue;
            }
            let sim = cosine(q, &ev);
            scored.push((key, sim));
        }

        if scored.is_empty() {
            return Ok(vec![]);
        }

        // 2) rank top_k
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if scored.len() > top_k {
            scored.truncate(top_k);
        }

        // 3) join to semantic_memory for text
        let conn_mem = Connection::open(&self.mem_path)?;
        let mut hits: Vec<MemoryHit> = Vec::with_capacity(scored.len());

        for (id, sim) in scored {
            // expected key format: "sm:{context}:{ts}"
            let parts: Vec<_> = id.split(':').collect();
            if parts.len() != 3 || parts[0] != "sm" {
                continue;
            }
            let ctx = parts[1];
            let ts: i64 = match parts[2].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };

            let mut st2 = conn_mem.prepare(
                "SELECT data FROM semantic_memory WHERE context = ?1 AND timestamp = ?2 LIMIT 1",
            )?;
            let text: Option<String> = st2
                .query_row(params![ctx, ts], |r| r.get(0))
                .optional()?;

            let text = text.unwrap_or_default();
            hits.push(MemoryHit { id, sim, text });
        }

        Ok(hits)
    }
}

/// Store one embedding vector keyed by "sm:{context}:{ts}".
/// This matches what SqliteVectorStore expects.
pub fn store_embedding(
    context: &str,
    timestamp: u64,
    embedding: &[f32],
    emb_path: &str,
) -> Result<()> {
    let conn = Connection::open(emb_path)?;
    conn.execute_batch(
        r#"
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA temp_store = MEMORY;

        CREATE TABLE IF NOT EXISTS semantic_embeddings (
            redis_key TEXT PRIMARY KEY,
            embedding BLOB NOT NULL
        );
        "#,
    )?;
    let key = format!("sm:{context}:{timestamp}");

    let blob = bincode::serialize(embedding)?;
    conn.execute(
        "INSERT OR REPLACE INTO semantic_embeddings (redis_key, embedding)
         VALUES (?1, ?2)",
        params![key, blob],
    )?;
    Ok(())
}

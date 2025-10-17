// src/Hippocampus/pipeline.rs

use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use redis::{Commands, Client};
use rusqlite::{params, Connection};

use crate::Hippocampus::infer::HippocampusModel; // <- your combined model module
use crate::Hippocampus::sqlite_backend::{SemanticMemoryEntry, recall_all};
use crate::Hippocampus::redis_backend::get_recent_keys;

// ---------------------------------
// Embedding SQLite (vector) helpers
// ---------------------------------

fn get_embedding_conn() -> Result<Connection, String> {
    fs::create_dir_all("db").map_err(|e| e.to_string())?;
    let conn = Connection::open("db/semantic_embeddings.db").map_err(|e| e.to_string())?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS semantic_embeddings (
            redis_key TEXT PRIMARY KEY,
            embedding BLOB NOT NULL
        )",
        [],
    )
        .map_err(|e| e.to_string())?;

    Ok(conn)
}

/// Store a learned embedding under a stable key (e.g., `sm:{context}:{ts}`).
pub fn store_embedding_sqlite(redis_key: &str, embedding: &[f32]) -> Result<(), String> {
    let conn = get_embedding_conn()?;
    let serialized =
        bincode::serialize(embedding).map_err(|e| format!("bincode serialize failed: {e}"))?;

    conn.execute(
        "INSERT OR REPLACE INTO semantic_embeddings (redis_key, embedding)
         VALUES (?1, ?2)",
        params![redis_key, serialized],
    )
        .map_err(|e| e.to_string())?;

    Ok(())
}

/// Load a learned embedding by key (handy for debugging, tests).
#[allow(dead_code)]
pub fn load_embedding_sqlite(redis_key: &str) -> Result<Vec<f32>, String> {
    let conn = get_embedding_conn()?;
    let mut stmt = conn
        .prepare("SELECT embedding FROM semantic_embeddings WHERE redis_key = ?1")
        .map_err(|e| e.to_string())?;
    let mut rows = stmt.query(params![redis_key]).map_err(|e| e.to_string())?;

    if let Some(row) = rows.next().map_err(|e| e.to_string())? {
        let blob: Vec<u8> = row.get(0).map_err(|e| e.to_string())?;
        let vec: Vec<f32> =
            bincode::deserialize(&blob).map_err(|e| format!("bincode decode failed: {e}"))?;
        Ok(vec)
    } else {
        Err(format!("No embedding found for key: {redis_key}"))
    }
}

// ------------------------
// Ingest + Recall pipeline
// ------------------------

type B = NdArray<f32>;

/// Ingest a text item end-to-end:
///  1) Encode text with the HippocampusModel to a learned vector
///  2) Store text row in SQLite `semantic_memory`
///  3) Store learned vector in SQLite `semantic_embeddings`
///  4) Cache text in Redis under `sm:{context}:{timestamp}` with TTL
fn l2_norm_in_place(v: &mut [f32]) {
    let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
    for x in v { *x /= n; }
}
pub fn ingest<B: Backend>(
    model: &HippocampusModel<B>,
    context: &str,
    text: &str,
    _ttl: u64,
    source: &str,
) -> Result<(), String> {
    // 1) use the provided model to embed
    let z_1xD = model.run_inference(text);                  // Tensor<B, 2>
    let _emb: Vec<f32> = z_1xD.to_data()
        .as_slice::<f32>().map_err(|e| format!("tensor to slice: {e:?}"))?
        .to_vec();

    // 2) store the text row
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| format!("timestamp: {e:?}"))?
        .as_secs();

    super::sqlite_backend::store_entry(context, text, Some(ts))
        .map_err(|e| format!("SQLite (semantic_memory) error: {e:?}"))?;

    // 3) store the learned embedding under a stable key
    let key = format!("sm:{context}:{ts}");
    let mut emb: Vec<f32> = z_1xD
        .to_data()
        .as_slice::<f32>()
        .map_err(|e| format!("{e:?}"))?   // <-- convert to String
        .to_vec();
    l2_norm_in_place(&mut emb);
    store_embedding_sqlite(&key, &emb)?;
    println!(
        "[SMIE::Ingest] ctx={context} src={source} first_dim={:.4} key={key}",
        emb.first().copied().unwrap_or(0.0)
    );

    Ok(())
}

/// Recall recent items (by Redis recency index). Returns text rows ready for context.
/// If you later want semantic recall, add a vector search over `semantic_embeddings`.
pub fn uriel_recall(context: &str, _filter: Option<String>) -> Result<Vec<SemanticMemoryEntry>, String> {
    let prefix = format!("sm:{context}");
    let keys = get_recent_keys(&prefix).map_err(|e| format!("Redis error: {e}"))?;

    let client = Client::open("redis://127.0.0.1/").map_err(|e| e.to_string())?;
    let mut con = client.get_connection().map_err(|e| e.to_string())?;

    let mut out = Vec::with_capacity(keys.len());
    for k in keys {
        if let Ok(data) = con.get::<_, String>(&k) {
            // try to parse ts from the key suffix "...:{ts}"
            let ts = k
                .rsplit(':')
                .next()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or_else(|| {
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                });

            out.push(SemanticMemoryEntry {
                context: context.to_string(),
                data,
                timestamp: ts,
            });
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct Retrieved {
    pub key: String,   // "sm:{context}:{ts}"
    pub sim: f32,      // cosine in [0,1]
    pub text: String,  // recovered from semantic_memory
}

fn l2_norm(v: &mut [f32]) {
    let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
    for x in v { *x /= n; }
}

fn parse_key(key: &str) -> Option<(&str, u64)> {
    // "sm:{context}:{ts}"
    let parts: Vec<&str> = key.split(':').collect();
    if parts.len() == 3 && parts[0] == "sm" {
        if let Ok(ts) = parts[2].parse::<u64>() {
            return Some((parts[1], ts));
        }
    }
    None
}

/// Semantic search within a context using the learned embeddings.
/// - Embeds the query with your Hippocampus model
/// - Scans SQLite `semantic_embeddings` rows for this context
/// - Cosine ranks and joins back to text via `semantic_memory` timestamp
pub fn search_by_context<B: Backend>(
    model: &HippocampusModel<B>,
    context: &str,
    query: &str,
    top_k: usize,
) -> Result<Vec<Retrieved>, String> {
    // 1) Query embedding (1xD -> Vec)
    let q = model.run_inference(query);
    let mut qv = q.to_data().as_slice::<f32>().map_err(|e| format!("{e:?}"))?.to_vec();
    l2_norm(&mut qv);

    // 2) Load all embeddings for this context
    let conn = get_embedding_conn()?;
    let like = format!("sm:{}:%", context);
    let mut stmt = conn
        .prepare("SELECT redis_key, embedding FROM semantic_embeddings WHERE redis_key LIKE ?1")
        .map_err(|e| e.to_string())?;
    let mut rows = stmt.query(params![like]).map_err(|e| e.to_string())?;

    let mut scored: Vec<(String, f32)> = Vec::new();
    while let Some(row) = rows.next().map_err(|e| e.to_string())? {
        let key: String = row.get(0).map_err(|e| e.to_string())?;
        let blob: Vec<u8> = row.get(1).map_err(|e| e.to_string())?;
        let mut ev: Vec<f32> =
            bincode::deserialize(&blob).map_err(|e| format!("bincode decode: {e}"))?;
        l2_norm(&mut ev);

        // cosine (unit vectors → dot)
        let sim = qv.iter().zip(ev.iter()).map(|(a, b)| a * b).sum::<f32>().clamp(0.0, 1.0);
        scored.push((key, sim));
    }

    // 3) rank + truncate
    use std::cmp::Ordering;

    if scored.len() > top_k {
        let k = top_k;
        // partition so top_k highest sims are in scored[..k] (order undefined)
        let _ = scored.select_nth_unstable_by(k, |a, b|            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
        );
        // sort just the top_k slice descending
        scored[..k].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(k);
    }
    // 4) join back to text by (context, timestamp)
    let mut out = Vec::with_capacity(scored.len());
    // recall_all returns all rows; we filter by exact timestamp match
    let rows_text = recall_all(context).map_err(|e| format!("{e:?}"))?;

    for (key, sim) in scored {
        if let Some((_ctx, ts)) = parse_key(&key) {
            if let Some(row) = rows_text.iter().find(|r| r.timestamp == ts) {
                out.push(Retrieved { key, sim, text: row.data.clone() });
            } else {
                // fallback if not found (schema drift etc.)
                out.push(Retrieved { key, sim, text: String::new() });
            }
        }
    }

    Ok(out)
}
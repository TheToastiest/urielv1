use serde::{Deserialize, Serialize};
use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

// ---------- Types you already use in runtime ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditInfo {
    pub arm: String,
    pub reward: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourcesInfo {
    pub answered_from: String,
    pub db_hit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaStep {
    pub turn_id: String,
    pub goal: String,
    pub hypothesis: String,
    pub plan: Vec<String>,
    pub needed: Vec<String>,
    pub evidence: Vec<String>,
    pub confidence: f32,
    pub depth: u8,
    pub outcome: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bandit: Option<BanditInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<SourcesInfo>,
}

// ---------- JSONL writer (Phase 1) ----------
//
// NOTE: `conn` is accepted for API compatibility, but not used here.
// If later you want a SQLite mirror, add an INSERT right below.

pub fn record_meta_trace(
    _conn: &rusqlite::Connection,
    _turn_id: &str,
    phase: &str,
    step: &MetaStep,
    ts: i64,
) {
    // Resolve base dir (env override -> default data/meta)
    let base: PathBuf = std::env::var_os("URIEL_META_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data").join("meta"));

    if let Err(e) = create_dir_all(&base) {
        eprintln!("[meta] create_dir_all({:?}) failed: {e}", base);
        return;
    }

    let path = base.join("meta_trace.jsonl");

    // Wrap with a small envelope so you can query basic fields quickly
    let envelope = serde_json::json!({
        "ts": ts,
        "phase": phase,
        "payload": step,            // full MetaStep serialized here
    });

    match serde_json::to_string(&envelope) {
        Ok(line) => {
            match OpenOptions::new().create(true).append(true).open(&path) {
                Ok(mut f) => {
                    if let Err(e) = writeln!(f, "{line}") {
                        eprintln!("[meta] write failed: {e}");
                    }
                }
                Err(e) => eprintln!("[meta] open {:?} failed: {e}", path),
            }
        }
        Err(e) => eprintln!("[meta] serialize failed: {e}"),
    }
}

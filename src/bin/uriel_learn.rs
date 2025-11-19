// src/bin/uriel_learn.rs
//! Consume learn_queue and promote high-quality answers into the knowledge graph.
//! SMIE / smie_core is gone; this only updates kg_fact + marks items as used.

use anyhow::{anyhow, Result};
use clap::Parser;
use chrono::Utc;
use regex::Regex;
use rusqlite::{params, Connection};

use URIELV1::knowledge::{ops as kops, types as ktypes};

/// Consume learn_queue and update kg_fact
#[derive(Parser, Debug)]
#[command(
    name = "uriel_learn",
    version,
    about = "Consume learn_queue and update knowledge graph (no SMIE index)"
)]
struct Args {
    /// Max items to process
    #[arg(short = 'n', long, default_value_t = 64)]
    max: usize,

    /// Path to semantic memory sqlite file
    #[arg(long, default_value = "data/semantic_memory.db")]
    db: String,
}

// ---------- helpers ----------

fn now_ts() -> i64 {
    Utc::now().timestamp()
}

fn normalize_topic(s: &str) -> String {
    let t = s.trim().to_lowercase();
    match t.as_str() {
        "c++" | "c plus plus" | "c-plus-plus" | "cpp" => "cpp".into(),
        "c#" | "c sharp" | "c-sharp" => "csharp".into(),
        "js" | "javascript" => "javascript".into(),
        "py" | "python" => "python".into(),
        _ => t,
    }
}

fn clean_text(s: &str) -> String {
    let mut t = s.trim().to_string();
    while t.ends_with(['.', ';']) {
        t.pop();
    }
    t
}

/// Extract the “topic” for definitional questions.
fn parse_def_topic(q: &str) -> Option<String> {
    // Match: what is X / what's X / define X / explain X
    static RE: once_cell::sync::Lazy<Regex> = once_cell::sync::Lazy::new(|| {
        Regex::new(r"(?i)^\s*(what\s+is|what'?s|define|explain)\s+(.+?)[\?\s]*$")
            .expect("def-topic regex")
    });
    let caps = RE.captures(q.trim())?;
    let topic = caps.get(2)?.as_str();
    Some(normalize_topic(topic))
}

fn looks_like_url(s: &str) -> bool {
    let l = s.trim().to_ascii_lowercase();
    l.starts_with("http://") || l.starts_with("https://") || l.contains("://")
}

fn looks_like_web_error(s: &str) -> bool {
    let l = s.to_ascii_lowercase();
    l.starts_with("please set a user-agent")
        || l.contains("access denied")
        || l.contains("forbidden")
        || l.contains("captcha")
        || l.contains("robots")
        || looks_like_url(s)
        || s.trim().is_empty()
}

fn main() -> Result<()> {
    let args = Args::parse();

    let conn = Connection::open(&args.db)?;
    conn.busy_timeout(std::time::Duration::from_millis(500))?;

    // fetch a batch from learn_queue
    let mut st = conn.prepare(
        "SELECT id, question, positive, COALESCE(hardness, 0.5) AS hardness
           FROM learn_queue
          WHERE used = 0
          ORDER BY id
          LIMIT ?1",
    )?;
    let rows = st.query_map([args.max as i64], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, f32>(3)?,
        ))
    })?;

    let mut n_done = 0usize;

    for row in rows {
        let (qid, q, p, hardness) = row?;

        // garbage / web errors -> mark used, skip
        if looks_like_url(&q) || looks_like_web_error(&p) {
            let _ = conn.execute("UPDATE learn_queue SET used=1 WHERE id=?1", [qid]);
            continue;
        }

        let conf = (1.0_f32 - hardness).clamp(0.0, 1.0);
        let ts = now_ts();

        if let Some(topic) = parse_def_topic(&q) {
            // Treat this as a definition: write def:<topic> into KG.
            let desc = clean_text(&p);
            let key = format!("def:{}", topic);

            let _ = kops::upsert_fact(
                &conn,
                ktypes::FactWrite {
                    subj: None,
                    predicate: key.clone(),
                    obj_text: desc.clone(),
                    obj_entity: None,
                    confidence: conf,
                    ts,
                    method: "critic".into(),
                    evidence: vec![format!("learn:q:{}|p:{}", &q, &desc)],
                },
            );
            let _ = kops::prefer_latest(&conn, None, &key);
        } else {
            // Non-def question/answer:
            // For now, just record as a generic “qa_snippet” fact keyed by the question hash.
            let q_norm = clean_text(&q);
            let key = format!("qa_snippet:{}", q_norm);

            let snippet = format!("Q: {}  A: {}", q_norm, clean_text(&p));

            let _ = kops::upsert_fact(
                &conn,
                ktypes::FactWrite {
                    subj: None,
                    predicate: key.clone(),
                    obj_text: snippet.clone(),
                    obj_entity: None,
                    confidence: conf,
                    ts,
                    method: "critic".into(),
                    evidence: vec![format!("learn:q:{}|p:{}", &q, &p)],
                },
            );
            // no prefer_latest needed; these are per-question keys
        }

        // mark as used and clear negatives
        let _ = conn.execute("UPDATE learn_queue SET used=1 WHERE id=?1", [qid]);
        let _ = conn.execute("DELETE FROM learn_negs WHERE qid=?1", [qid]);

        n_done += 1;
    }

    println!("Processed {} learn items into KG (no SMIE).", n_done);
    Ok(())
}

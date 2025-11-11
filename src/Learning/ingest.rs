// Learning/ingest.rs
use anyhow::Result;
use chrono::{Datelike, Timelike, Utc};
use serde::Serialize;
use std::{
    fs,
    fs::OpenOptions,
    io::{BufRead, BufReader, BufWriter, Write},
};
use std::collections::HashMap;
struct TaskCtx {
    query: Option<String>,
    source: Option<String>,
    positives: Vec<String>,
}

impl Default for TaskCtx {
    fn default() -> Self {
        Self { query: None, source: None, positives: Vec::new() }
    }
}
#[derive(Serialize)]
struct RouterRow {
    query: String,
    route_taken: String,
    candidates: Vec<String>,
    weight: f32,
    success: bool,
    conf: f32,
}

#[derive(Serialize)]
struct EncoderRow {
    query: String,
    positives: Vec<String>,
    hard_negatives: Vec<String>,
    weight: f32,
}

pub struct DatasetEmitter {
    meta_path: String,
    offset_path: String,
    out_root: String, // e.g., "datasets"
}

impl DatasetEmitter {
    pub fn new(meta_path: impl Into<String>, out_root: impl Into<String>) -> Self {
        let meta_path = meta_path.into();
        let offset_path = format!("{}.offset", meta_path);
        Self { meta_path, offset_path, out_root: out_root.into() }
    }

    pub fn tick(&self) -> Result<()> {
        // recover last offset
        let mut offset: u64 = fs::read_to_string(&self.offset_path)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let f = fs::File::open(&self.meta_path)?;
        let mut r = BufReader::new(f);
        let _ = r.seek_relative(offset as i64);

        // writers for this hour
        let mut router_w = open_jsonl(&self.out_root, "router")?;
        let mut encoder_w = open_jsonl(&self.out_root, "encoder")?;

        // ephemeral state keyed by task_id
        let mut ctx: HashMap<String, TaskCtx> = HashMap::with_capacity(256);

        loop {
            let mut line = String::new();
            let n = r.read_line(&mut line)?;
            if n == 0 { break; }
            offset += n as u64;
            if line.trim().is_empty() { continue; }

            let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else { continue };

            // only care about domain=chat (skip other domains if any)
            if v.get("domain").and_then(|x| x.as_str()) != Some("chat") { continue; }

            let phase = v.get("phase").and_then(|x| x.as_str()).unwrap_or("");
            let tid   = v.get("task_id").and_then(|x| x.as_str()).unwrap_or("").to_string();
            if tid.is_empty() { continue; }

            // fetch or insert
            let entry = ctx.entry(tid.clone()).or_default();

            match phase {
                // user input captured here
                "Sense" => {
                    if let Some(q) = v.get("inputs_ref").and_then(|x| x.as_str()) {
                        entry.query = Some(q.to_string());
                    }
                }

                // retrieval evidence: mark as retrieval and collect positives
                "RetrieveEvidence" => {
                    // where it came from (we stick it in 'source')
                    if let Some(s) = v.get("notes").and_then(|x| x.as_str()) {
                        entry.source = Some(s.to_string()); // e.g., "db"
                    }
                    // collect excerpt texts as positives
                    if let Some(arr) = v.get("evidence").and_then(|x| x.as_array()) {
                        for e in arr {
                            if let Some(ex) = e.get("excerpt").and_then(|x| x.as_str()) {
                                if !ex.is_empty() { entry.positives.push(ex.to_string()); }
                            }
                        }
                    }
                }

                // final-ish event in this schema
                "Report" => {
                    // Decide query
                    let mut query = entry.query.clone()
                        .or_else(|| v.get("inputs_ref").and_then(|x| x.as_str()).map(|s| s.to_string()))
                        .unwrap_or_default();
                    if query.starts_with("H: ") { query = query[3..].trim().to_string(); }
                    if query.is_empty() { continue; }

                    // confidence: array of floats OR single
                    let conf = if let Some(arr) = v.get("confidence").and_then(|x| x.as_array()) {
                        if arr.is_empty() { 0.0 } else {
                            let sum: f32 = arr.iter().filter_map(|x| x.as_f64()).map(|f| f as f32).sum();
                            sum / (arr.len() as f32)
                        }
                    } else {
                        v.get("confidence").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32
                    };

                    let loss = (1.0 - conf).clamp(0.0, 1.0);
                    let w    = 0.5 + 0.5 * loss;

                    // answered_from → route_taken
                    // rule: if we saw RetrieveEvidence, treat as retrieval; else direct
                    let answered_from = entry.source.clone().unwrap_or("direct".into());
                    let route_taken = match answered_from.as_str() {
                        "db" | "smie" | "rows" | "pfc" | "retrieval" => "retrieve_then_answer",
                        "tool:code" => "tool:code",
                        "tool:sql"  => "tool:sql",
                        _ => "direct",
                    }.to_string();

                    // router row
                    let rr = RouterRow {
                        query: query.clone(),
                        route_taken,
                        candidates: vec![
                            "direct".into(),
                            "retrieve_then_answer".into(),
                            "tool:code".into(),
                            "tool:sql".into(),
                        ],
                        weight: w,
                        success: true,   // 'Report' implies a final answer surfaced
                        conf,
                    };
                    serde_json::to_writer(&mut router_w, &rr)?; router_w.write_all(b"\n")?;

                    // encoder row (only if we have positives)
                    if !entry.positives.is_empty() {
                        let er = EncoderRow {
                            query: query.clone(),
                            positives: entry.positives.clone(),
                            hard_negatives: Vec::new(), // optional: mine later
                            weight: w,
                        };
                        serde_json::to_writer(&mut encoder_w, &er)?; encoder_w.write_all(b"\n")?;
                    }

                    // cleanup this task_id to bound memory
                    ctx.remove(&tid);
                }

                _ => { /* ignore other phases (Plan/Learn/…) */ }
            }

            // simple GC: cap the map
            if ctx.len() > 1024 {
                ctx.clear();
            }
        }

        fs::write(&self.offset_path, offset.to_string()).ok();
        Ok(())
    }

}

fn open_jsonl(out_root: &str, kind: &str) -> Result<BufWriter<std::fs::File>> {
    let now = Utc::now();
    let dir = format!(
        "{}/{}/{:04}/{:02}/{:02}/train",
        out_root,
        kind,
        now.year(),
        now.month(),
        now.day()
    );
    fs::create_dir_all(&dir)?;
    let path = format!("{}/part-{:02}.jsonl", dir, now.hour());
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    Ok(BufWriter::new(file))
}

// src/bin/uriel_ingest.rs
//! Autonomous .txt ingester for URIEL (SMIE + Hippocampus pipeline).
//!
//! Usage:
//!   cargo run --bin uriel_ingest -- [--from-start] <file-or-dir> [<file-or-dir> ...]
//!
//! Env (optional):
//!   SMIE_DB_PATH=data/semantic_memory.db
//!   SMIE_STORE_DIR=data/uriel
//!   URIEL_ENCODER_PATH=R:/uriel/models/checkpoints/uriel_encoder_step008500.bin
//!   OEP_WEIGHTS_PATH=data/oep_w.json
//!   INGEST_CTX=default
//!   INGEST_CTX_PER_FILE=1
//!   INGEST_TTL_SECS=604800
//!   INGEST_RESCAN_SECS=0
//!   FROM_START=0

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};

use anyhow::Result;
use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use crossbeam_channel::{unbounded, Receiver, Sender};
use ctrlc;
use regex::Regex;
use rusqlite;
use sha2::{Digest, Sha256};
use smie_core::{Embedder as SmieEmbedder, Metric, Smie, SmieConfig, tags};

use URIELV1::Hippocampus::infer::HippocampusModel;
use URIELV1::Hippocampus::pipeline;
use URIELV1::Hippocampus::sqlite_backend::{
    has_hash, read_concept_fact, record_hash, safe_upsert_concept_fact,
};

type B = NdArray<f32>;

// ---------------------- small utils ----------------------
fn now_sec() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn sanitize_input(s: &str) -> String {
    let mut t = s.trim();
    while let Some(c) = t.chars().next() {
        if matches!(c, '>' | '•' | '-' | '|') {
            t = t[1..].trim_start();
        } else {
            break;
        }
    }
    t.to_string()
}

fn normalize_for_hash(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_ws = false;
    for ch in s.chars().map(|c| c.to_ascii_lowercase()) {
        if ch.is_whitespace() {
            if !last_ws {
                out.push(' ');
            }
            last_ws = true;
        } else {
            last_ws = false;
            out.push(ch);
        }
    }
    out.trim().to_string()
}

fn sha256_hex(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    let bytes = h.finalize();
    let mut hex = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(hex, "{:02x}", b);
    }
    hex
}

// ---------------------- schema bootstrap ----------------------
fn ensure_runtime_schema(conn: &rusqlite::Connection) -> anyhow::Result<()> {
    conn.execute_batch(
        r#"
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA temp_store = MEMORY;

        -- concept_facts used by read_concept_fact/safe_upsert_concept_fact
        CREATE TABLE IF NOT EXISTS concept_facts (
            concept_key TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            updated_at  INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cf_updated_at
            ON concept_facts(updated_at DESC);

        CREATE VIEW IF NOT EXISTS concept_fact AS
        SELECT concept_key AS key, value, updated_at AS ts
        FROM   concept_facts;

        -- durable dedupe
        CREATE TABLE IF NOT EXISTS memory_hashes (
            ctx           TEXT NOT NULL,
            hash          TEXT NOT NULL,
            first_seen_at INTEGER NOT NULL,
            PRIMARY KEY (ctx, hash)
        );

        -- rows store used by Hippocampus pipeline::ingest recall
        CREATE TABLE IF NOT EXISTS semantic_memory (
            context   TEXT NOT NULL,
            data      TEXT NOT NULL,
            timestamp INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sm_context_ts
            ON semantic_memory (context, timestamp DESC);
        "#,
    )?;
    Ok(())
}

// ---------------------- SMIE embedder (HIPPO) ----------------------
#[derive(serde::Deserialize, serde::Serialize, Clone)]
struct OepWeightsFile {
    dim: usize,
    w: Vec<f32>,
    updated_at: i64,
}

/// Minimal HIPPO-backed embedder for SMIE with optional diagonal OEP weights.
struct SmieHippEmbed<Bk: Backend> {
    model: Arc<Mutex<HippocampusModel<Bk>>>,
    dim: usize,
    oep_w: RwLock<Option<Vec<f32>>>,
    oep_path: String,
    oep_mtime: RwLock<Option<SystemTime>>,
}

impl<Bk: Backend> SmieHippEmbed<Bk> {
    fn new(model: Arc<Mutex<HippocampusModel<Bk>>>) -> anyhow::Result<Self> {
        let dim = {
            let m = model
                .lock()
                .map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
            let z = m.run_inference("[dim-probe]");
            z.to_data()
                .as_slice::<f32>()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?
                .len()
        };

        let path = env::var("OEP_WEIGHTS_PATH").unwrap_or_else(|_| "data/oep_w.json".to_string());
        let mut w: Option<Vec<f32>> = None;
        let mut mt: Option<SystemTime> = None;
        if let Ok(md) = fs::metadata(&path) {
            if let Ok(txt) = fs::read_to_string(&path) {
                if let Ok(wf) = serde_json::from_str::<OepWeightsFile>(&txt) {
                    if wf.dim == dim && wf.w.len() == dim {
                        w = Some(wf.w);
                        mt = md.modified().ok();
                        eprintln!("[oep] loaded {}", path);
                    }
                }
            }
        }

        Ok(Self {
            model,
            dim,
            oep_w: RwLock::new(w),
            oep_path: path,
            oep_mtime: RwLock::new(mt),
        })
    }

    fn maybe_hot_reload(&self) {
        if let Ok(md) = fs::metadata(&self.oep_path) {
            let now_mt = md.modified().ok();
            let last_mt = *self.oep_mtime.read().unwrap();
            if now_mt.is_some() && now_mt != last_mt {
                if let Ok(txt) = fs::read_to_string(&self.oep_path) {
                    if let Ok(wf) = serde_json::from_str::<OepWeightsFile>(&txt) {
                        if wf.dim == self.dim && wf.w.len() == self.dim {
                            *self.oep_w.write().unwrap() = Some(wf.w);
                            *self.oep_mtime.write().unwrap() = now_mt;
                            eprintln!("[oep] hot-reloaded {}", self.oep_path);
                        }
                    }
                }
            }
        }
    }

    fn save(&self) {
        if let Some(w) = self.oep_w.read().unwrap().as_ref() {
            let obj = OepWeightsFile {
                dim: self.dim,
                w: w.clone(),
                updated_at: now_sec() as i64,
            };
            if let Ok(txt) = serde_json::to_string_pretty(&obj) {
                let _ = fs::create_dir_all("data");
                let _ = fs::write(&self.oep_path, txt);
            }
        }
    }
}

impl<Bk: Backend> SmieEmbedder for SmieHippEmbed<Bk> {
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        self.maybe_hot_reload();

        let m = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
        let z = m.run_inference(text);
        let mut v = z
            .to_data()
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?
            .to_vec();
        drop(m);

        // L2 normalize
        let n0 = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
        for x in &mut v {
            *x /= n0;
        }

        // Diagonal weights (if any) + renorm
        if let Some(w) = self.oep_w.read().unwrap().as_ref() {
            for i in 0..self.dim {
                v[i] *= w[i];
            }
            let n1 = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
            for x in &mut v {
                *x /= n1;
            }
        }
        Ok(v)
    }
}

// ---------------------- definition helpers ----------------------
fn ascii_norm(s: &str) -> String {
    s.replace('’', "'")
        .replace('‘', "'")
        .replace('“', "\"")
        .replace('”', "\"")
        .replace('—', "-")
        .replace('–', "-")
        .trim()
        .to_string()
}

fn strip_quotes(s: &str) -> &str {
    s.trim().trim_matches('"').trim_matches('\'')
}

fn strip_topic_parens(s: &str) -> String {
    let s = s.trim();
    if let Some(i) = s.find('(') {
        return s[..i].trim().to_string();
    }
    s.to_string()
}

// Accepts:
//  1) Topic (Paren)? is Desc
//  2) Topic: Desc   or   Topic - Desc / Topic — Desc
//  3) Desc is Topic  (reversed; we flip to Topic: Desc)
fn extract_definition(line: &str) -> Option<(String, String)> {
    let l = ascii_norm(line);
    let cop = r"(?:is|are|refers\s+to|means|=)";

    // 1) Topic (Paren)? cop Desc
    let re1 = Regex::new(&format!(
        r"(?i)^\s*([A-Za-z0-9+./#\-\s'’]{{1,64}}?)(?:\s*\([^)]+\))?\s+{}\s+(.+?)\s*$",
        cop
    ))
        .ok()?;
    if let Some(c) = re1.captures(&l) {
        let topic_raw = c.get(1)?.as_str();
        let topic = strip_quotes(&strip_topic_parens(topic_raw)).to_string();
        let desc = c.get(2)?.as_str().trim().to_string();
        if !topic.is_empty() && !desc.is_empty() {
            return Some((topic, desc));
        }
    }

    // 2) Topic:/-/— Desc
    let re2 = Regex::new(r"(?i)^\s*([A-Za-z0-9+./#\-\s'’]{1,64}?)\s*[:\-—]\s+(.+?)\s*$").ok()?;
    if let Some(c) = re2.captures(&l) {
        let topic = strip_quotes(c.get(1)?.as_str()).to_string();
        let desc = c.get(2)?.as_str().trim().to_string();
        if !topic.is_empty() && !desc.is_empty() {
            return Some((topic, desc));
        }
    }

    // 3) Desc cop Topic  (reversed)
    let re3 = Regex::new(&format!(
        r"(?i)^\s*(.+?)\s+{}\s+([A-Za-z0-9+./#\-\s'’]{{1,64}})\s*\.?\s*$",
        cop
    ))
        .ok()?;
    if let Some(c) = re3.captures(&l) {
        let desc = c.get(1)?.as_str().trim().to_string();
        let topic = strip_quotes(c.get(2)?.as_str()).to_string();
        if !topic.is_empty() && !desc.is_empty() {
            return Some((topic, desc));
        }
    }
    None
}

fn topic_surface_norm(s: &str) -> String {
    ascii_norm(s).to_ascii_lowercase().replace(['–', '—'], "-").trim().to_string()
}

fn maybe_singular(s: &str) -> String {
    let mut t = topic_surface_norm(s);
    if t.ends_with("ies") && t.len() > 3 {
        t.truncate(t.len() - 3);
        t.push('y');
    } else if t.ends_with('s') && !t.ends_with("ss") && !t.ends_with("us") {
        t.truncate(t.len() - 1);
    }
    t
}

fn canonicalize_surface(s: &str) -> String {
    let mut t = topic_surface_norm(s);
    for pre in ["the ", "a ", "an "] {
        if t.starts_with(pre) {
            t = t[pre.len()..].to_string();
            break;
        }
    }
    strip_topic_parens(&maybe_singular(&t))
}

fn norm_domain(s: &str) -> String {
    ascii_norm(s).to_ascii_lowercase().replace([' ', '\t'], "")
}

fn def_key(domain: Option<&str>, topic: &str) -> String {
    match domain {
        Some(d) if !d.is_empty() => format!("def:{}:{}", norm_domain(d), canonicalize_surface(topic)),
        _ => format!("def:{}", canonicalize_surface(topic)),
    }
}

// In <domain>, Topic is Desc   OR   Topic is Desc in <domain>
fn parse_domained_def(line: &str) -> Option<(String, String, String)> {
    let l = ascii_norm(line);
    let cop = r"(?:is|are|refers\s+to|means|=)";

    // In <domain>, Topic cop Desc
    if let Ok(re) = Regex::new(&format!(
        r"(?i)^\s*in\s+([a-z][a-z0-9 _\-]{{2,32}})\s*,\s*([A-Za-z0-9+./#\-\s'’]{{1,64}}?)\s+{}\s+(.+?)\s*$",
        cop
    )) {
        if let Some(c) = re.captures(&l) {
            let dom = norm_domain(c.get(1).unwrap().as_str());
            let topic = strip_quotes(c.get(2).unwrap().as_str()).to_string();
            let desc = c.get(3).unwrap().as_str().trim().to_string();
            if !dom.is_empty() && !topic.is_empty() && !desc.is_empty() {
                return Some((dom, topic, desc));
            }
        }
    }

    // Topic cop Desc in <domain>
    if let Ok(re) = Regex::new(&format!(
        r"(?i)^\s*([A-Za-z0-9+./#\-\s'’]{{1,64}}?)\s+{}\s+(.+?)\s+in\s+([a-z][a-z0-9 _\-]{{2,32}})\s*$",
        cop
    )) {
        if let Some(c) = re.captures(&l) {
            let topic = strip_quotes(c.get(1).unwrap().as_str()).to_string();
            let desc = c.get(2).unwrap().as_str().trim().to_string();
            let dom = norm_domain(c.get(3).unwrap().as_str());
            if !dom.is_empty() && !topic.is_empty() && !desc.is_empty() {
                return Some((dom, topic, desc));
            }
        }
    }

    None
}

// alias helpers stored in concept_facts
fn read_alias(surface_norm: &str) -> Option<String> {
    let k = format!("alias:{}", surface_norm);
    read_concept_fact(&k).ok().flatten()
}
fn write_alias(conn: &rusqlite::Connection, surface_norm: &str, canonical: &str, ts: u64) {
    let k = format!("alias:{}", surface_norm);
    let _ = safe_upsert_concept_fact(conn, "ingest", &k, canonical, ts, Some("alias"), None);
}

// ---------------------- tailers ----------------------
fn spawn_tail_source(
    path: PathBuf,
    from_start: bool,
    tx: Sender<(String, String)>, // (path, line)
) {
    std::thread::spawn(move || loop {
        match File::open(&path) {
            Ok(mut f) => {
                if !from_start {
                    let _ = f.seek(SeekFrom::End(0));
                }
                let mut rdr = BufReader::new(f);
                let mut buf = String::new();
                loop {
                    buf.clear();
                    match rdr.read_line(&mut buf) {
                        Ok(0) => std::thread::sleep(Duration::from_millis(25)),
                        Ok(_) => {
                            let s = buf.trim_end_matches(&['\n', '\r'][..]).to_string();
                            if !s.is_empty() {
                                let _ = tx.send((path.to_string_lossy().to_string(), s));
                            }
                        }
                        Err(_) => break, // reopen on error/rotation
                    }
                }
            }
            Err(_) => {
                std::thread::sleep(Duration::from_millis(200));
            }
        }
    });
}

fn list_txt_files_rec(root: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = fs::read_dir(root) {
        for ent in rd.flatten() {
            let p = ent.path();
            if p.is_dir() {
                list_txt_files_rec(&p, out);
            } else if p
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("txt"))
                .unwrap_or(false)
            {
                out.push(p);
            }
        }
    }
}

fn spawn_dir_rescanner(
    roots: Vec<PathBuf>,
    known: Arc<Mutex<HashSet<String>>>,
    tx: Sender<(String, String)>,
    from_start: bool,
    every_secs: u64,
) {
    if every_secs == 0 {
        return;
    }
    std::thread::spawn(move || loop {
        for root in &roots {
            let mut found = Vec::new();
            list_txt_files_rec(root, &mut found);
            for p in found {
                let key = p
                    .canonicalize()
                    .unwrap_or(p.clone())
                    .to_string_lossy()
                    .to_string();
                let mut guard = known.lock().unwrap();
                if !guard.contains(&key) {
                    guard.insert(key.clone());
                    drop(guard);
                    spawn_tail_source(PathBuf::from(&key), from_start, tx.clone());
                    eprintln!("[watch] new file: {}", key);
                }
            }
        }
        std::thread::sleep(Duration::from_secs(every_secs));
    });
}

// ---------------------- main ----------------------
fn main() -> Result<()> {
    // ---- parse args / env ----
    let mut args: Vec<String> = env::args().skip(1).collect();
    let mut from_start = env::var("FROM_START")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "on" | "ON"))
        .unwrap_or(false);
    if let Some(pos) = args.iter().position(|a| a == "--from-start") {
        from_start = true;
        args.remove(pos);
    }
    if args.is_empty() {
        eprintln!("Usage: uriel_ingest [--from-start] <file-or-dir> [...]");
        eprintln!("No inputs provided; exiting.");
        std::process::exit(2);
    }

    let smie_db_path =
        env::var("SMIE_DB_PATH").unwrap_or_else(|_| "data/semantic_memory.db".to_string());
    let smie_store_dir = env::var("SMIE_STORE_DIR").unwrap_or_else(|_| "data/uriel".to_string());
    let encoder_path = env::var("URIEL_ENCODER_PATH")
        .unwrap_or_else(|_| "R:/uriel/models/checkpoints/uriel_encoder_step008500.bin".to_string());
    let ttl_secs: u64 = env::var("INGEST_TTL_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7 * 24 * 3600);
    let ctx_default = env::var("INGEST_CTX").unwrap_or_else(|_| "default".to_string());
    let ctx_per_file = env::var("INGEST_CTX_PER_FILE")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "on" | "ON"))
        .unwrap_or(true);
    let rescan_secs: u64 = env::var("INGEST_RESCAN_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    fs::create_dir_all("data").ok();

    // ---- open DB + schema ----
    eprintln!("[db] opening {}", smie_db_path);
    let conn = rusqlite::Connection::open(&smie_db_path)?;
    ensure_runtime_schema(&conn)?;
    conn.busy_timeout(Duration::from_millis(500)).ok();

    // ---- model + SMIE ----
    let device = <B as Backend>::Device::default();
    let model = HippocampusModel::<B>::load_or_init(&device, &encoder_path);
    let model_for_smie =
        Arc::new(Mutex::new(HippocampusModel::<B>::load_or_init(&device, &encoder_path)));

    let embed = Arc::new(SmieHippEmbed::new(model_for_smie.clone())?);
    let smie = Arc::new(Smie::new(
        SmieConfig {
            dim: embed.dim(),
            metric: Metric::Cosine,
        },
        embed.clone(),
    )?);

    let _ = smie.load(&smie_store_dir);
    eprintln!(
        "[ingest] ctx_default='{}' per_file={} ttl={}s from_start={} rescan={}s",
        ctx_default, ctx_per_file, ttl_secs, from_start, rescan_secs
    );

    // ---- lines bus ----
    let (tx, rx): (Sender<(String, String)>, Receiver<(String, String)>) = unbounded();

    // Spawn tailers for inputs
    let mut roots_for_rescan: Vec<PathBuf> = Vec::new();
    let known = Arc::new(Mutex::new(HashSet::<String>::new()));

    for arg in args {
        let p = PathBuf::from(&arg);
        if p.is_dir() {
            let mut files = Vec::new();
            list_txt_files_rec(&p, &mut files);
            for f in files {
                let key = f
                    .canonicalize()
                    .unwrap_or(f.clone())
                    .to_string_lossy()
                    .to_string();
                let mut guard = known.lock().unwrap();
                if guard.insert(key.clone()) {
                    drop(guard);
                    spawn_tail_source(PathBuf::from(&key), from_start, tx.clone());
                    eprintln!("[watch] file: {}", key);
                }
            }
            roots_for_rescan.push(p);
        } else if p.is_file() {
            let key = p
                .canonicalize()
                .unwrap_or(p.clone())
                .to_string_lossy()
                .to_string();
            let mut guard = known.lock().unwrap();
            if guard.insert(key.clone()) {
                drop(guard);
                spawn_tail_source(PathBuf::from(&key), from_start, tx.clone());
                eprintln!("[watch] file: {}", key);
            }
        } else {
            eprintln!("[warn] not found: {}", p.display());
        }
    }

    // Optional periodic rescan for new .txt files
    spawn_dir_rescanner(roots_for_rescan, known.clone(), tx.clone(), from_start, rescan_secs);

    // ---- ctrl-c graceful save ----
    let smie_c = smie.clone();
    let embed_c = embed.clone();
    ctrlc::set_handler(move || {
        eprintln!("\n[ingest] SIGINT — saving SMIE and OEP weights...");
        let _ = smie_c.save("data/uriel");
        embed_c.save();
        std::process::exit(0);
    })?;

    // ---- ingest loop ----
    let mut inproc_seen: HashSet<String> = HashSet::with_capacity(8192);
    let mut stats: HashMap<&'static str, u64> =
        [("ingested", 0), ("dedup", 0), ("errors", 0)].into_iter().collect();

    loop {
        let (path, raw) = match rx.recv() {
            Ok(t) => t,
            Err(_) => break, // sender dropped
        };

        let mut line = sanitize_input(&raw);
        if line.is_empty() {
            continue;
        }
        let ts = now_sec();

        // ----------------- definitional ingestion (concept_facts) -----------------
        // do this BEFORE dedupe so repeated lines still update facts
        let mut ingest_text: Option<String> = None;

        if let Some((dom, topic_d, desc_d)) = parse_domained_def(&line) {
            let surf = canonicalize_surface(&topic_d);
            let canon = read_alias(&surf).unwrap_or_else(|| surf.clone());
            if canon != surf {
                write_alias(&conn, &surf, &canon, ts);
            }
            let key_cf = def_key(Some(&dom), &canon);
            if let Err(e) =
                safe_upsert_concept_fact(&conn, "ingest", &key_cf, &desc_d, ts, Some("def.domained"), None)
            {
                eprintln!("[cf] domained upsert error: {e}");
            } else {
                ingest_text = Some(format!("{} ({}): {}", canon, dom, desc_d));
            }
        } else if let Some((topic, desc)) = extract_definition(&line) {
            let surf = canonicalize_surface(&topic);
            let canon = read_alias(&surf).unwrap_or_else(|| surf.clone());
            if canon != surf {
                write_alias(&conn, &surf, &canon, ts);
            }
            let key_cf = def_key(None, &canon);
            if let Err(e) =
                safe_upsert_concept_fact(&conn, "ingest", &key_cf, &desc, ts, Some("def.generic"), None)
            {
                eprintln!("[cf] generic upsert error: {e}");
            } else {
                ingest_text = Some(format!("{}: {}", canon, desc));
            }
        }

        // ----------------- dedupe for SMIE/pipeline -----------------
        // canonical short form (if we built one) is what we store for retrieval
        let to_store = ingest_text.as_deref().unwrap_or(&line);

        let n = normalize_for_hash(to_store);
        let key = sha256_hex(&n);
        if !inproc_seen.insert(format!("{}::{}", &path, key)) {
            stats.entry("dedup").and_modify(|c| *c += 1);
            continue;
        }
        if has_hash(&path, &key).unwrap_or(false) {
            stats.entry("dedup").and_modify(|c| *c += 1);
            continue;
        }
        // FIX: record_hash expects u64, not i64
        if let Err(e) = record_hash(&path, &key, ts) {
            eprintln!("[dedupe] record_hash error: {e}");
        }

        // Decide SMIE tags by length (use canonical short text if any)
        let tmask = if to_store.len() > 180 {
            tags::SYSTEM | tags::LONG
        } else {
            tags::SYSTEM | tags::SHORT
        };

        // Ingest into SMIE
        if let Err(e) = smie.ingest(to_store, tmask) {
            eprintln!("[smie] ingest error: {e:?}");
            stats.entry("errors").and_modify(|c| *c += 1);
            continue;
        }

        // Ingest into Hippocampus rows
        // Build a String, then pass &str to pipeline::ingest
        let ctx_owned: String = if ctx_per_file {
            Path::new(&path)
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| format!("file:{}", s))
                .unwrap_or_else(|| ctx_default.clone())
        } else {
            ctx_default.clone()
        };

        // Ensure to_store is &str (not String)
        let to_store: &str = ingest_text.as_deref().unwrap_or(&line);

        // Ingest into Hippocampus rows
        if let Err(e) = pipeline::ingest(&model, &ctx_owned, to_store, ttl_secs, "ingest") {
            eprintln!("[hippo] ingest error: {e:?}");
            stats.entry("errors").and_modify(|c| *c += 1);
            continue;
        }


        stats.entry("ingested").and_modify(|c| *c += 1);

        if (stats["ingested"] + stats["dedup"] + stats["errors"]) % 200 == 0 {
            eprintln!(
                "[ingest] ok={} dedup={} err={} (last file='{}')",
                stats["ingested"], stats["dedup"], stats["errors"], path
            );
            let _ = smie.save(&smie_store_dir);
        }
    }

    let _ = smie.save(&smie_store_dir);
    embed.save();

    eprintln!(
        "[done] ingested={} dedup={} err={}",
        stats["ingested"], stats["dedup"], stats["errors"]
    );
    Ok(())
}

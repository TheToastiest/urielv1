// src/bin/uriel_ui.rs
#![allow(clippy::too_many_arguments)]

use std::cmp::Ordering;
use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering as AO},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use burn::record::{CompactRecorder, Recorder}; // <-- Recorder brings `.load` into scope
use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use eframe::egui;
use rand::Rng;
use regex::Regex;
use rusqlite;
use sha2::{Digest, Sha256};
use smie_core::{Embedder, Metric, Smie, SmieConfig, tags};

use URIELV1::Hippocampus::infer::HippocampusModel;
use URIELV1::Hippocampus::pipeline;
use URIELV1::Hippocampus::sqlite_backend::{
    has_hash, insert_belief, read_concept_fact, recall_latest, record_hash, upsert_concept_fact,
};
use URIELV1::concepts_and_contexts::{
    extractors::{CreatorConcept, IdentityConcept, LaunchCodeConcept, MentorConcept, NameConcept, PurposeConcept},
    registry::Registry,
    speech_concepts,
};

// ───────────────────────── helpers (kept aligned with runtime) ─────────────────────────
fn sanitize_input(s: &str) -> String {
    let mut t = s.trim();
    while let Some(c) = t.chars().next() {
        if matches!(c, '>' | '•' | '-' | '|') { t = t[1..].trim_start(); } else { break; }
    }
    t.to_string()
}
fn is_question(s: &str) -> bool {
    let l = s.trim().to_ascii_lowercase();
    l.ends_with('?')
        || l.starts_with("what ")
        || l.starts_with("who ")
        || l.starts_with("where ")
        || l.starts_with("why ")
        || l.starts_with("how ")
}
fn normalize_question(q: &str) -> String {
    let l = q.trim().to_ascii_lowercase();
    if l.starts_with("what is your name") || l.starts_with("what's your name") {
        "your name is".into()
    } else if l.starts_with("what is the launch code") {
        "the launch code is".into()
    } else if l.starts_with("who is your creator") {
        "your creator is".into()
    } else if l.starts_with("who is your mentor") {
        "your mentor is".into()
    } else if l.starts_with("who are you") || l.contains("who are you") {
        "you are".into()
    } else {
        q.to_string()
    }
}
fn concept_hint_from_norm(qn: &str) -> Option<&'static str> {
    if qn.starts_with("your name is") { Some("name") }
    else if qn.starts_with("the launch code is") { Some("launch code") }
    else { None }
}
fn ascii_norm(s: &str) -> String {
    s.replace('’', "'").replace('‘', "'")
        .replace('“', "\"").replace('”', "\"")
        .replace('—', "-").replace('–', "-")
        .trim()
        .to_string()
}
fn strip_quotes(s: &str) -> &str {
    s.trim().trim_matches('"').trim_matches('\'')
}
fn strip_topic_parens(s: &str) -> String {
    let s = s.trim();
    if let Some(i) = s.find('(') { return s[..i].trim().to_string(); }
    s.to_string()
}
fn canonical_topic(s: &str) -> String {
    let t = s.trim().to_ascii_lowercase();
    match t.as_str() {
        "c++" | "c plus plus" | "c-plus-plus" | "cpp" => "cpp".into(),
        "rust" | "rust lang" | "rustlang" => "rust".into(),
        "c#" | "c sharp" | "c-sharp" => "csharp".into(),
        "js" | "javascript" => "javascript".into(),
        "py" | "python" => "python".into(),
        _ => t,
    }
}
fn extract_definition(line: &str) -> Option<(String, String)> {
    let l = ascii_norm(line);

    // 1) Topic (Paren)? is Desc
    let re1 = Regex::new("(?i)^\\s*([A-Za-z0-9+./#\\-\\s'’]{1,64}?)(?:\\s*\\([^)]+\\))?\\s+is\\s+(.+?)\\s*$").ok()?;
    if let Some(c) = re1.captures(&l) {
        let topic_raw = c.get(1)?.as_str();
        let topic = strip_quotes(&strip_topic_parens(topic_raw)).to_string();
        let desc  = c.get(2)?.as_str().trim().to_string();
        if !topic.is_empty() && !desc.is_empty() { return Some((topic, desc)); }
    }

    // 2) Topic: Desc (or dash/em-dash)
    let re2 = Regex::new("(?i)^\\s*([A-Za-z0-9+./#\\-\\s'’]{1,64}?)\\s*[:\\-—]\\s+(.+?)\\s*$").ok()?;
    if let Some(c) = re2.captures(&l) {
        let topic = strip_quotes(c.get(1)?.as_str()).to_string();
        let desc  = c.get(2)?.as_str().trim().to_string();
        if !topic.is_empty() && !desc.is_empty() { return Some((topic, desc)); }
    }

    // 3) Desc is Topic (reversed)
    let re3 = Regex::new("(?i)^\\s*(.+?)\\s+is\\s+([A-Za-z0-9+./#\\-\\s'’]{1,64})\\s*\\.?\\s*$").ok()?;
    if let Some(c) = re3.captures(&l) {
        let desc  = c.get(1)?.as_str().trim().to_string();
        let topic = strip_quotes(c.get(2)?.as_str()).to_string();
        if !topic.is_empty() && !desc.is_empty() { return Some((topic, desc)); }
    }
    None
}
fn parse_definition_query(input: &str) -> Option<String> {
    let l = ascii_norm(input);
    const PS: [&str; 5] = [
        r"(?i)^\s*what\s+is\s+(.+?)\??\s*$",
        r"(?i)^\s*what'?s\s+(.+?)\??\s*$",
        r"(?i)^\s*define\s+(.+?)\s*$",
        r"(?i)^\s*explain\s+(.+?)\s*$",
        r"(?i)^\s*what\s+si\s+(.+?)\??\s*$",
    ];
    for p in PS {
        if let Some(c) = Regex::new(p).unwrap().captures(&l) {
            let raw = strip_quotes(c.get(1).unwrap().as_str()).trim();
            if !raw.is_empty() { return Some(raw.to_string()); }
        }
    }
    None
}
fn concept_of(text: &str) -> Option<&'static str> {
    let l = text.to_ascii_lowercase();
    if l.starts_with("the launch code is ") { Some("launch code") }
    else if l.starts_with("your name is ")  { Some("name") }
    else { None }
}
fn extract_answer(qn: &str, text: &str) -> Option<String> {
    let tl = text.trim();
    let ll = tl.to_ascii_lowercase();

    if qn.starts_with("your name is") {
        if let Some(idx) = ll.find("name is") {
            let ans = tl[idx + "name is".len()..].trim()
                .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace());
            if !ans.is_empty() { return Some(ans.to_string()); }
        }
        return None;
    }
    if qn.starts_with("the launch code is") {
        let re = Regex::new(r"(\d{3}[-\s]?\d{3}[-\s]?\d{3})").ok()?;
        if let Some(c) = re.captures(tl) { return Some(c.get(1).unwrap().as_str().to_string()); }
        return None;
    }
    if qn.starts_with("your creator is") {
        if let Some(idx) = ll.find("creator is") {
            let ans = tl[idx + "creator is".len()..].trim()
                .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace());
            if !ans.is_empty() { return Some(ans.to_string()); }
        }
        return None;
    }
    if qn.starts_with("your mentor is") {
        if let Some(idx) = ll.find("mentor is") {
            let ans = tl[idx + "mentor is".len()..].trim()
                .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace());
            if !ans.is_empty() { return Some(ans.to_string()); }
        }
        return None;
    }
    None
}
fn choose_best_for(qn: &str, hits: &[(u64, f32, u64, String)]) -> Option<(u64, f32, u64, String)> {
    let key = if qn.starts_with("your name is") { Some("name") }
    else if qn.starts_with("the launch code is") { Some("launch code") }
    else { None };

    if let Some(k) = key {
        let mut filtered: Vec<_> = hits.iter()
            .filter(|(_,_,_,text)| text.to_ascii_lowercase().contains(k))
            .cloned().collect();
        if !filtered.is_empty() {
            filtered.sort_by(|a,b| b.0.cmp(&a.0).then_with(|| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)));
            return filtered.into_iter().next();
        }
        let mut all = hits.to_vec();
        all.sort_by(|a,b| b.0.cmp(&a.0).then_with(|| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)));
        return all.into_iter().next();
    }
    let mut all = hits.to_vec();
    all.sort_by(|a,b| b.0.cmp(&a.0).then_with(|| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)));
    all.into_iter().next()
}
fn normalize_for_hash(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_ws = false;
    for ch in s.chars().map(|c| c.to_ascii_lowercase()) {
        if ch.is_whitespace() { if !last_ws { out.push(' '); } last_ws = true; }
        else { last_ws = false; out.push(ch); }
    }
    out.trim().to_string()
}
fn sha256_hex(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    let bytes = h.finalize();
    let mut hex = String::with_capacity(64);
    for b in bytes { use std::fmt::Write; let _ = write!(hex, "{:02x}", b); }
    hex
}
#[inline]
fn belief_implies_copula_long(text: &str) -> Option<bool> {
    let l = text.to_ascii_lowercase();
    if l.contains("say \"i am\"") || l.contains("say 'i am'") || l.contains("use i am")
        || l.contains("no contractions") || l.contains("use long copula")
    { return Some(true); }
    if l.contains("say \"i'm\"") || l.contains("say 'i'm'") || l.contains("use i'm")
        || l.contains("use contractions") || l.contains("use short copula")
    { return Some(false); }
    None
}
fn detect_beliefish(text: &str) -> Option<String> {
    let l = text.trim();
    let re_you_are = Regex::new(r"(?i)^\s*you\s+(?:are|'re)\s+(.+?)\s*[.!]?\s*$").unwrap();
    if let Some(c) = re_you_are.captures(l) {
        let val = c[1].trim();
        if !val.is_empty() { return Some(format!("you are {}", val)); }
    }
    let re_purpose_is = Regex::new(r"(?i)^\s*your\s+(?:purpose|mission|role)\s+(?:is|to)\s+(.+?)\s*[.!]?\s*$").unwrap();
    if let Some(c) = re_purpose_is.captures(l) {
        let val = c[1].trim();
        if !val.is_empty() { return Some(format!("your purpose is {}", val)); }
    }
    let re_created_to = Regex::new(r"(?i)^\s*you\s+(?:were\s+)?created\s+(?:to|for)\s+(.+?)\s*[.!]?\s*$").unwrap();
    if let Some(c) = re_created_to.captures(l) {
        let val = c[1].trim();
        if !val.is_empty() { return Some(format!("you were created to {}", val)); }
    }
    None
}
fn extract_copula_name_for_agent(text: &str) -> Option<String> {
    let role_words = ["creator", "mentor", "owner", "friend", "assistant"];
    let l = text.to_ascii_lowercase();

    if let Some(c) = Regex::new(r"(?i)^\s*your\s+name\s+is\s+(.+?)\s*[.!]?\s*$").unwrap().captures(text) {
        let cand = c[1].trim().to_string();
        if !cand.is_empty() { return Some(cand); }
    }
    if let Some(c) = Regex::new(r"(?i)^\s*you\s+(?:are|\'re)\s+(.+?)\s*[.!]?\s*$").unwrap().captures(text) {
        let cand = c[1].trim().to_string();
        if !cand.is_empty() && !role_words.iter().any(|w| l.contains(w)) { return Some(cand); }
    }
    None
}
fn now_sec() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}
#[inline]
fn fmt_tags(t: u64) -> String {
    let mut v = Vec::new();
    if t & tags::USER   != 0 { v.push("USER"); }
    if t & tags::SYSTEM != 0 { v.push("SYSTEM"); }
    if t & tags::SHORT  != 0 { v.push("SHORT"); }
    if t & tags::LONG   != 0 { v.push("LONG"); }
    if v.is_empty() { "NONE".into() } else { v.join("|") }
}

// ───────────────────────── Embedders ─────────────────────────
type Bx = NdArray<f32>;

struct SmieHippEmbed<B: Backend> { model: Arc<Mutex<HippocampusModel<B>>>, dim: usize }
impl<B: Backend> SmieHippEmbed<B> {
    fn new(model: Arc<Mutex<HippocampusModel<B>>>) -> anyhow::Result<Self> {
        let m = model.lock().map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
        let z = m.run_inference("[dim-probe]");
        let dim = z.to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?.len();
        drop(m);
        Ok(Self { model, dim })
    }
}
impl<B: Backend> Embedder for SmieHippEmbed<B> {
    fn dim(&self) -> usize { self.dim }
    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let m = self.model.lock().map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
        let z = m.run_inference(text);
        let mut v = z.to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?.to_vec();
        drop(m);
        let n = (v.iter().map(|x| x*x).sum::<f32>()).sqrt().max(1e-6);
        for x in &mut v { *x /= n; }
        Ok(v)
    }
}

// ───────────────────────── Unified persist for bulk (mirrors runtime) ─────────────────────────
#[allow(clippy::too_many_arguments)]
fn auto_persist_line_bulk<B: Backend>(
    conn: &rusqlite::Connection,
    model: &HippocampusModel<B>,
    smie_tx: Option<&Sender<(String, u64)>>, // <-- send (text, tag_bits)
    smie_tbits: u64,
    ctx: &str,
    text_raw: &str,
    concepts: &Registry,
    seen_hashes: &Arc<Mutex<HashSet<String>>>,
) -> anyhow::Result<bool> {
    let clean = sanitize_input(text_raw);
    if clean.is_empty() { return Ok(false); }

    // in-process dedupe
    let n = normalize_for_hash(&clean);
    let key = sha256_hex(&n);
    {
        let mut guard = seen_hashes.lock().unwrap();
        if !guard.insert(key.clone()) {
            return Ok(false);
        }
    }

    let ts = now_sec();

    // durable dedupe
    if has_hash(ctx, &key).map_err(to_any)? { return Ok(false); }
    let _ = record_hash(ctx, &key, ts).map_err(to_any)?;

    // style flip
    if !is_question(&clean) {
        if let Some(long) = belief_implies_copula_long(&clean) {
            speech_concepts::set_copula_style_long(long);
            let _ = upsert_concept_fact("speech.copula", if long { "long" } else { "short" }, ts);
        }
    }

    // definitions
    if let Some((topic, desc)) = extract_definition(&clean) {
        let k = format!("def:{}", canonical_topic(&topic));
        let _ = upsert_concept_fact(&k, &desc, ts);
        if let Some(tx) = smie_tx {
            let _ = tx.send((format!("{topic}: {desc}"), smie_tbits | tags::SYSTEM | tags::SHORT));
        }
    }

    // beliefs
    if !is_question(&clean) {
        if let Some(mut belief) = detect_beliefish(&clean) {
            belief = belief.trim().trim_end_matches(|c: char| c.is_ascii_punctuation()).to_string();
            if !belief.is_empty() {
                let _ = insert_belief(conn, &belieffix(belief.clone()), "user", 1.0);
                if let Some(long) = belief_implies_copula_long(&belief) {
                    speech_concepts::set_copula_style_long(long);
                }
            }
        }
    }

    // concepts
    let hits = concepts.extract_all_from_statement(&clean);
    if !hits.is_empty() {
        for h in &hits { let _ = upsert_concept_fact(h.key, &h.value, ts); }
    } else if let Some(name) = extract_copula_name_for_agent(&clean) {
        let _ = upsert_concept_fact("name", &name, ts);
        let _ = upsert_concept_fact("you are", &name, ts);
    } else if let Some(caps) = Regex::new(r"(?i)^\s*i\s+am\s+your\s+([a-z\s]+?),\s*([A-Za-z][A-Za-z .'\-]+)\s*[.!]?\s*$")?
        .captures(&clean) {
        let roles = caps[1].to_ascii_lowercase();
        let name  = caps[2].trim().to_string();
        if roles.contains("creator") { let _ = upsert_concept_fact("creator", &name, ts); }
        if roles.contains("mentor")  { let _ = upsert_concept_fact("mentor",  &name, ts); }
    }

    // legacy strict
    if let Some(con) = concept_of(&clean) {
        let qn = if con == "name" { "your name is" } else { "the launch code is" };
        if let Some(value) = extract_answer(qn, &clean) {
            let _ = upsert_concept_fact(con, &value, ts);
        }
    }

    // durability ingest
    pipeline::ingest(model, ctx, &clean, 7 * 24 * 3600, "ui")
        .map_err(|e| anyhow::anyhow!("ingest failed: {e}"))?;

    // SMIE enqueue (handled on UI thread)
    if let Some(tx) = smie_tx {
        let _ = tx.send((clean, smie_tbits));
    }

    Ok(true)
}

#[inline]
fn belieffix(s: String) -> String { s } // placeholder if you later normalize

#[inline]
fn to_any<E: std::fmt::Display + Send + Sync + 'static>(e: E) -> anyhow::Error {
    anyhow::anyhow!(e.to_string())
}

// ───────────────────────── App state ─────────────────────────
struct BulkParams {
    file_path: String,
    ctx: String,
    threads: usize,
    also_index_in_smie: bool,
}
struct BulkState {
    running: bool,
    start: Instant,
    processed: Arc<AtomicUsize>,
    total_enqueued: Arc<AtomicUsize>,
    errors: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
}
struct AppState {
    // core
    smie: Arc<Smie>,
    model: Arc<Mutex<HippocampusModel<Bx>>>,

    // concept registry
    concepts: Registry,

    // single-line ingest + recall
    input_text: String,
    query_text: String,
    top_k: usize,
    set_user: bool,
    set_system: bool,
    set_short: bool,
    set_long: bool,

    results: Vec<(u64, f32, u64, String)>,
    status: String,

    // SMIE queue for UI-thread ingestion
    smie_q_tx: Sender<(String, u64)>,
    smie_q_rx: Receiver<(String, u64)>,

    // bulk ingest
    bulk: Option<BulkState>,
    bulk_params: BulkParams,
    bulk_seen: Arc<Mutex<HashSet<String>>>,
}

impl AppState {
    fn new(smie: Arc<Smie>, model: Arc<Mutex<HippocampusModel<Bx>>>) -> Self {
        let concepts = Registry::new()
            .register(Box::new(NameConcept))
            .register(Box::new(LaunchCodeConcept))
            .register(Box::new(CreatorConcept))
            .register(Box::new(MentorConcept))
            .register(Box::new(PurposeConcept))
            .register(Box::new(IdentityConcept));

        let (tx, rx) = unbounded::<(String, u64)>();
        let threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

        Self {
            smie,
            model,
            concepts,

            input_text: String::new(),
            query_text: String::new(),
            top_k: 8,
            set_user: true,
            set_system: false,
            set_short: true,
            set_long: false,

            results: Vec::new(),
            status: String::new(),

            smie_q_tx: tx,
            smie_q_rx: rx,

            bulk: None,
            bulk_params: BulkParams {
                file_path: "data/seed_out.txt".into(),
                ctx: "default".into(),
                threads: threads.max(2),
                also_index_in_smie: false,
            },
            bulk_seen: Arc::new(Mutex::new(HashSet::with_capacity(1 << 16))),
        }
    }

    fn start_bulk_ingest(&mut self) {
        if self.bulk.as_ref().map(|b| b.running).unwrap_or(false) {
            self.status = "Bulk ingest already running".into();
            return;
        }
        let path = self.bulk_params.file_path.clone();
        let ctx_name = self.bulk_params.ctx.clone();
        let n_threads = self.bulk_params.threads.max(1);
        let also_smie = self.bulk_params.also_index_in_smie;

        // progress
        let processed = Arc::new(AtomicUsize::new(0));
        let total_enq = Arc::new(AtomicUsize::new(0));
        let errors = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));

        // queue
        let (line_tx, line_rx) = bounded::<String>(20_000);

        // SMIE enqueue options
        let smie_tx_opt = if also_smie { Some(self.smie_q_tx.clone()) } else { None };
        let smie_tbits = if also_smie { tags::USER | tags::SHORT } else { 0 };

        // Producer
        {
            let total_enq_c = total_enq.clone();
            let stop_c = stop.clone();
            let path_c = path.clone();
            thread::spawn(move || {
                let file = match File::open(&path_c) {
                    Ok(f) => f,
                    Err(_) => return,
                };
                let reader = BufReader::new(file);
                for line in reader.lines().flatten() {
                    if stop_c.load(AO::Relaxed) { break; }
                    let s = line.trim().to_string();
                    if s.is_empty() { continue; }
                    if line_tx.send(s).is_err() { break; }
                    total_enq_c.fetch_add(1, AO::Relaxed);
                }
            });
        }

        // Workers
        for _ in 0..n_threads {
            let rx = line_rx.clone();
            let stop_w = stop.clone();
            let processed_w = processed.clone();
            let errors_w = errors.clone();
            let seen = self.bulk_seen.clone();
            let ctx_local = ctx_name.clone();
            let model_arc = self.model.clone();
            let smie_tx = smie_tx_opt.clone();
            let reg = make_registry();

            thread::spawn(move || {
                let conn = match rusqlite::Connection::open("data/semantic_memory.db") {
                    Ok(c) => c,
                    Err(_) => return,
                };
                let _ = conn.execute_batch(
                    r#"
                    PRAGMA journal_mode=WAL;
                    PRAGMA synchronous=NORMAL;
                    PRAGMA temp_store=MEMORY;
                    PRAGMA cache_size=-200000;
                    PRAGMA page_size=32768;
                    PRAGMA mmap_size=30000000000;
                    PRAGMA busy_timeout=5000;
                "#,
                );

                while !stop_w.load(AO::Relaxed) {
                    let line = match rx.recv() {
                        Ok(s) => s,
                        Err(_) => break,
                    };
                    let ok = (|| -> anyhow::Result<bool> {
                        let model = match model_arc.lock() {
                            Ok(m) => m,
                            Err(_) => return Ok(false),
                        };
                        auto_persist_line_bulk(
                            &conn, &*model,
                            smie_tx.as_ref(), smie_tbits,
                            &ctx_local, &line, &reg, &seen,
                        )
                    })();

                    match ok {
                        Ok(true) => { processed_w.fetch_add(1, AO::Relaxed); }
                        Ok(false) => { /* duplicate/skip */ }
                        Err(_) => { errors_w.fetch_add(1, AO::Relaxed); }
                    }
                }
            });
        }

        self.bulk = Some(BulkState {
            running: true,
            start: Instant::now(),
            processed,
            total_enqueued: total_enq,
            errors,
            stop,
        });
        self.status = format!("⚡ Bulk ingest started on '{}' with {} worker(s).", path, n_threads);
    }

    fn stop_bulk(&mut self) {
        if let Some(b) = &self.bulk {
            b.stop.store(true, AO::Relaxed);
        }
    }
}

fn make_registry() -> Registry {
    Registry::new()
        .register(Box::new(NameConcept))
        .register(Box::new(LaunchCodeConcept))
        .register(Box::new(CreatorConcept))
        .register(Box::new(MentorConcept))
        .register(Box::new(PurposeConcept))
        .register(Box::new(IdentityConcept))
}

// ───────────────────────── UI app ─────────────────────────
struct UiApp {
    state: AppState,
}
impl UiApp {
    fn new() -> Self {
        type B = NdArray<f32>;
        let device = <B as Backend>::Device::default();
        let model = Arc::new(Mutex::new(HippocampusModel::<B>::load_or_init(
            &device,
            "R:/uriel/models/checkpoints/uriel_encoder_step008500.bin",
        )));

        // Optional: keep this (now compiles thanks to `Recorder` in scope)
        // let _ = CompactRecorder::new()
        //     .load("data/runs/router_v1/routercopy_step005000.bin".into(), &device);

        let smie_embedder = Arc::new(SmieHippEmbed::new(model.clone()).expect("smie embed init"));
        let smie = Arc::new(
            Smie::new(SmieConfig { dim: smie_embedder.dim(), metric: Metric::Cosine }, smie_embedder)
                .expect("smie init"),
        );
        std::fs::create_dir_all("data").ok();
        let _ = smie.load("data/uriel");

        UiApp { state: AppState::new(smie, model) }
    }
}
impl eframe::App for UiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // drain the SMIE queue on UI thread
        for _ in 0..2000 {
            if let Ok((text, tbits)) = self.state.smie_q_rx.try_recv() {
                let _ = self.state.smie.ingest(&text, if tbits == 0 { tags::USER | tags::SHORT } else { tbits });
            } else { break; }
        }

        // repaint while bulk running
        if self.state.bulk.as_ref().map(|b| b.running).unwrap_or(false) {
            ctx.request_repaint_after(Duration::from_millis(50));
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("URIEL UI — Memory + SMIE");
            ui.separator();

            // single-line ingest
            ui.label("Ingest (single line):");
            ui.text_edit_singleline(&mut self.state.input_text);
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.state.set_user, "USER");
                ui.checkbox(&mut self.state.set_system, "SYSTEM");
                ui.checkbox(&mut self.state.set_short, "SHORT");
                ui.checkbox(&mut self.state.set_long, "LONG");
                if ui.button("📥 Ingest").clicked() {
                    let text = sanitize_input(&self.state.input_text);
                    if text.is_empty() {
                        self.state.status = "⚠️ Nothing to ingest".into();
                    } else {
                        if let Ok(model) = self.state.model.lock() {
                            let _ = pipeline::ingest(&*model, "ui_context", &text, 30, "gui");
                        }
                        // SMIE
                        let mut tbits = 0u64;
                        if self.state.set_user   { tbits |= tags::USER; }
                        if self.state.set_system { tbits |= tags::SYSTEM; }
                        if self.state.set_short  { tbits |= tags::SHORT; }
                        if self.state.set_long   { tbits |= tags::LONG; }
                        match self.state.smie.ingest(&text, tbits) {
                            Ok(id) => { self.state.status = format!("✅ Ingested id {id} [{}]", fmt_tags(tbits)); self.state.input_text.clear(); }
                            Err(e) => { self.state.status = format!("❌ SMIE ingest failed: {e}"); }
                        }
                    }
                }
            });

            ui.separator();

            // recall
            ui.label("Recall:");
            ui.text_edit_singleline(&mut self.state.query_text);
            ui.horizontal(|ui| {
                ui.label("Top K:");
                ui.add(egui::Slider::new(&mut self.state.top_k, 1..=50));
                if ui.button("🔎 Recall").clicked() {
                    let q = sanitize_input(&self.state.query_text);
                    self.state.results.clear();
                    self.state.status.clear();
                    if q.is_empty() {
                        self.state.status = "⚠️ Empty query".into();
                    } else {
                        let qn = normalize_question(&q);
                        let mut probes: Vec<String> = vec![qn.clone()];
                        if let Some(key) = concept_hint_from_norm(&qn) { probes.push(key.into()); }
                        probes.push(q.clone());

                        let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
                        let mut answered: Option<String> = None;

                        'outer: for probe in probes {
                            if let Ok(hits) = self.state.smie.recall(&probe, self.state.top_k.max(8), mask) {
                                if !hits.is_empty() {
                                    self.state.results = hits.clone();
                                    if let Some(best) = choose_best_for(&qn, &hits) {
                                        answered = extract_answer(&qn, &best.3).or(Some(best.3));
                                        break 'outer;
                                    }
                                }
                            }
                        }
                        if answered.is_none() && is_question(&q) {
                            if let Ok(rows) = recall_latest("ui_context", 128) {
                                for r in rows {
                                    if let Some(ans) = extract_answer(&qn, &r.data) { answered = Some(ans); break; }
                                }
                            }
                        }
                        self.state.status = answered.unwrap_or_else(|| "(no answer)".into());
                    }
                }
            });

            ui.separator();
            ui.label("Status / Answer:");
            ui.monospace(&self.state.status);

            ui.separator();
            ui.label("Top Hits:");
            if self.state.results.is_empty() {
                ui.label("—");
            } else {
                for (i,(id,score,t,text)) in self.state.results.iter().enumerate() {
                    ui.monospace(format!("{}. id={}  score={:.3}  [{}]\n   {}", i+1, id, score, fmt_tags(*t), text));
                    ui.add_space(6.0);
                }
            }

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("💾 Save SMIE").clicked() {
                    std::fs::create_dir_all("data").ok();
                    match self.state.smie.save("data/uriel") {
                        Ok(_) => self.state.status = "✅ Saved SMIE metadata".into(),
                        Err(e) => self.state.status = format!("❌ Save failed: {e}"),
                    }
                }
                if ui.button("📂 Load SMIE").clicked() {
                    match self.state.smie.load("data/uriel") {
                        Ok(_) => self.state.status = "✅ Loaded SMIE metadata".into(),
                        Err(e) => self.state.status = format!("❌ Load failed: {e}"),
                    }
                }
            });

            ui.separator();
            // BULK
            ui.heading("⚡ Bulk Ingest");
            ui.label("File path:");
            ui.text_edit_singleline(&mut self.state.bulk_params.file_path);
            ui.label("Context name:");
            ui.text_edit_singleline(&mut self.state.bulk_params.ctx);
            ui.horizontal(|ui| {
                ui.label("Threads:");
                ui.add(egui::Slider::new(&mut self.state.bulk_params.threads, 1..=96));
                ui.checkbox(&mut self.state.bulk_params.also_index_in_smie, "Also index in SMIE (slower)");
            });
            ui.horizontal(|ui| {
                if self.state.bulk.as_ref().map(|b| b.running).unwrap_or(false) {
                    if ui.button("⏹ Cancel").clicked() {
                        self.state.stop_bulk();
                    }
                } else {
                    if ui.button("⚡ Start Bulk Ingest").clicked() {
                        self.state.bulk_seen.lock().unwrap().clear();
                        self.state.start_bulk_ingest();
                    }
                }
            });

            if let Some(b) = &self.state.bulk {
                let enq = b.total_enqueued.load(AO::Relaxed) as f32;
                let done = b.processed.load(AO::Relaxed) as f32;
                let errs = b.errors.load(AO::Relaxed);
                let prog = if enq <= 0.0 { 0.0 } else { (done / enq).min(1.0) };

                ui.add_space(8.0);
                ui.label(format!(
                    "Queued: {}   Ingested: {}   Errors: {}   Elapsed: {:.1}s",
                    enq as usize, done as usize, errs, b.start.elapsed().as_secs_f32()
                ));
                ui.add(egui::ProgressBar::new(prog).show_percentage());

                if !b.running && enq > 0.0 {
                    ui.label("Bulk ingest finished.");
                }
            }
        });

        // finish bulk if everything processed
        if let Some(b) = &mut self.state.bulk {
            let enq = b.total_enqueued.load(AO::Relaxed);
            let done = b.processed.load(AO::Relaxed);
            let errs = b.errors.load(AO::Relaxed);
            if b.stop.load(AO::Relaxed) || (enq > 0 && enq == done + errs) {
                b.running = false;
            }
        }
    }
}

// ───────────────────────── Entry ─────────────────────────
pub fn main() -> Result<(), eframe::Error> {
    // honor persisted copula style
    if let Ok(Some(style)) = read_concept_fact("speech.copula").map_err(to_any) {
        speech_concepts::set_copula_style_long(style.eq_ignore_ascii_case("long"));
    }

    let app = UiApp::new();
    let opts = eframe::NativeOptions::default();
    eframe::run_native("URIEL UI", opts, Box::new(|_cc| Box::new(app)))
}

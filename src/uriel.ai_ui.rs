// src/bin/uriel_gui.rs
#![allow(clippy::too_many_arguments)]
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering as AO},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use anyhow::Result;
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use crossbeam_channel::bounded;
use eframe::egui::{self, Color32, FontData, FontDefinitions, FontFamily, FontId, RichText, TextStyle};
use eframe::{egui::widgets::ProgressBar, App, Frame, NativeOptions};
use regex::Regex;
use rusqlite;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use sha2::{Digest, Sha256};
use std::thread;

use URIELV1::Hippocampus::infer::HippocampusModel;
use URIELV1::Hippocampus::pipeline;
use URIELV1::Hippocampus::sqlite_backend::{
    has_hash, insert_belief, read_concept_fact, recall_latest, record_hash, upsert_concept_fact,
};
use URIELV1::PFC::context_retriever::{ContextRetriever, Embedding, Embeddings, RetrieverCfg, VectorStore};
use URIELV1::PFC::cortex::{execute_cortex, CortexInputFrame, CortexOutput};
use URIELV1::parietal::parietal::SalienceScore;
use URIELV1::concepts_and_contexts::speech_concepts;
use URIELV1::Learning::routercopy::trainer::{RouterCopySystem, RouterCopySystemRecord};

// ========================= Small helpers (aligned with runtime) =========================
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
    let re1 = Regex::new("(?i)^\\s*([A-Za-z0-9+./#\\-\\s'’]{1,64}?)(?:\\s*\\([^)]+\\))?\\s+is\\s+(.+?)\\s*$").ok()?;
    if let Some(c) = re1.captures(&l) {
        let topic_raw = c.get(1)?.as_str();
        let topic = strip_quotes(&strip_topic_parens(topic_raw)).to_string();
        let desc  = c.get(2)?.as_str().trim().to_string();
        if !topic.is_empty() && !desc.is_empty() { return Some((topic, desc)); }
    }
    let re2 = Regex::new("(?i)^\\s*([A-Za-z0-9+./#\\-\\s'’]{1,64}?)\\s*[:\\-—]\\s+(.+?)\\s*$").ok()?;
    if let Some(c) = re2.captures(&l) {
        let topic = strip_quotes(c.get(1)?.as_str()).to_string();
        let desc  = c.get(2)?.as_str().trim().to_string();
        if !topic.is_empty() && !desc.is_empty() { return Some((topic, desc)); }
    }
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
        if !cand.is_empty() && !role_words.iter().any(|w| l.contains(*w)) { return Some(cand); }
    }
    None
}
fn now_sec() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
fn to_any<E: std::fmt::Display + Send + Sync + 'static>(e: E) -> anyhow::Error {
    anyhow::anyhow!(e.to_string())
}

// ========================= Router calibration + encoding (unchanged) =========================
#[derive(Deserialize, Clone, Copy, Default)]
struct Thresholds { promote: f32, refuse: f32 }
#[derive(Deserialize)]
struct OpCalib { #[serde(rename = "T")] t: f32, ece: f32, brier: f32 }
#[derive(Deserialize)]
struct CalibFile { op: OpCalib, #[serde(default)] thresholds: Thresholds }

fn load_routercopy_calib(path: &str) -> Option<(f32, Thresholds)> {
    fs::read_to_string(path).ok().and_then(|s| {
        serde_json::from_str::<CalibFile>(&s).ok().map(|c| {
            let thr = if c.thresholds.promote > 0.0 { c.thresholds } else { Thresholds { promote: 0.60, refuse: 0.40 } };
            (c.op.t, thr)
        })
    })
}
fn rc_encode_ascii95(s: &str, max_t: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(max_t);
    for ch in s.chars().take(max_t) {
        let idx = (ch as i32 - 32).clamp(0, 94) as usize;
        out.push(idx);
    }
    while out.len() < max_t { out.push(0); }
    out
}
fn rc_tokens_to_one_hot<B: Backend>(tokens: &[Vec<usize>], v: usize, dev: &B::Device) -> burn::tensor::Tensor<B, 3> {
    use burn::tensor::Tensor;
    let b = tokens.len();
    let t = tokens.first().map(|x| x.len()).unwrap_or(0);
    let mut data = vec![0.0f32; b * t * v];
    for (bi, row) in tokens.iter().enumerate() {
        for (ti, &idx) in row.iter().enumerate() {
            if idx < v {
                data[bi * t * v + ti * v + idx] = 1.0;
            }
        }
    }
    Tensor::<B, 1>::from_floats(data.as_slice(), dev).reshape([b, t, v])
}
fn rc_embed_seq<B: Backend>(oh: burn::tensor::Tensor<B,3>, emb: &URIELV1::embedder::embedder::FixedWeightEmbedder<B>) -> burn::tensor::Tensor<B,3> {
    let [b,t,v] = { let d = oh.dims(); [d[0],d[1],d[2]] };
    let flat = oh.reshape([b*t, v]);
    let proj = emb.forward(flat);
    proj.reshape([b,t,384])
}
fn rc_mean_over_time<B: Backend>(seq: burn::tensor::Tensor<B,3>) -> burn::tensor::Tensor<B,2> {
    seq.mean_dim(1).squeeze(1)
}

// ========================= PFC retriever stubs (no ANN) =========================
struct NullVS;
impl VectorStore for NullVS {
    fn search(&self, _q: &Embedding, _k: usize) -> std::result::Result<Vec<URIELV1::PFC::context_retriever::MemoryHit>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(vec![])
    }
}
struct ModelEmb<B: Backend> { model: Arc<Mutex<HippocampusModel<B>>> }
impl<B: Backend> ModelEmb<B> { fn new(model: Arc<Mutex<HippocampusModel<B>>>) -> Self { Self { model } } }
impl<B: Backend> Embeddings for ModelEmb<B> {
    fn embed(&self, text: &str) -> std::result::Result<Embedding, Box<dyn std::error::Error + Send + Sync>> {
        let m = self.model.lock().map_err(|_| io::Error::new(io::ErrorKind::Other, "model lock poisoned"))?;
        let z = m.run_inference(text);
        let mut v = z.to_data().as_slice::<f32>()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{e:?}")))?
            .to_vec();
        let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
        for x in &mut v { *x /= n; }
        Ok(v)
    }
}

// ========================= Auto-persist (DB durability; no SMIE) =========================
fn auto_persist_line_quick<B: Backend>(
    conn: &rusqlite::Connection,
    model: &HippocampusModel<B>,
    ctx: &str,
    text_raw: &str,
    seen_hashes: &Arc<Mutex<HashSet<String>>>,
) -> anyhow::Result<bool> {
    let clean = sanitize_input(text_raw);
    if clean.is_empty() { return Ok(false); }

    // in-process dedupe
    let n = normalize_for_hash(&clean);
    let key = sha256_hex(&n);
    {
        let mut guard = seen_hashes.lock().unwrap();
        if !guard.insert(key.clone()) { return Ok(false); }
    }

    let ts = now_sec();

    // durable dedupe
    if has_hash(ctx, &key).map_err(to_any)? { return Ok(false); }
    let _ = record_hash(ctx, &key, ts).map_err(to_any)?;

    // style
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
    }

    // beliefs
    if !is_question(&clean) {
        if let Some(mut belief) = detect_beliefish(&clean) {
            belief = belief.trim().trim_end_matches(|c: char| c.is_ascii_punctuation()).to_string();
            if !belief.is_empty() { let _ = insert_belief(conn, &belief, "user", 1.0); }
        }
    }

    // concepts (light)
    if let Some(name) = extract_copula_name_for_agent(&clean) {
        let _ = upsert_concept_fact("name", &name, ts);
        let _ = upsert_concept_fact("you are", &name, ts);
    } else if let Some(caps) = Regex::new(r"(?i)^\s*i\s+am\s+your\s+([a-z\s]+?),\s*([A-Za-z][A-Za-z .'\-]+)\s*[.!]?\s*$")?.captures(&clean) {
        let roles = caps[1].to_ascii_lowercase();
        let name  = caps[2].trim().to_string();
        if roles.contains("creator") { let _ = upsert_concept_fact("creator", &name, ts); }
        if roles.contains("mentor")  { let _ = upsert_concept_fact("mentor",  &name, ts); }
    }
    if let Some(con) = concept_of(&clean) {
        let qn = if con == "name" { "your name is" } else { "the launch code is" };
        if let Some(value) = extract_answer(qn, &clean) { let _ = upsert_concept_fact(con, &value, ts); }
    }

    // durable ingest (embedding + rows)
    pipeline::ingest(model, ctx, &clean, 7 * 24 * 3600, "gui")
        .map_err(|e| anyhow::anyhow!("ingest failed: {e}"))?;

    Ok(true)
}

// ========================= NEW: Trace (v2) model types & helpers =========================
#[derive(Serialize, Deserialize, Clone)]
enum MetaPhase {
    Sense, RetrieveEvidence, Hypothesize, Plan, Act, Observe, Critique,
    AskPermission, Learn, BeliefUpdate, Report
}

#[derive(Serialize, Deserialize, Clone, Default)]
struct EvidenceQuality { trust_w: f32, recency_w: f32, method_w: f32 }

#[derive(Serialize, Deserialize, Clone)]
struct EvidenceV2 {
    id: String,
    kind: String,       // "peerreview" | "preprint" | "doc" | "code" | "log"
    source_id: String,  // vector id or local id
    url: Option<String>,
    sha256: Option<String>,
    fetched_at: i64,
    excerpt: String,
    #[serde(default)]
    quality: EvidenceQuality,
}

#[derive(Serialize, Deserialize, Clone)]
struct MetaEventV2 {
    ts: i64,
    run_id: String,
    task_id: String,  // your turn_id
    domain: String,
    phase: MetaPhase,
    policy: Option<String>,
    arm: Option<String>,
    expected_value: Option<f64>,
    observed_value: Option<f64>,
    confidence: Option<(f32, f32, f32)>,
    cost: Option<(u32, u32, f32, f32)>,
    inputs_ref: Option<String>,
    outputs_ref: Option<String>,
    evidence: Vec<EvidenceV2>,
    belief_update: Option<Json>,
    notes: Option<String>,
}

// metacog loop JSONL
#[derive(Serialize, Deserialize, Clone)]
struct MetacogTrace {
    ts: i64,
    fact_key: String,
    old_value: String,
    prior_cert: f32,
    action: String,
    new_value: Option<String>,
    new_cert: Option<f32>,
    bandit: Option<TraceBandit>,
    notes: Option<String>,
}
#[derive(Serialize, Deserialize, Clone)]
struct TraceBandit { arm: String, reward: f32 }

// UI projection
#[derive(Clone)]
struct CogTurn {
    task_id: String,
    first_ts: i64,
    input_text: Option<String>,
    output_text: Option<String>,
    confidence: Option<(f32, f32, f32)>,
    events: Vec<MetaEventV2>, // sorted by ts
}

fn human_age(ts: i64) -> String {
    let now = now_sec() as i64;
    let d = (now - ts).max(0);
    if d < 60 { format!("{d}s") }
    else if d < 3600 { format!("{}m", d / 60) }
    else if d < 86_400 { format!("{}h", d / 3600) }
    else { format!("{}d", d / 86_400) }
}
fn recency_weight_from_ts(ts: i64, half_life_days: f32) -> f32 {
    let now = now_sec() as i64;
    let age_days = ((now - ts).max(0) as f32) / 86_400.0;
    (-age_days / half_life_days).exp() // 1.0 fresh, decays with half-life
}
fn text_entropy_s(s: &str) -> f32 {
    // very light Shannon entropy over bytes, normalized ~[0,1.0] for UI
    if s.is_empty() { return 0.0; }
    let mut freq = [0u32; 256];
    for b in s.as_bytes() { freq[*b as usize] += 1; }
    let n = s.len() as f32;
    let mut h = 0.0f32;
    for &c in &freq {
        if c == 0 { continue; }
        let p = (c as f32) / n;
        h += -p * p.log2();
    }
    (h / 8.0).clamp(0.0, 1.0) // 8-bit normalization
}
fn tail_lines(path: &str, max_lines: usize) -> io::Result<Vec<String>> {
    // simple, safe tail (for typical file sizes)
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut buf = Vec::new();
    for line in reader.lines().flatten() {
        buf.push(line);
        if buf.len() > max_lines { let _ = buf.remove(0); }
    }
    Ok(buf)
}

fn load_meta_events(path: &str, max_lines: usize) -> Vec<MetaEventV2> {
    match tail_lines(path, max_lines) {
        Ok(lines) => lines.into_iter().filter_map(|l| serde_json::from_str::<MetaEventV2>(&l).ok()).collect(),
        Err(_) => vec![]
    }
}
fn load_metacog(path: &str, max_lines: usize) -> Vec<MetacogTrace> {
    match tail_lines(path, max_lines) {
        Ok(lines) => lines.into_iter().filter_map(|l| serde_json::from_str::<MetacogTrace>(&l).ok()).collect(),
        Err(_) => vec![]
    }
}
fn group_turns(evts: Vec<MetaEventV2>) -> Vec<CogTurn> {
    let mut map: BTreeMap<String, Vec<MetaEventV2>> = BTreeMap::new();
    for e in evts { map.entry(e.task_id.clone()).or_default().push(e); }
    let mut out = Vec::with_capacity(map.len());
    for (task_id, mut v) in map {
        v.sort_by_key(|e| e.ts);
        let first_ts = v.first().map(|e| e.ts).unwrap_or(0);
        let input_text = v.iter().find(|e| matches!(e.phase, MetaPhase::Sense)).and_then(|e| e.inputs_ref.clone());
        let output_text = v.iter().rev().find(|e| matches!(e.phase, MetaPhase::Report)).and_then(|e| e.outputs_ref.clone());
        let conf = v.iter().rev().find(|e| matches!(e.phase, MetaPhase::Report)).and_then(|e| e.confidence.clone());
        out.push(CogTurn { task_id, first_ts, input_text, output_text, confidence: conf, events: v });
    }
    // newest first for the list
    out.sort_by(|a,b| b.first_ts.cmp(&a.first_ts));
    out
}

// ========================= The GUI =========================
type Bx = NdArray<f32>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab { Cognitive, Chat, Bulk, Settings }

struct BulkParams {
    file_path: String,
    ctx: String,
    threads: usize,
}
struct BulkState {
    running: bool,
    start: Instant,
    processed: Arc<AtomicUsize>,
    total_enqueued: Arc<AtomicUsize>,
    errors: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
}
struct RouterUi {
    sys: Option<RouterCopySystem<Bx>>,
    route_T: f32,
    thr: Thresholds,
    vocab: usize,
    arg_k: usize,
    p: [f32; 4], // Respond, Retrieve, Store, Defer
    pred: usize,
    conf: f32,
}
struct FontCfg {
    path: String,
    size_logo: f32,
    size_body: f32,
    applied: bool,
}

// ------------- NEW: Cognitive Window state -------------
struct CogState {
    meta_path: String,
    metacog_path: String,
    max_lines: usize,
    auto_refresh: bool,
    turns: Vec<CogTurn>,
    selected_idx: Option<usize>,
    metacog: Vec<MetacogTrace>,
    show_window: bool,
    filter: String,
}
impl Default for CogState {
    fn default() -> Self {
        Self {
            meta_path: "data/meta/meta_events.jsonl".into(),
            metacog_path: "data/meta/metacog_trace.jsonl".into(),
            max_lines: 50,
            auto_refresh: true,
            turns: vec![],
            selected_idx: None,
            metacog: vec![],
            show_window: true,
            filter: String::new(),
        }
    }
}
impl CogState {
    fn refresh(&mut self) {
        let ev = load_meta_events(&self.meta_path, self.max_lines);
        let mut turns = group_turns(ev);
        if !self.filter.trim().is_empty() {
            let f = self.filter.to_ascii_lowercase();
            turns.retain(|t|
                t.input_text.as_ref().map(|s| s.to_ascii_lowercase().contains(&f)).unwrap_or(false)
                    || t.output_text.as_ref().map(|s| s.to_ascii_lowercase().contains(&f)).unwrap_or(false)
            );
        }
        self.turns = turns;
        self.metacog = load_metacog(&self.metacog_path, self.max_lines);
        if self.selected_idx.map(|i| i >= self.turns.len()).unwrap_or(false) {
            self.selected_idx = None;
        }
    }
}

// ------------- App state -------------
struct AppState {
    // core
    model: Arc<Mutex<HippocampusModel<Bx>>>,

    // tabs
    tab: Tab,

    // brand
    font: FontCfg,
    accent: Color32,

    // chat
    chat_input: String,
    chat_output: String,
    ctx_name: String,
    last_answer_source: String,

    // router
    router: RouterUi,

    // bulk ingest
    bulk_params: BulkParams,
    bulk: Option<BulkState>,
    seen_hashes: Arc<Mutex<HashSet<String>>>,

    // status
    db_ok: bool,
    router_ok: bool,
    model_ok: bool,

    // cognitive
    cog: CogState,
}

impl AppState {
    fn new() -> Self {
        // model
        let device = <Bx as Backend>::Device::default();
        let model = Arc::new(Mutex::new(HippocampusModel::<Bx>::load_or_init(
            &device,
            "R:/uriel/models/checkpoints/uriel_encoder_step008500.bin",
        )));
        let model_ok = true;

        // router
        let (route_T, thr) = load_routercopy_calib("C:/Users/Brinn/rustroverprojects/urielv1/target/release/data/runs/router_v4/calib.json")
            .unwrap_or((1.0, Thresholds { promote: 0.60, refuse: 0.40 }));
        const RC_VOCAB: usize = 95;
        const RC_ARGK: usize = 6;

        let router_sys: Option<RouterCopySystem<Bx>> = match CompactRecorder::new()
            .load::<RouterCopySystemRecord<Bx>>(
                "data/runs/router_v4/routercopy_step010000.bin".into(),
                &device,
            ) {
            Ok(record) => {
                let sys = RouterCopySystem::<Bx>::new(95, 6, &device).load_record(record);
                Some(sys)
            }
            Err(_) => None,
        };
        let router_ok = router_sys.is_some();

        // DB reachable?
        let db_ok = rusqlite::Connection::open("data/semantic_memory.db").is_ok();

        // cognitive
        let mut cog = CogState::default();
        cog.refresh();

        Self {
            model,
            tab: Tab::Cognitive, // open on the Cognitive view
            font: FontCfg {
                path: "".into(),
                size_logo: 28.0,
                size_body: 16.0,
                applied: false,
            },
            accent: Color32::from_rgb(152, 99, 255),

            chat_input: String::new(),
            chat_output: String::new(),
            ctx_name: "default".into(),
            last_answer_source: "-".into(),

            router: RouterUi {
                sys: router_sys,
                route_T,
                thr,
                vocab: RC_VOCAB,
                arg_k: RC_ARGK,
                p: [0.0; 4],
                pred: 0,
                conf: 0.0,
            },

            bulk_params: BulkParams {
                file_path: "data/seed_out.txt".into(),
                ctx: "default".into(),
                threads: thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(2),
            },
            bulk: None,
            seen_hashes: Arc::new(Mutex::new(HashSet::with_capacity(1 << 16))),

            db_ok,
            router_ok,
            model_ok,

            cog,
        }
    }

    // Apply a font from disk (TTF/OTF). If path empty/missing, keep egui default.
    fn apply_font(&mut self, ctx: &egui::Context) {
        if self.font.path.trim().is_empty() || !Path::new(&self.font.path).exists() {
            // still adjust sizes on default family
            let mut style = (*ctx.style()).clone();
            style.text_styles.insert(TextStyle::Heading, FontId::new(self.font.size_logo, FontFamily::Proportional));
            style.text_styles.insert(TextStyle::Body, FontId::new(self.font.size_body, FontFamily::Proportional));
            ctx.set_style(style);
            self.font.applied = true;
            return;
        }
        let bytes = match fs::read(&self.font.path) {
            Ok(b) => b,
            Err(_) => { self.font.applied = false; return; }
        };
        let mut fonts = FontDefinitions::default();
        fonts.font_data.insert("urielfont".into(), FontData::from_owned(bytes));
        fonts.families.get_mut(&FontFamily::Proportional).unwrap().insert(0, "urielfont".into());
        ctx.set_fonts(fonts);

        let mut style = (*ctx.style()).clone();
        style.text_styles.insert(TextStyle::Heading, FontId::new(self.font.size_logo, FontFamily::Proportional));
        style.text_styles.insert(TextStyle::Body, FontId::new(self.font.size_body, FontFamily::Proportional));
        ctx.set_style(style);

        self.font.applied = true;
    }

    fn run_router(&mut self, text: &str) {
        if let Some(sys) = self.router.sys.as_ref() {
            let device = <Bx as Backend>::Device::default();
            let u_tok = rc_encode_ascii95(text, 128);
            let u_oh  = rc_tokens_to_one_hot::<Bx>(&[u_tok], self.router.vocab, &device);
            let u_emb = rc_embed_seq::<Bx>(u_oh, &sys.embed);
            let u_pool = rc_mean_over_time::<Bx>(u_emb);
            let (logits, _arg_logits) = sys.model.route_logits(u_pool);
            let probs = burn::tensor::activation::softmax(logits.div_scalar(self.router.route_T), 1);
            let p: Vec<f32> = probs.to_data().as_slice::<f32>().unwrap().to_vec();
            if p.len() >= 4 {
                self.router.p = [p[0], p[1], p[2], p[3]];
                let (pred, conf) = self.router.p.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,&v)|(i,v)).unwrap();
                self.router.pred = pred;
                self.router.conf = conf;
            }
        } else {
            self.router.p = [0.0; 4];
            self.router.pred = 0;
            self.router.conf = 0.0;
        }
    }

    fn ask_once(&mut self) {
        let input = sanitize_input(&self.chat_input);
        if input.is_empty() { return; }
        self.chat_output.clear();
        self.last_answer_source = "-".into();

        // router
        self.run_router(&input);
        let mut force_retrieve_only = false;
        let mut force_store = false;
        let mut defer = false;

        if self.router.sys.is_some() {
            if self.router.conf >= self.router.thr.promote {
                match self.router.pred {
                    1 => force_retrieve_only = true,
                    2 => force_store = true,
                    3 => defer = true,
                    _ => {}
                }
            } else if self.router.conf <= self.router.thr.refuse {
                force_retrieve_only = true;
            }
        }

        // DB connection per ask (keeps it simple)
        let conn = match rusqlite::Connection::open("data/semantic_memory.db") {
            Ok(c) => c,
            Err(e) => { self.chat_output = format!("(DB error) {e}"); return; }
        };

        // STORE (fast path) — only via explicit button in UI now
        if force_store && !is_question(&input) {
            if let Ok(m) = self.model.lock() {
                let _ = auto_persist_line_quick(&conn, &*m, &self.ctx_name, &self.chat_input, &self.seen_hashes);
            }
            self.chat_output = "Stored.".into();
            self.last_answer_source = "store".into();
            return;
        }

        // Definition queries from DB first
        if let Some(topic_raw) = parse_definition_query(&input) {
            let topic = canonical_topic(&topic_raw);
            let def_key = format!("def:{topic}");
            if let Ok(Some(def)) = read_concept_fact(&def_key).map_err(to_any) {
                self.chat_output = def;
                self.last_answer_source = "db:def".into();
                return;
            }
        }

        // Concept queries from DB (name/creator/mentor/you are)
        let qn = normalize_question(&input);
        if qn.starts_with("your name is") {
            if let Ok(Some(v)) = read_concept_fact("name").map_err(to_any) {
                self.chat_output = format!("Your name is {v}.");
                self.last_answer_source = "db:concept".into();
                return;
            }
        } else if qn.starts_with("your creator is") {
            if let Ok(Some(v)) = read_concept_fact("creator").map_err(to_any) {
                self.chat_output = format!("My creator is {v}.");
                self.last_answer_source = "db:concept".into();
                return;
            }
        } else if qn.starts_with("your mentor is") {
            if let Ok(Some(v)) = read_concept_fact("mentor").map_err(to_any) {
                self.chat_output = format!("My mentor is {v}.");
                self.last_answer_source = "db:concept".into();
                return;
            }
        } else if qn.starts_with("you are") {
            if let Ok(Some(v)) = read_concept_fact("you are").map_err(to_any) {
                self.chat_output = format!("I am {v}.");
                self.last_answer_source = "db:concept".into();
                return;
            }
        }

        // RETRIEVE (rows fallback)
        if force_retrieve_only || is_question(&input) {
            if let Ok(rows) = recall_latest(&self.ctx_name, 128) {
                for r in rows {
                    if let Some(ans) = extract_answer(&qn, &r.data) {
                        self.chat_output = ans;
                        self.last_answer_source = "rows".into();
                        return;
                    }
                }
            }
        }

        // DEFER (clarify)
        if defer {
            self.chat_output = "Could you clarify that a bit?".into();
            self.last_answer_source = "defer".into();
            return;
        }

        // RESPOND via PFC
        let emb = Arc::new(ModelEmb::new(self.model.clone()));
        let retriever = ContextRetriever::new(Arc::new(NullVS), emb, None, RetrieverCfg::default());
        let score = SalienceScore { urgency: 0.3, familiarity: 0.2, understanding: 0.5, drive_alignment: 0.0, final_weight: 0.35 };
        let frame = CortexInputFrame { raw_input: &input, score, drives: &std::collections::HashMap::new() };
        match execute_cortex(frame, &retriever) {
            CortexOutput::VerbalResponse(t) => { self.chat_output = t; self.last_answer_source = "pfc".into(); }
            CortexOutput::Multi(list) => {
                if let Some(CortexOutput::VerbalResponse(t)) = list.into_iter().find(|o| matches!(o, CortexOutput::VerbalResponse(_))) {
                    self.chat_output = t; self.last_answer_source = "pfc".into();
                } else { self.chat_output = "(no verbal response)".into(); self.last_answer_source = "pfc".into(); }
            }
            CortexOutput::Thought(t) => { self.chat_output = t; self.last_answer_source = "pfc:thought".into(); }
            CortexOutput::Action(a) => { self.chat_output = format!("(action) {a:?}"); self.last_answer_source = "pfc:action".into(); }
        }
    }

    // ================= Bulk ingest =================
    fn start_bulk_ingest(&mut self) {
        if self.bulk.as_ref().map(|b| b.running).unwrap_or(false) { return; }
        let path = self.bulk_params.file_path.clone();
        let ctx_name = self.bulk_params.ctx.clone();
        let n_threads = self.bulk_params.threads.max(1);

        let processed = Arc::new(AtomicUsize::new(0));
        let total_enq = Arc::new(AtomicUsize::new(0));
        let errors = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (line_tx, line_rx) = bounded::<String>(20_000);

        // Producer
        {
            let total_enq_c = total_enq.clone();
            let stop_c = stop.clone();
            thread::spawn(move || {
                let file = match File::open(&path) { Ok(f) => f, Err(_) => return };
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
            let seen = self.seen_hashes.clone();
            let ctx_local = ctx_name.clone();
            let model_arc = self.model.clone();

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
                        let model = match model_arc.lock() { Ok(m) => m, Err(_) => return Ok(false) };
                        auto_persist_line_quick(&conn, &*model, &ctx_local, &line, &seen)
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
    }
    fn stop_bulk(&mut self) { if let Some(b) = &self.bulk { b.stop.store(true, AO::Relaxed); } }
}

// ========================= eframe::App impl =========================
struct UrielGui { st: AppState }
impl UrielGui {
    fn new() -> Self { Self { st: AppState::new() } }

    fn brand_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let title = RichText::new("URIEL.AI")
                .color(self.st.accent)
                .size(self.st.font.size_logo)
                .strong();
            ui.label(title);
            ui.add_space(12.0);
            ui.separator();

            let dot = |ok| if ok { "●" } else { "○" };
            let okc = |ok| if ok { Color32::from_rgb(80, 200, 120) } else { Color32::from_rgb(220, 80, 80) };
            ui.label(RichText::new(format!("{} Model", dot(self.st.model_ok))).color(okc(self.st.model_ok)));
            ui.add_space(6.0);
            ui.label(RichText::new(format!("{} Router", dot(self.st.router_ok))).color(okc(self.st.router_ok)));
            ui.add_space(6.0);
            ui.label(RichText::new(format!("{} DB", dot(self.st.db_ok))).color(okc(self.st.db_ok)));
        });
    }

    fn tab_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            for (label, tab) in [("Cognitive", Tab::Cognitive), ("Input", Tab::Chat), ("Bulk Ingest", Tab::Bulk), ("Settings", Tab::Settings)] {                let selected = self.st.tab == tab;
                if ui.selectable_label(selected, label).clicked() { self.st.tab = tab; }
            }
        });
        ui.separator();
    }

    // ---------- NEW: Cognitive tab ----------
    fn tab_cognitive(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        // controls
        ui.horizontal(|ui| {
            ui.label("Meta events:");
            ui.text_edit_singleline(&mut self.st.cog.meta_path);
            ui.label("Metacog:");
            ui.text_edit_singleline(&mut self.st.cog.metacog_path);
        });
        ui.horizontal(|ui| {
            ui.label("Max lines:");
            ui.add(egui::Slider::new(&mut self.st.cog.max_lines, 200..=50_000).logarithmic(true));
            ui.checkbox(&mut self.st.cog.auto_refresh, "Auto refresh");
            if ui.button("Refresh now").clicked() { self.st.cog.refresh(); }
            ui.add_space(12.0);
            ui.label("Filter:");
            ui.text_edit_singleline(&mut self.st.cog.filter);
            if ui.button("Apply").clicked() { self.st.cog.refresh(); }
        });
        ui.separator();

        // layout: left list of turns, right details
        ui.columns(2, |cols| {
            // left: turn list
            cols[0].heading("Turns");
            let turns_len = self.st.cog.turns.len();
            if turns_len == 0 {
                cols[0].label("(no events yet)");
            } else {
                for (i, t) in self.st.cog.turns.iter().enumerate() {
                    let title = format!(
                        "🧭 {}  •  {}  •  {}",
                        human_age(t.first_ts),
                        t.task_id,
                        t.input_text.as_deref().unwrap_or("(no input)")
                    );
                    let sel = Some(i) == self.st.cog.selected_idx;
                    if cols[0].selectable_label(sel, title).clicked() {
                        self.st.cog.selected_idx = Some(i);
                        self.st.cog.show_window = true;
                    }
                }
            }

            // right: details of selected turn
            cols[1].heading("Cognitive Window");
            if let Some(idx) = self.st.cog.selected_idx {
                if let Some(turn) = self.st.cog.turns.get(idx).cloned() {
                    // Prompt context input box (display-only; pure input)
                    cols[1].group(|ui| {
                        ui.label(RichText::new("Prompt").strong());
                        let mut prompt = turn.input_text.clone().unwrap_or_default();
                        ui.add(egui::TextEdit::multiline(&mut prompt).desired_rows(3).interactive(false));
                    });

                    // Evidence timeline
                    cols[1].add_space(6.0);
                    cols[1].group(|ui| {
                        ui.label(RichText::new("Evidence & Reasoning").strong());
                        for e in turn.events.iter() {
                            match e.phase {
                                MetaPhase::Plan => {
                                    ui.label(format!("🗺️ Plan • policy={} arm={} notes={}",
                                                     e.policy.as_deref().unwrap_or("-"),
                                                     e.arm.as_deref().unwrap_or("-"),
                                                     e.notes.as_deref().unwrap_or("-")));
                                }
                                MetaPhase::RetrieveEvidence => {
                                    if e.evidence.is_empty() {
                                        ui.label("🔎 RetrieveEvidence • (no evidence recorded)");
                                    } else {
                                        ui.label("🔎 RetrieveEvidence:");
                                        for ev in &e.evidence {
                                            let rec_w = if ev.id.starts_with("rows:") {
                                                // rows:default:<ts>
                                                if let Some(ts_s) = ev.id.rsplit(':').next() {
                                                    if let Ok(ts) = ts_s.parse::<i64>() {
                                                        recency_weight_from_ts(ts, 7.0)
                                                    } else { ev.quality.recency_w }
                                                } else { ev.quality.recency_w }
                                            } else {
                                                // fall back to recorded recency weight
                                                ev.quality.recency_w
                                            };
                                            let ent = text_entropy_s(&ev.excerpt);
                                            let priority = (0.6 * rec_w + 0.4 * ent).clamp(0.0, 1.0);

                                            ui.separator();
                                            ui.label(RichText::new(format!("• id={}", ev.id)).strong());
                                            ui.label(format!("kind={}  source={}  age={}  recency_w={:.2}  entropy≈{:.2}  priority={:.2}",
                                                             ev.kind, ev.source_id, human_age(ev.fetched_at), rec_w, ent, priority));
                                            ui.add(ProgressBar::new(rec_w).text("recency"));
                                            ui.add(ProgressBar::new(ent).text("entropy/novelty"));
                                            ui.add(ProgressBar::new(priority).text("priority"));
                                            ui.label(RichText::new("excerpt:").italics());
                                            let mut ex = ev.excerpt.clone();
                                            ui.add(egui::TextEdit::multiline(&mut ex).desired_rows(3).interactive(false));
                                        }
                                    }
                                }
                                MetaPhase::Report => {
                                    ui.separator();
                                    ui.label(RichText::new("🧾 Report").strong());
                                    if let Some(out) = e.outputs_ref.as_ref() {
                                        let mut out_text = out.clone();
                                        ui.add(egui::TextEdit::multiline(&mut out_text).desired_rows(4).interactive(false));
                                    }
                                    if let Some((p, lo, hi)) = e.confidence {
                                        ui.label(format!("confidence p={:.2} (CI [{:.2},{:.2}])", p, lo, hi));
                                        ui.add(ProgressBar::new(p).text("final confidence"));
                                    }
                                }
                                MetaPhase::Learn => {
                                    if let Some(obs) = e.observed_value {
                                        ui.label(format!("📈 Learn • arm={} reward={:.2}", e.arm.as_deref().unwrap_or("-"), obs));
                                    }
                                }
                                MetaPhase::Sense => {
                                    // already surfaced as Prompt above; leave breadcrumb
                                    ui.label(format!("👂 Sense @ {}", human_age(e.ts)));
                                }
                                _ => {}
                            }
                        }
                    });

                    // Output panel at end
                    cols[1].add_space(6.0);
                    cols[1].group(|ui| {
                        ui.label(RichText::new("Final Answer").strong());
                        let mut fin = turn.output_text.clone().unwrap_or_default();
                        ui.add(egui::TextEdit::multiline(&mut fin).desired_rows(4).interactive(false));
                        if let Some((p, lo, hi)) = turn.confidence {
                            ui.label(format!("confidence p={:.2} (CI [{:.2},{:.2}])", p, lo, hi));
                            ui.add(ProgressBar::new(p).text("confidence"));
                        }
                    });
                }
            } else {
                cols[1].label("(select a turn on the left)");
            }
        });

        // metacognition stream
        ui.add_space(12.0);
        ui.separator();
        ui.heading("Background Metacognition");
        if self.st.cog.metacog.is_empty() {
            ui.label("(no metacog activity yet)");
        } else {
            egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                for m in &self.st.cog.metacog {
                    ui.separator();
                    ui.label(RichText::new(format!("{}  •  {}", human_age(m.ts), m.action)).strong());
                    ui.label(format!("key={}  prior={:.2}  new={:?}", m.fact_key, m.prior_cert, m.new_cert));
                    if let Some(n) = &m.notes { ui.label(format!("notes: {}", n)); }
                    if let Some(b) = &m.bandit { ui.label(format!("arm={} reward={:.2}", b.arm, b.reward)); }
                }
            });
        }

        // auto-refresh gently
        if self.st.cog.auto_refresh {
            self.st.cog.refresh();
            ctx.request_repaint_after(Duration::from_millis(500));
        }
    }

    // ---------- Chat ----------
    fn tab_chat(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Input (local demo)");
        ui.label("Enter input. Output here is local to the GUI (not the runtime); the Cognitive tab visualizes the runtime’s JSONL traces.");

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("Context:");
            ui.text_edit_singleline(&mut self.st.ctx_name);
        });

        ui.add_space(4.0);
        ui.label("Input:");
        ui.add(egui::TextEdit::multiline(&mut self.st.chat_input).desired_rows(3));

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("Input").clicked() { self.st.ask_once(); }
        });

        ui.add_space(8.0);
        ui.separator();

        // router panel
        ui.heading("Router");
        if self.st.router.sys.is_none() {
            ui.label(RichText::new("Router not loaded").color(Color32::from_rgb(220, 80, 80)));
        } else {
            let names = ["Respond", "Retrieve", "Store", "Defer"];
            for i in 0..4 {
                let v = self.st.router.p[i];
                let bar = ProgressBar::new(v as f32).text(format!("{}  {:.1}%", names[i], v * 100.0));
                ui.add(bar);
            }
            ui.label(format!(
                "Pred: {} @ {:.3} | T={:.2} promote={:.2} refuse={:.2}",
                ["Respond","Retrieve","Store","Defer"][self.st.router.pred],
                self.st.router.conf, self.st.router.route_T, self.st.router.thr.promote, self.st.router.thr.refuse
            ));
        }

        ui.add_space(8.0);
        ui.separator();

        ui.heading("Answer");
        ui.label(RichText::new(&self.st.chat_output).size(self.st.font.size_body));
        ui.label(RichText::new(format!("source: {}", self.st.last_answer_source)).italics());

        if self.st.cog.auto_refresh { self.st.cog.refresh(); }
        if self.st.bulk.as_ref().map(|b| b.running).unwrap_or(false) {
            ctx.request_repaint_after(Duration::from_millis(50));
        }
    }

    // ---------- Bulk ----------
    fn tab_bulk(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("⚡ Bulk Ingest");
        ui.label("Push a large text file into URIEL durable memory as fast as possible.");

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("File path:");
            ui.text_edit_singleline(&mut self.st.bulk_params.file_path);
        });
        ui.horizontal(|ui| {
            ui.label("Context:");
            ui.text_edit_singleline(&mut self.st.bulk_params.ctx);
            ui.label("Threads:");
            ui.add(egui::Slider::new(&mut self.st.bulk_params.threads, 1..=96));
        });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if self.st.bulk.as_ref().map(|b| b.running).unwrap_or(false) {
                if ui.button("⏹ Cancel").clicked() { self.st.stop_bulk(); }
            } else if ui.button("⚡ Start").clicked() {
                self.st.seen_hashes.lock().unwrap().clear();
                self.st.start_bulk_ingest();
            }
        });

        if let Some(b) = &self.st.bulk {
            let enq = b.total_enqueued.load(AO::Relaxed) as f32;
            let done = b.processed.load(AO::Relaxed) as f32;
            let errs = b.errors.load(AO::Relaxed);
            let prog = if enq <= 0.0 { 0.0 } else { (done / enq).min(1.0) };
            ui.add_space(8.0);
            ui.label(format!(
                "Queued: {}   Ingested: {}   Errors: {}   Elapsed: {:.1}s",
                enq as usize, done as usize, errs, b.start.elapsed().as_secs_f32()
            ));
            ui.add(ProgressBar::new(prog).show_percentage());

            if !b.running && enq > 0.0 {
                ui.label("Bulk ingest finished.");
            }
            ctx.request_repaint_after(Duration::from_millis(50));
        }
    }

    // ---------- Settings ----------
    fn tab_settings(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Settings");
        ui.add_space(4.0);

        ui.group(|ui| {
            ui.label("Brand font (TTF/OTF):");
            ui.text_edit_singleline(&mut self.st.font.path);
            ui.horizontal(|ui| {
                ui.label("Logo size:");
                ui.add(egui::Slider::new(&mut self.st.font.size_logo, 16.0..=64.0));
                ui.label("Body size:");
                ui.add(egui::Slider::new(&mut self.st.font.size_body, 12.0..=28.0));
                if ui.button("Apply Font").clicked() {
                    self.st.apply_font(ctx);
                }
            });
            if self.st.font.applied {
                ui.label(RichText::new("Font applied").color(Color32::from_rgb(80, 200, 120)));
            }
        });

        ui.add_space(6.0);
        ui.group(|ui| {
            ui.label("Accent color (RGB):");
            static mut R: u8 = 152; static mut G: u8 = 99; static mut B: u8 = 255;
            unsafe {
                let mut r = R as u32; let mut g = G as u32; let mut b = B as u32;
                ui.add(egui::Slider::new(&mut r, 0..=255).text("R"));
                ui.add(egui::Slider::new(&mut g, 0..=255).text("G"));
                ui.add(egui::Slider::new(&mut b, 0..=255).text("B"));
                R = r as u8; G = g as u8; B = b as u8;
                self.st.accent = Color32::from_rgb(R, G, B);
            }
        });

        ui.add_space(6.0);
        ui.group(|ui| {
            ui.label("Router settings (calibrated from data/runs/router_v1/calib.json if present):");
            ui.label(format!(
                "T = {:.2} | promote = {:.2} | refuse = {:.2}",
                self.st.router.route_T, self.st.router.thr.promote, self.st.router.thr.refuse
            ));
        });
    }
}

impl App for UrielGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        // Apply font once if user hasn’t yet
        if !self.st.font.applied { self.st.apply_font(ctx); }

        egui::TopBottomPanel::top("top").show(ctx, |ui| self.brand_bar(ui));

        egui::CentralPanel::default().show(ctx, |ui| {
            self.tab_bar(ui);
            match self.st.tab {
                Tab::Cognitive => self.tab_cognitive(ui, ctx),
                Tab::Chat => self.tab_chat(ui, ctx),
                Tab::Bulk => self.tab_bulk(ui, ctx),
                Tab::Settings => self.tab_settings(ui, ctx),
            }
        });

        // finish bulk if processed
        if let Some(b) = &mut self.st.bulk {
            let enq = b.total_enqueued.load(AO::Relaxed);
            let done = b.processed.load(AO::Relaxed);
            let errs = b.errors.load(AO::Relaxed);
            if b.stop.load(AO::Relaxed) || (enq > 0 && enq == done + errs) { b.running = false; }
        }
    }
}

// ========================= entry =========================
pub fn main() -> Result<(), eframe::Error> {
    // apply persisted copula style if present
    if let Ok(Some(style)) = read_concept_fact("speech.copula").map_err(to_any) {
        speech_concepts::set_copula_style_long(style.eq_ignore_ascii_case("long"));
    }

    let app = UrielGui::new();
    let opts = NativeOptions::default();
    eframe::run_native("URIEL.AI", opts, Box::new(|_cc| Box::new(app)))
}

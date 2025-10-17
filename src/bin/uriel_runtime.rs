// src/bin/uriel_runtime.rs
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use smie_core::Embedder;
use anyhow::Result;
use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use crossbeam_channel::{unbounded, Receiver, Sender};
use ctrlc;
use rand::Rng;
use regex::Regex;
use rusqlite;
use sha2::{Digest, Sha256};
use smie_core::tags;
// ── concepts & contexts
use URIELV1::concepts_and_contexts::{
    answer::answer_concept,
    extractors::{CreatorConcept, LaunchCodeConcept, MentorConcept, NameConcept},
    learner::ConceptLearner,
    registry::Registry,
};
use smie_core::Smie;
use smie_core::SmieConfig;
use smie_core::Metric;
// ── URIEL subsystems
use URIELV1::Frontal::frontal::{
    evaluate_frontal_control, FrontalContext, FrontalDecision, InhibitionConfig, InhibitionState,
    WorkingMemory,
};
use URIELV1::Hippocampus::infer::HippocampusModel;
use URIELV1::Hippocampus::pipeline;
use URIELV1::Hippocampus::primordial::{primordial, PRIMORDIAL_SHA256};
use URIELV1::Hippocampus::sqlite_backend::{
    has_hash, read_concept_fact, recall_latest, record_hash, upsert_concept_fact,
};
use URIELV1::Microthalamus::goal::{Ctx, Goal, GoalStatus};
use URIELV1::Microthalamus::microthalamus::{MicroThalamus, SyntheticDrive};
use URIELV1::PFC::context_retriever::{
    ContextRetriever, Embedding, Embeddings, MemoryHit, RetrieverCfg, VectorStore,
};
use URIELV1::PFC::cortex::{execute_cortex, CortexInputFrame, CortexOutput};
use URIELV1::parietal::parietal::SalienceScore;
use URIELV1::parietal::route_cfg::RouteCfg;

// bandit
use URIELV1::policy::retrieval::{Arm, Bandit};

// ─────────────────────────── helpers
fn assert_primordial_invariants() {
    let p = primordial();
    assert!(!p.axioms.is_empty());
    assert!(!p.prohibitions.is_empty());
    assert!(!p.drives.is_empty());
    let sum_w: f32 = p.drives.iter().map(|d| d.weight).sum();
    assert!((0.5..=2.0).contains(&sum_w));
    println!("Primordial OK (v{}, sha256 {}).", p.version, PRIMORDIAL_SHA256);
}

#[inline]
fn to_any<E: std::fmt::Display + Send + Sync + 'static>(e: E) -> anyhow::Error {
    anyhow::anyhow!(e.to_string())
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

fn is_question(s: &str) -> bool {
    let l = s.trim().to_ascii_lowercase();
    l.ends_with('?')
        || l.starts_with("what ")
        || l.starts_with("who ")
        || l.starts_with("where ")
        || l.starts_with("why ")
        || l.starts_with("how ")
}

// include common contractions
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

fn display_concept(key: &str, value: &str) -> String {
    match key {
        "name" => format!("Name: {value}"),
        "launch code" => format!("Launch code: {value}"),
        "creator" => format!("Creator: {value}"),
        "mentor" => format!("Mentor: {value}"),
        other => format!("{other}: {value}"),
    }
}

// concept-only extractor: only returns a value for the *asked* concept
fn extract_answer(qn: &str, text: &str) -> Option<String> {
    let tl = text.trim();
    let ll = tl.to_ascii_lowercase();

    if qn.starts_with("your name is") {
        if let Some(idx) = ll.find("name is") {
            let ans = tl[idx + "name is".len()..]
                .trim()
                .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace());
            if !ans.is_empty() {
                return Some(ans.to_string());
            }
        }
        return None;
    }
    if qn.starts_with("the launch code is") {
        let re = Regex::new(r"(\d{3}[-\s]?\d{3}[-\s]?\d{3})").ok()?;
        if let Some(c) = re.captures(tl) {
            return Some(c.get(1).unwrap().as_str().to_string());
        }
        return None;
    }
    if qn.starts_with("your creator is") {
        if let Some(idx) = ll.find("creator is") {
            let ans = tl[idx + "creator is".len()..]
                .trim()
                .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace());
            if !ans.is_empty() {
                return Some(ans.to_string());
            }
        }
        return None;
    }
    if qn.starts_with("your mentor is") {
        if let Some(idx) = ll.find("mentor is") {
            let ans = tl[idx + "mentor is".len()..]
                .trim()
                .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace());
            if !ans.is_empty() {
                return Some(ans.to_string());
            }
        }
        return None;
    }
    None
}

fn choose_best_for(
    qn: &str,
    hits: &[(u64, f32, u64, String)],
) -> Option<(u64, f32, u64, String)> {
    let key = if qn.starts_with("your name is") {
        Some("name")
    } else if qn.starts_with("the launch code is") {
        Some("launch code")
    } else {
        None
    };

    if let Some(k) = key {
        let mut filtered: Vec<_> = hits
            .iter()
            .filter(|(_, _, _, text)| text.to_ascii_lowercase().contains(k))
            .cloned()
            .collect();
        if !filtered.is_empty() {
            filtered.sort_by(|a, b| {
                b.0.cmp(&a.0)
                    .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal))
            });
            return filtered.into_iter().next();
        }
        let mut all = hits.to_vec();
        all.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal))
        });
        return all.into_iter().next();
    }
    let mut all = hits.to_vec();
    all.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal))
    });
    all.into_iter().next()
}

// STRICT statements-only fallback
fn concept_of(text: &str) -> Option<&'static str> {
    let l = text.to_ascii_lowercase();
    if l.starts_with("the launch code is ") {
        Some("launch code")
    } else if l.starts_with("your name is ") {
        Some("name")
    } else {
        None
    }
}

fn fmt_tags(t: u64) -> String {
    let mut v = Vec::new();
    if t & tags::USER != 0 {
        v.push("USER");
    }
    if t & tags::SYSTEM != 0 {
        v.push("SYSTEM");
    }
    if t & tags::SHORT != 0 {
        v.push("SHORT");
    }
    if t & tags::LONG != 0 {
        v.push("LONG");
    }
    if v.is_empty() {
        "NONE".into()
    } else {
        v.join("|")
    }
}

// ────────────────── “to be” verbiage (first step)
fn extract_copula_name_for_agent(text: &str) -> Option<String> {
    // Matches: "you are X", "you're X" (not roles), also "your name is X"
    // Avoid role words to not collide with creator/mentor.
    let role_words = ["creator", "mentor", "owner", "friend", "assistant"];
    let l = text.to_ascii_lowercase();

    // your name is X
    if let Some(c) =
        Regex::new(r"(?i)^\s*your\s+name\s+is\s+(.+?)\s*[.!]?\s*$")
            .unwrap()
            .captures(text)
    {
        let cand = c[1].trim().to_string();
        if !cand.is_empty() {
            return Some(cand);
        }
    }

    // you are / you're X
    if let Some(c) =
        Regex::new(r"(?i)^\s*you\s+(?:are|\'re)\s+(.+?)\s*[.!]?\s*$")
            .unwrap()
            .captures(text)
    {
        let cand = c[1].trim().to_string();
        if !cand.is_empty()
            && !role_words.iter().any(|w| l.contains(w))
        {
            return Some(cand);
        }
    }

    None
}

// ─────────────────────────── embedders
struct ModelEmb<B: Backend> {
    model: Arc<Mutex<HippocampusModel<B>>>,
}
impl<B: Backend> ModelEmb<B> {
    fn new(model: Arc<Mutex<HippocampusModel<B>>>) -> Self {
        Self { model }
    }
}
impl<B: Backend> Embeddings for ModelEmb<B> {
    fn embed(&self, text: &str) -> std::result::Result<Embedding, Box<dyn Error + Send + Sync>> {
        let model = self.model.lock().map_err(|_| {
            Box::<dyn Error + Send + Sync>::from(io::Error::new(
                io::ErrorKind::Other,
                "model lock poisoned",
            ))
        })?;
        let z = model.run_inference(text);
        let mut v = z
            .to_data()
            .as_slice::<f32>()
            .map_err(|e| {
                Box::<dyn Error + Send + Sync>::from(io::Error::new(
                    io::ErrorKind::Other,
                    format!("{e:?}"),
                ))
            })?
            .to_vec();
        let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
        for x in &mut v {
            *x /= n;
        }
        Ok(v)
    }
}

struct SmieHippEmbed<B: Backend> {
    model: Arc<Mutex<HippocampusModel<B>>>,
    dim: usize,
}
impl<B: Backend> SmieHippEmbed<B> {
    fn new(model: Arc<Mutex<HippocampusModel<B>>>) -> anyhow::Result<Self> {
        let m = model
            .lock()
            .map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
        let z = m.run_inference("[dim-probe]");
        let dim = z
            .to_data()
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?
            .len();
        drop(m);
        Ok(Self { model, dim })
    }
}
impl<B: Backend> Embedder for SmieHippEmbed<B> {
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
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
        let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
        for x in &mut v {
            *x /= n;
        }
        Ok(v)
    }
}

// kill SQLite ANN scans
struct NullVS;
impl VectorStore for NullVS {
    fn search(
        &self,
        _q: &Embedding,
        _k: usize,
    ) -> std::result::Result<Vec<MemoryHit>, Box<dyn Error + Send + Sync>> {
        Ok(vec![])
    }
}

// ─────────────────────────── MicroThalamus + goal helpers
fn drives_as_weights(
    drives: &HashMap<String, SyntheticDrive>,
) -> HashMap<String, f32> {
    drives.iter().map(|(k, d)| (k.clone(), d.weight)).collect()
}
fn push_goal(
    goals: &mut Vec<Goal>,
    mut g: Goal,
    drives: &HashMap<String, SyntheticDrive>,
    ctx: &Ctx,
) {
    g.score(&drives_as_weights(drives), ctx);
    goals.push(g);
}
fn rescore_goals(
    goals: &mut Vec<Goal>,
    drives: &HashMap<String, SyntheticDrive>,
    ctx: &Ctx,
) {
    let w = drives_as_weights(drives);
    for g in goals.iter_mut() {
        g.score(&w, ctx);
    }
    goals.sort_by(|a, b| {
        b.last_score
            .partial_cmp(&a.last_score)
            .unwrap_or(Ordering::Equal)
    });
    goals.retain(|g| !matches!(g.status, GoalStatus::Succeeded | GoalStatus::Failed | GoalStatus::Aborted));
}
fn mine_negatives_from_smie(
    smie: &Smie,
    query: &str,
    top_k: usize,
) -> Vec<(String, f32)> {
    // recall on the original text, take top_k as hard negatives
    let mut out = Vec::new();
    if let Ok(hits) = smie.recall(query, top_k, tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER) {
        for (_id, sim, _tags, txt) in hits {
            out.push((txt, sim));
        }
    }
    out
}

fn attach_negs(
    conn: &mut rusqlite::Connection,
    qid: i64,
    negs: &[(String, f32)],
) {
    if let Ok(tx) = conn.transaction() {
        for (neg, sim) in negs {
            let _ = tx.execute(
                "INSERT INTO learn_negs(qid,negative,sim) VALUES(?1,?2,?3)",
                rusqlite::params![qid, neg, *sim],
            );
        }
        let _ = tx.commit();
    }
}



// utils
fn now_sec() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// Persist a single line into SQLite + SMIE, with concept upsert and durable dedupe.
fn auto_persist_line<B: Backend>(
    model: &HippocampusModel<B>,
    smie: &Smie,
    ctx: &str,
    text_raw: &str,
    concepts: &Registry,
    _learner: &mut ConceptLearner, // reserved for online learning usage
    seen_hashes: &mut HashSet<String>,
) -> anyhow::Result<bool> {
    let clean = sanitize_input(text_raw);
    if clean.is_empty() {
        return Ok(false);
    }

    let n = normalize_for_hash(&clean);
    let key = sha256_hex(&n);

    // in-process guard (fast)
    if !seen_hashes.insert(key.clone()) {
        return Ok(false);
    }

    let ts = now_sec();

    // durable dedupe — if already recorded, skip
    if has_hash(ctx, &key).map_err(to_any)? {
        return Ok(false);
    }
    let _ = record_hash(ctx, &key, ts).map_err(to_any)?;

    // 1) registry extraction → DB upsert → tombstone same-concept in SMIE
    let hits = concepts.extract_all_from_statement(&clean);
    if !hits.is_empty() {
        for h in &hits {
            upsert_concept_fact(h.key, &h.value, ts).map_err(to_any)?;
            let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
            if let Ok(old) = smie.recall(h.key, 64, mask) {
                for (id, _s, _t, txt) in old {
                    if txt.to_ascii_lowercase().contains(h.key) {
                        let _ = smie.tombstone(id);
                    }
                }
            }
        }
    }
    // 1a) “to be” verbiage: "You're URIEL" → name/you are
    else if let Some(name) = extract_copula_name_for_agent(&clean) {
        upsert_concept_fact("name", &name, ts).map_err(to_any)?;
        upsert_concept_fact("you are", &name, ts).map_err(to_any)?;
        for k in ["name", "you are"] {
            let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
            if let Ok(old) = smie.recall(k, 64, mask) {
                for (id, _s, _t, txt) in old {
                    if txt.to_ascii_lowercase().contains(k) {
                        let _ = smie.tombstone(id);
                    }
                }
            }
        }
    }
    // 1b) Fallback for compound roles: "I am your creator and mentor, Brinn."
    else if let Some(caps) = Regex::new(
        r"(?i)^\s*i\s+am\s+your\s+([a-z\s]+?),\s*([A-Za-z][A-Za-z .'\-]+)\s*[.!]?\s*$",
    )
        .unwrap()
        .captures(&clean)
    {
        let roles = caps[1].to_ascii_lowercase();
        let name = caps[2].trim().to_string();
        if roles.contains("creator") {
            upsert_concept_fact("creator", &name, ts).map_err(to_any)?;
            let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
            if let Ok(old) = smie.recall("creator", 64, mask) {
                for (id, _s, _t, txt) in old {
                    if txt.to_ascii_lowercase().contains("creator") {
                        let _ = smie.tombstone(id);
                    }
                }
            }
        }
        if roles.contains("mentor") {
            upsert_concept_fact("mentor", &name, ts).map_err(to_any)?;
            let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
            if let Ok(old) = smie.recall("mentor", 64, mask) {
                for (id, _s, _t, txt) in old {
                    if txt.to_ascii_lowercase().contains("mentor") {
                        let _ = smie.tombstone(id);
                    }
                }
            }
        }
    }

    // 2) legacy strict fallback for name/launch-code sentences (optional)
    if let Some(con) = concept_of(&clean) {
        let qn = if con == "name" {
            "your name is"
        } else {
            "the launch code is"
        };
        if let Some(value) = extract_answer(qn, &clean) {
            upsert_concept_fact(con, &value, ts).map_err(to_any)?;
        }
        let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
        if let Ok(old) = smie.recall(con, 64, mask) {
            for (id, _s, _t, txt) in old {
                if concept_of(&txt) == Some(con) {
                    let _ = smie.tombstone(id);
                }
            }
        }
    }

    // 3) durability of the raw line
    pipeline::ingest(model, ctx, &clean, 7 * 24 * 3600, "runtime")
        .map_err(|e| anyhow::anyhow!("ingest failed: {e}"))?;

    // 4) SMIE embed raw content
    let _ = smie.ingest(&clean, tags::USER | tags::SHORT)?;

    Ok(true)
}

// ─────────── P1.2 / P1.3: stubs (safe) — wire to trainer/checkpointer when ready
fn maybe_stage_anchor_for_training(
    conn: &rusqlite::Connection,
    question: &str,
    positive: &str,
    reward: f32,
) {
    // ensure tables exist
    let _ = conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS learn_queue(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts INTEGER NOT NULL,
          question TEXT NOT NULL,
          positive TEXT NOT NULL,
          hardness REAL NOT NULL,
          used INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS learn_negs(
          qid INTEGER NOT NULL,
          negative TEXT NOT NULL,
          sim REAL NOT NULL,
          FOREIGN KEY(qid) REFERENCES learn_queue(id) ON DELETE CASCADE
        );
        "#,
    );

    // sample a small fraction of perfect passes as anchors
    if reward >= 0.9 {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() <= 0.05 {
            let _ = conn.execute(
                "INSERT INTO learn_queue(ts,question,positive,hardness,used)
                 VALUES(strftime('%s','now'),?1,?2,?3,0)",
                rusqlite::params![question, positive, 0.05_f32],
            );
        }
    }
}


fn enqueue_hard_example(
    conn: &rusqlite::Connection,
    question: &str,
    positive: &str,
    hardness: f32,
) -> Option<i64> {
    let ok = conn.execute(
        "INSERT INTO learn_queue(ts,question,positive,hardness,used)
         VALUES(strftime('%s','now'),?1,?2,?3,0)",
        rusqlite::params![question, positive, hardness],
    ).ok()?; // ok == 1 on success
    if ok == 1 { Some(conn.last_insert_rowid()) } else { None }
}

fn maybe_checkpoint_on_high_conf(_reward: f32) {
    // TODO: trigger encoder checkpoint when reward≥0.9 with throttling (e.g., every N events or T minutes)
}

// ─────────────────────────── runtime
fn main() -> Result<()> {
    assert_primordial_invariants();

    // Bandits
    let concept_arms: &[Arm] = &[
        Arm { name: "db" },
        Arm { name: "normalized" },
        Arm { name: "keyword" },
        Arm { name: "original" },
    ];
    let general_arms: &[Arm] = &[Arm { name: "normalized" }, Arm { name: "original" }];
    let concept_bandit = Bandit {
        intent: "concept_q",
        arms: concept_arms,
        epsilon: 0.15,
    };
    let general_bandit = Bandit {
        intent: "general_q",
        arms: general_arms,
        epsilon: 0.15,
    };
    let mut conn = rusqlite::Connection::open("data/semantic_memory.db")
        .expect("open data/semantic_memory.db");

    // concepts registry + online learner (learner reserved; not used for ranking yet)
    let concepts = Registry::new()
        .register(Box::new(NameConcept))
        .register(Box::new(LaunchCodeConcept))
        .register(Box::new(CreatorConcept))
        .register(Box::new(MentorConcept));
    let mut learner = ConceptLearner::new("data/learners");

    type B = NdArray<f32>;
    let device = <B as Backend>::Device::default();

    // PFC model
    let model_runtime =
        HippocampusModel::<B>::load_or_init(&device, "R:/uriel/models/checkpoints/uriel_encoder_step008500.bin");

    // shared encoder (PFC + SMIE)
    let model_for_emb = Arc::new(Mutex::new(HippocampusModel::<B>::load_or_init(
        &device,
        "R:/uriel/models/checkpoints/uriel_encoder_step008500.bin",
    )));

    // SMIE index
    let smie_embedder = Arc::new(SmieHippEmbed::new(model_for_emb.clone())?);
    let smie = Arc::new(Smie::new(
        SmieConfig {
            dim: smie_embedder.dim(),
            metric: Metric::Cosine,
        },
        smie_embedder,
    )?);
    std::fs::create_dir_all("data").ok();
    let _ = smie.load("data/uriel");

    // retriever without ANN
    let emb = Arc::new(ModelEmb::new(model_for_emb.clone()));
    let retriever = ContextRetriever::new(Arc::new(NullVS), emb, None, RetrieverCfg::default());

    // routing
    let mut drives: HashMap<String, SyntheticDrive> = Default::default();
    let _cfg = RouteCfg::default();
    let ctx = "default";

    // MicroThalamus + goals
    let (goal_tx, goal_rx): (Sender<Goal>, Receiver<Goal>) = unbounded();
    let mut thalamus = MicroThalamus::new(goal_tx.clone());
    thalamus.register_drive(SyntheticDrive::new(
        "Curiosity",
        0.25,
        0.01,
        Duration::from_secs(3),
        0.5,
        "drive:curiosity",
        vec![0.1, 0.2, 0.3, 0.0, 0.1, 0.4, 0.1, 0.2],
    ));
    thalamus.register_drive(SyntheticDrive::new(
        "ResolveConflict",
        0.35,
        0.01,
        Duration::from_secs(5),
        1.0,
        "drive:resolveconflict",
        vec![0.1, 0.2, 0.3, 0.0, 0.1, 0.4, 0.1, 0.2],
    ));
    drives = thalamus.drives.clone();

    let mut goals: Vec<Goal> = Vec::new();

    // frontal
    let mut wm = WorkingMemory::new(50);
    let inhib_state = InhibitionState::default();
    let inhib_cfg = InhibitionConfig::default();
    let mut last_out: Option<CortexOutput> = None;

    // dedupe for auto-persist
    let mut seen_hashes: HashSet<String> = HashSet::with_capacity(4096);

    // autosave
    {
        let smie_c = smie.clone();
        ctrlc::set_handler(move || {
            let _ = smie_c.save("data/uriel");
            eprintln!("\n💾 Saved SMIE. Bye.");
            std::process::exit(0);
        })?;
    }

    println!("URIEL runtime. Type lines. Prefix with 'store this:' to save, ask 'what … ?' to retrieve.");

    // repl
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        print!("> ");
        io::stdout().flush()?;
        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }

        // sanitize
        let raw = line.as_str();
        let input = sanitize_input(raw);
        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            break;
        }
        let lower = input.to_ascii_lowercase();

        // auto-persist: everything that isn't an explicit store and isn't a pure question
        if !lower.starts_with("store this:") && !is_question(&input) {
            let _ = auto_persist_line(
                &model_runtime,
                &smie,
                ctx,
                &input,
                &concepts,
                &mut learner,
                &mut seen_hashes,
            );
        }

        // MicroThalamus tick
        thalamus.inject_stimulus(ctx, &input);
        thalamus.tick();
        drives = thalamus.drives.clone();

        // drain goals
        let mut new_goals = Vec::new();
        while let Ok(g) = goal_rx.try_recv() {
            new_goals.push(g);
        }
        if !new_goals.is_empty() {
            let ctx_sig = Ctx {
                signals: [("weight".into(), 0.35)].into_iter().collect(),
            };
            for g in new_goals {
                push_goal(&mut goals, g, &drives, &ctx_sig);
            }
            rescore_goals(&mut goals, &drives, &ctx_sig);
        }

        // explicit STORE branch
        if lower.starts_with("store this:") {
            let rest = input.splitn(2, ':').nth(1).unwrap_or("").trim();
            pipeline::ingest(&model_runtime, ctx, rest, 7 * 24 * 3600, "runtime")
                .map_err(|e| anyhow::anyhow!("ingest failed: {e}"))?;
            if let Some(con) = concept_of(rest) {
                let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
                if let Ok(old) = smie.recall(con, 32, mask) {
                    for (id, _s, _t, txt) in old {
                        if concept_of(&txt) == Some(con) {
                            let _ = smie.tombstone(id);
                        }
                    }
                }
            }
            let tbits = tags::USER | tags::SHORT;
            let id = smie.ingest(rest, tbits)?;
            wm.set_focus(rest);
            println!("Stored ({id}) [{}].", fmt_tags(tbits));
            continue;
        }

        // SMIE-first answering
        let mut answer = String::new();
        if is_question(&input) {
            let q_norm = normalize_question(&input);
            let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;

            if let Some((key, norm)) = concepts.normalize_question(&input) {
                // ---- bandit: decide probe order for this question ----
                let chosen = concept_bandit.choose(&conn);
                let arms = ["db", "normalized", "keyword", "original"];
                let mut order = Vec::with_capacity(arms.len());
                order.push(chosen);
                for a in arms.iter() {
                    if *a != chosen {
                        order.push(*a);
                    }
                }

                let mut answered_source: Option<&'static str> = None;

                for arm in order {
                    match arm {
                        "db" => {
                            if let Ok(Some(v)) = read_concept_fact(key).map_err(to_any) {
                                answer = display_concept(key, &v);
                                answered_source = Some("db");
                                break;
                            }
                        }
                        "normalized" => {
                            if let Some(v) = answer_concept(&smie, key, &norm, &input)? {
                                if let Some(val) = extract_answer(&norm, &v) {
                                    answer = display_concept(key, &val);
                                } else {
                                    answer = display_concept(key, &v);
                                }
                                answered_source = Some("normalized");
                                break;
                            }
                        }
                        "keyword" => {
                            if let Ok(hits) = smie.recall(key, 8, mask) {
                                if let Some(best) = choose_best_for(&norm, &hits) {
                                    if let Some(val) = extract_answer(&norm, &best.3) {
                                        answer = display_concept(key, &val);
                                        answered_source = Some("keyword");
                                        break;
                                    }
                                }
                            }
                        }
                        "original" => {
                            if let Ok(hits) = smie.recall(&input, 8, mask) {
                                if let Some(best) = choose_best_for(&norm, &hits) {
                                    if let Some(val) = extract_answer(&norm, &best.3) {
                                        answer = display_concept(key, &val);
                                        answered_source = Some("original");
                                        break;
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }

                // Special case: "Who are you?" → derive from name if still empty
                if key == "you are" && answer.is_empty() {
                    if let Ok(Some(v)) = read_concept_fact("you are").map_err(to_any) {
                        answer = format!("You are {v}");
                        answered_source = Some("db");
                    } else if let Ok(Some(v)) = read_concept_fact("name").map_err(to_any) {
                        answer = format!("You are {v}");
                        let _ = upsert_concept_fact("you are", &v, now_sec());
                        answered_source = Some("db");
                    } else if let Some(v) =
                        answer_concept(&smie, "name", "your name is", &input)?
                    {
                        if let Some(name) = extract_answer("your name is", &v) {
                            answer = format!("You are {name}");
                            let _ = upsert_concept_fact("name", &name, now_sec());
                            let _ = upsert_concept_fact("you are", &name, now_sec());
                            answered_source = Some("normalized");
                        }
                    }
                }

                // newest rows fallback
                if answer.is_empty() {
                    if let Ok(rows) = recall_latest("default", 128) {
                        for r in rows {
                            if let Some(val) = extract_answer(&norm, &r.data) {
                                answer = display_concept(key, &val);
                                answered_source = Some("rows");
                                break;
                            }
                        }
                    }
                }

                // reward + optional learning/checkpoint hooks
                let reward = match answered_source {
                    Some("db") => 1.0,
                    Some("normalized") => 0.9,
                    Some("keyword") => 0.7,
                    Some("original") => 0.6,
                    Some("rows") => 0.5,
                    _ => 0.0,
                };
                concept_bandit.update(&conn, chosen, reward);

                // P1.2 staging: hard/borderline go to learn_queue; anchors on high conf
                if reward >= 0.35 && reward < 0.9 {
                    if let Some(qid) = enqueue_hard_example(&conn, &input, &answer, 1.0 - reward) {
                        let mut negs = mine_negatives_from_smie(&smie, &input, 6);
                        negs.retain(|(t,_s)| t != &answer);
                        negs.truncate(5);
                        attach_negs(&mut conn, qid, &negs);
                    }
                } else if reward >= 0.9 {
                    maybe_stage_anchor_for_training(&conn, &input, &answer, reward);
                    maybe_checkpoint_on_high_conf(reward);
                }
            } else {
                // ---- Non-concept: bandit-based simple probes ----
                let chosen = general_bandit.choose(&conn);
                let mut order = vec![chosen];
                for a in ["normalized", "original"].iter() {
                    if *a != chosen {
                        order.push(*a);
                    }
                }
                let mut answered = false;
                for arm in order {
                    let q = match arm {
                        "normalized" => q_norm.clone(),
                        _ => input.clone(),
                    };
                    if let Ok(hits) = smie.recall(&q, 8, mask) {
                        if let Some(best) = choose_best_for(&q_norm, &hits) {
                            if let Some(val) = extract_answer(&q_norm, &best.3) {
                                answer = val;
                            } else {
                                answer = best.3;
                            }
                            answered = true;
                            break;
                        }
                    }
                }
                let reward = if answered { 0.8 } else { 0.0 };
                general_bandit.update(&conn, chosen, reward);
                if reward >= 0.35 && reward < 0.9 {
                    enqueue_hard_example(&conn, &input, &answer, 1.0 - reward);
                } else if reward >= 0.9 {
                    maybe_stage_anchor_for_training(&conn, &input, &answer, reward);
                    maybe_checkpoint_on_high_conf(reward);
                }
            }
        }

        // If SMIE didn’t answer, try PFC (Thought only for concepts)
        if answer.trim().is_empty() {
            let score = SalienceScore {
                urgency: 0.3,
                familiarity: 0.2,
                understanding: 0.5,
                drive_alignment: 0.0,
                final_weight: 0.35,
            };
            let frame = CortexInputFrame {
                raw_input: &input,
                score,
                drives: &drives,
            };
            let out = execute_cortex(frame, &retriever);

            if concepts
                .normalize_question(&input)
                .is_some()
                || normalize_question(&input).starts_with("you are")
            {
                match out {
                    CortexOutput::Thought(t) => println!("{t}"),
                    CortexOutput::Multi(list) => {
                        for item in &list {
                            if let CortexOutput::Thought(t) = item {
                                println!("{t}");
                            }
                        }
                    }
                    CortexOutput::Action(a) => println!("(action) {:?}", a),
                    CortexOutput::VerbalResponse(_) => {
                        // ignore PFC verbal for concepts
                    }
                }
            } else {
                match out {
                    CortexOutput::VerbalResponse(text) => {
                        answer = text;
                    }
                    CortexOutput::Multi(list) => {
                        for item in &list {
                            if let CortexOutput::Thought(t) = item {
                                println!("{t}");
                            }
                        }
                        if let Some(CortexOutput::VerbalResponse(t)) =
                            list.into_iter().find(|o| matches!(o, CortexOutput::VerbalResponse(_)))
                        {
                            answer = t;
                        }
                    }
                    CortexOutput::Thought(t) => println!("{t}"),
                    CortexOutput::Action(a) => println!("(action) {:?}", a),
                }
            }
        }

        // Final fallback: newest rows scan
        if answer.trim().is_empty() && is_question(&input) {
            let qn = normalize_question(&input);
            if let Ok(rows) = recall_latest("default", 128) {
                for r in rows {
                    if let Some(val) = extract_answer(&qn, &r.data) {
                        answer = val;
                        break;
                    }
                }
            }
            if answer.trim().is_empty() {
                answer =
                    "(no memory yet — try `Your name is URIEL.` or `The launch code is 123-456-789`)."
                        .into();
            }
        }

        // Frontal gating
        let active_goal_labels: Vec<String> = goals.iter().map(|g| g.label.clone()).collect();
        let fctx = FrontalContext {
            active_goals: active_goal_labels,
            current_drive_weights: &drives,
            last_response: last_out.as_ref(),
            working_memory: std::cell::RefCell::new(wm.clone()),
            inhib_state: std::cell::RefCell::new(inhib_state.clone()),
            inhib_cfg: inhib_cfg.clone(),
        };

        let candidate = CortexOutput::VerbalResponse(answer.clone());
        match evaluate_frontal_control(&candidate, &fctx) {
            FrontalDecision::ExecuteImmediately(out) => {
                match out.clone() {
                    CortexOutput::VerbalResponse(text) => println!("{text}"),
                    other => println!("{other:?}"),
                }
                last_out = Some(out);
            }
            FrontalDecision::QueueForLater(out) => {
                eprintln!("(queued) {out:?}");
                last_out = Some(out);
            }
            FrontalDecision::SuppressResponse(msg) => {
                eprintln!("{msg}");
            }
            FrontalDecision::ReplaceWithGoal(msg) => {
                println!("{msg}");
                last_out = Some(CortexOutput::Thought(msg));
            }
        }
    }

    let _ = smie.save("data/uriel");
    Ok(())
}

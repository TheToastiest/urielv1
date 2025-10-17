// src/bin/uriel_runtime.rs
use std::cmp::Ordering;
use std::error::Error;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;

use URIELV1::Hippocampus::infer::HippocampusModel;
use URIELV1::Hippocampus::pipeline;
use URIELV1::Hippocampus::primordial::{primordial, PRIMORDIAL_SHA256};
use URIELV1::Hippocampus::sqlite_backend::recall_latest; // fact fallback
use URIELV1::Microthalamus::microthalamus::SyntheticDrive;
use URIELV1::PFC::context_retriever::{
    ContextRetriever, Embedding, Embeddings, MemoryHit, RetrieverCfg, VectorStore,
};
use URIELV1::PFC::cortex::CortexOutput;
use URIELV1::Frontal::frontal::{
    evaluate_frontal_control, FrontalContext, FrontalDecision, InhibitionConfig, InhibitionState,
    WorkingMemory,
};
use URIELV1::parietal::route_cfg::RouteCfg;

use ctrlc;
use regex::Regex;

// SMIE / raggedy_anndy
use smie_core::{Embedder, Smie, SmieConfig, Metric, tags};

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

fn sanitize_input(s: &str) -> String {
    // trim, then strip any leading prompt sigils like ">", "•", "-" repeatedly
    let mut t = s.trim();
    while let Some(c) = t.chars().next() {
        if c == '>' || c == '•' || c == '-' || c == '|' { t = t[1..].trim_start(); } else { break; }
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
    if l.starts_with("what is your name") { return "your name is".into(); }
    if l.starts_with("what is the launch code") { return "the launch code is".into(); }
    q.to_string()
}
fn concept_hint_from_norm(qn: &str) -> Option<&'static str> {
    if qn.starts_with("your name is") { Some("name") }
    else if qn.starts_with("the launch code is") { Some("launch code") }
    else { None }
}
fn extract_answer(qn: &str, text: &str) -> Option<String> {
    let tl = text.trim();
    let ll = tl.to_ascii_lowercase();

    if qn.starts_with("your name is") || ll.contains("your name is") || ll.contains("name is") {
        if let Some(idx) = ll.find("name is") {
            let ans = tl[idx + "name is".len()..]
                .trim()
                .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace());
            if !ans.is_empty() { return Some(ans.to_string()); }
        }
    }
    if qn.starts_with("the launch code is") || ll.contains("launch code") {
        let re = Regex::new(r"(\d{3}[-\s]?\d{3}[-\s]?\d{3})").ok()?;
        if let Some(c) = re.captures(tl) { return Some(c.get(1).unwrap().as_str().to_string()); }
    }
    None
}
fn choose_best_for(qn: &str, mut hits: Vec<(u64, f32, u64, String)>) -> Option<(u64, f32, u64, String)> {
    let key = if qn.starts_with("your name is") { "name" }
    else if qn.starts_with("the launch code is") { "launch code" }
    else { "" };
    hits.sort_by(|a, b| {
        let ab = b.3.to_ascii_lowercase().contains(key) as u8;
        let aa = a.3.to_ascii_lowercase().contains(key) as u8;
        ab.cmp(&aa)           // prefer containing key
            .then_with(|| b.0.cmp(&a.0)) // prefer NEWER id
            .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)) // then score
    });
    hits.into_iter().next()
}
fn concept_of(text: &str) -> Option<&'static str> {
    let l = text.to_ascii_lowercase();
    if l.starts_with("the launch code is") || l.contains("launch code") { return Some("launch code"); }
    if l.starts_with("your name is") || l.contains("your name is") || l.contains("name is") { return Some("name"); }
    None
}
fn fmt_tags(t: u64) -> String {
    let mut v = Vec::new();
    if t & tags::USER   != 0 { v.push("USER"); }
    if t & tags::SYSTEM != 0 { v.push("SYSTEM"); }
    if t & tags::SHORT  != 0 { v.push("SHORT"); }
    if t & tags::LONG   != 0 { v.push("LONG"); }
    if v.is_empty() { "NONE".into() } else { v.join("|") }
}

// ─────────────────────────── embedders

// PFC retriever embedder
struct ModelEmb<B: Backend> { model: Arc<Mutex<HippocampusModel<B>>> }
impl<B: Backend> ModelEmb<B> { fn new(model: Arc<Mutex<HippocampusModel<B>>>) -> Self { Self { model } } }
impl<B: Backend> Embeddings for ModelEmb<B> {
    fn embed(&self, text: &str) -> Result<Embedding, Box<dyn Error + Send + Sync>> {
        let model = self.model.lock()
            .map_err(|_| Box::<dyn Error + Send + Sync>::from(io::Error::new(io::ErrorKind::Other, "model lock poisoned")))?;
        let z = model.run_inference(text);
        let mut v = z.to_data().as_slice::<f32>()
            .map_err(|e| Box::<dyn Error + Send + Sync>::from(io::Error::new(io::ErrorKind::Other, format!("{e:?}"))))?
            .to_vec();
        let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
        for x in &mut v { *x /= n; }
        Ok(v)
    }
}

// SMIE embedder (index vectorizer)
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
        let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
        for x in &mut v { *x /= n; }
        Ok(v)
    }
}

// kill SQLite ANN scans
struct NullVS;
impl VectorStore for NullVS {
    fn search(&self, _q: &Embedding, _k: usize) -> Result<Vec<MemoryHit>, Box<dyn Error + Send + Sync>> {
        Ok(vec![])
    }
}

// ─────────────────────────── runtime
fn main() -> anyhow::Result<()> {
    assert_primordial_invariants();

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
        SmieConfig { dim: smie_embedder.dim(), metric: Metric::Cosine },
        smie_embedder,
    )?);
    std::fs::create_dir_all("data").ok();
    let _ = smie.load("data/uriel");

    // retriever without ANN
    let emb = Arc::new(ModelEmb::new(model_for_emb.clone()));
    let retriever = ContextRetriever::new(Arc::new(NullVS), emb, None, RetrieverCfg::default());

    // routing
    let drives: std::collections::HashMap<String, SyntheticDrive> = Default::default();
    let cfg = RouteCfg::default();
    let ctx = "default";

    // frontal
    let mut wm = WorkingMemory::new(50);
    let mut _inhib = InhibitionState::default();
    let inhib_cfg = InhibitionConfig::default();
    let mut last_out: Option<CortexOutput> = None;

    // autosave
    {
        let smie_c = smie.clone();
        ctrlc::set_handler(move || { let _ = smie_c.save("data/uriel"); eprintln!("\n💾 Saved SMIE. Bye."); std::process::exit(0); })?;
    }

    println!("URIEL runtime. Type lines. Prefix with 'store this:' to save, ask 'what … ?' to retrieve.");

    // repl
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        print!("> ");
        io::stdout().flush()?;
        line.clear();
        if stdin.read_line(&mut line)? == 0 { break; }

        // sanitize the user's input first
        let raw = line.as_str();
        let input = sanitize_input(raw);
        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") { break; }

        // STORE (case-insensitive)
        let lower = input.to_ascii_lowercase();
        if lower.starts_with("store this:") {
            let rest = input.splitn(2, ':').nth(1).unwrap_or("").trim();

            // durability path (DB/cache)
            pipeline::ingest(&model_runtime, ctx, rest, 7 * 24 * 3600, "runtime")
                .map_err(|e| anyhow::anyhow!("ingest failed: {e}"))?;

            // SMIE upsert single-truths
            if let Some(con) = concept_of(rest) {
                let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
                if let Ok(old) = smie.recall(con, 32, mask) {
                    for (id, _s, _t, txt) in old {
                        if concept_of(&txt) == Some(con) { let _ = smie.tombstone(id); }
                    }
                }
            }

            // SMIE embed raw content
            let tbits = tags::USER | tags::SHORT;
            let id = smie.ingest(rest, tbits)?;
            wm.set_focus(rest);
            println!("Stored ({id}) [{}].", fmt_tags(tbits));
            continue;
        }

        // SMIE FIRST for questions — try multiple probes
        let mut answer = String::new();
        if is_question(&input) {
            let q_norm = normalize_question(&input);
            let mut probes: Vec<String> = vec![q_norm.clone()];
            if let Some(key) = concept_hint_from_norm(&q_norm) { probes.push(key.into()); }
            probes.push(input.clone());

            let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
            for q in probes {
                if let Ok(hits) = smie.recall(&q, 8, mask) {
                    if !hits.is_empty() {
                        let best = choose_best_for(&q_norm, hits).unwrap();
                        if let Some(ans) = extract_answer(&q_norm, &best.3) { answer = ans; }
                        else { answer = best.3; }
                        break;
                    }
                }
            }
        }

        // If SMIE didn’t answer, try PFC
        if answer.trim().is_empty() {
            answer = match URIELV1::core::handle::handle_input_pfc(&model_runtime, ctx, &input, &drives, &cfg, &retriever) {
                Ok(out) => out,
                Err(err) => { eprintln!("ERR: {err}"); String::new() }
            };
        }

        // Final fallback: latest SQLite fact
        if answer.trim().is_empty() && is_question(&input) {
            let qn = normalize_question(&input);
            if let Ok(rows) = recall_latest("default", 128) {
                for r in rows {
                    if let Some(ans) = extract_answer(&qn, &r.data) { answer = ans; break; }
                }
            }
            if answer.trim().is_empty() {
                answer = "(no memory yet — try `store this: Your name is URIEL.`)".into();
            }
        }

        // Frontal gating
        let candidate = CortexOutput::VerbalResponse(answer.clone());
        let fctx = FrontalContext {
            active_goals: vec![],
            current_drive_weights: &drives,
            last_response: last_out.as_ref(),
            working_memory: std::cell::RefCell::new(wm.clone()),
            inhib_state: std::cell::RefCell::new(_inhib.clone()),
            inhib_cfg: inhib_cfg.clone(),
        };

        match evaluate_frontal_control(&candidate, &fctx) {
            FrontalDecision::ExecuteImmediately(out) => {
                if let CortexOutput::VerbalResponse(text) = out.clone() { println!("{text}"); }
                else { println!("{out:?}"); }
                last_out = Some(out);
            }
            FrontalDecision::QueueForLater(out) => { eprintln!("(queued) {out:?}"); last_out = Some(out); }
            FrontalDecision::SuppressResponse(msg) => { eprintln!("{msg}"); }
            FrontalDecision::ReplaceWithGoal(msg) => { println!("{msg}"); last_out = Some(CortexOutput::Thought(msg)); }
        }
    }

    let _ = smie.save("data/uriel");
    Ok(())
}

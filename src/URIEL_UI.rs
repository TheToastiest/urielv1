// src/URIEL_UI.rs
use std::cmp::Ordering;
use std::io;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;

use eframe::egui;

use URIELV1::Hippocampus::infer::HippocampusModel;
use URIELV1::Hippocampus::pipeline;                         // durability ingest
use URIELV1::Hippocampus::sqlite_backend::recall_latest;     // fact fallback

use URIELV1::Microthalamus::microthalamus::{MicroThalamus, SyntheticDrive};
use URIELV1::Microthalamus::goal::{Ctx, Goal, GoalStatus};

use smie_core::{Embedder, Smie, SmieConfig, Metric, tags};

use crossbeam_channel::{unbounded, Receiver, Sender};
use regex::Regex;

// -------------------------- small helpers --------------------------
fn sanitize_input(s: &str) -> String {
    let mut t = s.trim();
    while let Some(c) = t.chars().next() {
        if c == '>' || c == '•' || c == '-' || c == '|' { t = t[1..].trim_start(); } else { break; }
    }
    t.to_string()
}
fn is_question(s: &str) -> bool {
    let l = s.trim().to_ascii_lowercase();
    l.ends_with('?') || l.starts_with("what ") || l.starts_with("who ")
        || l.starts_with("where ") || l.starts_with("why ") || l.starts_with("how ")
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
        ab.cmp(&aa)               // prefer containing key
            .then_with(|| b.0.cmp(&a.0)) // prefer NEWER id
            .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal))
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

// -------------------------- Embedders --------------------------

// PFC-style embedder (if you later hook PFC to UI)
struct ModelEmb<B: Backend> { model: Arc<Mutex<HippocampusModel<B>>> }
impl<B: Backend> ModelEmb<B> { fn new(model: Arc<Mutex<HippocampusModel<B>>>) -> Self { Self { model } } }
impl<B: Backend> URIELV1::PFC::context_retriever::Embeddings for ModelEmb<B> {
    fn embed(&self, text: &str) -> Result<URIELV1::PFC::context_retriever::Embedding, Box<dyn std::error::Error + Send + Sync>> {
        let model = self.model.lock()
            .map_err(|_| Box::<dyn std::error::Error + Send + Sync>::from(io::Error::new(io::ErrorKind::Other, "model lock poisoned")))?;
        let z = model.run_inference(text);
        let mut v = z.to_data().as_slice::<f32>()
            .map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(io::Error::new(io::ErrorKind::Other, format!("{e:?}"))))?
            .to_vec();
        let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
        for x in &mut v { *x /= n; }
        Ok(v)
    }
}

// SMIE index embedder
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

// -------------------------- App state --------------------------
type Bx = NdArray<f32>;

struct AppState {
    smie: Arc<Smie>,
    model: Arc<Mutex<HippocampusModel<Bx>>>,
    thalamus: MicroThalamus,
    goal_rx: Receiver<Goal>,
    drives: std::collections::HashMap<String, SyntheticDrive>,
    goals: Vec<Goal>,

    // UI
    input_text: String,
    query_text: String,
    top_k: usize,
    set_user: bool,
    set_system: bool,
    set_short: bool,
    set_long: bool,

    results: Vec<(u64, f32, u64, String)>,
    status: String,
}

impl AppState {
    fn new(smie: Arc<Smie>, model: Arc<Mutex<HippocampusModel<Bx>>>) -> Self {
        // thalamus
        let (tx, rx) = unbounded::<Goal>();
        let mut th = MicroThalamus::new(tx);
        th.register_drive(SyntheticDrive::new(
            "Curiosity", 0.25, 0.01, Duration::from_secs(3), 0.5,
            "drive:curiosity", vec![0.1,0.2,0.3,0.0,0.1,0.4,0.1,0.2],
        ));
        th.register_drive(SyntheticDrive::new(
            "ResolveConflict", 0.35, 0.01, Duration::from_secs(5), 1.0,
            "drive:resolveconflict", vec![0.1,0.2,0.3,0.0,0.1,0.4,0.1,0.2],
        ));

        Self {
            smie,
            model,
            thalamus: th,
            goal_rx: rx,
            drives: Default::default(),
            goals: Vec::new(),

            input_text: String::new(),
            query_text: String::new(),
            top_k: 5,
            set_user: true,
            set_system: false,
            set_short: true,
            set_long: false,

            results: Vec::new(),
            status: String::new(),
        }
    }
}

// -------------------------- UI entry --------------------------
pub fn main() -> Result<(), eframe::Error> {
    // init model + SMIE (same as runtime)
    let device = <Bx as Backend>::Device::default();
    let model = Arc::new(Mutex::new(HippocampusModel::<Bx>::load_or_init(
        &device,
        "R:/uriel/models/checkpoints/uriel_encoder_step008500.bin",
    )));
    let smie_embedder = Arc::new(SmieHippEmbed::new(model.clone()).expect("smie embed init"));
    let smie = Arc::new(Smie::new(
        SmieConfig { dim: smie_embedder.dim(), metric: Metric::Cosine },
        smie_embedder,
    ).expect("smie init"));

    std::fs::create_dir_all("data").ok();
    let _ = smie.load("data/uriel");

    let mut state = AppState::new(smie.clone(), model.clone());

    let options = eframe::NativeOptions::default();
    eframe::run_simple_native("URIEL UI", options, move |ctx, _frame| {
        // tick thalamus (basic)
        state.thalamus.tick();
        state.drives = state.thalamus.drives.clone();

        // drain goals from thalamus
        let mut new_goals = Vec::new();
        while let Ok(g) = state.goal_rx.try_recv() { new_goals.push(g); }
        if !new_goals.is_empty() {
            let goal_ctx = Ctx { signals: [("weight".into(), 0.35)].into_iter().collect() };
            for g in new_goals {
                let w = state.drives.iter().map(|(k,d)| (k.clone(), d.weight)).collect();
                let mut g2 = g;
                g2.score(&w, &goal_ctx);
                state.goals.push(g2);
            }
            state.goals.sort_by(|a,b| b.last_score.partial_cmp(&a.last_score).unwrap_or(Ordering::Equal));
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("URIEL UI — SMIE + MicroThalamus");
            ui.separator();

            // Drives + Goals (compact readout)
            ui.horizontal(|ui| {
                ui.label("Drives:");
                for (k,d) in &state.drives { ui.monospace(format!("{k}:{:.2}  ", d.weight)); }
            });
            ui.horizontal(|ui| {
                ui.label("Goals:");
                if state.goals.is_empty() { ui.monospace("(none)"); }
                else {
                    for g in &state.goals { ui.monospace(format!("{}[{:.2}]  ", g.label, g.last_score)); }
                }
            });

            ui.separator();
            ui.label("Ingest memory:");
            ui.text_edit_singleline(&mut state.input_text);
            ui.horizontal(|ui| {
                ui.checkbox(&mut state.set_user, "USER");
                ui.checkbox(&mut state.set_system, "SYSTEM");
                ui.checkbox(&mut state.set_short, "SHORT");
                ui.checkbox(&mut state.set_long, "LONG");

                if ui.button("📥 Ingest").clicked() {
                    let text = sanitize_input(&state.input_text);
                    if text.is_empty() {
                        state.status = "⚠️ Nothing to ingest".into();
                    } else {
                        // durability
                        if let Ok(model) = state.model.lock() {
                            let _ = pipeline::ingest(&*model, "ui_context", &text, 30, "gui");
                        }
                        // upsert in SMIE for single-truths
                        if let Some(con) = concept_of(&text) {
                            let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
                            if let Ok(old) = state.smie.recall(con, 32, mask) {
                                for (id, _s, _t, txt) in old {
                                    if concept_of(&txt) == Some(con) { let _ = state.smie.tombstone(id); }
                                }
                            }
                        }
                        // tags
                        let mut tbits = 0u64;
                        if state.set_user   { tbits |= tags::USER; }
                        if state.set_system { tbits |= tags::SYSTEM; }
                        if state.set_short  { tbits |= tags::SHORT; }
                        if state.set_long   { tbits |= tags::LONG; }

                        match state.smie.ingest(&text, tbits) {
                            Ok(id) => { state.status = format!("✅ Ingested id {id} [{}]", fmt_tags(tbits)); state.input_text.clear(); }
                            Err(e) => { state.status = format!("❌ SMIE ingest failed: {e}"); }
                        }
                    }
                }
            });

            ui.separator();
            ui.label("Recall:");
            ui.text_edit_singleline(&mut state.query_text);
            ui.horizontal(|ui| {
                ui.label("Top K:");
                ui.add(egui::Slider::new(&mut state.top_k, 1..=20));
                if ui.button("🔎 Recall").clicked() {
                    let q = sanitize_input(&state.query_text);
                    state.results.clear();
                    state.status.clear();

                    if q.is_empty() {
                        state.status = "⚠️ Empty query".into();
                    } else {
                        // SMIE-first multiprobe
                        let qn = normalize_question(&q);
                        let mut probes: Vec<String> = vec![qn.clone()];
                        if let Some(key) = concept_hint_from_norm(&qn) { probes.push(key.into()); }
                        probes.push(q.clone());

                        let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
                        let mut answered = None::<String>;

                        for probe in probes {
                            if let Ok(hits) = state.smie.recall(&probe, state.top_k.max(8), mask) {
                                if !hits.is_empty() {
                                    state.results = hits.clone();
                                    if let Some(best) = choose_best_for(&qn, hits) {
                                        answered = extract_answer(&qn, &best.3).or(Some(best.3));
                                        break;
                                    }
                                }
                            }
                        }

                        if answered.is_none() && is_question(&q) {
                            // DB fact fallback (latest rows)
                            if let Ok(rows) = recall_latest("ui_context", 128) {
                                for r in rows {
                                    if let Some(ans) = extract_answer(&qn, &r.data) { answered = Some(ans); break; }
                                }
                            }
                        }

                        state.status = answered.unwrap_or_else(|| "(no answer)".into());
                    }
                }
            });

            ui.separator();
            ui.label("Status / Answer:");
            ui.monospace(&state.status);

            ui.separator();
            ui.label("Top Hits:");
            if state.results.is_empty() {
                ui.label("—");
            } else {
                for (i,(id,score,t,text)) in state.results.iter().enumerate() {
                    ui.monospace(format!("{}. id={}  score={:.3}  [{}]\n   {}", i+1, id, score, fmt_tags(*t), text));
                    ui.add_space(6.0);
                }
            }

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("💾 Save SMIE").clicked() {
                    std::fs::create_dir_all("data").ok();
                    match state.smie.save("data/uriel") {
                        Ok(_) => state.status = "✅ Saved SMIE metadata".into(),
                        Err(e) => state.status = format!("❌ Save failed: {e}"),
                    }
                }
                if ui.button("📂 Load SMIE").clicked() {
                    match state.smie.load("data/uriel") {
                        Ok(_) => state.status = "✅ Loaded SMIE metadata".into(),
                        Err(e) => state.status = format!("❌ Load failed: {e}"),
                    }
                }
            });
        });
    })
}

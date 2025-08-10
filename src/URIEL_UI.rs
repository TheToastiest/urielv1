use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;

use URIELV1::Frontal::frontal::{evaluate_frontal_control, FrontalContext, FrontalDecision};
use URIELV1::Hippocampus::burn_model::MyBackend;
use URIELV1::Hippocampus::embedding::{generate_embedding, store_embedding};
use URIELV1::Hippocampus::ingest; // ⟵ keep UI ingest
use URIELV1::Hippocampus::primordial::primordial_memory_set;
use URIELV1::Hippocampus::redis_backend::{cache_entry, get_recent_keys};
use URIELV1::Hippocampus::sqlite_backend::{recall_all, store_entry, SemanticMemoryEntry};
use URIELV1::Microthalamus::goal::{Ctx, Goal, GoalStatus};
use URIELV1::Microthalamus::microthalamus::{MicroThalamus, SyntheticDrive};
use URIELV1::PFC::cortex::{execute_cortex, CortexAction, CortexInputFrame, CortexOutput};
use URIELV1::PFC::context_retriever::{
    ContextRetriever, Embedding, Embeddings, MemoryHit, RetrieverCfg, VectorStore,
};
use URIELV1::Parietal::parietal::{ParietalRoute, SalienceScore};
use URIELV1::Temporal::temporal::{parse_phrase, ParsedCommand};
use URIELV1::Thalamus::thalamus::{handle_input_with_thalamus, InputType};
use URIELV1::embedder::embedder::FixedWeightEmbedder;

#[derive(Default)]
struct AppState {
    input_text: String,
    recall_results: Vec<SemanticMemoryEntry>,
    output_log: Vec<String>,
    thalamus: Option<MicroThalamus>,
    device: <MyBackend as burn::tensor::backend::Backend>::Device,
    embedder: Option<FixedWeightEmbedder<MyBackend>>,
    last_response: Option<CortexOutput>,
    goals: Vec<Goal>,
    cortex_tx: Option<Sender<Goal>>,
    cortex_rx: Option<Receiver<Goal>>,
    retriever: Option<Arc<ContextRetriever>>, // <-- holds the retriever
}

fn push_goal(
    state: &mut AppState,
    mut g: Goal,
    drives: &HashMap<String, SyntheticDrive>,
    ctx: &Ctx,
) {
    let drive_vec: HashMap<String, f32> =
        drives.iter().map(|(k, v)| (k.clone(), v.weight)).collect();
    g.score(&drive_vec, ctx);
    state.goals.push(g);
}

fn rescore_goals(state: &mut AppState, drives: &HashMap<String, SyntheticDrive>, ctx: &Ctx) {
    let drive_vec: HashMap<String, f32> =
        drives.iter().map(|(k, v)| (k.clone(), v.weight)).collect();
    for g in &mut state.goals {
        g.score(&drive_vec, ctx);
    }
    state.goals.sort_by(|a, b| {
        b.last_score
            .partial_cmp(&a.last_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    state.goals.retain(|g| {
        !matches!(
            g.status,
            GoalStatus::Succeeded | GoalStatus::Failed | GoalStatus::Aborted
        )
    });
}

fn main() -> Result<(), eframe::Error> {
    // Optional: background scanner if you expose one at crate root
    if let Err(e) = URIELV1::start_scanner() {
        eprintln!("Failed to start scanner: {e}");
    }

    // 🔹 Primordial seeding — CACHE + PERSIST + EMBED (UI kept)
    println!("⏳ Primordial Memory Being Inserted:");
    for (ctx, data, ttl) in primordial_memory_set() {
        let _ = cache_entry(ctx, data, ttl);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = store_entry(ctx, data, Some(timestamp));
        if let Ok(embedding_vec) = generate_embedding(data) {
            let _ = store_embedding(ctx, &embedding_vec);
        }
    }
    println!(
        "🔍 Keys in Redis: {:?}",
        get_recent_keys("smie:").unwrap_or_default()
    );

    // 🔹 App state
    let app_state = Rc::new(RefCell::new(AppState::default()));
    {
        let (ctx_tx, ctx_rx) = unbounded::<Goal>();
        let mut thalamus = MicroThalamus::new(ctx_tx.clone());

        thalamus.register_drive(SyntheticDrive::new(
            "Curiosity",
            0.2,
            0.01,
            Duration::from_secs(3),
            0.5,
            "drive:curiosity",
            vec![0.1, 0.2, 0.3, 0.0, 0.1, 0.4, 0.1, 0.2],
        ));
        thalamus.register_drive(SyntheticDrive::new(
            "ResolveConflict",
            0.3,
            0.01,
            Duration::from_secs(5),
            1.0,
            "drive:resolveconflict",
            vec![0.1, 0.2, 0.3, 0.0, 0.1, 0.4, 0.1, 0.2],
        ));

        let mut st = app_state.borrow_mut();
        st.thalamus = Some(thalamus);
        st.device = Default::default();
        st.embedder = Some(FixedWeightEmbedder::new_with_seed(32, 64, 1337, &st.device));
        st.goals = Vec::new();
        st.cortex_tx = Some(ctx_tx);
        st.cortex_rx = Some(ctx_rx);

        // -------------------- RETRIEVER INIT --------------------
        // Adapters are defined at the bottom of this file.
        let vs = Arc::new(UiSqliteVectorStore {
            contexts: vec!["ui_context".to_string(), "primordial".to_string()],
            max_scan: 512, // tune as needed
        });
        let emb = Arc::new(UiEmbeddings {});
        let retr = Arc::new(ContextRetriever::new(
            vs,
            emb,
            None,                   // no embedding cache yet
            RetrieverCfg::default(), // min_score/topk defaults
        ));
        st.retriever = Some(retr);
        // --------------------------------------------------------
    }

    let app_state_clone = app_state.clone();
    let options = eframe::NativeOptions::default();

    eframe::run_simple_native("SMIE UI", options, move |ctx, _frame| {
        let mut state = app_state_clone.borrow_mut();

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("SMIE — Semantic Memory Ingestion Engine");
            ui.separator();

            // 🔹 Working set
            ui.heading("Working Set");
            ui.horizontal(|ui| {
                ui.label("🎯 Active goals:");
                if state.goals.is_empty() {
                    ui.label("—");
                } else {
                    for g in &state.goals {
                        ui.monospace(format!(
                            "{} [{:.2}] {:?}",
                            g.label, g.last_score, g.status
                        ));
                    }
                }
            });

            if let Some(resp) = &state.last_response {
                match resp {
                    CortexOutput::VerbalResponse(t) => {
                        ui.label(format!("🧩 Last response: {}", t));
                    }
                    CortexOutput::Action(a) => match a {
                        CortexAction::ScanItem => {
                            ui.label("🛠️ Last action: ScanItem");
                        }
                        CortexAction::ScanEnvironment => {
                            ui.label("🛠️ Last action: ScanEnvironment");
                        }
                        CortexAction::SearchInternet(q) => {
                            ui.label(format!("🛠️ Last action: SearchInternet({q})"));
                        }
                        CortexAction::RunSubmodule(name) => {
                            ui.label(format!("🛠️ Last action: RunSubmodule({name})"));
                        }
                    },
                    CortexOutput::Thought(t) => {
                        ui.label(format!("💭 Last thought: {}", t));
                    }
                    CortexOutput::Multi(list) => {
                        ui.label(format!("🧩 Multi-output ({} items)", list.len()));
                    }
                };
            }

            ui.separator();
            // 🔹 Ingest text memory (UI preserved)
            ui.label("Enter memory data:");
            ui.text_edit_singleline(&mut state.input_text);

            if ui.button("Ingest").clicked() {
                let context = "ui_context".to_string();
                let data = state.input_text.clone();
                if data.is_empty() {
                    state.output_log.push("⚠️ No input text.".into());
                } else {
                    // Thalamus + Parietal evaluation
                    let drives_map = state
                        .thalamus
                        .as_ref()
                        .map(|t| t.drives.clone())
                        .unwrap_or_default();
                    match handle_input_with_thalamus(&context, &data, &InputType::Text, &drives_map)
                    {
                        Ok((salience, route)) => {
                            state.output_log.push(format!(
                                "🧠 Salience: urgency {:.2}, familiarity {:.2}, understanding {:.2}, drive {:.2}, final {:.2}",
                                salience.urgency,
                                salience.familiarity,
                                salience.understanding,
                                salience.drive_alignment,
                                salience.final_weight
                            ));
                            state.output_log.push(format!("🧭 Route: {:?}", route));

                            if salience.final_weight < 0.2 {
                                state.output_log.push(
                                    "🔇 Thalamus deprioritized this input.".into(),
                                );
                                state.input_text.clear();
                                return;
                            }

                            match route {
                                ParietalRoute::Suppress => {
                                    state.output_log.push("🔇 Parietal routed: Suppress".into());
                                    state.input_text.clear();
                                    return;
                                }
                                ParietalRoute::StoreOnly => {
                                    state.output_log.push("🗃️ Parietal routed: StoreOnly".into());
                                }
                                ParietalRoute::Respond => {
                                    state.output_log.push("✅ Parietal routed: Respond".into());

                                    let drives_snapshot: HashMap<String, SyntheticDrive> = state
                                        .thalamus
                                        .as_ref()
                                        .map(|t| t.drives.clone())
                                        .unwrap_or_default();

                                    let frame = CortexInputFrame {
                                        raw_input: &data,
                                        score: salience.clone(),
                                        drives: &drives_snapshot,
                                    };

                                    // ⬇️ Borrow the retriever from state
                                    let retr = state
                                        .retriever
                                        .as_ref()
                                        .expect("ContextRetriever not initialized");
                                    let cortex_out = execute_cortex(frame, retr);

                                    let goal_ctx = Ctx {
                                        signals: [
                                            ("urgency".into(), salience.urgency),
                                            ("familiarity".into(), salience.familiarity),
                                            ("understanding".into(), salience.understanding),
                                            ("drive".into(), salience.drive_alignment),
                                            ("weight".into(), salience.final_weight),
                                        ]
                                            .into_iter()
                                            .collect(),
                                    };

                                    // drain async goals from MicroThalamus → UI without immutable borrow during mutation
                                    let pending_goals: Vec<Goal> =
                                        if let Some(rx_ref) = state.cortex_rx.as_ref() {
                                            let rx = rx_ref.clone();
                                            let mut v = Vec::new();
                                            while let Ok(g) = rx.try_recv() {
                                                v.push(g);
                                            }
                                            v
                                        } else {
                                            Vec::new()
                                        };
                                    for g in pending_goals {
                                        push_goal(&mut state, g, &drives_snapshot, &goal_ctx);
                                        state
                                            .output_log
                                            .push("🎯 New goal from Thalamus".into());
                                    }

                                    let fctx = FrontalContext {
                                        active_goals: state
                                            .goals
                                            .iter()
                                            .map(|g| g.label.clone())
                                            .collect(),
                                        current_drive_weights: &drives_snapshot,
                                        last_response: state.last_response.as_ref(),
                                    };

                                    match evaluate_frontal_control(&cortex_out, &fctx) {
                                        FrontalDecision::ExecuteImmediately(out) => {
                                            match &out {
                                                CortexOutput::VerbalResponse(text) => {
                                                    ui.label(format!("🧩 Response: {}", text));
                                                }
                                                CortexOutput::Action(a) => match a {
                                                    CortexAction::ScanItem => {
                                                        ui.label("🛠️ Action: ScanItem");
                                                    }
                                                    CortexAction::ScanEnvironment => {
                                                        ui.label("🛠️ Action: ScanEnvironment");
                                                    }
                                                    CortexAction::SearchInternet(q) => {
                                                        ui.label(format!(
                                                            "🛠️ Action: SearchInternet({q})"
                                                        ));
                                                    }
                                                    CortexAction::RunSubmodule(name) => {
                                                        ui.label(format!(
                                                            "🛠️ Action: RunSubmodule({name})"
                                                        ));
                                                    }
                                                },
                                                CortexOutput::Thought(t) => {
                                                    ui.label(format!("💭 Thought: {}", t));
                                                }
                                                CortexOutput::Multi(list) => {
                                                    ui.label(format!(
                                                        "🧩 Multi-output ({} items)",
                                                        list.len()
                                                    ));
                                                }
                                            };
                                            state.last_response = Some(out);
                                        }
                                        FrontalDecision::QueueForLater(out) => {
                                            state.output_log.push("⏳ Queued response.".into());
                                            state.last_response = Some(out);
                                        }
                                        FrontalDecision::SuppressResponse(reason) => {
                                            state.output_log
                                                .push(format!("🔇 Suppressed: {reason}"));
                                        }
                                        FrontalDecision::ReplaceWithGoal(goal_text) => {
                                            let gid = (state.goals.len() as u64) + 1;
                                            let label = goal_text.clone();
                                            let g = Goal::new(
                                                gid,
                                                label,
                                                |dr, ctx| {
                                                    let c = *dr.get("Curiosity").unwrap_or(&0.0);
                                                    let r =
                                                        *dr.get("ResolveConflict").unwrap_or(&0.0);
                                                    (c * 0.6 + r * 0.4)
                                                        * (ctx.signals
                                                        .get("weight")
                                                        .copied()
                                                        .unwrap_or(0.2)
                                                        + 0.1)
                                                },
                                                |ctx| {
                                                    ctx.signals
                                                        .get("conflict")
                                                        .copied()
                                                        .unwrap_or(1.0)
                                                        < 0.1
                                                },
                                            )
                                                .with_decay(0.02);

                                            push_goal(&mut state, g, &drives_snapshot, &goal_ctx);
                                            state
                                                .output_log
                                                .push(format!("🎯 Goal queued: {}", goal_text));
                                        }
                                    }

                                    if let Some(thalamus) = state.thalamus.as_mut() {
                                        thalamus.inject_stimulus(&context, &data);
                                        thalamus.tick();
                                        let logs: Vec<String> = thalamus
                                            .drives
                                            .iter()
                                            .map(|(name, d)| {
                                                format!(
                                                    "🧠 Drive: {} → weight {:.2}",
                                                    name, d.weight
                                                )
                                            })
                                            .collect();
                                        state.output_log.extend(logs);
                                    }

                                    // re-score goals with snapshot & try completion
                                    let drives_snapshot2: HashMap<String, SyntheticDrive> =
                                        state
                                            .thalamus
                                            .as_ref()
                                            .map(|t| t.drives.clone())
                                            .unwrap_or_default();
                                    rescore_goals(&mut state, &drives_snapshot2, &goal_ctx);
                                    if let Some(label) = {
                                        if let Some(top) = state.goals.first_mut() {
                                            if top.try_finish(&goal_ctx) {
                                                Some(top.label.clone())
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        }
                                    } {
                                        state
                                            .output_log
                                            .push(format!("✅ Goal succeeded: {}", label));
                                    }
                                }
                            }

                            // 🔹 UI-level ingest trace (explicit, beyond handle_input)
                            let _ = ingest("ui_context".to_string(), data.clone(), 30, String::from("gui"));
                        }
                        Err(e) => state.output_log.push(format!("❌ Thalamus error: {}", e)),
                    }
                }
            }

            // 🔹 Voice command (kept)
            ui.separator();
            ui.label("Simulated Voice Command:");
            let mut voice_input = String::new();
            ui.text_edit_singleline(&mut voice_input);
            if ui.button("Process voice").clicked() {
                if !voice_input.trim().is_empty() {
                    let known_drives = vec!["curiosity", "resolve", "conflict"];
                    let parsed: ParsedCommand = parse_phrase(&voice_input, &known_drives);
                    state.output_log.clear();
                    state
                        .output_log
                        .push(format!("🎧 Addressed to URIEL: {}", parsed.addressed_to));
                    if let Some(cmd) = &parsed.command {
                        state.output_log.push(format!("🗣️ Command: {}", cmd));
                    }
                    if !parsed.args.is_empty() {
                        state.output_log.push(format!("🔣 Args: {:?}", parsed.args));
                    }
                    for (drive, weight) in parsed.drive_alignment.iter() {
                        state
                            .output_log
                            .push(format!("🧠 Drive Match: {} → {:.2}", drive, weight));
                    }
                    state
                        .output_log
                        .push(format!("⏱️ Timestamp: {}", parsed.timestamp));
                }
            }

            // 🔹 Recall
            if ui.button("Recall memory").clicked() {
                state.recall_results.clear();
                state.output_log.clear();
                if let Ok(mut entries) = recall_all("ui_context") {
                    entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
                    state.recall_results = entries.clone();
                    for entry in entries {
                        state
                            .output_log
                            .push(format!("- [{}] {}", entry.timestamp, entry.data));
                    }
                } else {
                    ui.label("⚠️ Failed to recall UI context memory.");
                }
            }

            if ui.button("Recall primordials").clicked() {
                state.recall_results.clear();
                state.output_log.clear();
                if let Ok(mut entries) = recall_all("primordial") {
                    entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
                    state.recall_results = entries.clone();
                    for entry in entries {
                        state
                            .output_log
                            .push(format!("- [{}] {}", entry.timestamp, entry.data));
                    }
                } else {
                    ui.label("⚠️ Failed to recall primordial memory.");
                }
            }

            ui.separator();
            ui.label("Recalled / Log:");
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .max_height(300.0)
                .show(ui, |ui| {
                    if state.output_log.is_empty() {
                        ui.label("No entries found.");
                    } else {
                        for line in &state.output_log {
                            ui.label(line);
                        }
                    }
                });
        });
    })
}

// -------------------- UI ADAPTERS FOR CONTEXT RETRIEVAL --------------------
use std::error::Error;

/// Cosine helper (expects unit-normalized query; we normalize items here)
fn cosine(a: &[f32], mut b: Vec<f32>) -> f32 {
    // normalize b
    let n = (b.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
    for x in b.iter_mut() {
        *x /= n;
    }
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    dot.clamp(0.0, 1.0)
}

/// Embeddings adapter that wraps your generate_embedding()
struct UiEmbeddings;

impl Embeddings for UiEmbeddings {
    fn embed(&self, text: &str) -> Result<Embedding, Box<dyn Error + Send + Sync>> {
        let mut v = generate_embedding(text).map_err(|e| format!("embed error: {e}"))?;
        // normalize to unit
        let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
        for x in v.iter_mut() {
            *x /= n;
        }
        Ok(v)
    }
}

/// Vector "store" that reads recent rows from SQLite and scores them on the fly.
/// This is a pragmatic bridge until FAISS/ANN is wired.
struct UiSqliteVectorStore {
    pub contexts: Vec<String>,
    pub max_scan: usize,
}

impl VectorStore for UiSqliteVectorStore {
    fn search(
        &self,
        query: &Embedding,
        top_k: usize,
    ) -> Result<Vec<MemoryHit>, Box<dyn Error + Send + Sync>> {
        let mut rows: Vec<(String, String)> = Vec::new();

        // Pull recent memory from the configured contexts
        for ctx in &self.contexts {
            if let Ok(mut entries) = recall_all(ctx) {
                // newest first
                entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
                for e in entries.into_iter().take(self.max_scan) {
                    rows.push((format!("{}:{}", ctx, e.timestamp), e.data));
                }
            }
        }

        // Score by cosine(query, embed(text))
        let mut scored: Vec<MemoryHit> = Vec::with_capacity(rows.len());
        for (id, text) in rows {
            if let Ok(vec) = generate_embedding(&text) {
                let sim = cosine(query, vec);
                scored.push(MemoryHit { id, sim, text });
            }
        }

        // sort & top-k
        scored.sort_by(|a, b| {
            b.sim
                .partial_cmp(&a.sim)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if scored.len() > top_k {
            scored.truncate(top_k);
        }
        Ok(scored)
    }
}
// --------------------------------------------------------------------------

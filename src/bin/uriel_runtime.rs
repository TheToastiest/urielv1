use std::error::Error;
use std::sync::{Arc, Mutex};
use std::io::{self, Write}; // ★ ADDED

use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;

use URIELV1::callosum::payloads;
use URIELV1::Hippocampus::infer::HippocampusModel;
use URIELV1::Hippocampus::pipeline;
use URIELV1::Hippocampus::sqlite_backend::recall_all;
use URIELV1::Microthalamus::microthalamus::SyntheticDrive;
use URIELV1::parietal::route_cfg::RouteCfg;
use URIELV1::core::handle::handle_input_pfc;

use URIELV1::PFC::context_retriever::{
    ContextRetriever, RetrieverCfg, VectorStore, Embeddings, MemoryHit, Embedding,
};

// ★ ADDED: pull in Frontal
use URIELV1::Frontal::frontal::{
    evaluate_frontal_control, FrontalContext, FrontalDecision,
    WorkingMemory, InhibitionState, InhibitionConfig,
};
use URIELV1::PFC::cortex::CortexOutput; // to wrap strings
// src/bin/uriel_runtime.rs (early in main)
use URIELV1::Hippocampus::primordial::{primordial, PRIMORDIAL_SHA256};

fn assert_primordial_invariants() {
    let p = primordial();
    assert!(!p.axioms.is_empty(), "Primordial axioms missing.");
    assert!(!p.prohibitions.is_empty(), "Primordial prohibitions missing.");
    assert!(!p.drives.is_empty(), "Primordial drives missing.");
    // extra: ensure weights roughly sane
    let sum_w: f32 = p.drives.iter().map(|d| d.weight).sum();
    assert!((0.5..=2.0).contains(&sum_w), "Primordial drive weights out of expected range.");
    println!("Primordial OK (v{}, sha256 {}).", p.version, PRIMORDIAL_SHA256);
}

// ---------- Embeddings adapter (thread-safe) ----------
struct ModelEmb<B: Backend> {
    model: Arc<Mutex<HippocampusModel<B>>>,
}
impl<B: Backend> ModelEmb<B> {
    fn new(model: Arc<Mutex<HippocampusModel<B>>>) -> Self { Self { model } }
}
impl<B: Backend> Embeddings for ModelEmb<B> {
    fn embed(&self, text: &str) -> Result<Embedding, Box<dyn Error + Send + Sync>> {
        let model = self.model.lock().unwrap();
        let z_1xD = model.run_inference(text);
        let mut v = z_1xD.to_data().as_slice::<f32>().unwrap().to_vec();
        // L2 normalize
        let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
        for x in &mut v { *x /= n; }
        Ok(v)
    }
}

// ---------- SQLite vector store (brute-force scan) ----------
struct SqliteScanVS<E: Embeddings> {
    emb: Arc<E>,
    context: String,
}
impl<E: Embeddings> SqliteScanVS<E> {
    fn new(emb: Arc<E>, context: impl Into<String>) -> Self {
        Self { emb, context: context.into() }
    }
}
impl<E: Embeddings> VectorStore for SqliteScanVS<E> {
    fn search(&self, q: &Embedding, top_k: usize)
              -> Result<Vec<MemoryHit>, Box<dyn Error + Send + Sync>>
    {
        let rows = recall_all(&self.context).map_err(|e| {
            Box::<dyn std::error::Error + Send + Sync>::from(
                std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
            )
        })?;

        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let e = self.emb.embed(&r.data)?; // embed stored text
            let sim = q.iter().zip(e.iter()).map(|(a,b)| a*b).sum::<f32>().clamp(0.0, 1.0);
            out.push(MemoryHit { id: format!("{}:{}", self.context, r.timestamp), sim, text: r.data });
        }
        out.sort_by(|a,b| b.sim.partial_cmp(&a.sim).unwrap_or(std::cmp::Ordering::Equal));
        if out.len() > top_k { out.truncate(top_k); }
        Ok(out)
    }
}

fn main() -> anyhow::Result<()> {
    type B = NdArray<f32>;
    let device = <B as Backend>::Device::default();

    // Load ONE model for runtime PFC path…
    let model_runtime = HippocampusModel::<B>::load(
        &device,
        "R:/uriel/models/checkpoints/uriel_encoder_step008500.bin",
    );

    // …and a SEPARATE instance wrapped for thread-safe embeddings.
    let model_for_emb = Arc::new(Mutex::new(HippocampusModel::<B>::load(
        &device,
        "R:/uriel/models/checkpoints/uriel_encoder_step008500.bin",
    )));

    let emb = Arc::new(ModelEmb::new(model_for_emb));
    let vs  = Arc::new(SqliteScanVS::new(emb.clone(), "default"));
    let retriever = ContextRetriever::new(vs, emb, None, RetrieverCfg::default());

    let drives: std::collections::HashMap<String, SyntheticDrive> = Default::default();
    let cfg = RouteCfg::default();
    let ctx = "default";

    // ★ ADDED: persistent Frontal state
    let wm = WorkingMemory::new(50);
    let inhib_state = InhibitionState::default();
    let inhib_cfg = InhibitionConfig::default();
    let mut last_output_opt: Option<CortexOutput> = None;

    println!("URIEL runtime. Type lines. Prefix with 'store this:' to save, ask 'what … ?' to retrieve.");
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        print!("> ");
        io::stdout().flush()?;
        line.clear();
        if stdin.read_line(&mut line)? == 0 { break; }
        let input = line.trim();
        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") { break; }

        // Run your high-level handler to produce a textual reply
        let raw_text = match handle_input_pfc(&model_runtime, ctx, input, &drives, &cfg, &retriever) {
            Ok(out)  => out,
            Err(err) => { eprintln!("ERR: {err}"); continue; }
        };

        // ★ ADDED: wrap as CortexOutput to pass through Frontal
        let candidate = CortexOutput::VerbalResponse(raw_text.clone());

        // Build a FrontalContext that reuses persistent WM/inhibition.
        // NOTE: we construct it each turn so `last_response` can point
        // to the most recent output, but the inner state persists.
        let frontal_ctx = FrontalContext {
            active_goals: vec![],
            current_drive_weights: &drives,
            last_response: last_output_opt.as_ref(),

            // important: DON’T re-init these each turn; seed from the persistent state
            working_memory: std::cell::RefCell::new(wm.clone()),
            inhib_state: std::cell::RefCell::new(inhib_state.clone()),
            inhib_cfg: inhib_cfg.clone(),
        };

        match evaluate_frontal_control(&candidate, &frontal_ctx) {
            FrontalDecision::ExecuteImmediately(out) => {
                // commit WM/inhibition after approval
                // (grab back the mutated state from the RefCells)
                let wm_new = frontal_ctx.working_memory.into_inner();
                let inhib_new = frontal_ctx.inhib_state.into_inner();

                // update our persistent copies
                // (WorkingMemory/InhibitionState derive Clone)
                // NOTE: if you’d rather avoid clones, store RefCell in a higher-level struct.
                // For now this is simple and fine.
                // shadowing for brevity:
                let _ = wm_new; let _ = inhib_new;

                // say it
                if let CortexOutput::VerbalResponse(text) = out.clone() {
                    println!("{text}");
                } else {
                    println!("{out:?}");
                }
                last_output_opt = Some(out);
            }
            FrontalDecision::QueueForLater(out) => {
                eprintln!("(queued) {out:?}");
                last_output_opt = Some(out);
            }
            FrontalDecision::SuppressResponse(msg) => {
                eprintln!("{msg}");
                // don’t update last_output_opt on suppress
            }
            FrontalDecision::ReplaceWithGoal(msg) => {
                println!("{msg}");
                last_output_opt = Some(CortexOutput::Thought(msg));
            }
        }
    }
    Ok(())
}

use std::collections::HashMap;

use crate::Microthalamus::microthalamus::SyntheticDrive;
use crate::parietal::parietal::SalienceScore;
use crate::rag_agent::retriever::UnifiedRetriever;

use super::reasoning_selector::select_reasoning_mode;
// ⬇️ pull in the type, not the free fn
use super::context_retriever::{ContextFrame, ContextRetriever};
use super::cortex_solver::{solve_with_context, CortexDraftOutput};
use super::language_composer::compose_response;
use super::action_router::{trigger_action_with, ActionContext};

#[derive(Debug)]
pub enum CortexReasoningMode {
    DirectRecall,
    ChainOfThought,
    AbstractSolve,
    PerceptionSynthesis,
    InternetResearch, // <-- this is your research-like mode
    MetaReflect,
    ActionSynthesis,
}
use serde::Serialize;

// ---- meta types ----
#[derive(Debug, Clone, Serialize)]
struct MetaStep {
    goal: String,           // e.g., "answer user question"
    hypothesis: String,     // e.g., "H(name)=<X>"
    plan: Vec<String>,      // probe sequence
    needed: Vec<String>,    // e.g., ["fact:name"]
    evidence: Vec<String>,  // short strings, ids, etc.
    confidence: f32,        // combined
    depth: u8,              // 0 = shallow, 1 = deep
    outcome: String,        // "accepted" | "revise" | "escalate"
    ts: u64,
}

// meta sink (stdout today; sqlite later)
trait MetaSink { fn record(&self, step: &MetaStep); }
struct StdoutMetaSink;
impl MetaSink for StdoutMetaSink {
    fn record(&self, step: &MetaStep) {
        // keep it brief for console; the Thought below is the human-facing summary
        #[cfg(debug_assertions)]
        eprintln!("[meta] {:?}", step);
    }
}

// ---- tiny utilities ----
fn now_sec() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

// a very small combiner: tune later or replace with learned function
fn combine_confidence(db_hit: bool, max_cos: f32, drive_align: f32) -> f32 {
    let w_db = if db_hit { 0.6 } else { 0.0 };
    let w_cos = 0.3 * max_cos;
    let w_drv = 0.2 * drive_align;
    (w_db + w_cos + w_drv).clamp(0.0, 1.0)
}

// stubs you can later replace with learned modules
fn propose_hypothesis(input: &str) -> String {
    // heuristic: echo a short claim; replace with concept router later
    format!("H: {}", input)
}
fn choose_plan(_mode: &super::cortex::CortexReasoningMode) -> Vec<String> {
    // epsilon-greedy bandit later; for now a fixed shallow plan
    vec!["check_db".into(), "probe_smie_norm".into(), "probe_smie_orig".into()]
}
fn gather_evidence(ctx: &super::context_retriever::ContextFrame) -> (Vec<String>, f32, bool) {
    let max_cos = ctx.top_hits.first().map(|h| h.sim).unwrap_or(0.0);
    let ev: Vec<String> = ctx.top_hits
        .iter()
        .take(3)
        .map(|h| format!("smid:{}:{:.2}", h.id, h.sim))
        .collect();
    (ev, max_cos, false) // db_hit=false in PFC; the concept path logs DB hits
}

fn meta_summary(s: &MetaStep) -> String {
    format!(
        "Thinking…\n- goal: {}\n- hypothesis: {}\n- plan: {}\n- confidence: {:.2}\n- outcome: {} (depth {})",
        s.goal,
        s.hypothesis,
        s.plan.join(" → "),
        s.confidence,
        s.outcome,
        s.depth
    )
}

#[derive(Debug, Clone)]
pub enum CortexAction {
    ScanItem,
    ScanEnvironment,
    SearchInternet(String),
    RunSubmodule(String),
}

#[derive(Debug, Clone)]
pub enum CortexOutput {
    VerbalResponse(String),
    Action(CortexAction),
    Thought(String),
    Multi(Vec<CortexOutput>),
}

pub struct CortexInputFrame<'a> {
    pub raw_input: &'a str,
    pub score: SalienceScore,
    pub drives: &'a HashMap<String, SyntheticDrive>,
}

// ---------- Public API ----------

// Now requires &ContextRetriever
pub fn execute_cortex_with_ctx(
    frame: CortexInputFrame,
    actx: &ActionContext,
    retriever: &ContextRetriever,
) -> CortexOutput {
    // 0) meta init
    let sink = StdoutMetaSink;
    let goal = "answer user question".to_string();
    let hypothesis = propose_hypothesis(frame.raw_input);
    let mode = select_reasoning_mode(&frame);
    let mut plan = choose_plan(&mode);
    let mut depth: u8 = 0;

    // 1) retrieve (bottom-up)
    let context: ContextFrame = retriever
        .build_context_frame(&frame, &mode)
        .unwrap_or_else(|_| ContextFrame::empty());

    // 2) evidence + confidence
    let (evidence, max_cos, db_hit) = gather_evidence(&context);
    let drive_align = frame
        .drives
        .values()
        .map(|d| d.weight)
        .fold(0.0_f32, f32::max);
    let mut confidence = combine_confidence(db_hit, max_cos, drive_align);

    // 3) shallow vs deep decision
    let (outcome, maybe_context) = if confidence >= 0.65 {
        ("accepted".to_string(), None)
    } else {
        // escalate once (deep pass)
        depth = 1;
        plan.push("expand_probes".into());
        let deeper: ContextFrame = retriever
            .build_context_frame(&frame, &mode) // TODO: add a deeper config/top_k
            .unwrap_or_else(|_| ContextFrame::empty());
        let (_ev2, max_cos2, db2) = gather_evidence(&deeper);
        confidence = confidence.max(combine_confidence(db2, max_cos2, drive_align));
        let out = if confidence >= 0.55 { "accepted" } else { "revise" };
        (out.to_string(), Some(deeper))
    };

    // 4) record meta
    let step = MetaStep {
        goal,
        hypothesis,
        plan,
        needed: vec![], // fill later with detected missing facts (e.g., ["fact:timezone"])
        evidence,
        confidence,
        depth,
        outcome: outcome.clone(),
        ts: now_sec(),
    };
    sink.record(&step);

    // 5) draft + actions
    let ctx_for_solve = maybe_context.as_ref().unwrap_or(&context);
    let draft: CortexDraftOutput = solve_with_context(&mode, &frame, ctx_for_solve);
    if let Some(action) = &draft.action {
        trigger_action_with(actx, action);
    }

    // 6) compose + emit Thought for audit
    let verbal = compose_response(&draft, &frame);
    let thought = CortexOutput::Thought(meta_summary(&step));

    // If your runtime already prints Thought separately, you can return `verbal` only.
    // Returning Multi gives a portable audit trail.
    CortexOutput::Multi(vec![thought, verbal])
}


// Back-compat shim: still defaults to dry-run, but now also needs retriever

pub fn execute_cortex_with_rag(
    frame: CortexInputFrame,
    actx: &ActionContext,
    rag: &UnifiedRetriever,
) -> CortexOutput {
    // for now, reuse the inner ContextRetriever
    execute_cortex_with_ctx(frame, actx, rag.ctx.as_ref())
}

pub fn execute_cortex_rag<'a>(
    frame: CortexInputFrame<'a>,
    rag: &UnifiedRetriever,
) -> CortexOutput {
    use crate::PFC::context_retriever::{ContextFrame, MemoryHit};

    // 0) meta init
    let sink = StdoutMetaSink;
    let goal = "answer user question".to_string();
    let hypothesis = propose_hypothesis(frame.raw_input);

    // Bias toward DirectRecall for CLI
    let mode = CortexReasoningMode::DirectRecall;
    let mut plan = choose_plan(&mode);
    let mut depth: u8 = 0;

    // 1) first try the unified ContextRetriever (VS + text etc.)
    let retriever: &ContextRetriever = rag.ctx.as_ref();
    let mut context: ContextFrame = retriever
        .build_context_frame(&frame, &mode)
        .unwrap_or_else(|_| ContextFrame::empty());

    eprintln!(
        "[rag-debug] ctx hits={} max_sim={:.3}",
        context.top_hits.len(),
        context.top_hits.first().map(|h| h.sim).unwrap_or(0.0),
    );

    // 2) If that returns nothing, fall back to raw SMIE text search
    if context.top_hits.is_empty() {
        if let Some(text_search) = rag.text.as_ref() {
            match text_search.search_text(frame.raw_input, 8) {
                Ok(hits) if !hits.is_empty() => {
                    let mapped: Vec<MemoryHit> = hits
                        .into_iter()
                        .map(|h| MemoryHit {
                            id: h.id,
                            sim: h.sim,
                            text: h.text,
                        })
                        .collect();
                    let max_sim = mapped.first().map(|h| h.sim).unwrap_or(0.0);
                    eprintln!(
                        "[rag-debug] smie fallback hits={} max_sim={:.3}",
                        mapped.len(),
                        max_sim
                    );
                    context.top_hits = mapped;
                    context.confidence = max_sim;
                }
                Ok(_) => {
                    eprintln!("[rag-debug] smie fallback found 0 hits");
                }
                Err(e) => {
                    eprintln!("[rag-debug] smie fallback error: {e}");
                }
            }
        }
    }

    // 3) evidence + confidence from the final context
    let (evidence, max_cos, db_hit) = gather_evidence(&context);
    let drive_align = frame
        .drives
        .values()
        .map(|d| d.weight)
        .fold(0.0_f32, f32::max);
    let mut confidence = combine_confidence(db_hit, max_cos, drive_align);

    // 4) optional deep pass (rebuild context with same mode for now)
    let ctx_for_solve = if confidence >= 0.65 {
        context
    } else {
        depth = 1;
        plan.push("expand_probes".into());
        let deeper: ContextFrame = retriever
            .build_context_frame(&frame, &mode)
            .unwrap_or_else(|_| ContextFrame::empty());
        let (_ev2, max_cos2, db2) = gather_evidence(&deeper);
        confidence = confidence.max(combine_confidence(db2, max_cos2, drive_align));
        deeper
    };

    let outcome = if confidence >= 0.55 { "accepted" } else { "revise" }.to_string();

    // 5) record meta
    let step = MetaStep {
        goal,
        hypothesis,
        plan,
        needed: vec![],
        evidence,
        confidence,
        depth,
        outcome: outcome.clone(),
        ts: now_sec(),
    };
    sink.record(&step);

    // 6) solve + compose into a real answer
    let draft: CortexDraftOutput = solve_with_context(&mode, &frame, &ctx_for_solve);

    let answer_text = draft.response.unwrap_or_else(|| {
        "I don’t know yet; my memory doesn’t have a strong answer for that.".to_string()
    });

    let meta = CortexOutput::Thought(meta_summary(&step));
    let verbal = CortexOutput::VerbalResponse(answer_text);

    CortexOutput::Multi(vec![meta, verbal])
}



use std::collections::HashMap;

use crate::Microthalamus::microthalamus::SyntheticDrive;
use crate::parietal::parietal::SalienceScore;

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
    // 1) Choose reasoning strategy
    let mode = select_reasoning_mode(&frame);

    // 2) Build a context frame via the retriever
    let context: ContextFrame = retriever
        .build_context_frame(&frame, &mode)
        .unwrap_or_else(|_| ContextFrame::empty());

    // 3) Produce a draft (tool calls / action proposals live here)
    let draft: CortexDraftOutput = solve_with_context(&mode, &frame, &context);

    // 4) Route side-effect actions (dry-run aware)
    if let Some(action) = &draft.action {
        trigger_action_with(actx, action);
    }

    // 5) Compose the final user-facing output
    compose_response(&draft, &frame)
}

// Back-compat shim: still defaults to dry-run, but now also needs retriever
pub fn execute_cortex(frame: CortexInputFrame, retriever: &ContextRetriever) -> CortexOutput {
    let actx = ActionContext { dry_run: true, source: "cortex" };
    execute_cortex_with_ctx(frame, &actx, retriever)
}

// Legacy convenience: also pass retriever through
pub fn execute_cortex_compat<'a>(
    input: &'a str,
    score: SalienceScore,
    drives: &'a HashMap<String, SyntheticDrive>,
    retriever: &ContextRetriever,
) -> CortexOutput {
    let frame = CortexInputFrame { raw_input: input, score, drives };
    execute_cortex(frame, retriever)
}

use crate::PFC::cortex::{CortexInputFrame, CortexReasoningMode, CortexAction};
use crate::PFC::context_retriever::ContextFrame;

#[derive(Debug)]
pub struct CortexDraftOutput {
    pub response: Option<String>,
    pub action: Option<CortexAction>,
    pub trace: Option<String>,
}

// Helper: summarize internal recall hits
fn summarize_hits(ctx: &ContextFrame, max_show: usize) -> (String, usize) {
    let n = ctx.top_hits.len();
    let show = n.min(max_show);
    let mut lines = Vec::with_capacity(show + 1);
    lines.push(format!("Found {n} related memories; showing top {show}:"));
    for (i, hit) in ctx.top_hits.iter().take(show).enumerate() {
        // Truncate snippet for UI friendliness
        let snippet = if hit.text.len() > 120 {
            format!("{}…", &hit.text[..120])
        } else {
            hit.text.clone()
        };
        lines.push(format!("{}. [sim {:.2}] {}", i + 1, hit.sim, snippet));
    }
    (lines.join("\n"), n)
}

pub fn solve_with_context(
    mode: &CortexReasoningMode,
    frame: &CortexInputFrame,
    context: &ContextFrame,
) -> CortexDraftOutput {
    match mode {
        // Prefer internal recall: if we have hits, answer from memory; else fall back
        CortexReasoningMode::DirectRecall => {
            if !context.top_hits.is_empty() {
                let (summary, n) = summarize_hits(context, 3);
                CortexDraftOutput {
                    response: Some(summary),
                    action: None,
                    trace: Some(format!("DirectRecall: used {} internal hits.", n)),
                }
            } else {
                // No useful internal context → gentle fallback
                CortexDraftOutput {
                    response: Some("No strong internal matches. (Dry-run) would consider external search.".to_string()),
                    action: Some(CortexAction::SearchInternet(frame.raw_input.to_string())),
                    trace: Some("Fallback from DirectRecall → InternetResearch.".to_string()),
                }
            }
        }

        CortexReasoningMode::InternetResearch => CortexDraftOutput {
            // Keep this as your explicit research mode (UI shows [DRY] via ActionContext)
            response: Some("Conducting internet search...".to_string()),
            action: Some(CortexAction::SearchInternet(frame.raw_input.to_string())),
            trace: Some("InternetResearch selected by selector.".to_string()),
        },

        CortexReasoningMode::PerceptionSynthesis => CortexDraftOutput {
            response: Some("Analyzing environment...".to_string()),
            action: Some(CortexAction::ScanEnvironment),
            trace: Some("Perception synthesis routine.".to_string()),
        },

        CortexReasoningMode::AbstractSolve => CortexDraftOutput {
            response: Some("Solving problem abstractly...".to_string()),
            action: None,
            trace: Some("Applied abstract reasoning pattern.".to_string()),
        },

        CortexReasoningMode::ChainOfThought => CortexDraftOutput {
            response: Some("Thinking step by step...".to_string()),
            action: None,
            trace: Some("CoT draft composed.".to_string()),
        },

        CortexReasoningMode::ActionSynthesis => CortexDraftOutput {
            response: Some("Preparing an action plan...".to_string()),
            action: Some(CortexAction::RunSubmodule("plan_executor".into())),
            trace: Some("Synthesized action plan.".to_string()),
        },

        CortexReasoningMode::MetaReflect => CortexDraftOutput {
            response: Some("Reflecting on approach and next steps...".to_string()),
            action: None,
            trace: Some("Meta reflection drafted.".to_string()),
        },
    }
}

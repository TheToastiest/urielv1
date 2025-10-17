use crate::PFC::cortex::{CortexOutput, CortexInputFrame};
use crate::PFC::cortex_solver::CortexDraftOutput;

pub fn compose_response(draft: &CortexDraftOutput, _frame: &CortexInputFrame) -> CortexOutput {
    // If a tool/action was selected, emit it directly
    if let Some(a) = &draft.action {
        return CortexOutput::Action(a.clone());
    }

    // Otherwise emit scaffold as a Thought. Runtime will *show it*,
    // but still fallback to SMIE/DB to produce a final answer.
    CortexOutput::Thought(
        "Thinking step by step…\n- First, recall the most relevant memory.\n- Next, relate it to the question and produce an answer."
            .into(),
    )
}
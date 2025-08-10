use crate::PFC::cortex::{CortexOutput, CortexInputFrame};
use crate::PFC::cortex_solver::CortexDraftOutput;

pub fn compose_response(draft: &CortexDraftOutput, _frame: &CortexInputFrame) -> CortexOutput {
    if let Some(response) = &draft.response {
        CortexOutput::VerbalResponse(response.clone())
    } else if let Some(action) = &draft.action {
        CortexOutput::Action(action.clone())
    } else {
        CortexOutput::Thought("No direct response generated.".to_string())
    }
}



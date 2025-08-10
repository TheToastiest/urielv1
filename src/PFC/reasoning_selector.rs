use crate::PFC::cortex::{CortexReasoningMode, CortexInputFrame};

pub fn select_reasoning_mode(frame: &CortexInputFrame) -> CortexReasoningMode {
    let input = frame.raw_input.to_lowercase();

    if input.contains("search") || input.contains("internet") {
        CortexReasoningMode::InternetResearch
    } else if input.contains("scan") {
        CortexReasoningMode::PerceptionSynthesis
    } else if input.contains("solve") || input.contains("equation") {
        CortexReasoningMode::AbstractSolve
    } else {
        CortexReasoningMode::ChainOfThought
    }
}

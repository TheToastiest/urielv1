use crate::PFC::cortex::{CortexOutput, CortexAction};
use crate::Microthalamus::microthalamus::SyntheticDrive;

use std::collections::HashMap;

#[derive(Debug)]
pub enum FrontalDecision {
    ExecuteImmediately(CortexOutput),
    QueueForLater(CortexOutput),
    SuppressResponse(String),
    ReplaceWithGoal(String),
}

#[derive(Debug)]
pub struct FrontalContext<'a> {
    pub active_goals: Vec<String>,
    pub current_drive_weights: &'a HashMap<String, SyntheticDrive>,
    pub last_response: Option<&'a CortexOutput>,
}

pub fn evaluate_frontal_control(output: &CortexOutput, context: &FrontalContext) -> FrontalDecision {
    // Very basic stub logic for now

    if let CortexOutput::VerbalResponse(text) = output {
        if text.contains("loop") {
            return FrontalDecision::SuppressResponse("Looping detected.".to_string());
        }

        if text.contains("let me think") && context.active_goals.contains(&"urgent_action".to_string()) {
            return FrontalDecision::ReplaceWithGoal("This is urgent — respond directly.".to_string());
        }
    }

    // Later: Add working memory + inhibition checks here
    FrontalDecision::ExecuteImmediately(output.clone())
}

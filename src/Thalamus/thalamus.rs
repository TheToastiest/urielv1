use crate::Hippocampus::{ingest_text, SmieError};
use crate::Microthalamus::microthalamus::SyntheticDrive;
use crate::Parietal::parietal::{evaluate_input, ParietalRoute, SalienceScore};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum InputType {
    Voice,
    Text,
    Image,
    Sensor,
    Other(String),
}

impl std::fmt::Display for InputType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputType::Voice => write!(f, "voice"),
            InputType::Text => write!(f, "text"),
            InputType::Image => write!(f, "image"),
            InputType::Sensor => write!(f, "sensor"),
            InputType::Other(s) => write!(f, "{s}"),
        }
    }
}

/// Returns (salience, route). Ingests when route != Suppress.
pub fn handle_input_with_thalamus(
    context: &str,
    input: &str,
    source: &InputType,
    drives: &HashMap<String, SyntheticDrive>,
) -> Result<(SalienceScore, ParietalRoute), SmieError> {
    let ttl_secs = 60 * 5;

    // Parietal salience + routing
    let (salience, route) = evaluate_input(input, drives);

    println!(
        "🧠 Salience: urgency {:.2}, familiarity {:.2}, drive {:.2}, final {:.2} → {:?}",
        salience.urgency,
        salience.familiarity,
        salience.drive_alignment,
        salience.final_weight,
        route
    );

    // Only ingest on Respond or StoreOnly, and only if input has substance
    if !input.trim().is_empty() {
        match route {
            ParietalRoute::Respond | ParietalRoute::StoreOnly => {
                ingest_text(context, input, ttl_secs, &source.to_string())?;
            }
            ParietalRoute::Suppress => {
                println!("🧠 Suppressed: '{input}'");
            }
        }
    }

    Ok((salience, route))
}
pub fn handle_input_with_thalamus_legacy(
    context: &str,
    input: &str,
    source: &InputType,
    drives: &HashMap<String, SyntheticDrive>,
) -> Result<f32, SmieError> {
    let (salience, route) = handle_input_with_thalamus(context, input, source, drives)?;
    Ok(salience.final_weight)
}

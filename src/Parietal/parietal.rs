use std::collections::{HashMap, HashSet};
use crate::Microthalamus::microthalamus::SyntheticDrive;
use crate::Hippocampus::embedding::generate_embedding;

// ---------- Tunables ----------
const W_URGENCY: f32 = 0.35;
const W_FAMILIARITY: f32 = 0.20;
const W_UNDERSTANDING: f32 = 0.20;
const W_DRIVE: f32 = 0.25;

const RESPOND_THRESH: f32 = 0.60;
const STORE_THRESH: f32   = 0.30;
const RESEARCH_OVERRIDE_MIN_WEIGHT: f32 = 0.35;

// ---------- Model ----------
#[derive(Debug, Clone)]
pub struct SalienceScore {
    pub urgency: f32,
    pub familiarity: f32,
    pub understanding: f32,
    pub drive_alignment: f32,
    pub final_weight: f32,
}

impl SalienceScore {
    pub fn new(urgency: f32, familiarity: f32, understanding: f32, drive_alignment: f32) -> Self {
        let final_weight =
            urgency * W_URGENCY +
                familiarity * W_FAMILIARITY +
                understanding * W_UNDERSTANDING +
                drive_alignment * W_DRIVE;

        Self { urgency, familiarity, understanding, drive_alignment, final_weight }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ParietalRoute {
    Respond,
    StoreOnly,
    Suppress,
}

// ---------- text helpers ----------
trait ContainsAny {
    fn contains_any_word(&self, terms: &[&str]) -> bool;
}
impl ContainsAny for str {
    fn contains_any_word(&self, terms: &[&str]) -> bool {
        let lower = self.to_lowercase();
        let words: HashSet<String> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(|w| w.to_string())
            .collect();

        terms.iter().any(|t| {
            let tl = t.to_lowercase();
            words.contains(tl.as_str())
        })
    }
}

trait ContainsAnySubstr {
    fn contains_any_substr(&self, terms: &[&str]) -> bool;
}
impl ContainsAnySubstr for str {
    fn contains_any_substr(&self, terms: &[&str]) -> bool {
        let lower = self.to_lowercase();
        terms.iter().any(|t| {
            let tl = t.to_lowercase();
            lower.contains(tl.as_str())
        })
    }
}

// ---------- math ----------
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

// ---------- main ----------
pub fn evaluate_input(input: &str, drives: &HashMap<String, SyntheticDrive>) -> (SalienceScore, ParietalRoute) {
    // Urgency: detect boost words, then downshift if negated
    let urgent_word = input.contains_any_word(&["urgent","immediately","asap","now","quickly","promptly","swiftly"])
        || input.contains_any_substr(&["right now","as soon as","need this now",""]);
    let negation = input.contains_any_substr(&["not urgent","no rush","later","whenever"]);
    let urgency = if urgent_word && !negation { 0.80 } else if negation { 0.20 } else { 0.30 };

    // Familiarity: self/you/memory verbs
    let familiarity = if input.contains_any_word(&["you","your","remember","commit","memory","i","me","my","we","they"]) {
        0.60
    } else { 0.20 };

    // Understanding / research cues
    let understanding = if input.contains_any_word(&["research","search","internet","define","explain","look","find","help","assist","scan","render","edit"])
    || input.contains_any_substr(&["help me","assist me","scan for me","render for me","edit for me", "edit this"])
    {
        0.60
    } else { 0.25 };

    // Research intent override flag
    let research_intent = input.contains_any_word(&["research","search","lookup","investigate","google","web","internet"])
        || input.contains_any_substr(&["look up","look this up","do some research","find info","search for"]);

    // Drive alignment via embedding similarity (max scaled)
    let drive_alignment = match generate_embedding(input) {
        Ok(input_vec) => {
            let mut max_scaled = 0.0f32;
            for drive in drives.values() {
                let sim = cosine_similarity(&input_vec, &drive.vector);
                let scaled = sim * drive.weight;
                #[cfg(debug_assertions)]
                println!("Drive '{}': similarity {:.2}, scaled weight {:.2}", drive.name, sim, scaled);
                if scaled > max_scaled { max_scaled = scaled; }
            }
            max_scaled.clamp(0.0, 1.0)
        }
        Err(_) => 0.0,
    };

    let score = SalienceScore::new(urgency, familiarity, understanding, drive_alignment);

    // Routing
    let route = if (research_intent && score.final_weight >= RESEARCH_OVERRIDE_MIN_WEIGHT) || score.final_weight >= RESPOND_THRESH {
        #[cfg(debug_assertions)]
        if research_intent && score.final_weight < RESPOND_THRESH {
            println!("🔎 Research intent override: weight {:.2} → Respond", score.final_weight);
        }
        ParietalRoute::Respond
    } else if score.final_weight >= STORE_THRESH {
        ParietalRoute::StoreOnly
    } else {
        ParietalRoute::Suppress
    };

    #[cfg(debug_assertions)]
    println!(
        "🧠 Salience: urgency {:.2}, familiarity {:.2}, understanding {:.2}, drive {:.2} → weight {:.2} ⇒ {:?}",
        score.urgency, score.familiarity, score.understanding, score.drive_alignment, score.final_weight, route
    );

    (score, route)
}

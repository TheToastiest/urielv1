// src/parietal/infer_route.rs

use std::collections::HashMap;

use crate::Microthalamus::microthalamus::SyntheticDrive;
use crate::parietal::parietal::{evaluate_input, ParietalRoute, SalienceScore};

/// Unified route used by the rest of the pipeline.
/// We extend your ParietalRoute with a Retrieve option.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferredRoute {
    Respond,
    Retrieve,
    Store,
    Suppress,
}

/// Decide the operational route for an input by:
/// 1) Running your parietal salience model (`evaluate_input`)
/// 2) Applying a light retrieval-intent override for question/recall queries
///
/// Returns (Sa
/// lienceScore, InferredRoute) so callers can log/inspect the weights.
fn is_question(t: &str) -> bool {
    let t = t.trim().to_lowercase();
    t.ends_with('?')
        || t.starts_with("what ")
        || t.starts_with("who ")
        || t.starts_with("where ")
        || t.starts_with("when ")
        || t.starts_with("why ")
        || t.starts_with("how ")
        || t.starts_with("which ")
}

pub fn decide_route(
    text: &str,
    drives: &HashMap<String, SyntheticDrive>,
) -> (SalienceScore, InferredRoute) {
    let (score, base) = evaluate_input(text, drives);
    let wants_retrieval = detect_retrieval_intent(text);

    let route = match (base, wants_retrieval) {
        // If the salience says "Respond" but the user is clearly asking to recall/search,
        // we route to Retrieve instead.
        (ParietalRoute::Respond, true) => InferredRoute::Retrieve,

        // If salience is too low but user is explicitly asking to retrieve, allow it.
        (ParietalRoute::Suppress, true) => InferredRoute::Retrieve,

        // Normal mappings
        (ParietalRoute::Respond, false) => InferredRoute::Respond,
        (ParietalRoute::StoreOnly, _)   => InferredRoute::Store,
        (ParietalRoute::Suppress, false)=> InferredRoute::Suppress,
    };

    (score, route)
}

/// Heuristic for retrieval intent:
/// - interrogatives ("what/when/who/where/how/which")
/// - verbs ("recall", "retrieve", "show", "list", "search", "lookup", "find")
/// - trailing question mark
pub fn detect_retrieval_intent(text: &str) -> bool {
    let t = text.to_ascii_lowercase();

    // quick check: any question mark
    if t.trim_end().ends_with('?') {
        return true;
    }

    const INTERROGATIVES: [&str; 6] = ["what", "when", "who", "where", "how", "which"];
    const VERBS: [&str; 8] = ["recall", "retrieve", "show", "list", "search", "lookup", "look up", "find"];

    contains_any_word(&t, &INTERROGATIVES) || contains_any_substr(&t, &VERBS)
}

/* ------------ small string helpers (local copy) ------------ */

fn contains_any_word(s: &str, terms: &[&str]) -> bool {
    use std::collections::HashSet;
    let words: HashSet<&str> = s
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    terms.iter().any(|t| words.contains(&t.to_ascii_lowercase()[..]))
}

fn contains_any_substr(s: &str, terms: &[&str]) -> bool {
    terms
        .iter()
        .any(|t| s.contains(&t.to_ascii_lowercase()))
}

use crate::parietal::route_cfg::RouteCfg;

// New: cfg-aware version (doesn’t break callers using the old one)
pub fn decide_route_with_cfg(
    text: &str,
    drives: &HashMap<String, SyntheticDrive>,
    cfg: &RouteCfg,
) -> (SalienceScore, InferredRoute) {
    let (score, base) = evaluate_input(text, drives);
    let wants_retrieval = detect_retrieval_intent_with_extra(text, cfg.extra_retrieval_terms);

    let route = if cfg.retrieval_override && wants_retrieval {
        InferredRoute::Retrieve
    } else {
        match base {
            ParietalRoute::Respond   => InferredRoute::Respond,
            ParietalRoute::StoreOnly => InferredRoute::Store,
            ParietalRoute::Suppress  => InferredRoute::Suppress,
        }
    };
    (score, route)
}

fn detect_retrieval_intent_with_extra(text: &str, extra: &[&str]) -> bool {
    detect_retrieval_intent(text) || contains_any_substr(&text.to_ascii_lowercase(), extra)
}
// src/parietal/infer_route.rs

use std::collections::HashMap;

use URIELV1::Microthalamus::microthalamus::SyntheticDrive;
use URIELV1::parietal::parietal::{evaluate_input, ParietalRoute, SalienceScore};

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
/// Returns (SalienceScore, InferredRoute) so callers can log/inspect the weights.
pub fn decide_route(
    text: &str,
    drives: &HashMap<String, SyntheticDrive>,
) -> (SalienceScore, InferredRoute) {
    let (score, base) = evaluate_input(text, drives);
    let wants_retrieval = detect_retrieval_intent(text);

    // 🔧 Retrieval intent overrides *any* base route
    let route = if wants_retrieval {
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

/* ------------ optional quick test (enable with `cargo test`) ------------ */
#[cfg(test)]
mod tests {
    use super::*;
    use URIELV1::Microthalamus::microthalamus::SyntheticDrive;

    #[test]
    fn retrieval_heuristic_detects_questions() {
        let drives: HashMap<String, SyntheticDrive> = HashMap::new();
        let (_score, route) = decide_route("What did we store yesterday?", &drives);
        assert!(matches!(route, InferredRoute::Retrieve));
    }

    #[test]
    fn store_mapping_passes_through() {
        // Make sure "store this" flows through your parietal to StoreOnly
        // We can't force evaluate_input here without drives, but at least
        // sanity-check the heuristic doesn't override store.
        assert!(detect_retrieval_intent("store this: foo") == false);
    }
}

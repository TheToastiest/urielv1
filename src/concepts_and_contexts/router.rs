use super::learner::ConceptLearner;

/// Predict which concept a question is asking for.
/// Returns (key, prob) if above threshold.
pub fn predict_concept_for_question<'a>(
    learner: &mut ConceptLearner,
    question: &str,
    candidate_keys: &'a [&'static str],
    min_conf: f32,
) -> Option<(&'a str, f32)> {
    let mut best: Option<(&str, f32)> = None;
    for &k in candidate_keys {
        let p = learner.predict(k, question);
        if p >= min_conf && best.as_ref().map(|(_,bp)| p > *bp).unwrap_or(true) {
            best = Some((k, p));
        }
    }
    best
}

/// Train the router after a successful answer.
/// Positives for the gold key; negatives for others.
pub fn train_router_from_outcome(
    learner: &mut ConceptLearner,
    question: &str,
    gold_key: &str,
    candidate_keys: &[&'static str],
) {
    learner.update(gold_key, question, 1.0);
    for &k in candidate_keys {
        if k != gold_key {
            learner.update(k, question, 0.0);
        }
    }
}

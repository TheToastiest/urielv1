use URIELV1::Parietal::parietal::*;
use std::collections::HashMap;
#[test]
fn understanding_not_always_high() {
    let drives = HashMap::new();
    let (s1, _) = evaluate_input("hello world", &drives);
    let (s2, _) = evaluate_input("please research this on the internet", &drives);
    assert!(s2.understanding > s1.understanding);
}

#[test]
fn urgency_phrase_hits() {
    let drives = HashMap::new();
    let (s1, _) = evaluate_input("do this right now please", &drives);
    let (s2, _) = evaluate_input("maybe later", &drives);
    assert!(s1.urgency > s2.urgency);
}

#[test]
fn routing_bands_behave() {
    let drives = HashMap::new();
    let (s, r) = evaluate_input("research this right now", &drives);
    // depending on weights/thresholds, this should not be Suppress
    assert!(r != ParietalRoute::Suppress);
}

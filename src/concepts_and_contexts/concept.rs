#[derive(Debug, Clone)]
pub struct ConceptHit {
    pub key:   &'static str, // "name", "launch code", "creator", ...
    pub value: String,
    pub confidence: f32,     // 0..1 (rule based → 0.8..0.95; learner can refine)
}

pub trait Concept: Send + Sync {
    fn key(&self) -> &'static str;
    fn single_truth(&self) -> bool { true } // most URIEL “identity” facts are single-truth
    fn extract_from_statement(&self, line: &str) -> Vec<ConceptHit>;
    fn normalize_question(&self, q: &str) -> Option<String>; // -> normalized “your name is”
}

use super::concept::{Concept, ConceptHit};
use std::collections::HashMap;

pub struct Registry {
    pub concepts: HashMap<&'static str, Box<dyn Concept>>,
}
impl Registry {
    pub fn new() -> Self { Self { concepts: HashMap::new() } }
    pub fn register(mut self, c: Box<dyn Concept>) -> Self { self.concepts.insert(c.key(), c); self }
    pub fn extract_all_from_statement(&self, line: &str) -> Vec<ConceptHit> {
        let mut out = Vec::new();
        for c in self.concepts.values() { out.extend(c.extract_from_statement(line)); }
        out
    }
    pub fn normalize_question(&self, q: &str) -> Option<(&'static str, String)> {
        for (k, c) in &self.concepts {
            if let Some(norm) = c.normalize_question(q) { return Some((k, norm)); }
        }
        None
    }
}

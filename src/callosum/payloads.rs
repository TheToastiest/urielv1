use std::fmt::Debug;
use crate::parietal::parietal::{SalienceScore, ParietalRoute};

#[derive(Clone, Debug)]
pub struct RouteDecision {
    pub context: String,
    pub input: String,
    pub route: ParietalRoute,
    pub score: SalienceScore,
}

#[derive(Clone, Debug)]
pub struct StoreRequest {
    pub context: String,
    pub text: String,
    pub ts_unix_ms: i64,
}

#[derive(Clone, Debug)]
pub struct RetrieveRequest {
    pub context: String,
    pub query: String,
}

#[derive(Clone, Debug)]
pub struct RetrievedContext {
    pub context: String,
    pub query: String,
    pub snippets: Vec<String>, // simple for now
}
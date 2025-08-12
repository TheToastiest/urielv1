// Core types both sides can see
#[derive(Clone, Debug)]
pub struct Evidence {
    pub id: String,
    pub emb: Vec<f32>,        // normalized
    pub cosine: f32,
    pub time_delta_s: f32,
    pub ttl_s: f32,
    pub trust: f32,
    pub context_id: u16,
    pub modality_id: u8,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct InferenceInput<'a> {
    pub raw_input: &'a str,
    pub drives: std::collections::HashMap<String, f32>,
    pub hits: Vec<Evidence>,      // top_k from PFC
}

#[derive(Clone, Debug)]
pub enum Action {
    Respond, Ask, StoreOnly,
    SearchInternet(String),
    RunSubmodule(String),
}

#[derive(Clone, Debug)]
pub struct InferenceDecision {
    pub action: Action,
    pub chosen_ids: Vec<String>,
    pub confidence: f32,
    pub rationale: String,
}

// Trait lives in core so any engine impl can plug in.
pub trait InferenceEngine: Send + Sync {
    fn infer(&self, inp: &InferenceInput) -> InferenceDecision;
}

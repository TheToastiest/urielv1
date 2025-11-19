use crate::PFC::context_retriever::MemoryHit;

#[derive(Clone, Debug)]
pub struct RetrievedSegment {
    pub source: String,     // "smie", "episodic", "concept", "rows"
    pub score: f32,         // normalized
    pub text: String,
    pub meta: SegmentMeta,
}

#[derive(Clone, Debug, Default)]
pub struct SegmentMeta {
    pub similarity: Option<f32>,
    pub recency: Option<f32>,
    pub key: Option<String>,
    pub context: Option<String>,
}

#[derive(Clone, Debug)]
pub struct RAGFrame {
    pub smie_hits: Vec<RetrievedSegment>,
    pub epi_hits: Vec<RetrievedSegment>,
    pub concept_hits: Vec<RetrievedSegment>,
    pub row_hits: Vec<RetrievedSegment>,
    pub combined: Vec<RetrievedSegment>,
}
impl RAGFrame {
    pub fn empty() -> Self {
        Self {
            smie_hits: Vec::new(),
            epi_hits: Vec::new(),
            concept_hits: Vec::new(),
            row_hits: Vec::new(),
            combined: Vec::new(),
        }
    }
}

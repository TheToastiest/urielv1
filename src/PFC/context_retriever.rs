    //! context_retriever.rs
    use std::error::Error;
    use std::sync::Arc;
    use crate::PFC::cortex::{CortexInputFrame, CortexReasoningMode};

    /// A single retrieved memory item to feed the solver/context composer.
    #[derive(Debug, Clone)]
    pub struct MemoryHit {
        pub id: String,    // e.g., sqlite row id / redis key
        pub sim: f32,      // cosine similarity in [0,1]
        pub text: String,  // snippet or full text
    }

    #[derive(Debug, Clone)]
    pub struct ContextFrame {
        /// Top-k memory hits sorted by similarity desc.
        pub top_hits: Vec<MemoryHit>,
        /// Confidence that the assembled context is sufficient for answering.
        pub confidence: f32,
    }

    impl ContextFrame {
        pub fn empty() -> Self { Self { top_hits: Vec::new(), confidence: 0.0 } }
    }

    /// ---------- ports (swappable adapters) ----------
    pub type Embedding = Vec<f32>;

    pub trait Embeddings: Send + Sync {
        fn embed(&self, text: &str) -> Result<Embedding, Box<dyn Error + Send + Sync>>;
    }

    pub trait EmbeddingCache: Send + Sync {
        fn get(&self, key: &str) -> Option<Embedding>;
        fn put(&self, key: &str, emb: &Embedding);
    }

    /// Vector store must return top-k hits already sorted by similarity desc,
    /// with similarity expressed as cosine in [0,1].
    pub trait VectorStore: Send + Sync {
        fn search(&self, query: &Embedding, top_k: usize) -> Result<Vec<MemoryHit>, Box<dyn Error + Send + Sync>>;
    }

    /// ---------- retriever ----------
    #[derive(Debug, Clone)]
    pub struct RetrieverCfg {
        pub top_k_default: usize,
        pub top_k_research: usize,
        pub min_score: f32,        // drop anything below this
        pub use_cache: bool,
    }

    impl Default for RetrieverCfg {
        fn default() -> Self {
            Self {
                top_k_default: 8,
                top_k_research: 16,
                min_score: 0.35,
                use_cache: true,
            }
        }
    }
pub trait TextSearch: Send + Sync {
    fn search_text(&self, text: &str, top_k: usize) -> Result<Vec<MemoryHit>, Box<dyn Error + Send + Sync>>;
}
pub struct ContextRetriever {
    vs: Arc<dyn VectorStore>,
    emb: Arc<dyn Embeddings>,
    cache: Option<Arc<dyn EmbeddingCache>>,
    cfg: RetrieverCfg,
    ts: Option<Arc<dyn TextSearch>>,
}

    impl ContextRetriever {
        pub fn new(
            vs: Arc<dyn VectorStore>,
            emb: Arc<dyn Embeddings>,
            cache: Option<Arc<dyn EmbeddingCache>>,
            cfg: RetrieverCfg,
        ) -> Self {
            Self { vs, emb, cache, cfg, ts: None }
        }
        pub fn new_with_textsearch(
            vs: Arc<dyn VectorStore>,
            emb: Arc<dyn Embeddings>,
            cache: Option<Arc<dyn EmbeddingCache>>,
            cfg: RetrieverCfg,
            ts: Option<Arc<dyn TextSearch>>,
        ) -> Self {
            Self { vs, emb, cache, cfg, ts }
        }

        fn get_or_embed(&self, text: &str) -> Result<Embedding, Box<dyn Error + Send + Sync>> {
            if self.cfg.use_cache {
                if let Some(c) = &self.cache {
                    if let Some(e) = c.get(text) { return Ok(e); }
                }
            }
            let mut e = self.emb.embed(text)?;
            normalize_unit(&mut e);
            if self.cfg.use_cache {
                if let Some(c) = &self.cache { c.put(text, &e); }
            }
            Ok(e)
        }

        /// confidence heuristic:
        ///  - 0 if no hits
        ///  - otherwise 0.6*max + 0.4*mean(top3)
        fn confidence(&self, hits: &[MemoryHit]) -> f32 {
            if hits.is_empty() { return 0.0; }
            let max = hits[0].sim;
            let m = hits.iter().take(3).map(|h| h.sim).sum::<f32>() / hits.len().min(3) as f32;
            (0.6 * max + 0.4 * m).clamp(0.0, 1.0)
        }

        fn k_for(&self, _mode: &CortexReasoningMode) -> usize {
            // If you want research to pull deeper, switch to:
            // match mode { CortexReasoningMode::InternetResearch => self.cfg.top_k_research, _ => self.cfg.top_k_default }
            self.cfg.top_k_default
        }

        /// Build a context frame for the given input/mode (main path).
        pub fn build_context_frame(
            &self,
            frame: &CortexInputFrame,
            mode: &CortexReasoningMode,
        ) -> Result<ContextFrame, Box<dyn Error + Send + Sync>> {
            self.build_context_frame_from_str(frame.raw_input, mode)
        }

        /// Helper: same as above but takes a plain &str (useful in tests/tools).
        pub fn build_context_frame_from_str(
            &self,
            query_text: &str,
            mode: &CortexReasoningMode,
        ) -> Result<ContextFrame, Box<dyn Error + Send + Sync>> {
            // 0) Prefer text search (SMIE) if available
            if let Some(ts) = &self.ts {
                let mut hits = ts.search_text(query_text, self.k_for(mode))?
                    .into_iter()
                    .filter(|h| h.sim >= self.cfg.min_score)
                    .collect::<Vec<_>>();
                hits.sort_by(|a,b| b.sim.partial_cmp(&a.sim).unwrap_or(std::cmp::Ordering::Equal));
                let confidence = self.confidence(&hits);
                return Ok(ContextFrame { top_hits: hits, confidence });
            }

            // 1) Fallback to embed + ANN vector store
            let q = self.get_or_embed(query_text)?;
            let mut hits = self.vs.search(&q, self.k_for(mode))?
                .into_iter()
                .filter(|h| h.sim >= self.cfg.min_score)
                .collect::<Vec<_>>();
            hits.sort_by(|a,b| b.sim.partial_cmp(&a.sim).unwrap_or(std::cmp::Ordering::Equal));
            let confidence = self.confidence(&hits);
            Ok(ContextFrame { top_hits: hits, confidence })
        }

    }

    /// public shim to keep old callsites happy (requires &ContextRetriever)
    pub fn build_context_frame(
        retriever: &ContextRetriever,
        frame: &CortexInputFrame,
        mode: &CortexReasoningMode,
    ) -> ContextFrame {
        retriever.build_context_frame(frame, mode).unwrap_or_else(|_| ContextFrame::empty())
    }

    /// ---------- helpers ----------
    fn normalize_unit(v: &mut [f32]) {
        let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
        for x in v { *x /= n; }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        struct DummyEmb;
        impl Embeddings for DummyEmb {
            fn embed(&self, text: &str) -> Result<Embedding, Box<dyn Error + Send + Sync>> {
                // toy embedding: byte sum repeated (DON'T use in prod)
                let s = text.bytes().map(|b| b as f32).sum::<f32>();
                Ok(vec![s, 1.0, 0.0])
            }
        }

        struct MemStore { rows: Vec<(String, String, Embedding)> } // (id, text, emb)
        impl MemStore {
            fn new(rows: Vec<(String, String)>, emb: &dyn Embeddings) -> Self {
                let rows = rows.into_iter().map(|(id, text)| {
                    let mut e = emb.embed(&text).unwrap();
                    super::normalize_unit(&mut e);
                    (id, text, e)
                }).collect();
                Self { rows }
            }
        }
        impl VectorStore for MemStore {
            fn search(&self, q: &Embedding, top_k: usize) -> Result<Vec<MemoryHit>, Box<dyn Error + Send + Sync>> {
                let mut scored: Vec<MemoryHit> = self.rows.iter().map(|(id, text, e)| {
                    let sim = cosine(q, e);
                    MemoryHit { id: id.clone(), sim, text: text.clone() }
                }).collect();
                scored.sort_by(|a,b| b.sim.partial_cmp(&a.sim).unwrap());
                scored.truncate(top_k);
                Ok(scored)
            }
        }

        fn cosine(a: &[f32], b: &[f32]) -> f32 {
            let dot = a.iter().zip(b).map(|(x,y)| x*y).sum::<f32>();
            dot.clamp(-1.0, 1.0).max(0.0) // keep [0,1] for tests
        }

        #[test]
        fn retrieves_and_confidence() {
            let emb = Arc::new(DummyEmb);
            let store = Arc::new(MemStore::new(
                vec![
                    ("1".into(), "hello world".into()),
                    ("2".into(), "metacognitive drive research".into()),
                    ("3".into(), "rust context retriever".into()),
                ],
                emb.as_ref()
            ));
            let retriever = ContextRetriever::new(store, emb, None, RetrieverCfg::default());

            // Use the from_str helper so we don't depend on CortexInputFrame fields/lifetimes
            let mode = CortexReasoningMode::DirectRecall; // or InternetResearch
            let cf = retriever.build_context_frame_from_str(
                "do some research of the metacognitive drive",
                &mode
            ).unwrap();

            assert!(!cf.top_hits.is_empty());
            assert!((0.0..=1.0).contains(&cf.confidence));
            assert!(cf.top_hits.first().unwrap().sim >= cf.top_hits.last().unwrap().sim);
        }
    }

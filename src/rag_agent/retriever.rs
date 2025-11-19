// retriever.rs
use std::sync::Arc;

use crate::rag_agent::{RAGFrame, RetrievedSegment};
use crate::PFC::context_retriever::{
    ContextRetriever, TextSearch, VectorStore, Embeddings,
};
use crate::knowledge::ops::read_best_fact_text;
use crate::Hippocampus::sqlite_backend::recall_latest;

pub struct UnifiedRetriever {
    pub ctx: Arc<ContextRetriever>,
    pub emb: Arc<dyn Embeddings>,
    pub text: Option<Arc<dyn TextSearch>>,
    pub vs: Arc<dyn VectorStore>,
    pub conn: Arc<rusqlite::Connection>,
}

impl UnifiedRetriever {
    fn embed_query(&self, q: &str) -> Vec<f32> {
        match self.emb.embed(q) {
            Ok(mut v) => {
                normalize(&mut v);
                v
            }
            Err(_) => Vec::new(),
        }
    }

    pub fn retrieve(&self, query: &str, context: &str) -> RAGFrame {
        let mut frame = RAGFrame::empty();

        // --- SMIE text ANN ---
        if let Some(ts) = &self.text {
            if let Ok(hits) = ts.search_text(query, 8) {
                frame.smie_hits = hits.into_iter()
                    .map(|h| RetrievedSegment {
                        source: "smie".into(),
                        score: h.sim,
                        text: h.text,
                        meta: Default::default(),
                    })
                    .collect();
            }
        }

        // --- VectorStore ANN ---
        let qv = self.embed_query(query);
        if !qv.is_empty() {
            if let Ok(hits) = self.vs.search(&qv, 8) {
                frame.epi_hits = hits.into_iter()
                    .map(|h| RetrievedSegment {
                        source: "episodic".into(),
                        score: h.sim,
                        text: h.text,
                        meta: Default::default(),
                    })
                    .collect();
            }
        }

        // --- Concept DB ---
        if let Ok(Some(fact)) = read_best_fact_text(&self.conn, None, query) {
            frame.concept_hits.push(RetrievedSegment {
                source: "concept".into(),
                score: 1.0,
                text: fact,
                meta: Default::default(),
            });
        }

        // --- Recent rows fallback ---
        if let Ok(rows) = recall_latest(context, 8) {
            frame.row_hits = rows.into_iter()
                .map(|r| RetrievedSegment {
                    source: "rows".into(),
                    score: 0.5,
                    text: r.data,
                    meta: Default::default(),
                })
                .collect();
        }

        frame
    }
}

fn normalize(v: &mut [f32]) {
    let n = (v.iter().map(|x| x * x).sum::<f32>())
        .sqrt()
        .max(1e-6);
    for x in v {
        *x /= n;
    }
}

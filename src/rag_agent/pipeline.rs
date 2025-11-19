use crate::rag_agent::{RAGFrame, RetrievedSegment};
use crate::rag_agent::scoring::unified_score;

pub fn fuse_rag(frame: &mut RAGFrame, drive_alignment: f32) {
    let mut combined = Vec::new();

    for src in [
        &frame.smie_hits,
        &frame.epi_hits,
        &frame.concept_hits,
        &frame.row_hits,
    ] {
        for seg in src.iter() {
            let score = unified_score(
                seg.meta.similarity,
                seg.meta.recency,
                drive_alignment,
                &seg.source,
            );

            let mut ns = seg.clone();
            ns.score = score;
            combined.push(ns);
        }
    }

    combined.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    frame.combined = combined;
}

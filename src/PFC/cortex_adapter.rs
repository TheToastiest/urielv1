// src/Adapters/cortex_adapter.rs
use std::sync::Arc;
use std::thread;

use crate::callosum::callosum::CorpusCallosum;
use crate::callosum::event::{CallosumEvent};
use crate::callosum::payloads::{RetrievedContext, RouteDecision};

use crate::PFC::cortex::{CortexInputFrame, CortexReasoningMode};
use crate::PFC::context_retriever::ContextRetriever;
use crate::PFC::cortex_solver::solve_with_context;
use crate::parietal::parietal::SalienceScore;
use crate::Microthalamus::microthalamus::SyntheticDrive;

use std::collections::HashMap;

pub fn spawn_cortex_adapter(
    cc: Arc<CorpusCallosum>,
    retriever: Arc<ContextRetriever>,
    drives: Arc<HashMap<String, SyntheticDrive>>,
    name: &str,
) {
    let rx = cc.subscribe(name);
    thread::spawn(move || {
        // keep last route decision per correlation id
        use std::collections::HashMap as Map;
        let mut last_route: Map<u64, RouteDecision> = Map::new();

        while let Ok(env) = rx.recv() {
            if let Some(rd) = env.event.dynamic_as::<RouteDecision>() {
                if let Some(cid) = env.correlation_id {
                    last_route.insert(cid, rd.clone());
                }
            }

            if let Some(rc) = env.event.dynamic_as::<RetrievedContext>() {
                // find the matching route decision to extract salience score (if available)
                let score = if let Some(cid) = env.correlation_id {
                    last_route.get(&cid).map(|r| r.score.clone())
                } else { None };
                let score = score.unwrap_or(SalienceScore {
                    urgency: 0.3, familiarity: 0.2, understanding: 0.3, drive_alignment: 0.0,
                    final_weight: 0.0
                });

                // Build a tiny synthetic context frame for the solver
                let top_hits = rc.snippets.iter().map(|s| crate::PFC::context_retriever::MemoryHit {
                    id: "sqlite".into(),
                    sim: 1.0, // unknown, treat as strong
                    text: s.clone(),
                }).collect::<Vec<_>>();
                let ctx_frame = crate::PFC::context_retriever::ContextFrame {
                    top_hits,
                    confidence: 0.9,
                };

                // Choose a mode: use DirectRecall if we have hits, else CoT
                let mode = if !ctx_frame.top_hits.is_empty() {
                    CortexReasoningMode::DirectRecall
                } else {
                    CortexReasoningMode::ChainOfThought
                };

                let frame = CortexInputFrame { raw_input: &rc.query, score, drives: &drives };
                let draft = solve_with_context(&mode, &frame, &ctx_frame);

                if let Some(resp) = draft.response {
                    cc.publish("cortex", env.correlation_id, CallosumEvent::ResponseGenerated(resp));
                }
            }
        }
    });
}

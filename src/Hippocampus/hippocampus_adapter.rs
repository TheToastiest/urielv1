// src/Adapters/hippocampus_adapter.rs
use std::sync::Arc;
use std::thread;

use crate::callosum::callosum::CorpusCallosum;
use crate::callosum::event::{CallosumEvent};
use crate::callosum::payloads::{StoreRequest, RetrieveRequest, RetrievedContext};

use crate::Hippocampus::memory_bridge::{write_memory, search_memory_by_context, MemWrite};
use crate::PFC::context_retriever::ContextRetriever;

pub fn spawn_hippocampus_adapter(
    cc: Arc<CorpusCallosum>,
    retriever: Arc<ContextRetriever>,
    name: &str,
) {
    let rx = cc.subscribe(name);
    thread::spawn(move || {
        while let Ok(env) = rx.recv() {
            // Store
            if let Some(req) = env.event.dynamic_as::<StoreRequest>() {
                let _ = write_memory(MemWrite {
                    context: req.context.clone(),
                    text: req.text.clone(),
                    ts_unix_ms: req.ts_unix_ms,
                    z: None, // optional, you can add embeddings later
                });
                // You could also publish an ack here if needed.
                continue;
            }
            // Retrieve
            if let Some(req) = env.event.dynamic_as::<RetrieveRequest>() {
                // Option A: your SQLite by-context helper
                let hits = search_memory_by_context(&req.context, 8).unwrap_or_default();
                let snippets: Vec<String> = hits.into_iter().map(|h| h.snippet).collect();

                // Option B (in parallel or instead): vector retriever
                // let mode = crate::PFC::cortex::CortexReasoningMode::DirectRecall;
                // let cf = retriever.build_context_frame_from_str(&req.query, &mode).unwrap();
                // let mut snippets = cf.top_hits.iter().map(|h| h.text.clone()).collect::<Vec<_>>();

                // Publish both a legacy “MemoryRecalled” and a richer payload
                if !snippets.is_empty() {
                    cc.publish("hippocampus", env.correlation_id,
                               CallosumEvent::MemoryRecalled(snippets[0].clone()));
                }
                cc.publish("hippocampus", env.correlation_id,
                           CallosumEvent::dynamic(RetrievedContext {
                               context: req.context.clone(),
                               query: req.query.clone(),
                               snippets,
                           }));
                continue;
            }
        }
    });
}

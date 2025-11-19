// core/handle.rs (refactor)

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use burn::tensor::{backend::Backend};
use crate::rag_agent::retriever::UnifiedRetriever;

use crate::Microthalamus::microthalamus::SyntheticDrive;
use crate::Hippocampus::infer::HippocampusModel;
use crate::Hippocampus::memory_bridge::{write_memory, MemWrite};
use crate::parietal::route_cfg::RouteCfg;
use crate::parietal::infer_route::{decide_route_with_cfg, InferredRoute};
use crate::PFC::cortex::{
    execute_cortex_rag, CortexInputFrame, CortexOutput, CortexReasoningMode
};
use crate::PFC::context_retriever::{ContextFrame};
use crate::Hippocampus::error::SmieError;

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

fn stringify_output(out: CortexOutput) -> String {
    match out {
        CortexOutput::VerbalResponse(s) => s,
        CortexOutput::Thought(t)        => format!("(thought) {t}"),
        CortexOutput::Action(a)         => format!("(action) {a:?}"),
        CortexOutput::Multi(items)      => {
            let parts: Vec<String> = items.into_iter().map(stringify_output).collect();
            parts.join("\n")
        }
    }
}

/// End-to-end: parietal → (store | cortex) with retriever-managed context.
pub fn handle_input_pfc<B: Backend>(
    model: &HippocampusModel<B>,
    context_key: &str,
    text: &str,
    drives: &HashMap<String, SyntheticDrive>,
    route_cfg: &RouteCfg,
    rag: &UnifiedRetriever,
) -> Result<String, SmieError> {

    let (_score, route) = decide_route_with_cfg(text, drives, route_cfg);
    let ts = now_ms();

    match route {
        InferredRoute::Store => {
            let z_1xD = model.run_inference(text);
            let z = z_1xD.to_data().as_slice::<f32>().unwrap().to_vec();
            let id = write_memory(MemWrite {
                context: context_key.into(),
                text: text.into(),
                ts_unix_ms: ts,
                z: Some(z),
            })?;
            Ok(format!("Stored ({id})."))
        }

        InferredRoute::Retrieve => {
            let mode = CortexReasoningMode::DirectRecall;
            let cf = rag
                .ctx
                .build_context_frame_from_str(text, &mode)
                .unwrap_or_else(|_| ContextFrame::empty());
            let _ctx_strings: Vec<String> = cf.top_hits.iter().map(|h| h.text.clone()).collect();

            let frame = CortexInputFrame {
                raw_input: text,
                score: _score, // assuming decide_route_with_cfg returns a SalienceScore
                drives,
            };
            let out = execute_cortex_rag(frame, rag);
            Ok(stringify_output(out))
        }

        InferredRoute::Respond => {
            let mode = CortexReasoningMode::DirectRecall;
            let cf = rag
                .ctx
                .build_context_frame_from_str(text, &mode)
                .unwrap_or_else(|_| ContextFrame::empty());

            if let Some(best) = cf.top_hits.first() {
                return Ok(best.text.clone());
            }

            let frame = CortexInputFrame {
                raw_input: text,
                score: _score,
                drives,
            };
            let out = execute_cortex_rag(frame, rag);
            Ok(stringify_output(out))
        }


        InferredRoute::Suppress => Ok("Suppressed (low salience).".into()),
    }
}

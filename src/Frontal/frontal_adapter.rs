// src/Frontal/frontal_adapter.rs
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::callosum::callosum::CorpusCallosum;
use crate::callosum::event::{CallosumEvent, EventEnvelope};
use crate::Frontal::frontal::{
    evaluate_frontal_control, FrontalContext, FrontalDecision,
    InhibitionConfig, InhibitionState, WorkingMemory,
};
use crate::Microthalamus::microthalamus::SyntheticDrive;
use crate::PFC::cortex::CortexOutput;

pub fn spawn_frontal_adapter(
    cc: Arc<CorpusCallosum>,
    drives: Arc<HashMap<String, SyntheticDrive>>,
    name: &str,
) {
    // NOTE: subscribe returns Receiver<Arc<EventEnvelope>>
    let rx = cc.subscribe(name);

    std::thread::spawn(move || {
        // Persistent state across turns in this adapter thread
        let mut wm = WorkingMemory::new(50);
        let mut inhib_state = InhibitionState::default();
        let inhib_cfg = InhibitionConfig::default();
        let mut last_output: Option<CortexOutput> = None;

        while let Ok(env_arc) = rx.recv() {
            // Work with the envelope, then match on the inner event
            let env: &EventEnvelope = &*env_arc;

            match &env.event {
                CallosumEvent::ResponseGenerated(text) => {
                    // Build context (wrap persistent state in RefCell for interior mutability)
                    let ctx = FrontalContext {
                        active_goals: vec![],
                        current_drive_weights: &drives,
                        last_response: last_output.as_ref(),
                        working_memory: RefCell::new(wm),
                        inhib_state: RefCell::new(inhib_state),
                        inhib_cfg,
                    };

                    let candidate = CortexOutput::VerbalResponse(text.clone());
                    let decision = evaluate_frontal_control(&candidate, &ctx);

                    // Persist any mutations made during frontal control
                    let FrontalContext { working_memory, inhib_state: is, .. } = ctx;
                    wm = working_memory.into_inner();
                    inhib_state = is.into_inner();

                    // Act on decision (you could also publish a new envelope here)
                    match decision {
                        FrontalDecision::ExecuteImmediately(out) => {
                            println!("{out:?}");
                            last_output = Some(out);
                        }
                        FrontalDecision::QueueForLater(out) => {
                            eprintln!("(queued by frontal control) {out:?}");
                            last_output = Some(out);
                        }
                        FrontalDecision::SuppressResponse(msg) => {
                            eprintln!("{msg}");
                            last_output = Some(CortexOutput::Thought(format!("suppressed: {msg}")));
                        }
                        FrontalDecision::ReplaceWithGoal(msg) => {
                            println!("[goal injected] {msg}");
                            last_output = Some(CortexOutput::Thought(format!("goal: {msg}")));
                        }
                    }
                }

                // Other events: no-op for now (but you can use them to prime WM/inhibition).
                CallosumEvent::EvaluatedInput { .. }
                | CallosumEvent::RawInput(_)
                | CallosumEvent::MemoryRecalled(_)
                | CallosumEvent::Dynamic(_) => {
                    // no-op
                }
            }
        }
    });
}

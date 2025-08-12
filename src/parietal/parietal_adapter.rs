// src/Adapters/parietal_adapter.rs
use std::{collections::HashMap, sync::Arc, thread};
use crate::Microthalamus::microthalamus::SyntheticDrive;
use super::parietal::{evaluate_input, ParietalRoute};
use super::route_cfg::RouteCfg; // if you prefer decide_route_with_cfg, swap it in
use crate::callosum::callosum::CorpusCallosum;
use crate::callosum::event::{CallosumEvent};
use crate::callosum::payloads::{RouteDecision, StoreRequest, RetrieveRequest};

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64
}

pub fn spawn_parietal_adapter(
    cc: Arc<CorpusCallosum>,
    drives: Arc<HashMap<String, SyntheticDrive>>,
    route_cfg: RouteCfg,        // not used in this minimal demo, but keep if you have heuristics
    name: &str,
    context_key: String,        // “default” or session id
) {
    let rx = cc.subscribe(name);
    thread::spawn(move || {
        while let Ok(env) = rx.recv() {
            match &env.event {
                CallosumEvent::RawInput(text) => {
                    // Evaluate salience & routing
                    let (score, route) = evaluate_input(text, &drives);
                    cc.publish(
                        "parietal",
                        env.correlation_id,
                        CallosumEvent::EvaluatedInput {
                            input: text.clone(),
                            urgency: score.urgency,
                            familiarity: score.familiarity,
                            understanding: score.understanding,
                            drive_align: score.drive_alignment,
                            tags: vec![],
                        },
                    );

                    // Ship the explicit route decision
                    cc.publish(
                        "parietal",
                        env.correlation_id,
                        CallosumEvent::dynamic(RouteDecision {
                            context: context_key.clone(),
                            input: text.clone(),
                            route,
                            score,
                        }),
                    );

                    // Kick off next stage
                    match route {
                        ParietalRoute::StoreOnly => {
                            cc.publish(
                                "parietal",
                                env.correlation_id,
                                CallosumEvent::dynamic(StoreRequest {
                                    context: context_key.clone(),
                                    text: text.clone(),
                                    ts_unix_ms: now_ms(),
                                }),
                            );
                        }
                        ParietalRoute::Respond => {
                            // treat as retrieval question — ask hippocampus for context
                            cc.publish(
                                "parietal",
                                env.correlation_id,
                                CallosumEvent::dynamic(RetrieveRequest {
                                    context: context_key.clone(),
                                    query: text.clone(),
                                }),
                            );
                        }
                        ParietalRoute::Suppress => {
                            // do nothing
                        }
                    }
                }
                _ => {}
            }
        }
    });
}

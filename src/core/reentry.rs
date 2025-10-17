// src/Core/reentry.rs
use std::time::{Duration, Instant};

use crate::callosum::callosum::CorpusCallosum;
use crate::callosum::event::{CallosumEvent, EventEnvelope};
use crate::parietal::parietal::evaluate_input;
use crate::PFC::context_retriever::ContextRetriever;
use crate::PFC::cortex::{execute_cortex_with_ctx, CortexInputFrame, CortexOutput};
use crate::Frontal::frontal::{evaluate_frontal_control, FrontalContext, WorkingMemory, InhibitionState, InhibitionConfig};
use crate::Microthalamus::microthalamus::SyntheticDrive;
use crate::Hippocampus::primordial::primordial;

pub struct ReentryCfg {
    pub max_cycles: usize,         // e.g., 3..5
    pub max_duration_ms: u64,      // e.g., 1500
}

impl Default for ReentryCfg {
    fn default() -> Self {
        Self { max_cycles: 4, max_duration_ms: 1500 }
    }
}

pub struct ReentryEngine<'a> {
    pub callosum: &'a CorpusCallosum,
    pub retriever: &'a ContextRetriever,
    pub drives: &'a std::collections::HashMap<String, SyntheticDrive>,
}

impl<'a> ReentryEngine<'a> {
    pub fn run(
        &self,
        input_text: &str,
        cfg: ReentryCfg,
    ) -> String {
        // 0) Primordial must exist (touch to ensure it’s loaded)
        let _p = primordial();
        let start = Instant::now();

        // 1) Perception/parietal
        let (score, _route) = evaluate_input(input_text, self.drives);

        self.publish("parietal", None, CallosumEvent::EvaluatedInput {
            input: input_text.to_string(),
            urgency: score.urgency,
            familiarity: score.familiarity,
            understanding: score.understanding,
            drive_align: score.drive_alignment,
            tags: vec![], // you can add tags later
        });

        // 2) Initialize WM + Inhibition per interaction
        let mut wm = WorkingMemory::new(64);
        let mut inhib_state = InhibitionState::default();
        let inhib_cfg = InhibitionConfig::default();

        // 3) Reentry cycles
        let mut last_output: Option<CortexOutput> = None;
        for cycle in 0..cfg.max_cycles {
            if start.elapsed() > Duration::from_millis(cfg.max_duration_ms) {
                break;
            }

            // Build Cortex frame from current salience & drives
            let frame = CortexInputFrame {
                raw_input: input_text,
                score: score.clone(),
                drives: self.drives,
            };

            // Reason with context (retrieval happens inside)
            let out = execute_cortex_with_ctx(frame, &Default::default(), self.retriever);

            // Frontal control & global policies
            let ctx = crate::Frontal::frontal::FrontalContext {
                active_goals: vec![],
                current_drive_weights: self.drives,
                last_response: last_output.as_ref(),
                working_memory: std::cell::RefCell::new(wm),
                inhib_state: std::cell::RefCell::new(inhib_state),
                inhib_cfg,
            };

            let decision = evaluate_frontal_control(&out, &ctx);

            // Persist WM/Inhibition across cycles
            let FrontalContext { working_memory, inhib_state: is, .. } = ctx;
            wm = working_memory.into_inner();
            inhib_state = is.into_inner();

            match decision {
                crate::Frontal::frontal::FrontalDecision::ExecuteImmediately(res) => {
                    self.publish("frontal", None, CallosumEvent::ResponseGenerated(render(&res)));
                    last_output = Some(res.clone());

                    // termination: if content looks final, stop
                    if is_final(&res) { return render(&res); }
                    // else: loop—this allows the system to *revise* using WM/inhibition
                }
                crate::Frontal::frontal::FrontalDecision::QueueForLater(res) => {
                    self.publish("frontal", None, CallosumEvent::ResponseGenerated(format!("[queued] {}", render(&res))));
                    last_output = Some(res);
                    break;
                }
                crate::Frontal::frontal::FrontalDecision::SuppressResponse(msg) => {
                    self.publish("frontal", None, CallosumEvent::ResponseGenerated(format!("[suppressed] {msg}")));
                    // If suppressed, either continue to try alternative mode or end.
                    if cycle + 1 == cfg.max_cycles { return msg; }
                }
                crate::Frontal::frontal::FrontalDecision::ReplaceWithGoal(msg) => {
                    self.publish("frontal", None, CallosumEvent::ResponseGenerated(format!("[goal] {msg}")));
                    return msg;
                }
            }
        }

        // Fallback if we didn’t return earlier
        last_output
            .as_ref()
            .map(render)
            .unwrap_or_else(|| "No response.".to_string())
            }

    fn publish(&self, source: &'static str, corr: Option<u64>, ev: CallosumEvent) {
        use std::time::SystemTime;
        let env = EventEnvelope {
            id: 0, // you can generate a monotonic id if desired
            ts: SystemTime::now(),
            source,
            correlation_id: corr,
            event: ev,
        };

        let _id = self.callosum.publish_env(env);
    }
}

fn render(o: &CortexOutput) -> String {
    match o {
        CortexOutput::VerbalResponse(s) => s.clone(),
        CortexOutput::Action(a) => format!("{a:?}"),
        CortexOutput::Thought(t) => format!("[thought] {t}"),
        CortexOutput::Multi(v) => v.iter().map(render).collect::<Vec<_>>().join("\n"),
    }
}

fn is_final(o: &CortexOutput) -> bool {
    matches!(o, CortexOutput::VerbalResponse(_)) // tighten later with heuristics
}

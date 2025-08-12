use std::any::Any;
use std::fmt::Debug;
use std::time::SystemTime;

/// Trait for cloneable, type-erased payloads
pub trait CloneableAny: Any + Send + Sync + Debug {
    fn clone_box(&self) -> Box<dyn CloneableAny>;
    fn as_any(&self) -> &dyn Any;
}
impl<T> CloneableAny for T
where
    T: Any + Send + Sync + Clone + Debug + 'static,
{
    fn clone_box(&self) -> Box<dyn CloneableAny> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
}
impl Clone for Box<dyn CloneableAny> {
    fn clone(&self) -> Box<dyn CloneableAny> { self.clone_box() }
}

#[derive(Debug, Clone)]
pub enum CallosumEvent {
    RawInput(String),
    EvaluatedInput {
        input: String,
        urgency: f32,
        familiarity: f32,
        understanding: f32,
        drive_align: f32,
        tags: Vec<String>,
    },
    ResponseGenerated(String),
    MemoryRecalled(String),
    Dynamic(Box<dyn CloneableAny>),
}

#[derive(Debug, Clone)]
pub struct EventEnvelope {
    pub id: u64,                     // monotonic
    pub ts: SystemTime,              // when published
    pub source: &'static str,        // "parietal", "cortex", "ui", ...
    pub correlation_id: Option<u64>, // per-turn id
    pub event: CallosumEvent,
}

impl CallosumEvent {
    pub fn dynamic_as<T: Any>(&self) -> Option<&T> {
        match self {
            CallosumEvent::Dynamic(b) => b.as_any().downcast_ref::<T>(),
            _ => None,
        }
    }
    pub fn dynamic<T>(val: T) -> Self
    where
        T: Any + Send + Sync + Clone + Debug + 'static,
    {
        CallosumEvent::Dynamic(Box::new(val))
    }
}

#[derive(Debug)]
pub enum ForwardAct {
    Silent,
    Text(String),
    Speak(String),
    Both { text: String, speech: String },
    NewGoal { title: String, notes: String },
}

pub struct ForwardPolicy {
    pub speak_threshold: f32,
    pub text_threshold: f32,
    pub autospeak_enabled: bool,
}
impl Default for ForwardPolicy {
    fn default() -> Self {
        Self { speak_threshold: 0.60, text_threshold: 0.35, autospeak_enabled: true }
    }
}

pub struct ForwardProcessor {
    policy: ForwardPolicy,
}
impl ForwardProcessor {
    pub fn new(policy: ForwardPolicy) -> Self { Self { policy } }
    pub fn decide(&self, evt: &CallosumEvent) -> Option<ForwardAct> {
        match evt {
            CallosumEvent::EvaluatedInput { input, urgency, familiarity:_, understanding, drive_align, tags:_ } => {
                let score = 0.5*urgency + 0.3*understanding + 0.2*drive_align;
                if score >= self.policy.speak_threshold && self.policy.autospeak_enabled {
                    Some(ForwardAct::Both { text: input.clone(), speech: input.clone() })
                } else if score >= self.policy.text_threshold {
                    Some(ForwardAct::Text(input.clone()))
                } else {
                    Some(ForwardAct::Silent)
                }
            }
            CallosumEvent::ResponseGenerated(s) => {
                if self.policy.autospeak_enabled {
                    Some(ForwardAct::Both { text: s.clone(), speech: s.clone() })
                } else {
                    Some(ForwardAct::Text(s.clone()))
                }
            }
            _ => None
        }
    }
}

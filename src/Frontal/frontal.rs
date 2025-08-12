use crate::PFC::cortex::{CortexAction, CortexOutput};
use crate::Microthalamus::microthalamus::SyntheticDrive;

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/* ---------------- utilities ---------------- */

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

/* ---------------- working memory ---------------- */

#[derive(Debug, Clone)]
pub struct WorkingMemory {
    // very simple: just keep last N assistant outputs (for loop detection)
    recent_bot: Vec<String>,
    max_len: usize,
}

impl WorkingMemory {
    pub fn new(max_len: usize) -> Self {
        Self {
            recent_bot: Vec::with_capacity(max_len.min(256)),
            max_len,
        }
    }

    pub fn remember_bot(&mut self, s: &str) {
        if s.trim().is_empty() {
            return;
        }
        self.recent_bot.push(s.to_string());
        if self.recent_bot.len() > self.max_len {
            let overflow = self.recent_bot.len() - self.max_len;
            self.recent_bot.drain(0..overflow);
        }
    }

    /// Count how many times the same response text appears
    /// in the last `window` messages (including the newest one
    /// if you’ve already pushed it).
    fn duplicate_count(&self, text: &str, window: usize) -> usize {
        let n = self.recent_bot.len();
        let start = n.saturating_sub(window);
        self.recent_bot[start..]
            .iter()
            .filter(|t| t.as_str() == text)
            .count()
    }
}

/* ---------------- inhibition ---------------- */

#[derive(Debug, Clone, Copy)]
pub struct InhibitionConfig {
    /// If the same response text appears >= this many times recently, suppress.
    pub duplicate_response_limit: usize,
    /// How many last bot messages to consider for duplicate counting.
    pub loop_window: usize,
    /// Cooldown between identical actions, in milliseconds.
    pub action_cooldown_ms: u128,
}

impl Default for InhibitionConfig {
    fn default() -> Self {
        Self {
            duplicate_response_limit: 3,
            loop_window: 5,
            action_cooldown_ms: 2_000, // 2s
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct InhibitionState {
    /// last execution time per action key
    pub last_action_ms: HashMap<String, u128>,
}

fn action_key(a: &CortexAction) -> String {
    // cheap, good enough: stringify the enum
    format!("{a:?}")
}

/* ---------------- decisions & context ---------------- */

#[derive(Debug)]
pub enum FrontalDecision {
    ExecuteImmediately(CortexOutput),
    QueueForLater(CortexOutput),
    SuppressResponse(String),
    ReplaceWithGoal(String),
}

#[derive(Debug)]
pub struct FrontalContext<'a> {
    pub active_goals: Vec<String>,
    pub current_drive_weights: &'a HashMap<String, SyntheticDrive>,
    pub last_response: Option<&'a CortexOutput>,

    // NEW: interior-mutable state so callers don’t need &mut
    pub working_memory: RefCell<WorkingMemory>,
    pub inhib_state: RefCell<InhibitionState>,
    pub inhib_cfg: InhibitionConfig,
}

/* ---------------- core policy ---------------- */

pub fn evaluate_frontal_control(output: &CortexOutput, context: &FrontalContext) -> FrontalDecision {
    // 0) hard filters you already had
    if let CortexOutput::VerbalResponse(text) = output {
        if text.contains("loop") {
            return FrontalDecision::SuppressResponse("Looping detected.".to_string());
        }
        if text.contains("let me think")
            && context.active_goals.contains(&"urgent_action".to_string())
        {
            return FrontalDecision::ReplaceWithGoal("This is urgent — respond directly.".to_string());
        }
    }

    // 1) inhibition: duplicate responses / loopiness
    if let CortexOutput::VerbalResponse(text) = output {
        let mut wm = context.working_memory.borrow_mut();

        // We check *before* remembering to allow immediate suppression on repeats
        let dup = wm.duplicate_count(text, context.inhib_cfg.loop_window);
        if dup + 1 /* include this one */ >= context.inhib_cfg.duplicate_response_limit {
            return FrontalDecision::SuppressResponse("Suppressed duplicate response.".into());
        }

        // allow it; remember for future checks
        wm.remember_bot(text);
    }

    // 2) inhibition: action cooldowns
    if let CortexOutput::Action(a) = output {
        let key = action_key(a);
        let now = now_ms();
        let mut st = context.inhib_state.borrow_mut();
        if let Some(&last) = st.last_action_ms.get(&key) {
            if now.saturating_sub(last) < context.inhib_cfg.action_cooldown_ms {
                // Either queue (if you want to run later) or suppress.
                return FrontalDecision::QueueForLater(output.clone());
            }
        }
        // record allowance
        st.last_action_ms.insert(key, now);
    }

    // 3) (optional) last_response guard: if we literally just said the same thing, suppress
    if let (Some(CortexOutput::VerbalResponse(prev)), CortexOutput::VerbalResponse(curr)) =
        (context.last_response, output)
    {
        if prev == curr {
            return FrontalDecision::SuppressResponse("Redundant response.".into());
        }
    }

    // 4) all good
    FrontalDecision::ExecuteImmediately(output.clone())
}

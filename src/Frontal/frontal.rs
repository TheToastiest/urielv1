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
/* ---------------- acks/policy ---------------- */
const ACK_MINIMAL: &str   = "(noted)";
const ACK_DUPLICATE: &str = "(same as before)";
const ACK_LOOP: &str      = "(loop prevented)";

/* ---------------- working memory ---------------- */


fn now_sec() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

#[derive(Debug, Clone)]
pub struct WorkingMemory {
    recent_bot: Vec<String>,
    max_len: usize,
    // NEW:
    current_focus: Option<(u64, String)>, // (ts, text)
}

impl WorkingMemory {
    pub fn new(max_len: usize) -> Self {
        Self { recent_bot: Vec::with_capacity(max_len.min(256)),
            max_len,
            current_focus: None }
    }

    pub fn set_focus(&mut self, text: &str) {
        self.current_focus = Some((now_sec(), text.trim().to_string()));
    }
    pub fn focus_if_recent(&self, max_age_sec: u64) -> Option<&str> {
        self.current_focus.as_ref()
            .filter(|(ts, _)| now_sec().saturating_sub(*ts) <= max_age_sec)
            .map(|(_, s)| s.as_str())
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
    // 0) hard filters -> minimal acks, not silence
    if let CortexOutput::VerbalResponse(text) = output {
        // Be stricter than substring "loop" if you want; keeping it simple for now.
        if text.to_ascii_lowercase().contains("loop") {
            // speak minimally, do not update WM (we didn't accept the content)
            return FrontalDecision::ExecuteImmediately(
                CortexOutput::VerbalResponse(ACK_LOOP.into())
            );
        }
        if text.contains("let me think")
            && context.active_goals.contains(&"urgent_action".to_string())
        {
            // reroute as goal (still visible)
            return FrontalDecision::ReplaceWithGoal("This is urgent — respond directly.".to_string());
        }
    }

    // 1) inhibition: duplicate responses / loopiness
    if let CortexOutput::VerbalResponse(text) = output {
        let mut wm = context.working_memory.borrow_mut();

        // Check BEFORE remembering to allow immediate downgrade on repeats.
        let dup = wm.duplicate_count(text, context.inhib_cfg.loop_window);
        if dup + 1 >= context.inhib_cfg.duplicate_response_limit {
            // minimal ack instead of hard suppress; DO NOT remember the repeated text
            return FrontalDecision::ExecuteImmediately(
                CortexOutput::VerbalResponse(ACK_DUPLICATE.into())
            );
        }

        // accept; remember for future duplicate detection
        wm.remember_bot(text);
    }

    // 2) inhibition: action cooldowns (queue actions, still "say something" via caller)
    if let CortexOutput::Action(a) = output {
        let key = action_key(a);
        let now = now_ms();
        let mut st = context.inhib_state.borrow_mut();
        if let Some(&last) = st.last_action_ms.get(&key) {
            if now.saturating_sub(last) < context.inhib_cfg.action_cooldown_ms {
                // caller prints "(queued)"; state not updated
                return FrontalDecision::QueueForLater(output.clone());
            }
        }
        // record allowance
        st.last_action_ms.insert(key, now);
    }

    // 3) last_response guard -> minimal ack vs silence
    if let (Some(CortexOutput::VerbalResponse(prev)), CortexOutput::VerbalResponse(curr)) =
        (context.last_response, output)
    {
        if prev == curr {
            return FrontalDecision::ExecuteImmediately(
                CortexOutput::VerbalResponse(ACK_DUPLICATE.into())
            );
        }
    }

    // 4) all good
    FrontalDecision::ExecuteImmediately(output.clone())
}


use std::collections::HashMap;
use std::time::{Duration, Instant};
use crossbeam_channel::Sender;
use super::goal::Goal;

/// Fast-ish keyword scan (lowercases once)
pub fn evaluate_priority(context: &str, input: &str) -> f32 {
    const KEYWORDS: [&str; 5] = ["urgent", "now", "critical", "important", "immediately"];

    let input_lower = input.to_lowercase();
    let mut score: f32 = 0.0;

    // keyword hits (max ~1.5 if all match; clamp later)
    for kw in KEYWORDS.iter() {
        if input_lower.contains(kw) {
            score += 0.3;
        }
    }

    // longer inputs likely carry more detail
    if input.len() > 50 {
        score += 0.2;
    }

    // context flags
    if context.contains("alert") || context.contains("warning") {
        score += 0.2;
    }

    score.clamp(0.0, 1.0)
}

#[derive(Clone, Debug)]
pub struct SyntheticDrive {
    pub name: String,
    pub weight: f32,          // 0..1
    pub decay_rate: f32,      // per-second linear decay (e.g., 0.01 = 1%/s)
    pub cooldown: Duration,   // min time between activations
    pub last_activated: Instant,
    pub reward_scale: f32,
    pub enabled: bool,
    pub vector: Vec<f32>,
    pub vector_key: String,
    last_decay_check: Instant,
}

impl SyntheticDrive {
    pub fn new(
        name: impl Into<String>,
        weight: f32,
        decay_rate: f32,
        cooldown: Duration,
        reward_scale: f32,
        vector_key: impl Into<String>,
        vector: Vec<f32>,
    ) -> Self {
        let now = Instant::now();
        Self {
            name: name.into(),
            weight: weight.clamp(0.0, 1.0),
            decay_rate: decay_rate.max(0.0),
            cooldown,
            last_activated: now - cooldown, // allow immediate activation if needed
            reward_scale,
            enabled: true,
            vector,
            vector_key: vector_key.into(),
            last_decay_check: now,
        }
    }

    /// Apply time-based decay since last check.
    pub fn tick(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_decay_check).as_secs_f32();
        if dt > 0.0 && self.decay_rate > 0.0 {
            self.weight = (self.weight - self.decay_rate * dt).clamp(0.0, 1.0);
        }
        self.last_decay_check = now;
    }

    pub fn reinforce(&mut self, reward: f32) {
        self.weight = (self.weight + reward * self.reward_scale).clamp(0.0, 1.0);
    }

    pub fn apply_urgency(&mut self, urgency: f32) {
        self.weight = (self.weight + urgency).clamp(0.0, 1.0);
    }

    #[inline]
    pub fn can_fire(&self) -> bool {
        self.last_activated.elapsed() >= self.cooldown
    }
}

pub struct MicroThalamus {
    pub drives: HashMap<String, SyntheticDrive>,
    pub active_goal: Option<Goal>,
    pub cortex_sender: Sender<Goal>,
    next_goal_id: u64,
}

impl MicroThalamus {
    pub fn new(cortex_sender: Sender<Goal>) -> Self {
        Self {
            drives: HashMap::new(),
            active_goal: None,
            cortex_sender,
            next_goal_id: 1,
        }
    }

    #[inline]
    fn next_goal_id(&mut self) -> u64 {
        let id = self.next_goal_id;
        self.next_goal_id += 1;
        id
    }

    pub fn register_drive(&mut self, drive: SyntheticDrive) {
        self.drives.insert(drive.name.clone(), drive);
    }

    /// Periodic maintenance: decay, choose top drive, optionally emit a goal.
    pub fn tick(&mut self) {
        // decay first
        for d in self.drives.values_mut() {
            if d.enabled {
                d.tick();
            }
        }

        // pick top drive by weight (enabled, above threshold)
        let top_key_opt = self
            .drives
            .iter()
            .filter(|(_, d)| d.enabled && d.weight > 0.1)
            .max_by(|a, b| a.1.weight.partial_cmp(&b.1.weight).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone());

        if let Some(top_key) = top_key_opt {
            // Check cooldown without taking &mut first
            let can_fire = self.drives.get(&top_key).map(|d| d.can_fire()).unwrap_or(false);

            if can_fire {
                // Generate ID while nothing is mut-borrowed
                let gid = self.next_goal_id();

                // Grab the name immutably (or fallback to key)
                let drive_name = self
                    .drives
                    .get(&top_key)
                    .map(|d| d.name.clone())
                    .unwrap_or_else(|| top_key.clone());

                let goal = Goal::from_drive(gid, &drive_name);

                if self.cortex_sender.send(goal.clone()).is_ok() {
                    // Now take the mutable borrow to update fields
                    if let Some(top_drive) = self.drives.get_mut(&top_key) {
                        top_drive.last_activated = Instant::now();
                        top_drive.weight = (top_drive.weight - 0.05).clamp(0.0, 1.0);
                    }
                    self.active_goal = Some(goal);
                }
            }
        }

    }

    pub fn reinforce_drive(&mut self, drive_name: &str, reward: f32) {
        if let Some(drive) = self.drives.get_mut(drive_name) {
            drive.reinforce(reward);
        }
    }

    /// Map external stimulus into drive urgency. Keep cheap and explicit.
    pub fn inject_stimulus(&mut self, context: &str, input: &str) {
        let urgency_score = evaluate_priority(context, input);

        // Curiosity tracks general “interesting/novel” stimuli
        if let Some(curiosity) = self.drives.get_mut("Curiosity") {
            curiosity.apply_urgency(urgency_score * 0.6);
        }

        // Conflict resolution when mismatch is present
        let has_conflict = context.contains("conflict")
            || input.contains("contradiction")
            || input.contains("conflict")
            || input.contains("disagree")
            || input.contains("inconsistent");

        if has_conflict {
            if let Some(resolve) = self.drives.get_mut("ResolveConflict") {
                resolve.apply_urgency(urgency_score * 0.75);
            }
        }
    }
}

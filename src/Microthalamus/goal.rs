use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

pub type DriveVec = HashMap<String, f32>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GoalStatus {
    Pending,
    Active,
    Succeeded,
    Failed,
    Aborted,
}

#[derive(Clone, Debug)]
pub struct Constraint {
    pub key: String,
    pub min: f32,
    pub max: f32,
}

#[derive(Clone)]
pub struct Goal {
    pub id: u64,
    pub label: String,
    pub status: GoalStatus,
    pub constraints: Vec<Constraint>,

    /// Priority scorer: higher = sooner. Hot-swappable per goal.
    pub priority_fn: Arc<dyn Fn(&DriveVec, &Ctx) -> f32 + Send + Sync>,

    /// Explicit termination condition.
    pub success_test: Arc<dyn Fn(&Ctx) -> bool + Send + Sync>,

    /// Soft decay per second (0.02 = 2%/s). 0.0 disables.
    pub decay: f32,

    created_at: Instant,
    last_scored: Instant,
    pub last_score: f32,
}

#[derive(Default, Clone)]
pub struct Ctx {
    /// Signals available to scorers/terminators (e.g., "weight", "conflict").
    pub signals: HashMap<String, f32>,
}

impl std::fmt::Debug for Goal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Goal")
            .field("id", &self.id)
            .field("label", &self.label)
            .field("status", &self.status)
            .field("decay", &self.decay)
            .field("constraints", &self.constraints)
            .field("last_score", &self.last_score)
            .finish()
    }
}

#[inline]
fn get_drive(drives: &DriveVec, key: &str) -> f32 {
    if let Some(v) = drives.get(key) {
        *v
    } else {
        // case-insensitive fallback
        let lower = key.to_lowercase();
        let upper = key.to_uppercase();
        drives
            .get(&lower)
            .copied()
            .or_else(|| drives.get(&upper).copied())
            .unwrap_or(0.0)
    }
}

impl Goal {
    pub fn new<L, P, S>(id: u64, label: L, priority_fn: P, success_test: S) -> Self
    where
        L: Into<String>,
        P: Fn(&DriveVec, &Ctx) -> f32 + Send + Sync + 'static,
        S: Fn(&Ctx) -> bool + Send + Sync + 'static,
    {
        Self {
            id,
            label: label.into(),
            status: GoalStatus::Pending,
            constraints: Vec::new(),
            priority_fn: Arc::new(priority_fn),
            success_test: Arc::new(success_test),
            decay: 0.0,
            created_at: Instant::now(),
            last_scored: Instant::now(),
            last_score: 0.0,
        }
    }

    /// Drive-origin goal that respects case-insensitive drive names.
    pub fn from_drive(id: u64, drive_name: &str) -> Self {
        let dn = drive_name.to_string();
        Self::new(
            id,
            format!("GoalFromDrive:{dn}"),
            move |drives, ctx| {
                // priority = drive weight × (1 + ctx.weight)
                let w = get_drive(drives, &dn);
                let modw = 1.0 + ctx.signals.get("weight").copied().unwrap_or(0.0).max(0.0);
                w * modw
            },
            |_ctx| false,
        )
    }

    /// Mark as active when scheduled/selected.
    pub fn activate(&mut self) {
        self.status = GoalStatus::Active;
    }

    pub fn with_decay(mut self, per_sec: f32) -> Self {
        self.decay = per_sec.max(0.0);
        self
    }

    pub fn with_constraint(mut self, c: Constraint) -> Self {
        self.constraints.push(c);
        self
    }

    /// Convenience: succeed when signal `key` < `threshold`.
    pub fn success_on_signal_below(mut self, key: &str, threshold: f32) -> Self {
        let k = key.to_string();
        self.success_test = Arc::new(move |ctx: &Ctx| ctx.signals.get(&k).copied().unwrap_or(1.0) < threshold);
        self
    }

    /// Score with decay + constraint penalties. Caches last_score for UI.
    pub fn score(&mut self, drives: &DriveVec, ctx: &Ctx) -> f32 {
        let base = (self.priority_fn)(drives, ctx);

        // Constraint penalty
        let penalty: f32 = self
            .constraints
            .iter()
            .map(|c| {
                let v = *ctx.signals.get(&c.key).unwrap_or(&0.0);
                if v < c.min {
                    c.min - v
                } else if v > c.max {
                    v - c.max
                } else {
                    0.0
                }
            })
            .sum();

        let dt = self.last_scored.elapsed().as_secs_f32();
        let decay_factor = if self.decay > 0.0 {
            (1.0 - self.decay).powf(dt.max(0.0))
        } else {
            1.0
        };

        self.last_scored = Instant::now();
        self.last_score = (base - penalty).max(0.0) * decay_factor;
        self.last_score
    }

    pub fn try_finish(&mut self, ctx: &Ctx) -> bool {
        if (self.success_test)(ctx) {
            self.status = GoalStatus::Succeeded;
            true
        } else {
            false
        }
    }

    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

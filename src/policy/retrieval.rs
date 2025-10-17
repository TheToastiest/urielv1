use rusqlite::Connection;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy)]
pub struct Arm { pub name: &'static str }

pub struct Bandit {
    pub intent: &'static str,           // e.g. "concept_q" | "general_q"
    pub arms: &'static [Arm],
    pub epsilon: f32,                   // exploration
}
fn now() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() }

impl Bandit {
    pub fn choose(&self, conn: &Connection) -> &'static str {
        use rand::{thread_rng, Rng};
        let mut rng = thread_rng();
        if rng.gen::<f32>() < self.epsilon {
            return self.arms[rng.gen_range(0..self.arms.len())].name;
        }
        // exploit
        let mut best = (self.arms[0].name, f32::MIN);
        for a in self.arms {
            let v: f32 = conn.query_row(
                "SELECT value FROM retrieval_policy WHERE intent=?1 AND arm=?2",
                rusqlite::params![self.intent, a.name],
                |r| r.get::<_, f32>(0),
            ).unwrap_or(0.0);
            if v > best.1 { best = (a.name, v); }
        }
        best.0
    }

    pub fn update(&self, conn: &Connection, arm: &str, reward: f32) {
        let (mut n, mut v):(i64,f32) = conn.query_row(
            "SELECT n, value FROM retrieval_policy WHERE intent=?1 AND arm=?2",
            rusqlite::params![self.intent, arm],
            |r| Ok((r.get(0)?, r.get(1)?)),
        ).unwrap_or((0, 0.0));
        n += 1;
        v += (reward - v) / (n as f32);
        let _ = conn.execute(
            "INSERT INTO retrieval_policy(intent,arm,n,value,updated_at)
             VALUES(?1,?2,?3,?4,?5)
             ON CONFLICT(intent,arm) DO UPDATE SET n=excluded.n, value=excluded.value, updated_at=excluded.updated_at",
            rusqlite::params![self.intent, arm, n, v, now() as i64],
        );
    }
}

// src/api/runtime_api.rs
use std::sync::Arc;
use anyhow::Result;
use serde::Serialize;

use crate::callosum::callosum::CorpusCallosum;
use crate::callosum::metacog_subscriber::EpisodeEvent;
use crate::knowledge::{schema as kschema, ops as kops, types as ktypes};
use rusqlite;

fn now_sec() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64
}

pub struct RuntimeCore {
    pub bus: Arc<CorpusCallosum>,
    pub conn: rusqlite::Connection,
}

impl RuntimeCore {
    pub fn new_default() -> Result<Self> {
        let conn = rusqlite::Connection::open("data/semantic_memory.db")?;
        conn.busy_timeout(std::time::Duration::from_millis(500)).ok();
        kschema::apply_cognition_migration(&conn)?;
        kops::ensure_global_entity(&conn)?;

        let bus = Arc::new(CorpusCallosum::new());
        Ok(Self { bus, conn })
    }

    /// Minimal “ask” that records episodes + echoes. We’ll swap the echo for
    /// your real answering path next (SMIE/cortex).
    pub fn ask_json(&mut self, prompt: &str) -> Result<String> {
        let ts = now_sec();
        let turn_id = format!("ffi-{}-{}", ts, rand::random::<u32>());

        // Store + publish user episode
        let _ = kops::record_episode(
            &self.conn,
            ktypes::EpisodeWrite {
                ts,
                role: "user",
                text: prompt,
                turn_id: Some(&turn_id),
                smie_id: None,
                ctx_json: None,
            },
        );
        self.bus.publish_dynamic("ffi", None, EpisodeEvent {
            role: "user".into(),
            text: prompt.to_string(),
            turn_id: Some(turn_id.clone()),
        });

        // TODO: replace this with real answer pipeline (SMIE / cortex / KB)
        let reply = format!("(URIEL runtime) {}", prompt);

        // Store + publish agent episode
        let _ = kops::record_episode(
            &self.conn,
            ktypes::EpisodeWrite {
                ts,
                role: "agent",
                text: &reply,
                turn_id: Some(&turn_id),
                smie_id: None,
                ctx_json: None,
            },
        );
        self.bus.publish_dynamic("ffi", None, EpisodeEvent {
            role: "agent".into(),
            text: reply.clone(),
            turn_id: Some(turn_id),
        });

        #[derive(Serialize)]
        struct Resp<'a> { text: &'a str }
        Ok(serde_json::to_string(&Resp { text: &reply })?)
    }
}

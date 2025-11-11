// src/callosum/metacog_subscriber.rs
use std::sync::Arc;
use std::thread;

use rusqlite::Connection;

use crate::callosum::callosum::CorpusCallosum;
use crate::knowledge::{ops as kops};
// AFTER
use crate::callosum::event::{CallosumEvent, EventEnvelope};

/// ---- Dynamic payloads carried inside CallosumEvent::Dynamic ----
#[derive(Clone, Debug)]
pub struct EpisodeEvent {
    pub role: String,            // "user" | "agent" | "system" | "tool"
    pub text: String,
    pub turn_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct FactUpsertEvent {
    pub predicate: String,       // e.g., "name", "you are", "def:cpp"
    pub value: String,           // obj_text (already canonicalized if you used canon_value_for)
    pub ts: i64,
}
/// Start a background thread that subscribes to the Callosum and reacts to events.
pub fn spawn_metacog_subscriber(bus: Arc<CorpusCallosum>, conn: Connection) -> thread::JoinHandle<()> {
    let rx = bus.subscribe("metacog");
    thread::spawn(move || {
        // NOTE: keep this loop lightweight; heavy work should be fanned out if needed.
        while let Ok(env) = rx.recv() {
            if let Some(act) = handle_event(&conn, &env) {
                eprintln!("[metacog] {act}");
            }
        }
        eprintln!("[metacog] subscriber exited");
    })
}

fn handle_event(conn: &Connection, env: &Arc<EventEnvelope>) -> Option<String> {
    match &env.event {
        CallosumEvent::Dynamic(b) => {
            // Episode
            if let Some(ep) = b.as_any().downcast_ref::<EpisodeEvent>() {
                // Minimal: you could mine beliefs/defs here later
                return Some(format!("episode role={} len={}", ep.role, ep.text.len()));
            }
            // Fact upsert
            if let Some(fu) = b.as_any().downcast_ref::<FactUpsertEvent>() {
                // TMS-lite: prefer latest for this predicate (GLOBAL subject)
                let _ = kops::prefer_latest(conn, None, &fu.predicate);
                return Some(format!("fact_upsert {}='{}'", fu.predicate, fu.value));
            }
            None
        }
        // You can listen to other built-in events if you use them.
        CallosumEvent::RawInput(s) => Some(format!("raw_input len={}", s.len())),
        CallosumEvent::ResponseGenerated(s) => Some(format!("response len={}", s.len())),
        _ => None,
    }
}

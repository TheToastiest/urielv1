// src/api/runtime_api.rs
use std::sync::Arc;
use anyhow::Result;
use serde::Serialize;

use crate::callosum::callosum::CorpusCallosum;
use crate::callosum::metacog_subscriber::EpisodeEvent;
use crate::knowledge::{schema as kschema, ops as kops, types as ktypes};
use rusqlite::{self, params};

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

    /// Heuristic router (deploy-legal; no label leakage).
    /// Returns one of: "direct", "retrieve_then_answer", "tool:code", "tool:sql".
    fn route_query(&self, q: &str) -> &'static str {
        let l = q.to_ascii_lowercase();

        let is_sql =
            l.contains(" select ") || l.contains(" insert ") || l.contains(" update ")
                || l.contains(" delete ") || l.contains(" create table ")
                || (l.contains(" from ") && l.contains(" where "));

        let is_code =
            l.contains("```") || l.contains('`') || l.contains('{') || l.contains('}')
                || l.contains(';') || l.contains("fn ") || l.contains("let ")
                || l.contains("#include") || l.contains("class ")
                || l.contains("std::") || l.contains("traceback")
                || l.contains("panic!") || l.contains("error");

        let is_retr =
            l.contains('?')
                || l.starts_with("what ") || l.starts_with("how ") || l.starts_with("why ")
                || l.starts_with("when ") || l.starts_with("who ") || l.starts_with("where ")
                || l.contains(" vs ") || l.contains(" difference ");

        if is_sql { "tool:sql" }
        else if is_code { "tool:code" }
        else if is_retr { "retrieve_then_answer" }
        else { "direct" }
    }

    /// Best-effort retrieval from the episodes log.
    /// Tries a simple LIKE search; if schema differs, returns empty Vec.
    fn retrieve_topk(&self, q: &str, k: usize) -> Vec<String> {
        let pat = format!("%{}%", q.replace('%', "\\%").replace('_', "\\_"));
        let mut out: Vec<String> = Vec::new();

        // Prefer newest content first.
        if let Ok(mut stmt) = self.conn.prepare(
            "SELECT text FROM episodes
             WHERE text LIKE ?1 ESCAPE '\\'
             ORDER BY ts DESC
             LIMIT ?2"
        ) {
            let rows = stmt.query_map(params![pat, k as i64], |row| row.get::<_, String>(0));
            if let Ok(iter) = rows {
                for r in iter.flatten() {
                    if !r.trim().is_empty() { out.push(r); }
                }
            }
        }
        out
    }

    /// Minimal “ask” that routes and (for retrieve) pulls context from memory.
    /// This replaces the echo with a real, deploy-legal pipeline.
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

        // -------- real routing + answering --------
        let route = self.route_query(prompt);

        let reply = match route {
            "retrieve_then_answer" => {
                let hits = self.retrieve_topk(prompt, 5);
                if hits.is_empty() {
                    // No hits yet; degrade gracefully.
                    format!("(retrieve) No matching memory yet.\n\nQ: {prompt}")
                } else {
                    // Naive extractive answer: show top snippets (newest first).
                    let mut s = String::new();
                    s.push_str("(retrieve)\n");
                    for (i, h) in hits.iter().take(3).enumerate() {
                        s.push_str(&format!("{}. {}\n", i + 1, h.trim()));
                    }
                    s.push_str("\nQ: ");
                    s.push_str(prompt);
                    s
                }
            }
            "tool:code" => {
                // Surface the route; actual code toolchain can subscribe via the bus.
                format!("(code-route) I can draft or debug code for:\n{prompt}")
            }
            "tool:sql" => {
                format!("(sql-route) I can help with SQL for:\n{prompt}")
            }
            _ => {
                // Direct: keep it terse and useful for now.
                // If you wire SMIE/cortex later, replace this arm to call them.
                format!("(direct) {prompt}")
            }
        };
        // ------------------------------------------

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

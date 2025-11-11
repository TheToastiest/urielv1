use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension};
use crate::knowledge::types::{EpisodeWrite, FactWrite};

pub fn ensure_global_entity(conn: &Connection) -> Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO kg_entity(id,name,kind,created_at)
         VALUES(1,'__GLOBAL__','system',strftime('%s','now'))", [],
    )?;
    Ok(())
}

pub fn record_episode(conn: &Connection, e: EpisodeWrite<'_>) -> Result<i64> {
    conn.execute(
        "INSERT INTO episode(ts,role,text,turn_id,smie_id,ctx)
         VALUES(?1,?2,?3,?4,?5,?6)",
        params![e.ts, e.role, e.text, e.turn_id, e.smie_id, e.ctx_json],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn upsert_fact(conn: &Connection, w: FactWrite) -> Result<i64> {
    let subj = w.subj.unwrap_or(1);
    conn.execute(
        "INSERT INTO kg_fact(subj_id,predicate,obj_text,obj_entity_id,ts,confidence,preferred)
         VALUES(?1,?2,?3,?4,?5,?6,0)
         ON CONFLICT(subj_id,predicate,obj_text) DO UPDATE SET
           ts=excluded.ts,
           confidence=excluded.confidence",
        params![subj, w.predicate, w.obj_text, w.obj_entity, w.ts, w.confidence],
    )?;

    let fact_id: i64 = conn.prepare(
        "SELECT id FROM kg_fact WHERE subj_id=?1 AND predicate=?2 AND obj_text IS ?3"
    )?.query_row(params![subj, w.predicate, w.obj_text], |r| r.get(0))?;

    for ev in w.evidence {
        conn.execute(
            "INSERT INTO justification(fact_id,evidence_ref,method,score,added_at)
             VALUES(?1,?2,?3,?4,?5)",
            params![fact_id, ev, w.method, 0.5_f32, w.ts],
        )?;
    }
    Ok(fact_id)
}
/// Preferred read: prefer `preferred=1`, else by confidence then recency.
pub fn read_best_fact_text(
    conn: &Connection,
    subj: Option<i64>,
    predicate: &str,
) -> Result<Option<String>> {
    let subj = subj.unwrap_or(1);
    let mut st = conn.prepare(
        "SELECT obj_text
           FROM kg_fact
          WHERE subj_id=?1 AND predicate=?2
          ORDER BY preferred DESC, confidence DESC, ts DESC
          LIMIT 1",
    )?;
    let v: Option<String> =
        st.query_row(params![subj, predicate], |r| r.get(0)).optional()?;
    Ok(v)
}

/// Quick “prefer latest best” until TMS lands.
pub fn prefer_latest(conn: &Connection, subj: Option<i64>, predicate: &str) -> Result<()> {
    let subj = subj.unwrap_or(1);
    let best: Option<i64> = conn
        .prepare(
            "SELECT id
               FROM kg_fact
              WHERE subj_id=?1 AND predicate=?2
              ORDER BY confidence DESC, ts DESC
              LIMIT 1",
        )?
        .query_row(params![subj, predicate], |r| r.get(0))
        .optional()?;
    conn.execute(
        "UPDATE kg_fact SET preferred=0 WHERE subj_id=?1 AND predicate=?2",
        params![subj, predicate],
    )?;
    if let Some(id) = best {
        conn.execute("UPDATE kg_fact SET preferred=1 WHERE id=?1", params![id])?;
    }
    Ok(())
}
// ===== Router policy stats (running mean reward per arm) =====
pub fn router_stats_init(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS router_stats(
          arm   TEXT PRIMARY KEY,
          n     INTEGER NOT NULL DEFAULT 0,
          mean  REAL    NOT NULL DEFAULT 0.0
        );
        INSERT OR IGNORE INTO router_stats(arm,n,mean)
          VALUES ('db',0,0.0),('normalized',0,0.0),('keyword',0,0.0),
                 ('original',0,0.0),('rows',0,0.0),('pfc',0,0.0);
        "#
    )
}

pub fn router_stats_update(conn: &Connection, arm: &str, reward: f32) -> rusqlite::Result<()> {
    router_stats_init(conn)?;
    // one-pass update: new_mean = (mean*n + reward) / (n+1)
    conn.execute_batch(&format!(
        "UPDATE router_stats
           SET mean = (mean * n + {r}) / (n + 1),
               n    = n + 1
         WHERE arm = '{a}';",
        r = reward, a = arm.replace('\'', "''")
    ))
}


// ========= Critic: justification score + learning queue =========

#[derive(Clone, Copy, Debug)]
pub struct CriticInput {
    pub is_concept: bool,
    pub answered_from: &'static str,   // "db","normalized","keyword","original","rows","pfc",...
    pub extract_ok: bool,              // value passed validation for the key
    pub consistency: Option<f32>,      // Some(1.0 match, 0.5 unknown, 0.0 mismatch)
    pub recency: f32,                  // 0..1 from rows_recency or similar
    pub retrieval_cos: Option<f32>,    // best cosine if retrieval used
    pub plan_success: Option<f32>,     // 0..1 (optional; use 0.5 if None)
}

#[derive(Clone, Copy, Debug)]
pub struct CriticOutcome {
    pub j: f32,                        // justification score (0..1)
    pub reward: f32,                   // training reward (same as j for now)
    pub verdict: &'static str,         // "good" | "hard" | "bad"
}

/// Map source → trust prior
fn source_trust(s: &str) -> f32 {
    match s {
        "db" => 0.95,
        "normalized" => 0.85,
        "keyword" => 0.70,
        "original" => 0.60,
        "rows" => 0.55,
        "pfc" => 0.50,
        _ => 0.20,
    }
}

/// Convert cosine-like sim (0..1) → 0..1 with a hinge around 0.5..0.8
fn retrieval_to01(cos: Option<f32>) -> f32 {
    match cos { Some(s) => ((s - 0.50) / 0.30).clamp(0.0, 1.0), None => 0.0 }
}

/// Compute justification score J and a coarse verdict.
pub fn critic_justify(ci: CriticInput) -> CriticOutcome {
    let base = source_trust(ci.answered_from);
    let retr = retrieval_to01(ci.retrieval_cos);
    let cons = ci.consistency.unwrap_or(0.5);
    let extr = if ci.extract_ok { 1.0 } else { 0.0 };
    let rec  = ci.recency.clamp(0.0, 1.0);
    let plan = ci.plan_success.unwrap_or(0.5);

    let j = if ci.is_concept {
        0.35*base + 0.25*retr + 0.15*extr + 0.15*cons + 0.05*rec + 0.05*plan
    } else {
        0.30*base + 0.30*retr + 0.15*extr + 0.10*cons + 0.10*rec + 0.05*plan
    }.clamp(0.0, 1.0);

    let verdict = if j >= 0.90 { "good" } else if j >= 0.35 { "hard" } else { "bad" };
    CriticOutcome { j, reward: j, verdict }
}

/// Ensure tables exist for online learning items.
fn ensure_learn_tables(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS learn_queue(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts INTEGER NOT NULL,
          question TEXT NOT NULL,
          positive TEXT NOT NULL,
          hardness REAL NOT NULL,     -- 1 - reward
          used INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS learn_negs(
          qid INTEGER NOT NULL,
          negative TEXT NOT NULL,
          sim REAL NOT NULL DEFAULT 0.0,
          FOREIGN KEY(qid) REFERENCES learn_queue(id) ON DELETE CASCADE
        );
        "#
    )?;
    Ok(())
}

/// Enqueue a single (question, positive) with reward ∈ [0,1]. Returns qid.
pub fn critic_enqueue_learning(
    conn: &Connection,
    question: &str,
    positive: &str,
    reward: f32,
) -> anyhow::Result<i64> {
    ensure_learn_tables(conn)?;
    let hardness = (1.0 - reward).clamp(0.0, 1.0);
    conn.execute(
        "INSERT INTO learn_queue(ts,question,positive,hardness,used)
         VALUES(strftime('%s','now'),?1,?2,?3,0)",
        params![question, positive, hardness],
    )?;
    let qid = conn.last_insert_rowid();
    Ok(qid)
}

/// Optionally attach a few negatives (you can mine from SMIE elsewhere).
pub fn critic_attach_negative(conn: &Connection, qid: i64, negative: &str, sim: f32) -> anyhow::Result<()> {
    ensure_learn_tables(conn)?;
    conn.execute(
        "INSERT INTO learn_negs(qid,negative,sim) VALUES(?1,?2,?3)",
        params![qid, negative, sim],
    )?;
    Ok(())
}

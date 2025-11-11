use anyhow::Result;
use rusqlite::Connection;

const MIGRATION: &str = r#"
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

BEGIN;

CREATE TABLE IF NOT EXISTS schema_migrations(
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  applied_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS kg_entity(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT '',
  created_at INTEGER NOT NULL,
  UNIQUE(name, kind)
);
CREATE INDEX IF NOT EXISTS idx_entity_name ON kg_entity(name);

INSERT OR IGNORE INTO kg_entity(id, name, kind, created_at)
VALUES (1, '__GLOBAL__', 'system', strftime('%s','now'));

CREATE TABLE IF NOT EXISTS kg_fact(
  id INTEGER PRIMARY KEY,
  subj_id INTEGER NOT NULL DEFAULT 1,
  predicate TEXT NOT NULL,
  obj_text TEXT,
  obj_entity_id INTEGER,
  ts INTEGER NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.5,
  preferred INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY(subj_id) REFERENCES kg_entity(id) ON DELETE CASCADE,
  FOREIGN KEY(obj_entity_id) REFERENCES kg_entity(id) ON DELETE CASCADE,
  UNIQUE(subj_id, predicate, obj_text)
);
CREATE INDEX IF NOT EXISTS idx_fact_lookup
  ON kg_fact(subj_id, predicate, preferred DESC, ts DESC);

CREATE TABLE IF NOT EXISTS episode(
  id INTEGER PRIMARY KEY,
  ts INTEGER NOT NULL,
  role TEXT NOT NULL,
  text TEXT NOT NULL,
  turn_id TEXT,
  smie_id TEXT,
  ctx TEXT
);
CREATE INDEX IF NOT EXISTS idx_episode_ts ON episode(ts);

CREATE TABLE IF NOT EXISTS justification(
  id INTEGER PRIMARY KEY,
  fact_id INTEGER NOT NULL,
  evidence_ref TEXT NOT NULL,
  method TEXT NOT NULL,
  score REAL NOT NULL DEFAULT 0.5,
  added_at INTEGER NOT NULL,
  FOREIGN KEY(fact_id) REFERENCES kg_fact(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_just_fact ON justification(fact_id);

CREATE TABLE IF NOT EXISTS plan_run(
  id INTEGER PRIMARY KEY,
  turn_id TEXT NOT NULL,
  status TEXT NOT NULL,
  target_conf REAL NOT NULL DEFAULT 0.75,
  started_at INTEGER NOT NULL,
  finished_at INTEGER
);

CREATE TABLE IF NOT EXISTS plan_step(
  id INTEGER PRIMARY KEY,
  plan_id INTEGER NOT NULL,
  step_idx INTEGER NOT NULL,
  action TEXT NOT NULL,
  args_json TEXT NOT NULL,
  exp_value REAL NOT NULL DEFAULT 0.0,
  started_at INTEGER,
  finished_at INTEGER,
  success INTEGER,
  result_json TEXT,
  FOREIGN KEY(plan_id) REFERENCES plan_run(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_plan_step_plan ON plan_step(plan_id);

CREATE TABLE IF NOT EXISTS tool_run(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  args_json TEXT NOT NULL,
  result_json TEXT,
  success INTEGER,
  latency_ms INTEGER,
  started_at INTEGER NOT NULL,
  finished_at INTEGER
);

CREATE TABLE IF NOT EXISTS policy_event(
  id INTEGER PRIMARY KEY,
  ts INTEGER NOT NULL,
  policy TEXT NOT NULL,
  context_json TEXT NOT NULL,
  chosen TEXT NOT NULL,
  reward REAL
);

COMMIT;
"#;

pub fn apply_cognition_migration(conn: &Connection) -> Result<()> {
    conn.execute_batch(MIGRATION)?;
    Ok(())
}

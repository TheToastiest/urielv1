use anyhow::Result;
use rusqlite::Connection;

// stubs for later phases — safe no-ops
pub fn recompute_confidence(_conn: &Connection, _fact_id: i64) -> Result<f32> { Ok(0.5) }
pub fn choose_preferred(_conn: &Connection, _subj: Option<i64>, _predicate: &str) -> Result<()> { Ok(()) }

use rusqlite::OptionalExtension;
use rusqlite::{params, Connection};
use std::time::{SystemTime, UNIX_EPOCH};

fn conn() -> rusqlite::Result<Connection> {
    let c = Connection::open("semantic_memory.db")?;
    c.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS kv_facts (
            k_norm TEXT PRIMARY KEY,
            key_raw TEXT NOT NULL,
            value  TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS kv_alias (
            alias_norm TEXT PRIMARY KEY,
            k_norm     TEXT NOT NULL,
            FOREIGN KEY(k_norm) REFERENCES kv_facts(k_norm)
        );
        "#
    )?;
    Ok(c)
}

fn norm(s: &str) -> String {
    s.trim().to_lowercase().replace(char::is_whitespace, " ")
}

fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64
}

pub fn upsert_fact(key: &str, value: &str) -> rusqlite::Result<()> {
    let c = conn()?;
    let k_norm = norm(key);
    c.execute(
        r#"
        INSERT INTO kv_facts (k_norm, key_raw, value, updated_at)
        VALUES (?1, ?2, ?3, ?4)
        ON CONFLICT(k_norm) DO UPDATE SET
            key_raw=excluded.key_raw,
            value=excluded.value,
            updated_at=excluded.updated_at
        "#,
        params![k_norm, key, value, now()],
    )?;
    Ok(())
}

pub fn add_alias(alias: &str, key: &str) -> rusqlite::Result<()> {
    let c = conn()?;
    c.execute(
        r#"INSERT OR REPLACE INTO kv_alias (alias_norm, k_norm) VALUES (?1, ?2)"#,
        params![norm(alias), norm(key)],
    )?;
    Ok(())
}

pub fn get_fact(term: &str) -> rusqlite::Result<Option<(String,String)>> {
    let c = conn()?;
    let t = norm(term);

    // 1) exact key
    if let Ok(mut st) = c.prepare("SELECT key_raw, value FROM kv_facts WHERE k_norm = ?1") {
        if let Some(row) = st.query_row([t.clone()], |r| Ok((r.get(0)?, r.get(1)?))).optional()? {
            return Ok(Some(row));
        }
    }
    // 2) alias
    if let Ok(mut st) = c.prepare(
        "SELECT f.key_raw, f.value
           FROM kv_alias a JOIN kv_facts f ON a.k_norm = f.k_norm
          WHERE a.alias_norm = ?1"
    ) {
        if let Some(row) = st.query_row([t.clone()], |r| Ok((r.get(0)?, r.get(1)?))).optional()? {
            return Ok(Some(row));
        }
    }
    // 3) fuzzy-ish LIKE fallback
    let like = format!("%{}%", t.replace('%', "").replace('_', ""));
    if let Ok(mut st) = c.prepare(
        "SELECT key_raw, value FROM kv_facts WHERE k_norm LIKE ?1 LIMIT 1"
    ) {
        if let Some(row) = st.query_row([like], |r| Ok((r.get(0)?, r.get(1)?))).optional()? {
            return Ok(Some(row));
        }
    }
    Ok(None)
}

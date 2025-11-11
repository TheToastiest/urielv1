// src/lexicon/mod.rs
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::io::BufRead;

#[derive(Debug, Deserialize)]
pub struct LexForm { pub form: String, pub features: serde_json::Value }
#[derive(Debug, Deserialize)]
pub struct LexEntry {
    pub lemma: String,
    pub pos: String,
    pub forms: Option<Vec<LexForm>>,
    pub senses: Option<Vec<String>>,   // glosses
    pub synonyms: Option<Vec<String>>,
    pub source: Option<String>,
}

pub fn init_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(r#"
    CREATE TABLE IF NOT EXISTS lex_lemmas(lemma TEXT PRIMARY KEY, pos TEXT, added_at INTEGER);
    CREATE TABLE IF NOT EXISTS lex_forms(lemma TEXT, form TEXT PRIMARY KEY, features TEXT);
    CREATE TABLE IF NOT EXISTS lex_senses(id INTEGER PRIMARY KEY AUTOINCREMENT, lemma TEXT, gloss TEXT, weight REAL, source TEXT);
    CREATE TABLE IF NOT EXISTS lex_rel(lemma TEXT, rel TEXT, target TEXT, weight REAL);
    "#)
}

pub fn upsert_entry(conn: &Connection, e: &LexEntry, ts: i64) -> rusqlite::Result<()> {
    conn.execute("INSERT OR REPLACE INTO lex_lemmas(lemma,pos,added_at) VALUES(?,?,?)",
                 params![e.lemma, e.pos, ts])?;
    if let Some(forms) = &e.forms {
        for f in forms {
            let feats = serde_json::to_string(&f.features).unwrap_or("{}".into());
            conn.execute("INSERT OR REPLACE INTO lex_forms(lemma,form,features) VALUES(?,?,?)",
                         params![e.lemma, f.form, feats])?;
        }
    }
    if let Some(senses) = &e.senses {
        for g in senses {
            conn.execute("INSERT INTO lex_senses(lemma,gloss,weight,source) VALUES(?,?,?,?)",
                         params![e.lemma, g, 1.0f32, e.source.as_deref().unwrap_or("manual")])?;
        }
    }
    if let Some(syns) = &e.synonyms {
        for s in syns {
            conn.execute("INSERT INTO lex_rel(lemma,rel,target,weight) VALUES(?,?,?,?)",
                         params![e.lemma, "syn", s, 1.0f32])?;
        }
    }
    Ok(())
}

// Convenience
pub fn synonyms(conn:&Connection, lemma:&str, limit:usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut st = conn.prepare("SELECT target FROM lex_rel WHERE lemma=?1 AND rel='syn' ORDER BY weight DESC LIMIT ?2").ok();
    if let Some(mut st) = st {
        let rows = st.query_map(params![lemma, limit as i64], |r| Ok(r.get::<_,String>(0)?)).ok();
        if let Some(rows) = rows {
            for r in rows.flatten() { out.push(r); }
        }
    }
    out
}
pub fn load_jsonl(conn: &Connection, path: &str) -> std::io::Result<()> {
    let p = std::path::Path::new(path);
    if !p.exists() { return Ok(()); }

    let f = std::fs::File::open(p)?;
    let r = std::io::BufReader::new(f);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;

    for line in r.lines().flatten() {
        let line = line.trim();
        if line.is_empty() { continue; }
        if let Ok(entry) = serde_json::from_str::<crate::lexicon::LexEntry>(line) {
            let _ = crate::lexicon::upsert_entry(conn, &entry, ts);
        }
    }
    Ok(())
}
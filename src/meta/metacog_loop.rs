// src/meta/metacog_loop.rs
use std::{fs, io::Write, path::Path, thread, time::Duration};

use anyhow::Result;
use rand::Rng;
use rusqlite::{params, Connection};
use serde::Serialize;

use crate::Hippocampus::primordial::primordial;
use crate::Hippocampus::sqlite_backend::{
    belief_stability, beliefs_agg, upsert_concept_fact,
};
use crate::policy::retrieval::{Arm, Bandit};

#[inline]
fn now_sec() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

/// Configuration for the metacognition loop.
#[derive(Clone)]
pub struct MetacogCfg {
    pub interval_secs: u64,
    pub batch_size: usize,
    pub promote_threshold: f32,
    pub demote_threshold: f32,
    pub ambiguous_low: f32,
    pub ambiguous_high: f32,
    pub peer_domains: Vec<&'static str>,
    pub max_web_bytes_per_page: usize,
    pub max_web_chunks: usize,
    pub metacog_log_path: String,
}

impl Default for MetacogCfg {
    fn default() -> Self {
        Self {
            interval_secs: 3600,
            batch_size: 16,
            promote_threshold: 0.60,
            demote_threshold: 0.30,
            ambiguous_low: 0.31,
            ambiguous_high: 0.59,
            peer_domains: vec![
                "wikipedia.org",
                "nih.gov", "ncbi.nlm.nih.gov",
                "nature.com", "science.org", "acm.org", "ieee.org",
                "arxiv.org", "openaccess.thecvf.com",
                "harvard.edu", "mit.edu", "stanford.edu",
                "who.int", "cdc.gov", "europa.eu",
            ],
            max_web_bytes_per_page: 300_000,
            max_web_chunks: 80,
            metacog_log_path: "data/meta/metacog_trace.jsonl".to_string(),
        }
    }
}

/// A single fact row from `concept_facts`.
#[derive(Debug, Clone)]
struct FactRow {
    key: String,
    value: String,
    updated_at: i64,
}

#[derive(Serialize)]
struct MetacogTrace {
    ts: i64,
    fact_key: String,
    old_value: String,
    prior_cert: f32,
    action: String,
    new_value: Option<String>,
    new_cert: Option<f32>,
    bandit: Option<TraceBandit>,
    notes: Option<String>,
}

#[derive(Serialize)]
struct TraceBandit {
    arm: String,
    reward: f32,
}

/// Arms must be 'static for the Bandit
const FACT_ARMS: &[Arm] = &[
    Arm { name: "local" },
    Arm { name: "web" },
];

/// Spawn a background thread that periodically fact-checks concept facts.
pub fn spawn_metacog_loop(
    db_path: String,
    cfg: MetacogCfg,
    shutdown_rx: Option<crossbeam_channel::Receiver<()>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        if let Err(e) = run_loop(db_path, cfg, shutdown_rx) {
            eprintln!("[metacog] loop error: {e:?}");
        }
    })
}

fn run_loop(
    db_path: String,
    cfg: MetacogCfg,
    shutdown_rx: Option<crossbeam_channel::Receiver<()>>,
) -> Result<()> {
    fs::create_dir_all("data/meta").ok();

    // This thread owns its own SQLite connection (thread-affine).
    let conn = Connection::open(&db_path)?;

    // Ensure metacog log file exists
    if !Path::new(&cfg.metacog_log_path).exists() {
        let _ = fs::File::create(&cfg.metacog_log_path);
    }

    // Bandit to decide order of checks (local vs web)
    let fact_bandit = Bandit {
        intent: "fact_check",
        arms: FACT_ARMS,
        epsilon: 0.15,
    };

    // Primordial “do not touch” guard
    let pri = primordial();
    let deny_set = build_primordial_guard(&pri);

    loop {
        if let Some(rx) = &shutdown_rx {
            if rx.try_recv().is_ok() {
                break;
            }
        }

        let facts = load_concept_facts(&conn, cfg.batch_size)?;

        for f in facts {
            if deny_set.contains(&f.key) {
                continue;
            }

            let now = now_sec();
            let prior = estimate_local_certainty(&conn, &f, now);

            let chosen = fact_bandit.choose(&conn);
            let mut used_arm = chosen.to_string();
            let mut reward = 0.0;

            let mut action = "skip".to_string();
            let mut new_val: Option<String> = None;
            let mut new_cert: Option<f32> = None;
            let mut notes: Option<String> = None;

            // Decide if we should web-verify
            let should_web = prior <= cfg.demote_threshold
                || (prior >= cfg.ambiguous_low && prior <= cfg.ambiguous_high);

            if should_web || chosen == "web" {
                let (candidate, wcert, wnote) = web_verify(&conn, &f, &cfg)?;
                used_arm = "web".into();
                notes = wnote;

                if let Some((k, v)) = candidate {
                    if wcert >= cfg.promote_threshold {
                        upsert_concept_fact(&k, &v, now as u64)?;
                        action = "overwrite".into();
                        new_val = Some(v);
                        new_cert = Some(wcert);
                        reward = 1.0;
                    } else {
                        action = "web_insufficient".into();
                        new_cert = Some(wcert);
                        reward = 0.2;
                    }
                } else {
                    action = "web_no_candidate".into();
                    reward = 0.0;
                }
            } else {
                if prior >= cfg.promote_threshold {
                    upsert_concept_fact(&f.key, &f.value, now as u64)?;
                    action = "keep_refresh".into();
                    reward = 0.8;
                } else {
                    action = "keep".into();
                    reward = 0.4;
                }
            }

            log_metacog(
                &cfg.metacog_log_path,
                MetacogTrace {
                    ts: now,
                    fact_key: f.key.clone(),
                    old_value: f.value.clone(),
                    prior_cert: prior,
                    action,
                    new_value: new_val,
                    new_cert,
                    bandit: Some(TraceBandit { arm: used_arm, reward }),
                    notes,
                },
            )?;

            fact_bandit.update(&conn, chosen, reward);
        }

        // sleep until next tick (interruptible)
        let mut slept = 0u64;
        while slept < cfg.interval_secs {
            if let Some(rx) = &shutdown_rx {
                if rx.try_recv().is_ok() {
                    return Ok(());
                }
            }
            thread::sleep(Duration::from_secs(1));
            slept += 1;
        }
    }

    Ok(())
}

/// Load a fair batch of facts (round-robin by updated_at asc).
fn load_concept_facts(conn: &Connection, n: usize) -> Result<Vec<FactRow>> {
    let mut stmt = conn.prepare(
        "SELECT concept_key, value, updated_at
         FROM concept_facts
         ORDER BY updated_at ASC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map([n as i64], |r| {
        Ok(FactRow {
            key: r.get(0)?,
            value: r.get(1)?,
            updated_at: r.get::<_, i64>(2)?,
        })
    })?;

    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

/// Very light primordial guard; extend with any keys you never want overwritten.
fn build_primordial_guard(
    _pri: &crate::Hippocampus::primordial::Primordial,
) -> std::collections::HashSet<String> {
    let mut set = std::collections::HashSet::new();
    set.insert("axiom".into());
    set.insert("prohibition".into());
    set.insert("drive".into());
    set.insert("primordial".into());
    set
}

/// Map a concept key to belief “key_norm” prefix used by beliefs_agg().
fn belief_prefix_for(key: &str) -> &'static str {
    match key {
        "name"         => "your name is",
        "you are"      => "you are",
        "creator"      => "your creator is",
        "mentor"       => "your mentor is",
        "purpose"      => "your purpose is",
        "launch code"  => "the launch code is",
        _              => "",
    }
}

/// Estimate local certainty from belief stability + recency.
fn estimate_local_certainty(conn: &Connection, f: &FactRow, now: i64) -> f32 {
    let mut base = 0.0f32;

    if let Ok(aggs) = beliefs_agg(conn, 64) {
        let pref = belief_prefix_for(&f.key);
        if !pref.is_empty() {
            // NOTE: key_norm is a FIELD on BeliefAgg, not a method
            if let Some(best) = aggs.iter().find(|a| a.key_norm.starts_with(pref)) {
                let s = belief_stability(best, now);
                base = base.max(s);
            }
        }
    }

    // Recency bonus (7-day half-life style)
    let age_days = ((now - f.updated_at).max(0) as f32) / 86_400.0;
    let recency = (-age_days / 7.0).exp();

    (0.65 * base + 0.35 * recency).clamp(0.0, 1.0)
}

/// Try to verify or improve a fact using web evidence.
/// Returns: (candidate fact (key,val) to upsert, new_certainty, notes)
fn web_verify(conn: &Connection, f: &FactRow, cfg: &MetacogCfg) -> Result<(Option<(String, String)>, f32, Option<String>)> {
    let icfg = crate::tools::web_ingest::IngestCfg {
        whitelist_domains: &cfg.peer_domains,
        max_bytes_per_page: cfg.max_web_bytes_per_page,
        max_chunks: cfg.max_web_chunks,
    };

    // Heuristic candidate URLs for the fact
    let mut urls: Vec<String> = Vec::new();
    if f.key == "name" || f.key == "you are" {
        let slug = slugify(&f.value);
        urls.push(format!("https://en.wikipedia.org/wiki/{slug}"));
    } else if f.key == "purpose" {
        urls.push("https://en.wikipedia.org/wiki/Purpose".to_string());
    } else if f.key == "creator" || f.key == "mentor" {
        urls.push("https://en.wikipedia.org/wiki/Mentor".to_string());
    } else if f.key == "launch code" {
        urls.push("https://en.wikipedia.org/wiki/Launch_code".to_string());
    }

    let mut notes = String::new();
    if !urls.is_empty() {
        let url_refs: Vec<&str> = urls.iter().map(|s| s.as_str()).collect();
        // We don’t care whether ingest_urls returns usize or ()
        match crate::tools::web_ingest::ingest_urls(conn, &url_refs, &icfg) {
            Ok(_) => notes.push_str("ingest_ok; "),
            Err(e) => notes.push_str(&format!("ingest_err:{e:?}; ")),
        }
    } else {
        notes.push_str("no candidate urls; ");
    }

    // Very light “web certainty”
    let mut web_cert: f32 = if urls.is_empty() { 0.0_f32 } else { 0.45_f32 };
    if matches!(f.key.as_str(), "name" | "you are" | "purpose") && web_cert > 0.0 {
        web_cert += 0.10_f32;
    }

    let candidate = Some((f.key.clone(), f.value.clone()));
    Ok((candidate, web_cert.min(1.0_f32), Some(notes)))
}

fn slugify(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else if ch.is_whitespace() || ch == '-' || ch == '_' {
            out.push('_');
        }
    }
    if out.is_empty() {
        let r: u32 = rand::thread_rng().gen();
        out = format!("Item_{r}");
    }
    out
}

/// Append one JSON line to the audit file.
fn log_metacog(path: &str, rec: MetacogTrace) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let line = serde_json::to_string(&rec)?;
    writeln!(file, "{line}")?;
    Ok(())
}

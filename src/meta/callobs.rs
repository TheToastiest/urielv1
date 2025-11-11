use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{fs, io::Write, path::PathBuf};

use crate::meta::trace_v2::{Evidence, EvidenceQuality}; // reuse your existing types

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum Op { None, ReadFact, Recall, RowsScan }

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Arg {
    ConceptKey { value: String },
    QuerySpan  { on: String, byte: (usize, usize) },
    MemoryKey  { value: String },
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalLabel { OK, ASK, IDK }

#[derive(Serialize, Deserialize, Clone)]
pub struct CopySpan { pub snippet_id: String, pub start_byte: usize, pub end_byte: usize }

#[derive(Serialize, Deserialize)]
struct Envelope {
    ts: i64,
    trace_id: String,
    turn_id: String,
    domain: String,
    event: String,   // "CALL" | "OBS" | "FINAL"
    role: String,    // "agent" | "tool" | "user"
}

fn now_sec() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64
}

pub struct Writer {
    file: std::fs::File,
    trace_id: String,
    domain: String,
}
impl Writer {
    /// base_dir like "data/logs". Creates file: data/logs/trace-<ts>-<rand>.jsonl
    pub fn new(base_dir: &str) -> Result<Self> {
        fs::create_dir_all(base_dir)?;
        let ts = now_sec();
        let rnd: u32 = rand::random();
        let trace_id = format!("{}-{:08x}", ts, rnd);
        let path: PathBuf = [base_dir, &format!("trace-{}-{:08x}.jsonl", ts, rnd)].iter().collect();
        let file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { file, trace_id, domain: "chat".into() })
    }

    #[inline] fn env(&self, turn_id: &str, event: &str, role: &str) -> Envelope {
        Envelope {
            ts: now_sec(),
            trace_id: self.trace_id.clone(),
            turn_id: turn_id.to_string(),
            domain: self.domain.clone(),
            event: event.to_string(),
            role: role.to_string(),
        }
    }
    fn write<T: Serialize>(&mut self, rec: &T) -> Result<()> {
        let line = serde_json::to_string(rec)?;
        writeln!(self.file, "{}", line)?;
        Ok(())
    }

    pub fn log_call(&mut self, turn_id: &str, op: Op, arg: Option<Arg>, policy: &str) -> Result<()> {
        #[derive(Serialize)]
        struct Call<'a> { #[serde(flatten)] env: Envelope, op: Op,
            #[serde(skip_serializing_if="Option::is_none")] arg: Option<Arg>, policy: &'a str }
        self.write(&Call { env: self.env(turn_id, "CALL", "agent"), op, arg, policy })
    }

    /// Log OBS using your existing Evidence shape.
    pub fn log_obs_evidence(&mut self, turn_id: &str, tool: &str, obs: Vec<Evidence>) -> Result<()> {
        #[derive(Serialize)]
        struct Obs<'a> { #[serde(flatten)] env: Envelope, tool: &'a str, obs: Vec<Evidence> }
        self.write(&Obs { env: self.env(turn_id, "OBS", "tool"), tool, obs })
    }

    pub fn log_final(&mut self, turn_id: &str, label: FinalLabel,
                     text: &str, must_copy: &[CopySpan], calibrated_p: Option<f32>) -> Result<()> {
        #[derive(Serialize)]
        struct Final<'a> {
            #[serde(flatten)] env: Envelope,
            label: FinalLabel,
            text: &'a str,
            #[serde(skip_serializing_if="Vec::is_empty")] must_copy: Vec<CopySpan>,
            #[serde(skip_serializing_if="Option::is_none")] calibrated_p: Option<f32>
        }
        self.write(&Final {
            env: self.env(turn_id, "FINAL", "agent"),
            label, text, must_copy: must_copy.to_vec(), calibrated_p
        })
    }
}

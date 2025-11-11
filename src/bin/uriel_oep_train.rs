// src/bin/uriel_oep_train.rs
use anyhow::{anyhow, Result};
use clap::Parser;
use chrono::Utc;
use regex::Regex;
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use URIELV1::Hippocampus::infer::HippocampusModel;

/// CLI to learn a diagonal post-projection vector `w` from learn_queue.
#[derive(Parser, Debug)]
#[command(name="uriel_oep_train", version, about="Online Embedding Post-projection (OEP) trainer")]
struct Args {
    /// How many recent learning items to sample
    #[arg(short = 'n', long = "batch", default_value_t = 128)]
    batch: usize,

    /// Learning rate for w updates
    #[arg(long, default_value_t = 0.05)]
    lr: f32,

    /// Margin for hinge (s_pos - s_neg should exceed this)
    #[arg(long, default_value_t = 0.20)]
    margin: f32,

    /// DB path
    #[arg(long, default_value = "data/semantic_memory.db")]
    db: String,

    /// Encoder checkpoint path (same as runtime)
    #[arg(long = "ckpt", default_value = "R:/uriel/models/checkpoints/uriel_encoder_step008500.bin")]
    ckpt: String,

    /// Output JSON file for learned weights
    #[arg(long, default_value = "data/oep_w.json")]
    out: String,
}

#[derive(Serialize, Deserialize)]
struct OepWeights {
    dim: usize,
    w: Vec<f32>,
    updated_at: i64,
}

impl OepWeights {
    fn new(dim: usize) -> Self {
        Self { dim, w: vec![1.0; dim], updated_at: Utc::now().timestamp() }
    }
}

fn load_weights(path: &str, dim: usize) -> OepWeights {
    if Path::new(path).exists() {
        if let Ok(text) = fs::read_to_string(path) {
            if let Ok(mut obj) = serde_json::from_str::<OepWeights>(&text) {
                if obj.dim == dim {
                    return obj;
                }
            }
        }
    }
    OepWeights::new(dim)
}

fn save_weights(path: &str, w: &OepWeights) -> Result<()> {
    let txt = serde_json::to_string_pretty(w)?;
    fs::create_dir_all(Path::new(path).parent().unwrap_or_else(|| Path::new(".")))?;
    fs::write(path, txt)?;
    Ok(())
}

fn looks_like_url(s: &str) -> bool {
    let l = s.trim().to_ascii_lowercase();
    l.starts_with("http://") || l.starts_with("https://") || l.contains("://")
}

fn looks_like_web_error(s: &str) -> bool {
    let l = s.to_ascii_lowercase();
    l.starts_with("please set a user-agent")
        || l.contains("access denied")
        || l.contains("forbidden")
        || l.contains("captcha")
        || l.contains("robots")
        || looks_like_url(s)
        || s.trim().is_empty()
}

fn parse_one_negative(conn: &Connection, qid: i64) -> Option<String> {
    conn.prepare("SELECT negative FROM learn_negs WHERE qid=?1 ORDER BY COALESCE(sim,0.0) DESC LIMIT 1")
        .ok()?
        .query_row([qid], |r| r.get::<_, String>(0))
        .optional()
        .ok()
        .flatten()
}

fn main() -> Result<()> {
    // ---- args / setup
    let args = Args::parse();
    type B = NdArray<f32>;
    let device = <B as Backend>::Device::default();
    let model = Arc::new(Mutex::new(HippocampusModel::<B>::load_or_init(&device, &args.ckpt)));

    // probe embedding dim
    let dim = {
        let m = model.lock().map_err(|_| anyhow!("model lock poisoned"))?;
        let z = m.run_inference("[dim-probe]");
        let d = z.to_data().as_slice::<f32>().map_err(|e| anyhow!("{e:?}"))?.len();
        d
    };

    let mut oep = load_weights(&args.out, dim);

    // ---- load batch from learn_queue (most recent items, used or not)
    let conn = Connection::open(&args.db)?;
    conn.busy_timeout(std::time::Duration::from_millis(500))?;

    let mut st = conn.prepare(
        "SELECT id, question, positive
           FROM learn_queue
          ORDER BY id DESC
          LIMIT ?1"
    )?;
    let rows = st.query_map([args.batch as i64], |r| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
    })?;

    // Precompile a lax “definitiony” detector just to skip super-noisy rows
    let def_re = Regex::new(r"(?i)^\s*(what\s+is|what'?s|define|explain)\s+").unwrap();

    let mut used = 0usize;
    let mut skipped = 0usize;
    let lr = args.lr;
    let margin = args.margin;

    // ---- training loop over samples
    for rec in rows {
        let (qid, q, p) = rec?;
        // basic guards
        if looks_like_url(&q) || looks_like_web_error(&p) {
            skipped += 1;
            continue;
        }
        // prefer meaningful questions
        if !def_re.is_match(&q) && q.len() < 6 {
            skipped += 1;
            continue;
        }

        let n = match parse_one_negative(&conn, qid) {
            Some(s) if !looks_like_web_error(&s) => s,
            _ => { skipped += 1; continue; }
        };

        // embed q, p, n
        let (eq, ep, en) = {
            let m = model.lock().map_err(|_| anyhow!("model lock poisoned"))?;
            let tq = m.run_inference(&q).to_data().as_slice::<f32>().map_err(|e| anyhow!("{e:?}"))?.to_vec();
            let tp = m.run_inference(&p).to_data().as_slice::<f32>().map_err(|e| anyhow!("{e:?}"))?.to_vec();
            let tn = m.run_inference(&n).to_data().as_slice::<f32>().map_err(|e| anyhow!("{e:?}"))?.to_vec();
            (tq, tp, tn)
        };
        if eq.len()!=dim || ep.len()!=dim || en.len()!=dim { skipped += 1; continue; }

        // scores with current w: s_pos = <w.*q, w.*p>, s_neg = <w.*q, w.*n>
        let mut s_pos = 0.0f32;
        let mut s_neg = 0.0f32;
        for i in 0..dim {
            let qh = oep.w[i] * eq[i];
            s_pos += qh * (oep.w[i] * ep[i]);
            s_neg += qh * (oep.w[i] * en[i]);
        }

        if s_pos - s_neg < margin {
            // simple diagonal update: w_i += lr * q_i * (p_i - n_i)
            for i in 0..dim {
                oep.w[i] += lr * (eq[i] * (ep[i] - en[i]));
            }
            // floor and renormalize average to 1.0 to avoid blow-up
            let mut sum = 0.0f32;
            for wi in &mut oep.w {
                if *wi < 0.05 { *wi = 0.05; }
                sum += *wi;
            }
            let mean = (sum / dim as f32).max(1e-6);
            for wi in &mut oep.w { *wi /= mean; }
            used += 1;
        }
    }

    oep.updated_at = Utc::now().timestamp();
    save_weights(&args.out, &oep)?;
    println!(
        "OEP training: dim={}, updated={} samples, skipped={}, saved → {}",
        oep.dim, used, skipped, &args.out
    );
    Ok(())
}

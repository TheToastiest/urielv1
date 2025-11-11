// training/dataset.rs
use anyhow::Result;
use globwalk::GlobWalkerBuilder;
use serde::Deserialize;
use std::io::BufRead;

#[derive(Deserialize)]
struct RouterRow {
    query: String,
    route_taken: String,
    candidates: Vec<String>,
    weight: f32,
    conf: f32,
    success: bool,
}

#[derive(Clone)]
pub struct RouterSample {
    pub tokens: Vec<usize>,  // [T]
    pub label: usize,        // class idx
    pub weight: f32,
}

#[inline]
fn to_idx(c: char) -> usize {
    // 0 is PAD; printable ASCII maps to 1..=95
    ((c as i32 - 32).clamp(0, 94) as usize) + 1
}

// Make this Copy so passing it into inject_band doesn't move it.
#[derive(Clone, Copy, Debug)]
pub enum BandMode { None, Label, Heuristic }

#[inline]
fn tag_for_label(label: usize) -> usize {
    // 4 distinct tag ids so 4-way routing works later
    match label { 0 => 92, 1 => 93, 2 => 94, 3 => 95, _ => 91 }
}

#[inline]
fn heuristic_label(q: &str) -> usize {
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

    if is_sql { 3 } else if is_code { 2 } else if is_retr { 1 } else { 0 }
}

fn inject_band(
    tokens: &mut [usize],
    seq_len: usize,
    label: usize,
    q: &str,
    mode: BandMode,
    band_len_frac: f32,
    rng: &mut fastrand::Rng,
) {
    if matches!(mode, BandMode::None) { return; }
    let len = ((seq_len as f32 * band_len_frac).round() as usize).clamp(0, seq_len);
    if len == 0 { return; }

    // Randomized offset so the model can't memorize "position 0..k == tag"
    let max_start = seq_len.saturating_sub(len);
    let start = if max_start > 0 { rng.usize(0..=max_start) } else { 0 };

    let tag = match mode {
        BandMode::Label => tag_for_label(label),
        BandMode::Heuristic => tag_for_label(heuristic_label(q)),
        BandMode::None => unreachable!(),
    };
    for i in start..start + len {
        tokens[i] = tag;
    }
}

pub fn load_router_samples(
    glob_pat: &str,
    seq_len: usize,
    min_conf: f32,
    band_mode: BandMode,   // train: Heuristic or Label; val: None
    band_len_frac: f32,    // e.g., 0.50
    band_prob: f32,        // e.g., 1.0 means always attempt injection
    seed: u64,             // determinism
) -> Result<Vec<RouterSample>> {
    let mut out = Vec::new();
    let mut rng = fastrand::Rng::with_seed(seed);

    // deterministic file order
    let mut files: Vec<std::path::PathBuf> = GlobWalkerBuilder::from_patterns(".", &[glob_pat])
        .build()?
        .into_iter()
        .filter_map(|e| e.ok())
        .map(|e| e.path().to_path_buf())
        .filter(|p| p.is_file())
        .collect();
    files.sort();

    for path in files {
        if let Ok(f) = std::fs::File::open(&path) {
            let rdr = std::io::BufReader::new(f);
            for line in rdr.lines().flatten() {
                if line.is_empty() { continue; }
                if let Ok(row) = serde_json::from_str::<RouterRow>(&line) {
                    if !row.success { continue; }
                    if row.conf < min_conf { continue; }

                    // map route string -> class index (must match head order)
                    let label = match row.route_taken.as_str() {
                        "retrieve_then_answer" => 1usize,
                        "tool:code"            => 2usize,
                        "tool:sql"             => 3usize,
                        _                      => 0usize, // direct
                    };

                    // Build text WITHOUT label leakage (no "[RET]"/"[DIR]" prefix)
                    let q = row.query.trim();
                    let boosted = q.to_string();

                    // Tokenize to fixed length with PAD=0 (0 is PAD; printable ASCII maps to 1..=95)
                    let mut tks: Vec<usize> = boosted.chars().map(to_idx).collect();
                    if tks.len() > seq_len { tks.truncate(seq_len); }
                    while tks.len() < seq_len { tks.push(0); }

                    // Optional band injection (label or heuristic). For val, call with BandMode::None.
                    if band_prob > 0.0 && rng.f32() < band_prob {
                        inject_band(&mut tks, seq_len, label, q, band_mode, band_len_frac, &mut rng);
                    }

                    out.push(RouterSample {
                        tokens: tks,
                        label,
                        weight: row.weight.max(0.1),
                    });
                }
            }
        }
    }
    Ok(out)
}

use burn::tensor::Tensor;

pub fn one_hot_batch<B: burn::tensor::backend::Backend>(
    batch: &[RouterSample],
    vocab: usize,
    dev: &B::Device,
) -> Tensor<B, 3> {
    let t = batch[0].tokens.len();
    let mut data = vec![0.0f32; batch.len() * t * vocab];
    for (bi, s) in batch.iter().enumerate() {
        for (ti, &ix) in s.tokens.iter().enumerate() {
            data[bi * t * vocab + ti * vocab + ix] = 1.0;
        }
    }
    Tensor::<B, 1>::from_floats(data.as_slice(), dev).reshape([batch.len(), t, vocab])
}

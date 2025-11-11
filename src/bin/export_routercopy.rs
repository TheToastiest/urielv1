use anyhow::{Context, Result};
use clap::Parser;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
struct Args {
    /// Folder with CALL/OBS/FINAL jsonl (e.g., data/logs)
    #[arg(long)]
    input: String,
    /// Output dataset folder (will create train/dev/test.jsonl)
    #[arg(long)]
    out: String,
    /// Optional meta events file to recover user_text (e.g., data/meta/meta_events.jsonl)
    #[arg(long)]
    meta: Option<String>,
    /// Train/dev/test split as 3 numbers summing to 1.0
    #[arg(long, num_args = 3, default_values_t = [0.8f32, 0.1, 0.1])]
    split: Vec<f32>,
    /// Whitelist domains (comma-sep); URLs outside are dropped (DB/SMIE/rows kept)
    #[arg(long, default_value = "example.com,wikipedia.org,github.com,sciencedirect.com,jstor.org")]
    whitelist: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct ObsText {
    snippet_id: String,
    text: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct RowOut {
    user_text: String,
    obs_text: Vec<ObsText>,
    final_text: String,
    op_id: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    arg_class_id: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arg_span: Option<[usize; 2]>,
    #[serde(default)]
    copy_mask: Vec<u8>,
    #[serde(default)]
    ptr_index: Vec<i32>,
}

#[derive(Default)]
struct TurnAgg {
    user_text: Option<String>,
    op: Option<String>,
    arg_key: Option<String>,
    obs: Vec<ObsText>,
    final_text: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let wl: HashSet<String> = args
        .whitelist
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();

    let mut by_turn: HashMap<String, TurnAgg> = HashMap::new();

    // 1) Optional meta: turn_id -> user_text (phase Sense)
    if let Some(meta_path) = &args.meta {
        if Path::new(meta_path).exists() {
            let f = fs::File::open(meta_path)?;
            for line in BufReader::new(f).lines() {
                let s = match line {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                if s.trim().is_empty() {
                    continue;
                }
                let v: Value = match serde_json::from_str(&s) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if v.get("phase").and_then(|x| x.as_str()) != Some("Sense") {
                    continue;
                }
                let turn_id = if let Some(t) = v.get("task_id").and_then(|x| x.as_str()) {
                    t.to_string()
                } else {
                    continue
                };
                if let Some(inp) = v.get("inputs_ref").and_then(|x| x.as_str()) {
                    by_turn.entry(turn_id).or_default().user_text = Some(inp.to_string());
                }
            }
        }
    }

    // 2) Walk logs folder (handles nested day folders)
    for entry in fs::read_dir(&args.input).with_context(|| format!("read_dir {}", &args.input))? {
        let entry = entry?;
        let p = entry.path();
        if p.is_dir() {
            for e2 in fs::read_dir(&p)? {
                let e2 = e2?;
                if e2.path().is_file() {
                    ingest_file(&e2.path(), &mut by_turn, &wl)?;
                }
            }
        } else if p.is_file() {
            ingest_file(&p, &mut by_turn, &wl)?;
        }
    }

    // 3) Build rows
    let mut rows: Vec<RowOut> = Vec::new();
    for (_tid, agg) in by_turn.into_iter() {
        let final_text = match agg.final_text {
            Some(x) => x,
            None => continue,
        };
        let user_text = agg.user_text.unwrap_or_default();
        let obs = agg.obs;

        let op_id = map_op(agg.op.as_deref());
        let arg_class_id = agg.arg_key.as_ref().and_then(|k| map_concept_key(k));

        let (copy_mask, ptr_index) = build_copy_targets(&final_text, &obs);

        rows.push(RowOut {
            user_text,
            obs_text: obs,
            final_text,
            op_id,
            arg_class_id,
            arg_span: None,
            copy_mask,
            ptr_index,
        });
    }

    // 4) Shuffle + split
    let mut rng = StdRng::seed_from_u64(42);
    rows.shuffle(&mut rng);
    let n = rows.len();
    let a = (args.split[0] * n as f32).round() as usize;
    let b = (args.split[1] * n as f32).round() as usize;
    let (train, rem) = rows.split_at(a.min(n));
    let (dev, test) = rem.split_at(b.min(rem.len()));

    fs::create_dir_all(&args.out)?;
    write_jsonl(&PathBuf::from(&args.out).join("train.jsonl"), train)?;
    write_jsonl(&PathBuf::from(&args.out).join("dev.jsonl"), dev)?;
    write_jsonl(&PathBuf::from(&args.out).join("test.jsonl"), test)?;

    // Stats
    let sr_train = mean_support(train);
    let sr_dev = mean_support(dev);
    let sr_test = mean_support(test);
    println!("router_v1 dataset → {}", &args.out);
    println!("counts: train={} dev={} test={}", train.len(), dev.len(), test.len());
    println!(
        "support_ratio: train={:.3} dev={:.3} test={:.3}",
        sr_train, sr_dev, sr_test
    );
    Ok(())
}

fn write_jsonl(path: &Path, items: &[RowOut]) -> Result<()> {
    let mut f = fs::File::create(path)?;
    for it in items {
        let s = serde_json::to_string(it)?;
        use std::io::Write;
        writeln!(f, "{}", s)?;
    }
    Ok(())
}

fn mean_support(rows: &[RowOut]) -> f32 {
    let mut num = 0usize;
    let mut den = 0usize;
    for r in rows {
        for &m in &r.copy_mask {
            den += 1;
            if m != 0 {
                num += 1;
            }
        }
    }
    if den == 0 {
        0.0
    } else {
        num as f32 / den as f32
    }
}

fn ingest_file(path: &Path, by_turn: &mut HashMap<String, TurnAgg>, whitelist: &HashSet<String>) -> Result<()> {
    let f = fs::File::open(path)?;
    for line in BufReader::new(f).lines() {
        let s = match line {
            Ok(s) => s,
            Err(_) => continue,
        };
        if s.trim().is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&s) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let event = v.get("event").and_then(|x| x.as_str()).unwrap_or("");
        let turn = v.get("turn_id").and_then(|x| x.as_str()).unwrap_or("");
        if turn.is_empty() {
            continue;
        }
        let rec = by_turn.entry(turn.to_string()).or_default();

        match event {
            "CALL" => {
                if let Some(op) = v.get("op").and_then(|x| x.as_str()) {
                    rec.op = Some(op.to_string());
                }
                if let Some(arg) = v.get("arg") {
                    if arg.get("type").and_then(|x| x.as_str()) == Some("concept_key") {
                        if let Some(val) = arg.get("value").and_then(|x| x.as_str()) {
                            rec.arg_key = Some(val.to_string());
                        }
                    }
                }
            }
            "OBS" => {
                let tool = v.get("tool").and_then(|x| x.as_str()).unwrap_or("unknown").to_string();
                let obs_arr = v
                    .get("obs")
                    .and_then(|x| x.as_array())
                    .cloned()
                    .unwrap_or_default();
                for o in obs_arr {
                    // Filter URL if present
                    let url_opt = o.get("url").and_then(|x| x.as_str());
                    if let Some(u) = url_opt {
                        if !is_whitelisted(u, whitelist) {
                            continue;
                        }
                    }
                    let text = o
                        .get("excerpt")
                        .or_else(|| o.get("text"))
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();
                    if text.trim().is_empty() {
                        continue;
                    }
                    let id = o
                        .get("id")
                        .or_else(|| o.get("snippet_id"))
                        .and_then(|x| x.as_str())
                        .unwrap_or_else(|| tool.as_str());
                    rec.obs.push(ObsText {
                        snippet_id: id.to_string(),
                        text,
                    });
                }
            }
            "FINAL" => {
                if let Some(t) = v.get("text").and_then(|x| x.as_str()) {
                    rec.final_text = Some(t.to_string());
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn host_from_url(url: &str) -> Option<String> {
    // crude parse: scheme://host[:port]/...
    static RE: once_cell::sync::Lazy<Regex> =
        once_cell::sync::Lazy::new(|| Regex::new(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://([^/]+)").unwrap());
    if let Some(c) = RE.captures(url) {
        return Some(c[1].to_ascii_lowercase());
    }
    None
}

fn is_whitelisted(url: &str, wl: &HashSet<String>) -> bool {
    if let Some(host) = host_from_url(url) {
        return wl.iter().any(|d| host == *d || host.ends_with(&format!(".{}", d)));
    }
    // If URL can't be parsed, keep it (DB/SMIE/rows usually have no URL)
    true
}

fn map_op(op: Option<&str>) -> usize {
    match op.map(|s| s.to_ascii_lowercase()).as_deref() {
        // All tool-based retrieval paths -> Retrieve (1)
        Some("read_fact") | Some("recall") | Some("rows_scan") => 1,

        // No tools used -> Respond (0)
        Some("none") => 0,

        // If you ever log store/defer explicitly:
        Some("store") | Some("write_fact") => 2,  // Store
        Some("defer")                          => 3,  // Defer

        // Default conservatively to Retrieve
        _ => 1,
    }
}


fn map_concept_key(key: &str) -> Option<usize> {
    match key.to_ascii_lowercase().as_str() {
        "name" => Some(0),
        "creator" => Some(1),
        "mentor" => Some(2),
        "launch code" => Some(3),
        "you are" => Some(4),
        "purpose" => Some(5),
        _ => None,
    }
}

fn tokenize_lc(s: &str) -> Vec<String> {
    let re = Regex::new(r"[A-Za-z0-9']+").unwrap();
    re.find_iter(s)
        .map(|m| m.as_str().to_ascii_lowercase())
        .collect()
}

fn build_copy_targets(final_text: &str, obs: &[ObsText]) -> (Vec<u8>, Vec<i32>) {
    let ftoks = tokenize_lc(final_text);
    let mut obs_tokens: Vec<String> = Vec::new();
    for o in obs {
        obs_tokens.extend(tokenize_lc(&o.text));
    }
    // inverted index
    let mut idx: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, t) in obs_tokens.iter().enumerate() {
        idx.entry(t.as_str()).or_default().push(i);
    }

    let mut mask = vec![0u8; ftoks.len()];
    let mut ptr = vec![-1i32; ftoks.len()];
    for (i, t) in ftoks.iter().enumerate() {
        if let Some(list) = idx.get(t.as_str()) {
            mask[i] = 1;
            ptr[i] = *list.first().unwrap() as i32;
        }
    }
    (mask, ptr)
}

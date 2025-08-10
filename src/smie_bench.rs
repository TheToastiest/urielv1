use std::{fs, io::Write, path::PathBuf};
use clap::{Parser, ValueEnum};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use burn::tensor::{backend::Backend, Tensor};
use URIELV1::Hippocampus::burn_model::{embed_text, MyBackend, MyModel};
use URIELV1::Hippocampus::ingest as smie_ingest;
use URIELV1::PFC::context_retriever::MemoryHit;

#[derive(Clone, ValueEnum)]
enum Mode { Query, Mixed }

#[derive(Parser)]
struct Args {
    /// Folder with .txt docs
    #[arg(long)]
    docs: PathBuf,
    /// Top-K
    #[arg(long, default_value_t = 8)]
    k: usize,
    /// Trials
    #[arg(long, default_value_t = 100)]
    trials: usize,
    /// RNG seed
    #[arg(long, default_value_t = 1337)]
    seed: u64,
    /// Bench mode: query-only or mixed (query + ingest each trial)
    #[arg(long, value_enum, default_value_t = Mode::Query)]
    mode: Mode,
    /// Ingest context name
    #[arg(long, default_value = "bench")]
    context: String,
    /// Ingest TTL (seconds)
    #[arg(long, default_value_t = 30)]
    ttl: u64,
    /// Output CSV path (if omitted, stdout)
    #[arg(long)]
    out: Option<PathBuf>,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let nb = (b.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
    a.iter().zip(b.iter()).map(|(x, y)| x * (y / nb)).sum::<f32>()
}

fn embed_pipeline(text: &str) -> anyhow::Result<Vec<f32>> {
    let device = <MyBackend as Backend>::Device::default();
    let input_vec = embed_text(text);
    let d = input_vec.len();
    let input = Tensor::<MyBackend, 1>::from_floats(input_vec.as_slice(), &device).reshape([1, d]);
    let model = MyModel::<MyBackend>::new(d, 128, &device);
    let output = model.forward(input);
    let v: Vec<f32> = output
        .into_data()
        .convert::<f32>()
        .into_vec()
        .map_err(|e| anyhow::anyhow!("tensor->vec failed: {:?}", e))?;
    Ok(v)
}

struct MemStore {
    items: Vec<(String, Vec<f32>, String)>, // id, emb, text
}
impl MemStore {
    fn new() -> Self { Self { items: Vec::new() } }
    fn add(&mut self, id: String, text: String) -> anyhow::Result<()> {
        let emb = embed_pipeline(&text)?;
        self.items.push((id, emb, text));
        Ok(())
    }
    fn search(&self, q: &[f32], k: usize) -> Vec<MemoryHit> {
        let mut hits: Vec<_> = self.items.iter().map(|(id, emb, text)| MemoryHit {
            id: id.clone(), sim: cosine(q, emb), text: text.clone()
        }).collect();
        hits.sort_by(|a,b| b.sim.partial_cmp(&a.sim).unwrap_or(std::cmp::Ordering::Equal));
        if hits.len() > k { hits.truncate(k); }
        hits
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if !args.docs.exists() || !args.docs.is_dir() {
        anyhow::bail!("--docs not found or not a directory: {:?}", args.docs);
    }

    // Load corpus & pre-embed into in-mem store
    let mut store = MemStore::new();
    let mut corpus: Vec<String> = Vec::new();
    for entry in fs::read_dir(&args.docs)? {
        let p = entry?.path();
        if p.is_file() {
            let text = fs::read_to_string(&p)?;
            let id = p.file_name().unwrap().to_string_lossy().to_string();
            store.add(id, text.clone())?;
            corpus.push(text);
        }
    }
    if corpus.is_empty() {
        eprintln!("No files under {:?}", args.docs);
        return Ok(());
    }

    // CSV writer (file or stdout)
    let mut writer: Box<dyn Write> = if let Some(path) = args.out.as_ref() {
        Box::new(fs::File::create(path)?)
    } else {
        Box::new(std::io::stdout())
    };

    writeln!(writer, "trial,embed_us,search_us,ingest_us,top1_sim,top1_id")?;

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut search_lats = Vec::with_capacity(args.trials);
    let mut ingest_lats = Vec::with_capacity(args.trials);

    for t in 0..args.trials {
        // --- Embed query ---
        let tq0 = std::time::Instant::now();
        let q = corpus.choose(&mut rng).unwrap();
        let q_emb = embed_pipeline(q)?;
        let embed_us = tq0.elapsed().as_micros() as f64;

        // --- Search ---
        let ts0 = std::time::Instant::now();
        let hits = store.search(&q_emb, args.k);
        let search_us = ts0.elapsed().as_micros() as f64;
        search_lats.push(search_us);

        // --- Ingest (only in mixed mode) ---
        let mut ingest_us = 0.0;
        if matches!(args.mode, Mode::Mixed) {
            let ti0 = std::time::Instant::now();
            // reuse a random doc; in real tests, feed live text
            let text = corpus.choose(&mut rng).unwrap().clone();
            let _ = smie_ingest(args.context.clone(), text, args.ttl, "bench".to_string());
            ingest_us = ti0.elapsed().as_micros() as f64;
            ingest_lats.push(ingest_us);
        }

        let (sim, id) = hits.get(0)
            .map(|h| (h.sim, h.id.clone()))
            .unwrap_or((0.0, "".into()));

        writeln!(writer, "{t},{embed_us:.1},{search_us:.1},{ingest_us:.1},{sim:.6},{id}")?;
    }

    // Summaries to stderr (don’t pollute CSV)
    search_lats.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let p = |v:&[f64], q:f64| v[ ((v.len()-1) as f64 * q).floor() as usize ];
    eprintln!(
        "SEARCH  trials={}  P50={:.1}us  P95={:.1}us  P99={:.1}us",
        search_lats.len(), p(&search_lats,0.50), p(&search_lats,0.95), p(&search_lats,0.99)
    );
    if !ingest_lats.is_empty() {
        ingest_lats.sort_by(|a,b| a.partial_cmp(b).unwrap());
        eprintln!(
            "INGEST  trials={}  P50={:.1}us  P95={:.1}us  P99={:.1}us",
            ingest_lats.len(), p(&ingest_lats,0.50), p(&ingest_lats,0.95), p(&ingest_lats,0.99)
        );
    }

    Ok(())
}

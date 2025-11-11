    use anyhow::{anyhow, Result};
    use clap::Parser;
    use chrono::Utc;
    use regex::Regex;
    use rusqlite::{params, Connection};
    use std::sync::{Arc, Mutex};

    use burn::tensor::backend::Backend;
    use burn_ndarray::NdArray;
    use smie_core::{Embedder, Metric, Smie, SmieConfig, tags};

    use URIELV1::Hippocampus::infer::HippocampusModel;
    use URIELV1::knowledge::{ops as kops, types as ktypes};

    /// Consume learn_queue and update kg_fact + SMIE (retrieval index)
    #[derive(Parser, Debug)]
    #[command(name = "uriel_learn", version, about = "Consume learn_queue and update memory/index")]
    struct Args {
        /// Max items to process
        #[arg(short = 'n', long, default_value_t = 64)]
        max: usize,

        /// Path to semantic memory sqlite file
        #[arg(long, default_value = "data/semantic_memory.db")]
        db: String,

        /// Path to SMIE index directory
        #[arg(long = "smie-dir", default_value = "data/uriel")]
        smie_dir: String,

        /// Path to encoder checkpoint (.bin), same as runtime
        #[arg(long = "ckpt", default_value = "R:/uriel/models/checkpoints/uriel_encoder_step008500.bin")]
        ckpt: String,
    }

    /// Minimal embedder adapting HippocampusModel<B> to smie_core::Embedder.
    struct SmieHippEmbed<B: Backend> {
        model: Arc<Mutex<HippocampusModel<B>>>,
        dim: usize,
    }
    impl<B: Backend> SmieHippEmbed<B> {
        fn new(model: Arc<Mutex<HippocampusModel<B>>>) -> Result<Self> {
            let m = model.lock().map_err(|_| anyhow!("model lock poisoned"))?;
            let z = m.run_inference("[dim-probe]");
            let dim = z.to_data().as_slice::<f32>().map_err(|e| anyhow!("{e:?}"))?.len();
            drop(m);
            Ok(Self { model, dim })
        }
    }
    impl<B: Backend> Embedder for SmieHippEmbed<B> {
        fn dim(&self) -> usize { self.dim }
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            let m = self.model.lock().map_err(|_| anyhow!("model lock poisoned"))?;
            let z = m.run_inference(text);
            let mut v = z.to_data().as_slice::<f32>().map_err(|e| anyhow!("{e:?}"))?.to_vec();
            let n = (v.iter().map(|x| x * x).sum::<f32>()).max(1e-6).sqrt();
            for x in &mut v { *x /= n; }
            Ok(v)
        }
    }

    // ---------- helpers ----------
    fn now_ts() -> i64 { Utc::now().timestamp() }

    fn normalize_topic(s: &str) -> String {
        let t = s.trim().to_lowercase();
        match t.as_str() {
            "c++" | "c plus plus" | "c-plus-plus" | "cpp" => "cpp".into(),
            "c#" | "c sharp" | "c-sharp" => "csharp".into(),
            "js" | "javascript" => "javascript".into(),
            "py" | "python" => "python".into(),
            _ => t,
        }
    }

    fn clean_text(s: &str) -> String {
        let mut t = s.trim().to_string();
        while t.ends_with(['.', ';']) { t.pop(); }
        t
    }

    fn parse_def_topic(q: &str) -> Option<String> {
        // Match: what is X / what's X / define X / explain X
        static RE: once_cell::sync::Lazy<Regex> = once_cell::sync::Lazy::new(|| {
            Regex::new(r"(?i)^\s*(what\s+is|what'?s|define|explain)\s+(.+?)[\?\s]*$").unwrap()
        });
        let caps = RE.captures(q.trim())?;
        let topic = caps.get(2)?.as_str();
        Some(normalize_topic(topic))
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

    fn main() -> Result<()> {
        let args = Args::parse();

        let conn = Connection::open(&args.db)?;
        conn.busy_timeout(std::time::Duration::from_millis(500))?;

        // SMIE + encoder (same CKPT as runtime)
        type B = NdArray<f32>;
        let device = <B as Backend>::Device::default();
        let model = Arc::new(Mutex::new(HippocampusModel::<B>::load_or_init(&device, &args.ckpt)));
        let emb = Arc::new(SmieHippEmbed::<B>::new(model.clone())?);
        let smie = Arc::new(Smie::new(SmieConfig { dim: emb.dim(), metric: Metric::Cosine }, emb)?);
        std::fs::create_dir_all(&args.smie_dir).ok();
        let _ = smie.load(&args.smie_dir);

        // fetch a batch from learn_queue
        let mut st = conn.prepare(
            "SELECT id, question, positive, COALESCE(hardness, 0.5) AS hardness
               FROM learn_queue
              WHERE used = 0
              ORDER BY id
              LIMIT ?1"
        )?;
        let rows = st.query_map([args.max as i64], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, f32>(3)?,
            ))
        })?;

        let mut n_done = 0usize;
        for row in rows {
            let (qid, q, p, hardness) = row?;

            // guard against garbage
            if looks_like_url(&q) || looks_like_web_error(&p) {
                let _ = conn.execute("UPDATE learn_queue SET used=1 WHERE id=?1", [qid]);
                continue;
            }

            let conf = (1.0_f32 - hardness).clamp(0.0, 1.0);
            let ts = now_ts();

            if let Some(topic) = parse_def_topic(&q) {
                // definition-like → update def:topic and seed SMIE
                let desc = clean_text(&p);
                let key = format!("def:{}", topic);

                let _ = kops::upsert_fact(&conn, ktypes::FactWrite {
                    subj: None,
                    predicate: key.clone(),
                    obj_text: desc.clone(),
                    obj_entity: None,
                    confidence: conf,
                    ts,
                    method: "critic".into(),
                    evidence: vec![format!("learn:q:{}|p:{}", &q, &desc)],
                });
                let _ = kops::prefer_latest(&conn, None, &key);

                let short = format!("{}: {}", topic, &desc);
                let _ = smie.ingest(&short, tags::SYSTEM | tags::SHORT);
            } else {
                // non-def → seed SMIE with a QA snippet
                let snippet = format!("{} {}", clean_text(&q), clean_text(&p));
                let _ = smie.ingest(&snippet, tags::LONG);
            }

            let _ = conn.execute("UPDATE learn_queue SET used=1 WHERE id=?1", [qid]);
            let _ = conn.execute("DELETE FROM learn_negs WHERE qid=?1", [qid]);
            n_done += 1;
        }

        smie.save(&args.smie_dir)?;
        println!("Processed {} learn items. SMIE saved → {}", n_done, args.smie_dir);
        Ok(())
    }

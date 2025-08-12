use std::sync::Arc;
use tempfile::tempdir;

use URIELV1::Hippocampus::sqlite_backend::{store_entry, recall_all};
use URIELV1::PFC::context_retriever::{
    ContextRetriever, RetrieverCfg, VectorStore, Embeddings, Embedding, MemoryHit
};

/// --- tiny, deterministic embedder for tests (no Burn!) ---
/// 64-dim bag-of-characters with simple hashing + L2 norm.
struct SimpleEmb;

impl SimpleEmb {
    fn normalize(text: &str) -> String {
        let mut s = String::with_capacity(text.len());
        for ch in text.chars() {
            if ch.is_ascii_alphanumeric() { s.push(ch.to_ascii_lowercase()); }
            else { s.push(' '); }
        }
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    fn words(text: &str) -> Vec<String> {
        Self::normalize(text)
            .split_whitespace()
            .map(|w| w.to_string())
            .collect()
    }

    fn word_bigrams(words: &[String]) -> Vec<String> {
        words.windows(2)
            .map(|w| format!("{} {}", w[0], w[1]))
            .collect()
    }

    fn char_ngrams(text: &str, n: usize) -> Vec<String> {
        let s: String = Self::normalize(text).chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        if s.len() < n { return vec![]; }
        s.as_bytes()
            .windows(n)
            .map(|win| String::from_utf8(win.to_vec()).unwrap())
            .collect()
    }

    // Tiny synonym hook to tie related surface forms together.
    fn synonym_tokens(text: &str) -> Vec<String> {
        let t = Self::normalize(text);
        let mut syn = Vec::new();

        let nn = t.contains("nearest neighbor") || t.contains("approximate nearest neighbor") || t.contains("ann");
        let hnsw = t.contains("hnsw") || t.contains("hierarchical navigable small world");
        if nn || hnsw {
            syn.push("__syn:ann__".into());
        }
        // add more if you need: faiss, efSearch, etc.
        syn
    }

    // tiny FNV-1a for hashing into dims
    fn hash_to_dim(token: &str, dims: usize) -> usize {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in token.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        (h as usize) % dims
    }
}

impl Embeddings for SimpleEmb {
    fn embed(&self, text: &str) -> Result<Embedding, Box<dyn std::error::Error + Send + Sync>> {
        const D: usize = 512; // bump a bit to reduce collisions
        let mut v = vec![0.0f32; D];

        let words = Self::words(text);
        let bigrams = Self::word_bigrams(&words);
        let trigrams = Self::char_ngrams(text, 3);
        let syn = Self::synonym_tokens(text);

        // 1) full words (primary signal)
        for w in &words {
            let wgt = 1.0 + (w.len() as f32 / 10.0).min(1.0);
            let idx = Self::hash_to_dim(w, D);
            v[idx] += wgt;
        }

        // 2) bigrams help “nearest neighbor” act like a unit
        for bg in &bigrams {
            let idx = Self::hash_to_dim(bg, D);
            v[idx] += 0.8;
        }

        // 3) char trigrams: subword glue
        for g in &trigrams {
            let idx = Self::hash_to_dim(g, D);
            v[idx] += 0.4;
        }

        // 4) synonym features: ties “nearest neighbor” <-> “HNSW”
        for s in &syn {
            let idx = Self::hash_to_dim(s, D);
            v[idx] += 1.2;
        }

        // L2 normalize
        let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
        for x in &mut v { *x /= n; }
        Ok(v)
    }
}

/// brute-force vector store over SQLite rows for a given `context`
struct SqliteScanVS<E: Embeddings> { emb: Arc<E>, context: String }
impl<E: Embeddings> SqliteScanVS<E> {
    fn new(emb: Arc<E>, context: impl Into<String>) -> Self {
        Self { emb, context: context.into() }
    }
}
impl<E: Embeddings> VectorStore for SqliteScanVS<E> {
    fn search(&self, q: &Embedding, top_k: usize)
              -> Result<Vec<MemoryHit>, Box<dyn std::error::Error + Send + Sync>>
    {
        let rows = recall_all(&self.context).map_err(|e| {
            Box::<dyn std::error::Error + Send + Sync>::from(
                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
            )
        })?;

        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let e = self.emb.embed(&r.data)?;
            let sim = cosine(q, &e);
            out.push(MemoryHit { id: format!("{}:{}", self.context, r.timestamp), sim, text: r.data });
        }
        out.sort_by(|a,b| b.sim.partial_cmp(&a.sim).unwrap_or(std::cmp::Ordering::Equal));
        if out.len() > top_k { out.truncate(top_k); }
        Ok(out)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b).map(|(x,y)| x*y).sum::<f32>();
    dot.clamp(0.0, 1.0) // both vectors are unit, keep [0,1] for simplicity
}

/// helper: build retriever bound to a context
fn build_retriever(context: &str) -> ContextRetriever {
    let emb = Arc::new(SimpleEmb);
    let vs  = Arc::new(SqliteScanVS::new(emb.clone(), context.to_string()));
    let cfg = RetrieverCfg { top_k_default: 8, top_k_research: 16, min_score: 0.0, use_cache: false };
    ContextRetriever::new(vs, emb, None, cfg)
}

#[test]
fn e2e_multiple_seeds_and_queries() {
    // isolate DB in a temp file (don’t use :memory: because each Connection is separate)
    let td = tempdir().expect("tempdir");
    let db_path = td.path().join("semantic_memory_e2e.db");
    std::env::set_var("SMIE_DB_PATH", &db_path);

    let ctx = "default";

    // --- seed facts ---
    let seeds = [
        "the launch code is zebra-13",
        "project URIEL uses a FixedWeightEmbedder",
        "my favorite color is ultramarine",
        "the database runs on SQLite",
        "HNSW enables fast ANN search",
    ];
    let mut t0 = 1u64;
    for s in &seeds {
        // stable timestamps so IDs are deterministic in assertions if needed
        store_entry(ctx, s, Some(t0)).expect("store");
        t0 += 1;
    }

    // --- build retriever ---
    let retriever = build_retriever(ctx);

    // --- table-driven queries -> expected substrings ---
    let cases = [
        ("what is the launch code?",           "zebra-13"),
        ("what does URIEL use for embeddings", "FixedWeightEmbedder"),
        ("what's my favorite color?",          "ultramarine"),
        ("which database do we use",           "SQLite"),
        ("what enables ANN?",           "HNSW"),
    ];

    for (q, expect) in cases {
        let mode = URIELV1::PFC::cortex::CortexReasoningMode::DirectRecall; // depth doesn’t matter here
        let cf = retriever.build_context_frame_from_str(q, &mode).expect("context frame");

        assert!(
            !cf.top_hits.is_empty(),
            "query {:?} produced no hits", q
        );

        // top-1 should contain the expected substring
        let top = &cf.top_hits[0].text;
        assert!(
            top.to_lowercase().contains(&expect.to_lowercase()),
            "top-1 {:?} does not contain expected {:?} for query {:?}",
            top, expect, q
        );
    }
}

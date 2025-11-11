    // src/bin/uriel_runtime.rs
    use std::cmp::Ordering;
    use std::collections::{HashMap, HashSet};
    use std::error::Error;
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use serde::{Deserialize, Serialize};
    use std::fs;
    use std::sync::RwLock;
    use std::time::SystemTime;
    use burn::record::Recorder;
    use burn::module::Module;
    use anyhow::Result;
    use burn::tensor::backend::Backend;
    use burn::tensor::Tensor;
    use burn_ndarray::NdArray;
    use crossbeam_channel::{unbounded, Receiver, Sender};
    use ctrlc;
    use rand::Rng;
    use regex::Regex;
    use rusqlite;
    use sha2::{Digest, Sha256};
    use URIELV1::riftnet::RiftClientDyn;
    use std::fs::File;
    use std::io::{BufRead, BufReader, Seek, SeekFrom};
    use URIELV1::Learning::ingest::DatasetEmitter;

    use smie_core::{Embedder, Metric, Smie, SmieConfig, tags};
    // Callosum bus + metacog subscriber
    use URIELV1::callosum::callosum::CorpusCallosum;
    use URIELV1::callosum::metacog_subscriber::{spawn_metacog_subscriber, EpisodeEvent, FactUpsertEvent};
    // Soarus bridge (background worker API)
    use URIELV1::soarus::{start_worker, SoarusJob, SoarusMsg, SoarusIcpConfig};
    use soarus_bridge::{IcpMode as SoarusIcpMode};
    use URIELV1::policy::types::{Route, RtFeatures, RtProbs, ClarifyDecision};
    use URIELV1::policy::features::extract_features;
    use URIELV1::policy::guardrail::default_guardrails;
    use URIELV1::policy::router_infer::RouterInfer;
    use URIELV1::policy::policy_infer::RtPolicy;
    use URIELV1::policy::clarification::Clarifier;
    // ── concepts & contexts
    use URIELV1::concepts_and_contexts::{
        answer::answer_concept,
        extractors::{CreatorConcept, LaunchCodeConcept, MentorConcept, NameConcept, PurposeConcept, IdentityConcept},
        learner::ConceptLearner,
        registry::Registry,
        speech_concepts::realize_concept,
    };
    use burn::record::CompactRecorder;
    use URIELV1::Learning::routercopy::trainer::RouterCopySystem;
    // before main(), with the other use lines:
    use URIELV1::knowledge::{schema as kschema, ops as kops, types as ktypes};
    use URIELV1::meta::trace_v2::{TraceSink, Evidence, EvidenceQuality};
    use URIELV1::meta::callobs::{Writer as CofWriter, Op, Arg, FinalLabel};

    // ── meta trace (Phase 1)
    use URIELV1::meta::trace::{MetaStep, BanditInfo, SourcesInfo, record_meta_trace};
    use URIELV1::meta::metacog_loop::{MetacogCfg, spawn_metacog_loop};
    use crossbeam_channel as xchan; // we’ll use a small stop channel

    // ── URIEL subsystems
    use URIELV1::Frontal::frontal::{
        evaluate_frontal_control, FrontalContext, FrontalDecision, InhibitionConfig, InhibitionState,
        WorkingMemory,
    };
    use URIELV1::Hippocampus::infer::HippocampusModel;
    use URIELV1::Hippocampus::pipeline;
    use URIELV1::Hippocampus::primordial::{primordial, PRIMORDIAL_SHA256};
    use URIELV1::Hippocampus::sqlite_backend::{
        has_hash, read_concept_fact, recall_latest, record_hash, upsert_concept_fact,insert_belief, safe_upsert_concept_fact,
    };
    use URIELV1::Microthalamus::goal::{Ctx, Goal};
    use URIELV1::Microthalamus::microthalamus::{MicroThalamus, SyntheticDrive};
    use URIELV1::PFC::context_retriever::{
        TextSearch, ContextRetriever, Embedding, Embeddings, MemoryHit, RetrieverCfg, VectorStore,
    };
    use URIELV1::PFC::cortex::{execute_cortex, CortexInputFrame, CortexOutput};
    use URIELV1::parietal::parietal::SalienceScore;

    // bandit
    use URIELV1::policy::retrieval::{Arm, Bandit};

    fn persist_soarus_pose(conn: &mut rusqlite::Connection, o: &URIELV1::soarus::SoarusIcpOutput) -> anyhow::Result<()> {
        // Ensure a tiny table exists (idempotent)
        conn.execute_batch(r#"
        CREATE TABLE IF NOT EXISTS soarus_pose (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            qx REAL, qy REAL, qz REAL, qw REAL,
            tx REAL, ty REAL, tz REAL,
            rmse REAL, p99 REAL, inliers INTEGER
        );
    "#)?;

        let q = o.pose.rotation;
        let t = o.pose.translation.vector;

        conn.execute(
            "INSERT INTO soarus_pose(ts,qx,qy,qz,qw,tx,ty,tz,rmse,p99,inliers)
         VALUES(strftime('%s','now'),?,?,?,?,?,?,?,?,?,?)",
            rusqlite::params![
            q.i, q.j, q.k, q.w,
            t.x, t.y, t.z,
            o.rmse, o.p99, o.inliers as i64,
        ],
        )?;
        Ok(())
    }

    fn cognition_reads_enabled() -> bool {
        std::env::var("URIEL_COGNITION_READ")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on"))
            .unwrap_or(true)
    }
    fn map_source_to_method(s: &str) -> &'static str {
        match s {
            "db" => "db",
            "db+smie" | "keyword" | "normalized" | "original" => "smie",
            "rows" => "rows",
            "pfc" => "pfc",
            _ => "unknown",
        }
    }
    // ---------- Autonomous input sources ----------
    fn spawn_stdin_source(tx: Sender<String>) {
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                line.clear();
                match stdin.read_line(&mut line) {
                    Ok(0) => { std::thread::sleep(std::time::Duration::from_millis(10)); }
                    Ok(_) => {
                        let s = line.trim_end_matches(&['\n','\r'][..]).to_string();
                        if s.is_empty() { continue; }
                        // Allow manual quit from stdin if someone types it
                        if s.eq_ignore_ascii_case("exit") || s.eq_ignore_ascii_case("quit") {
                            // best-effort: still send so main loop can see it
                            let _ = tx.send("__QUIT__".into());
                            break;
                        }
                        let _ = tx.send(s);
                    }
                    Err(_) => break,
                }
            }
        });
    }

    /// Tail a file like `tail -f`. If `from_start=false`, begin at EOF.
    fn spawn_tail_source(path: String, from_start: bool, tx: Sender<String>) {
        std::thread::spawn(move || {
            loop {
                match File::open(&path) {
                    Ok(mut f) => {
                        if !from_start {
                            let _ = f.seek(SeekFrom::End(0));
                        }
                        let mut reader = BufReader::new(f);
                        let mut buf = String::new();
                        loop {
                            buf.clear();
                            match reader.read_line(&mut buf) {
                                Ok(0) => { std::thread::sleep(std::time::Duration::from_millis(25)); }
                                Ok(_) => {
                                    let s = buf.trim_end_matches(&['\n','\r'][..]).to_string();
                                    if !s.is_empty() {
                                        let _ = tx.send(s);
                                    }
                                }
                                Err(_) => break, // file rotated or error -> reopen
                            }
                        }
                    }
                    Err(_) => {
                        // file not present yet; retry
                        std::thread::sleep(std::time::Duration::from_millis(200));
                    }
                }
            }
        });
    }
    fn cognition_writes_enabled() -> bool {
        std::env::var("URIEL_COGNITION_WRITE")
            .map(|v| v=="1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on"))
            .unwrap_or(true)
    }
    // --- Session context: decaying domain/topic salience ---
    #[derive(Debug, Default, Clone)]
    struct SessionCtx {
        // domain -> weight in [0,1]
        dom_weights: HashMap<String, f32>,
        // time bookkeeping
        last_tick: u64,
        // hyperparams
        half_life_secs: u64,
        // optional: recent topics for lightweight threading
        recent_topics: Vec<String>, // small LRU
        max_topics: usize,
    }

    impl SessionCtx {
        fn new(half_life_secs: u64) -> Self {
            Self {
                dom_weights: HashMap::new(),
                last_tick: now_sec(),
                half_life_secs,
                recent_topics: Vec::with_capacity(8),
                max_topics: 8,
            }
        }

        fn decay(&mut self) {
            let now = now_sec();
            let dt = now.saturating_sub(self.last_tick) as f32;
            self.last_tick = now;
            if self.half_life_secs == 0 { return; }
            // decay factor e^(-ln(2)*dt/half_life)
            let k = -(std::f32::consts::LN_2) * (dt / self.half_life_secs as f32);
            let factor = k.exp();
            for w in self.dom_weights.values_mut() {
                *w *= factor;
                if *w < 1e-3 { *w = 0.0; } // prune tiny
            }
            self.dom_weights.retain(|_,v| *v > 0.0);
        }

        fn bump_domain(&mut self, dom: &str, strength: f32) {
            let d = norm_domain(dom);
            let entry = self.dom_weights.entry(d).or_insert(0.0);
            *entry = (*entry + strength).min(1.5);
        }

        /// Choose the current domain if any has sufficient weight.
        fn current_domain(&self, min_conf: f32) -> Option<String> {
            self.dom_weights
                .iter()
                .max_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .and_then(|(d,w)| if *w >= min_conf { Some(d.clone()) } else { None })
        }

        fn observe_topic(&mut self, topic: &str) {
            let t = canonicalize_surface(topic);
            if t.is_empty() { return; }
            if let Some(pos) = self.recent_topics.iter().position(|x| x == &t) {
                self.recent_topics.remove(pos);
            }
            self.recent_topics.insert(0, t);
            if self.recent_topics.len() > self.max_topics {
                self.recent_topics.pop();
            }
        }
    }
    fn split_lead_context(line: &str) -> (Option<String>, String) {
        // Split once on a sentence terminator; keep the main clause.
        let l = line.trim();
        for sep in ['.', '!', '?', ';'] {
            if let Some(i) = l.find(sep) {
                let (lead, rest) = l.split_at(i);
                let rest = rest.trim_start_matches(|c: char| c == sep || c.is_whitespace()).trim();
                if !rest.is_empty() {
                    return (Some(lead.trim().to_string()), rest.to_string());
                }
            }
        }
        (None, l.to_string())
    }

    // Accept "tell me about shading (in art)", "talk about shaders", "about X in Y"
    fn parse_about_query(input: &str) -> Option<(String /*topic*/, Option<String> /*domain*/)> {
        let l = ascii_norm(input);
        let re = Regex::new(
            r"(?ix)^(?:\s*(?:tell\s+me|talk|chat|speak)\s+)?(?:about|on|regarding)\s+(.+?)\s*$"
        ).ok()?;
        let caps = re.captures(&l)?;
        let raw = caps.get(1)?.as_str().trim();

        // optional “… in <domain>”
        if let Some(ci) = Regex::new(r"(?i)^(.+?)\s+in\s+([a-z][a-z0-9 _\-]{2,32})\s*$").ok()?.captures(raw) {
            let topic = ci.get(1)?.as_str().trim().to_string();
            let dom   = norm_domain(ci.get(2)?.as_str());
            if !topic.is_empty() { return Some((topic, Some(dom))); }
            return None;
        }
        if raw.is_empty() { None } else { Some((raw.to_string(), None)) }
    }
    pub fn tms_preferred(conn: &rusqlite::Connection, key: &str) -> rusqlite::Result<Option<(String,f32,i64)>> {
        let mut st = conn.prepare("SELECT value,support,last_seen FROM v_preferred WHERE predicate=?1")?;
        let mut rows = st.query([key])?;
        if let Some(r) = rows.next()? {
            Ok(Some((r.get(0)?, r.get(1)?, r.get(2)?)))
        } else {
            Ok(None)
        }
    }

    pub fn tms_upsert_support(
        conn: &rusqlite::Connection,
        predicate: &str,
        value: &str,
        method: &str,
        confidence: f32,
        evidence: &[String],
    ) -> rusqlite::Result<()> {
        let _ = URIELV1::knowledge::ops::upsert_fact(conn, URIELV1::knowledge::types::FactWrite {
            subj: None,
            predicate: predicate.to_string(),
            obj_text: value.to_string(),
            obj_entity: None,
            confidence,
            ts: now_sec() as i64,
            method: method.to_string(),
            evidence: evidence.to_vec(),
        });
        let _ = URIELV1::knowledge::ops::prefer_latest(conn, None, predicate);
        Ok(())
    }
    fn read_launch_code_any(conn: &rusqlite::Connection) -> Option<(String,&'static str)> {
        for (k,src) in [("launch code","cf"), ("launch_code","cf")] {
            if let Ok(Some(v)) = read_concept_fact(k).map_err(to_any) { return Some((v, src)); }
        }
        if cognition_reads_enabled() {
            for (k,src) in [("launch code","kb"), ("launch_code","kb")] {
                if let Ok(Some(v)) = kops::read_best_fact_text(conn, None, k) { return Some((v, src)); }
            }
        }
        None
    }
    // ─────────────────────────── helpers
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum IntentKind {
        Ask,            // “what / why / how / ?”
        DefineAssert,   // “X is …”, “In <domain>, X is …”
        Correct,        // “No, X is …”, “X isn’t …”
        Compare,        // “A vs B”
        Instruct,       // “explain …”, “define …”, “how to …”
        Style,          // “from now on say …”, “use ‘I am’ …”
        Chat,           // small talk / unclear
    }

    fn is_negation_prefix(s: &str) -> bool {
        let l = ascii_norm(s).to_ascii_lowercase();
        l.starts_with("no, ") || l.starts_with("no ") || l.starts_with("not ") ||
            l.starts_with("actually ") || l.starts_with("correction:") || l.starts_with("that is wrong")
    }
    fn ensure_runtime_schema(conn: &rusqlite::Connection) -> anyhow::Result<()> {
        // Create the tables/views we actually read/write in the runtime.
        conn.execute_batch(r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA temp_store = MEMORY;

            -- Core concept store
            CREATE TABLE IF NOT EXISTS concept_facts (
                concept_key TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cf_updated_at
                ON concept_facts(updated_at DESC);

            -- Back-compat view for legacy SELECT key,value,ts ...
            CREATE VIEW IF NOT EXISTS concept_fact AS
            SELECT concept_key AS key, value, updated_at AS ts
            FROM   concept_facts;

            -- Durable dedupe
            CREATE TABLE IF NOT EXISTS memory_hashes (
                ctx           TEXT NOT NULL,
                hash          TEXT NOT NULL,
                first_seen_at INTEGER NOT NULL,
                PRIMARY KEY (ctx, hash)
            );

            -- Beliefs + policy + audit (names spelled correctly)
            CREATE TABLE IF NOT EXISTS self_beliefs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER,
                text TEXT,
                source TEXT,
                weight REAL
            );

            CREATE TABLE IF NOT EXISTS policy_kv (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS write_audit (
                ts INTEGER,
                actor TEXT,
                op TEXT,
                k TEXT,
                old_v TEXT,
                new_v TEXT,
                source TEXT,
                notes TEXT
            );

            -- Stream storage used by recall_latest()
            CREATE TABLE IF NOT EXISTS semantic_memory (
                context   TEXT NOT NULL,
                data      TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sm_context_ts
                ON semantic_memory (context, timestamp DESC);
        "#)?;
        Ok(())
    }

    // Basic speech-act/intent rules (cheap, fast). You can replace with a tiny head later.
    fn infer_intent(line: &str) -> IntentKind {
        let l  = ascii_norm(line);
        let ll = l.to_ascii_lowercase();

        // 1) Questions must win first
        if is_question(&l) {
            return IntentKind::Ask;
        }

        // 2) Comparisons
        if parse_vs_query(&l).is_some() || parse_compare_query(&l).is_some() {
            return IntentKind::Compare;
        }

        // 3) Style toggles (copula)
        if belief_implies_copula_long(&l).is_some() {
            return IntentKind::Style;
        }

        // 4) Definitions/assertions
        if parse_domained_def(&l).is_some() || extract_definition(&l).is_some() {
            return IntentKind::DefineAssert;
        }

        // 5) Imperative prompts
        if is_prompt_like(&l) {
            return IntentKind::Instruct;
        }

        IntentKind::Chat
    }



    #[inline]
    fn add_token(mut v: serde_json::Value, token: Option<&str>) -> serde_json::Value {
        if let Some(t) = token {
            if let Some(obj) = v.as_object_mut() {
                obj.insert("token".into(), serde_json::Value::String(t.to_string()));
            }
        }
        v
    }

    #[derive(Serialize, Deserialize, Clone, Default)]
    struct MeshTogglesCfg {
        forward: Option<bool>,
        broadcast_episodes: Option<bool>,
        broadcast_facts: Option<bool>,
        accept_episodes: Option<bool>,
        accept_facts: Option<bool>,
        token: Option<String>,
    }

    #[derive(Serialize, Deserialize, Clone, Default)]
    struct RiftCfg {
        dll: Option<String>,
        host: Option<String>,
        port: Option<u16>,
    }

    #[derive(Serialize, Deserialize, Clone, Default)]
    struct MeshConfig {
        riftnet: Option<RiftCfg>,
        mesh:    Option<MeshTogglesCfg>,
    }

    fn load_mesh_config() -> Option<MeshConfig> {
        let path = std::env::var("MESH_CONFIG").ok()?;          // e.g. data/mesh.json
        let txt  = fs::read_to_string(&path).ok()?;
        serde_json::from_str::<MeshConfig>(&txt).ok()
    }

    #[inline]
    fn env_bool(key: &str, default: bool) -> bool {
        std::env::var(key)
            .map(|v| matches!(v.as_str(), "1"|"true"|"TRUE"|"on"|"ON"|"yes"|"YES"))
            .unwrap_or(default)
    }

    #[derive(Serialize, Deserialize, Clone)]
    struct RouterBiasFile { bias: [f32;4], updated_at: i64 }

    fn load_router_bias(path: &str) -> [f32;4] {
        if let Ok(txt) = fs::read_to_string(path) {
            if let Ok(rb) = serde_json::from_str::<RouterBiasFile>(&txt) {
                return rb.bias;
            }
        }
        [0.0;4]
    }
    fn save_router_bias(path: &str, bias: [f32;4]) {
        let obj = RouterBiasFile { bias, updated_at: now_sec() as i64 };
        if let Ok(txt) = serde_json::to_string_pretty(&obj) {
            let _ = fs::create_dir_all("data");
            let _ = fs::write(path, txt);
        }
    }
    fn apply_bias_to_probs(p: &mut [f32;4], bias: [f32;4]) {
        let mut s = 0.0f32;
        for i in 0..4 { p[i] *= bias[i].exp(); s += p[i]; }
        if s > 0.0 { for i in 0..4 { p[i] /= s; } }
    }

    #[derive(Serialize, Deserialize, Clone)]
    struct OepWeightsFile { dim: usize, w: Vec<f32>, updated_at: i64 }

    fn load_oep_weights(path: &str, expected_dim: usize) -> Option<Vec<f32>> {
        if let Ok(txt) = fs::read_to_string(path) {
            if let Ok(wf) = serde_json::from_str::<OepWeightsFile>(&txt) {
                if wf.dim == expected_dim && wf.w.len() == expected_dim {
                    return Some(wf.w);
                }
            }
        }
        None
    }

    fn looks_like_url(s: &str) -> bool {
        let l = s.trim().to_ascii_lowercase();
        l.starts_with("http://") || l.starts_with("https://") || l.contains("://")
    }

    fn looks_like_web_error(s: &str) -> bool {
        let l = s.to_ascii_lowercase();
        l.contains("please set a user-agent")
            || l.contains("access denied")
            || l.contains("forbidden")
            || l.contains("captcha")
            || l.contains("robots")
            || s.contains("http://") || s.contains("https://")
            || s.len() > 600
    }

    fn answered_from_static(s: &str) -> &'static str {
        match s {
            "db" => "db",
            "normalized" => "normalized",
            "keyword" => "keyword",
            "original" => "original",
            "rows" => "rows",
            "pfc" => "pfc",
            _ => "unknown",
        }
    }

    fn assert_primordial_invariants() {
        let p = primordial();
        assert!(!p.axioms.is_empty());
        assert!(!p.prohibitions.is_empty());
        assert!(!p.drives.is_empty());
        let sum_w: f32 = p.drives.iter().map(|d| d.weight).sum();
        assert!((0.5..=2.0).contains(&sum_w));
        println!("Primordial OK (v{}, sha256 {}).", p.version, PRIMORDIAL_SHA256);
    }
    fn turn_hash_u64(s: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut h);
        h.finish()
    }

    #[inline]
    fn to_any<E: std::fmt::Display + Send + Sync + 'static>(e: E) -> anyhow::Error {
        anyhow::anyhow!(e.to_string())
    }

    fn sanitize_input(s: &str) -> String {
        let mut t = s.trim();
        while let Some(c) = t.chars().next() {
            if matches!(c, '>' | '•' | '-' | '|') { t = t[1..].trim_start(); } else { break; }
        }
        t.to_string()
    }

    // 1) types near the top of uriel_runtime.rs
    #[derive(serde::Deserialize, Clone, Copy, Default)]
    struct Thresholds { promote: f32, refuse: f32 }
    #[derive(serde::Deserialize)]
    struct OpCalib { #[serde(rename="T")] t: f32, ece: f32, brier: f32 }
    #[derive(serde::Deserialize)]
    struct CalibFile { op: OpCalib, #[serde(default)] thresholds: Thresholds }

    // 2) helper
    fn load_routercopy_calib(path: &str) -> Option<(f32, Thresholds)> {
        std::fs::read_to_string(path).ok().and_then(|s| {
            serde_json::from_str::<CalibFile>(&s).ok()
                .map(|c| (c.op.t, if c.thresholds.promote > 0.0 { c.thresholds } else { Thresholds{promote:0.60, refuse:0.40} }))
        })
    }

    fn rc_encode_ascii95(s: &str, max_t: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(max_t);
        for ch in s.chars().take(max_t) {
            let idx = (ch as i32 - 32).clamp(0, 94) as usize;
            out.push(idx);
        }
        while out.len() < max_t { out.push(0); }
        out
    }
    // Consider prompts that should *not* be auto-stored (even if they don't end in '?').
    fn is_prompt_like(s: &str) -> bool {
        let l = ascii_norm(s).to_ascii_lowercase();

        if is_question(&l) {
            return true;
        }

        // Common instruction/prompt stems (keep ASCII-only; add more as needed)
        const STEMS: [&str; 34] = [
            "define ", "explain ", "what is ", "what's ", "what si ", "hwat is ",
            "tell me about ", "give me an overview of ", "give me a summary of ",
            "what are the key points of ", "how does ", "how do i ", "how to ",
            "what are the main features of ", "what is the purpose of ",
            "what are the benefits of ", "what are the drawbacks of ",
            "in what scenarios is ", "what are some examples of ", "how is ",
            "what should i know about ", "why is ", "what are the applications of ",
            "how do i get started with ", "what are the best practices for ",
            "what are common mistakes when using ", "how has ", "what is the history of ",
            "who are the key figures associated with ", "when was ", "where can i find tutorials on ",
            "what are the future trends related to ", "when should i consider using ", "when not to use "
        ];

        if STEMS.iter().any(|p| l.starts_with(p)) {
            return true;
        }
        // Comparisons
        if l.contains(" vs ") || l.starts_with("compare ") || l.contains(" instead of ") {
            return true;
        }

        false
    }

    // Env gate: set URIEL_AUTO_STORE=0 to disable auto-store during seeding.
    fn auto_store_allowed() -> bool {
        match std::env::var("URIEL_AUTO_STORE") {
            Ok(v) => v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on"),
            Err(_) => true, // default: allow (keeps previous behavior)
        }
    }

    fn rc_tokens_to_one_hot<B: Backend>(
        tokens: &[Vec<usize>],
        v: usize,
        dev: &B::Device,
    ) -> Tensor<B, 3> {
        let b = tokens.len();
        let t = tokens.first().map(|x| x.len()).unwrap_or(0);
        let mut data = vec![0.0f32; b * t * v];

        for (bi, row) in tokens.iter().enumerate() {
            for (ti, &idx) in row.iter().enumerate() {
                if idx < v {
                    data[bi * t * v + ti * v + idx] = 1.0;
                }
            }
        }
        // pass a slice, not Vec
        Tensor::<B, 1>::from_floats(data.as_slice(), dev).reshape([b, t, v])
    }



    fn rc_embed_seq<B: Backend>(oh: Tensor<B,3>, emb: &URIELV1::embedder::embedder::FixedWeightEmbedder<B>) -> Tensor<B,3> {
        let [b,t,v] = { let d = oh.dims(); [d[0],d[1],d[2]] };
        let flat = oh.reshape([b*t, v]);
        let proj = emb.forward(flat);
        proj.reshape([b,t,384])
    }

    fn rc_mean_over_time<B: Backend>(seq: Tensor<B,3>) -> Tensor<B,2> { seq.mean_dim(1).squeeze(1) }


    fn parse_vs_query(input: &str) -> Option<(String, String)> {
        let l = ascii_norm(input);
        let re = Regex::new(r"(?i)^\s*([A-Za-z0-9+./#\-\s']+)\s+v(?:s|\.|ersus)\.?\s+([A-Za-z0-9+./#\-\s']+)\s*\??\s*$").ok()?;
        if let Some(c) = re.captures(&l) {
            let a = strip_quotes(c.get(1)?.as_str()).trim();
            let b = strip_quotes(c.get(2)?.as_str()).trim();
            if !a.is_empty() && !b.is_empty() { return Some((a.to_string(), b.to_string())); }
        }
        None
    }
    fn parse_compare_query(input: &str) -> Option<(String,String)> {
        let l = ascii_norm(input);
        let re = Regex::new(
            r"(?ix)
              ^\s*(?:compare|contrast)\s+(.+?)\s+(?:and|with)\s+(.+?)\s*\??\s*$|
              ^\s*(?:what(?:'s|\s+is)\s+)?the\s+difference\s+between\s+(.+?)\s+and\s+(.+?)\s*\??\s*$
            "
        ).ok()?;
        if let Some(c) = re.captures(&l) {
            let a = c.get(1).or_else(|| c.get(3)).map(|m| m.as_str().trim())?;
            let b = c.get(2).or_else(|| c.get(4)).map(|m| m.as_str().trim())?;
            if !a.is_empty() && !b.is_empty() { return Some((a.to_string(), b.to_string())); }
        }
        None
    }

    fn short_def_for(topic: &str) -> Option<String> {
        // 1) def:topic
        let key = format!("def:{}", canonical_topic(topic));
        if let Ok(Some(def)) = read_concept_fact(&key).map_err(to_any) { return Some(def); }
        None
    }
    struct CompareOut { table: String, summary: String, evidence: Vec<String> }

    fn load_compare_facets(conn: &rusqlite::Connection) -> Vec<String> {
        if let Ok(Some((json, _, _))) = tms_preferred(conn, "op:compare:facets") {
            serde_json::from_str::<Vec<String>>(&json).unwrap_or_else(|_| vec!["definition".into()])
        } else {
            vec!["definition".into()] // fallback minimum
        }
    }

    fn facet_value(
        conn: &rusqlite::Connection,
        smie: &Smie,
        topic: &str,
        facet: &str,
        dom: Option<&str>,
    ) -> Option<(String,String)> {
        // 1) Try KB forms first (data-driven)
        for key in [
            format!("prop:{}:{}", facet, topic),
            format!("facet:{}:{}", facet, topic),
            match dom {
                Some(d) if !d.is_empty() => format!("def:{}:{}", norm_domain(d), topic),
                _ => String::new(),
            },
            format!("def:{}", topic),
        ] {
            if key.is_empty() { continue; }
            if let Ok(Some(v)) = URIELV1::knowledge::ops::read_best_fact_text(conn, None, &key) {
                return Some((v, format!("kb:{key}")));
            }
            if let Some(v) = read_concept_fact(&key).ok().flatten() {
                return Some((v, format!("cf:{key}")));
            }

        }

        // 2) Salvage via SMIE: "<topic> <facet>"
        let q = format!("{} {}", topic, facet.replace('_', " "));
        if let Ok(hits) = smie.recall(
            &q, 6, tags::SHORT|tags::LONG|tags::SYSTEM|tags::USER
        ) {
            if let Some((_id, _sim, _tg, txt)) = hits.first() {
                if let Some((_t, desc)) = extract_definition(txt) {
                    return Some((desc.clone(), "smie".into()));
                } else {
                    let sentence = txt.split('.').next().unwrap_or(txt).trim().to_string();
                    if !sentence.is_empty() { return Some((sentence, "smie".into())); }
                }
            }
        }
        None
    }

    fn act_compare(
        conn: &rusqlite::Connection,
        smie: &Smie,
        lhs_raw: &str,
        rhs_raw: &str,
        dom_hint: Option<&str>,
    ) -> Option<CompareOut> {
        let lhs = canonical_topic(lhs_raw);
        let rhs = canonical_topic(rhs_raw);
        let dom = dom_hint.map(norm_domain);

        let mut facets = load_compare_facets(conn);
        if !facets.iter().any(|f| f == "definition") {
            facets.insert(0, "definition".into());
        }

        let mut rows: Vec<(String,String,String)> = Vec::new();
        let mut evidence: Vec<String> = Vec::new();

        for facet in facets.iter() {
            let l = facet_value(conn, smie, &lhs, facet, dom.as_deref());
            let r = facet_value(conn, smie, &rhs, facet, dom.as_deref());
            if let (Some((lv, ls)), Some((rv, rs))) = (l, r) {
                rows.push((facet.clone(), lv.clone(), rv.clone()));
                evidence.push(ls);
                evidence.push(rs);
            }
        }
        if rows.is_empty() { return None; }

        // Markdown table
        let mut table = String::from("| facet | ");
        table.push_str(lhs_raw); table.push_str(" | ");
        table.push_str(rhs_raw); table.push_str(" |\n|---|---|---|\n");
        for (f, lv, rv) in &rows {
            table.push_str(&format!("| {} | {} | {} |\n", f, lv, rv));
        }

        // Tiny TL;DR
        let mut tl = Vec::new();
        let find = |name: &str| rows.iter().find(|(f,_,_)| f == name).map(|(_,a,b)| (a,b));
        if let Some((a,b)) = find("when_use") {
            tl.push(format!("{}: best when {}", lhs_raw, a));
            tl.push(format!("{}: best when {}", rhs_raw, b));
        } else if let Some((a,b)) = find("use_cases") {
            tl.push(format!("{}: {}", lhs_raw, a));
            tl.push(format!("{}: {}", rhs_raw, b));
        } else if let Some((a,b)) = find("pros") {
            tl.push(format!("+ {}: {}", lhs_raw, a));
            tl.push(format!("+ {}: {}", rhs_raw, b));
        }
        let summary = if tl.is_empty() {
            format!("{} vs {}: see table.", lhs_raw, rhs_raw)
        } else { tl.join(" • ") };

        Some(CompareOut { table, summary, evidence })
    }

    // --- small edit distance (no crates) ---
    fn lev_distance(a: &str, b: &str) -> usize {
        let (a, b) = (a.as_bytes(), b.as_bytes());
        let (n, m) = (a.len(), b.len());
        if n == 0 { return m; }
        if m == 0 { return n; }
        let mut prev: Vec<usize> = (0..=m).collect();
        let mut curr = vec![0usize; m + 1];

        for i in 0..n {
            curr[0] = i + 1;
            for j in 0..m {
                let cost = if a[i] == b[j] { 0 } else { 1 };
                curr[j + 1] = (prev[j + 1] + 1)            // deletion
                    .min(curr[j] + 1)                      // insertion
                    .min(prev[j] + cost);                  // substitution
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        prev[m]
    }

    // Try to find a nearby definition key by querying SMIE for definitional snippets
    // and choosing the closest surface form to the requested topic.
    fn nearest_def_key_for(
        smie: &Smie,
        topic: &str,
        dom: Option<&str>,
        top_k: usize,
    ) -> Option<String> {
        let want = canonicalize_surface(topic);
        let mask: u64 = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
        let probes = [format!("{}:", want), format!("{} is", want), want.clone()];

        let mut best: Option<(usize, String)> = None; // (lev distance, key)

        for q in &probes {
            if let Ok(hits) = smie.recall(q, top_k, mask) {
                for (_id, _sim, _tg, txt) in hits {
                    if let Some((cand_topic, _desc)) = extract_definition(&txt) {
                        let cand = canonicalize_surface(&cand_topic);
                        let d = lev_distance(&want, &cand);
                        if d <= 2 && best.as_ref().map(|(bd, _)| d < *bd).unwrap_or(true) {
                            best = Some((d, def_key(dom, &cand)));
                        }
                    }
                }
            }
        }
        best.map(|(_, k)| k)
    }


    // ---------- Definition helpers (NEW) ----------

    fn canonical_topic(s: &str) -> String {
        let t = s.trim().to_ascii_lowercase();
        match t.as_str() {
            "c++" | "c plus plus" | "c-plus-plus" | "cpp" => "cpp".into(),
            "rust" | "rust lang" | "rustlang" => "rust".into(),
            "c#" | "c sharp" | "c-sharp" => "csharp".into(),
            "js" | "javascript" => "javascript".into(),
            "py" | "python" => "python".into(),
            _ => t,
        }
    }
    fn parse_definition_query(input: &str) -> Option<String> {
        let l = ascii_norm(input);

        // Optional domain preface like: "In art, what is shading?"
        let re_lead = r"(?:in\s+[a-z][a-z0-9 _\-]{2,32}\s*,\s*)?";

        let patterns = [
            format!(r"(?i)^\s*{}what\s+(?:is|are)\s+(.+?)\??\s*$", re_lead),
            format!(r"(?i)^\s*{}what'?s\s+(.+?)\??\s*$", re_lead),
            format!(r"(?i)^\s*{}define\s+(.+?)\s*$", re_lead),
            format!(r"(?i)^\s*{}explain\s+(.+?)\s*$", re_lead),
            // common typo
            format!(r"(?i)^\s*{}what\s+si\s+(.+?)\??\s*$", re_lead),
        ];

        for p in patterns {
            if let Some(c) = Regex::new(&p).ok()?.captures(&l) {
                let raw = strip_quotes(c.get(1).unwrap().as_str()).trim();
                if !raw.is_empty() { return Some(raw.to_string()); }
            }
        }
        None
    }

    fn ascii_norm(s: &str) -> String {
        s.replace('’', "'").replace('‘', "'")
            .replace('“', "\"").replace('”', "\"")
            .replace('—', "-").replace('–', "-")
            .trim()
            .to_string()
    }
    fn strip_quotes(s: &str) -> &str {
        s.trim().trim_matches('"').trim_matches('\'')
    }
    fn strip_topic_parens(s: &str) -> String {
        // Keep leading token; drop a single parenthetical like "(Algorithmic Language)"
        let s = s.trim();
        if let Some(i) = s.find('(') {
            return s[..i].trim().to_string();
        }
        s.to_string()
    }

    /// Accepts:
    ///  1) Topic (Paren)? is Desc
    ///  2) Topic: Desc   or   Topic - Desc / Topic — Desc
    ///  3) Desc is Topic   (reversed; we flip to Topic: Desc)
    fn extract_definition(line: &str) -> Option<(String, String)> {
        let l = ascii_norm(line);
        let cop = r"(?:is|are|refers\s+to|means|=)";

        // 1) Topic (Paren)? cop Desc
        let re1 = Regex::new(&format!(
            r"(?i)^\s*([A-Za-z0-9+./#\-\s'’]{{1,64}}?)(?:\s*\([^)]+\))?\s+{}\s+(.+?)\s*$",
            cop
        )).ok()?;
        if let Some(c) = re1.captures(&l) {
            let topic_raw = c.get(1)?.as_str();
            let topic = strip_quotes(&strip_topic_parens(topic_raw)).to_string();
            let desc  = c.get(2)?.as_str().trim().to_string();
            if !topic.is_empty() && !desc.is_empty() { return Some((topic, desc)); }
        }

        // 2) Topic:/-/— Desc
        let re2 = Regex::new(r"(?i)^\s*([A-Za-z0-9+./#\-\s'’]{1,64}?)\s*[:\-—]\s+(.+?)\s*$").ok()?;
        if let Some(c) = re2.captures(&l) {
            let topic = strip_quotes(c.get(1)?.as_str()).to_string();
            let desc  = c.get(2)?.as_str().trim().to_string();
            if !topic.is_empty() && !desc.is_empty() { return Some((topic, desc)); }
        }

        // 3) Desc cop Topic  (reversed)
        let re3 = Regex::new(&format!(
            r"(?i)^\s*(.+?)\s+{}\s+([A-Za-z0-9+./#\-\s'’]{{1,64}})\s*\.?\s*$",
            cop
        )).ok()?;
        if let Some(c) = re3.captures(&l) {
            let desc  = c.get(1)?.as_str().trim().to_string();
            let topic = strip_quotes(c.get(2)?.as_str()).to_string();
            if !topic.is_empty() && !desc.is_empty() { return Some((topic, desc)); }
        }
        None
    }



    fn is_question(s: &str) -> bool {
        let l = s.trim().to_ascii_lowercase();
        l.ends_with('?')
            || l.starts_with("what ")
            || l.starts_with("who ")
            || l.starts_with("where ")
            || l.starts_with("why ")
            || l.starts_with("how ")
    }

    // include common contractions
    fn normalize_question(q: &str) -> String {
        let l = q.trim().to_ascii_lowercase();
        if l.starts_with("what is your name") || l.starts_with("what's your name") {
            "your name is".into()
        } else if l.starts_with("what is the launch code") {
            "the launch code is".into()
        } else if l.starts_with("who is your creator") {
            "your creator is".into()
        } else if l.starts_with("who is your mentor") {
            "your mentor is".into()
        } else if l.starts_with("who are you") || l.contains("who are you") {
            "you are".into()
        } else {
            q.to_string()
        }
    }

    fn normalize_for_hash(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut last_ws = false;
        for ch in s.chars().map(|c| c.to_ascii_lowercase()) {
            if ch.is_whitespace() { if !last_ws { out.push(' '); } last_ws = true; }
            else { last_ws = false; out.push(ch); }
        }
        out.trim().to_string()
    }

    fn sha256_hex(s: &str) -> String {
        let mut h = Sha256::new();
        h.update(s.as_bytes());
        let bytes = h.finalize();
        let mut hex = String::with_capacity(64);
        for b in bytes { use std::fmt::Write; let _ = write!(hex, "{:02x}", b); }
        hex
    }

    fn display_concept(key: &str, value: &str) -> String {
        match key {
            "name"         => format!("Name: {value}"),
            "launch code"  => format!("Launch code: {value}"),
            "creator"      => format!("Creator: {value}"),
            "mentor"       => format!("Mentor: {value}"),
            other          => format!("{other}: {value}"),
        }
    }

    // concept-only extractor: only returns a value for the *asked* concept
    fn extract_answer(qn: &str, text: &str) -> Option<String> {
        let tl = text.trim();
        let ll = tl.to_ascii_lowercase();

        if qn.starts_with("your name is") {
            // 1) "your name is X"
            if let Some(idx) = ll.find("name is") {
                let ans = tl[idx + "name is".len()..].trim()
                    .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace());
                if !ans.is_empty() { return Some(ans.to_string()); }
            }
            // 2) fallback: "I am X" / "I'm X"
            if let Some(caps) = Regex::new(r"(?i)\b(?:i\s+am|i['’]m)\s+([A-Za-z][A-Za-z .'\-]{1,59})\b")
                .ok()
                .and_then(|re| re.captures(tl))
            {
                let ans = caps.get(1).unwrap().as_str().trim()
                    .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace());
                if !ans.is_empty() { return Some(ans.to_string()); }
            }
            return None;
        }

        if qn.starts_with("the launch code is") {
            let re = Regex::new(r"(\d{3}[-\s]?\d{3}[-\s]?\d{3})").ok()?;
            if let Some(c) = re.captures(tl) { return Some(c.get(1).unwrap().as_str().to_string()); }
            return None;
        }
        if qn.starts_with("your creator is") {
            if let Some(idx) = ll.find("creator is") {
                let ans = tl[idx + "creator is".len()..].trim()
                    .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace());
                if !ans.is_empty() { return Some(ans.to_string()); }
            }
            return None;
        }
        if qn.starts_with("your mentor is") {
            if let Some(idx) = ll.find("mentor is") {
                let ans = tl[idx + "mentor is".len()..].trim()
                    .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace());
                if !ans.is_empty() { return Some(ans.to_string()); }
            }
            return None;
        }
        None
    }
    struct SmieTextSearch {
        smie: Arc<Smie>,
        mask: u64,
    }
    impl SmieTextSearch {
        fn new(smie: Arc<Smie>) -> Self {
            Self { smie, mask: tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER }
        }
    }
    impl TextSearch for SmieTextSearch {
        fn search_text(&self, text: &str, top_k: usize)
                       -> Result<Vec<MemoryHit>, Box<dyn std::error::Error + Send + Sync>>
        {
            let hits = self.smie.recall(text, top_k, self.mask)?;
            Ok(hits.into_iter().map(|(id, sim, _t, txt)| MemoryHit {
                id: id.to_string(),
                sim,
                text: txt,
            }).collect())
        }
    }

    // Simple “beliefish” detector v1. Returns Some(canonical_text) or None.
    fn detect_beliefish(text: &str) -> Option<String> {
        let l = text.trim();
        // identity
        let re_you_are = Regex::new(r"(?i)^\s*you\s+(?:are|'re)\s+(.+?)\s*[.!]?\s*$").unwrap();
        if let Some(c) = re_you_are.captures(l) {
            let val = c[1].trim();
            if !val.is_empty() { return Some(format!("you are {}", val)); }
        }
        // purpose
        let re_purpose_is = Regex::new(r"(?i)^\s*your\s+(?:purpose|mission|role)\s+(?:is|to)\s+(.+?)\s*[.!]?\s*$").unwrap();
        if let Some(c) = re_purpose_is.captures(l) {
            let val = c[1].trim();
            if !val.is_empty() { return Some(format!("your purpose is {}", val)); }
        }
        // created to/for
        let re_created_to = Regex::new(r"(?i)^\s*you\s+(?:were\s+)?created\s+(?:to|for)\s+(.+?)\s*[.!]?\s*$").unwrap();
        if let Some(c) = re_created_to.captures(l) {
            let val = c[1].trim();
            if !val.is_empty() { return Some(format!("you were created to {}", val)); }
        }
        None
    }

    fn choose_best_for(qn: &str, hits: &[(u64, f32, u64, String)]) -> Option<(u64, f32, u64, String)> {
        let key = if qn.starts_with("your name is") { Some("name") }
        else if qn.starts_with("the launch code is") { Some("launch code") }
        else { None };

        if let Some(k) = key {
            let mut filtered: Vec<_> = hits.iter()
                .filter(|(_,_,_,text)| text.to_ascii_lowercase().contains(k))
                .cloned().collect();
            if !filtered.is_empty() {
                filtered.sort_by(|a,b| b.0.cmp(&a.0).then_with(|| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)));
                return filtered.into_iter().next();
            }
            let mut all = hits.to_vec();
            all.sort_by(|a,b| b.0.cmp(&a.0).then_with(|| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)));
            return all.into_iter().next();
        }
        let mut all = hits.to_vec();
        all.sort_by(|a,b| b.0.cmp(&a.0).then_with(|| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)));
        all.into_iter().next()
    }
    #[inline]
    fn belief_implies_copula_long(text: &str) -> Option<bool> {
        let l = text.to_ascii_lowercase();
        // “From now on, say ‘I am’”, “use I am”, “no contractions”, etc.
        if l.contains("say \"i am\"") || l.contains("say 'i am'") || l.contains("use i am")
            || l.contains("no contractions") || l.contains("use long copula")
        { return Some(true); }
        if l.contains("say \"i'm\"") || l.contains("say 'i'm'") || l.contains("use i'm")
            || l.contains("use contractions") || l.contains("use short copula")
        { return Some(false); }
        None
    }

    #[inline]
    fn fmt_tags(t: u64) -> String {
        let mut v = Vec::new();
        if t & tags::USER   != 0 { v.push("USER"); }
        if t & tags::SYSTEM != 0 { v.push("SYSTEM"); }
        if t & tags::SHORT  != 0 { v.push("SHORT"); }
        if t & tags::LONG   != 0 { v.push("LONG"); }
        if v.is_empty() { "NONE".into() } else { v.join("|") }
    }

    // STRICT statements-only fallback
    fn concept_of(text: &str) -> Option<&'static str> {
        let l = text.to_ascii_lowercase();
        if l.starts_with("the launch code is ") { Some("launch code") }
        else if l.starts_with("your name is ")  { Some("name") }
        else { None }
    }

    // ────────────────── “to be” verbiage (first step)
    fn extract_copula_name_for_agent(text: &str) -> Option<String> {
        // Matches: "you are X", "you're X" (not roles), also "your name is X"
        // Avoid role words to not collide with creator/mentor.
        let role_words = ["creator", "mentor", "owner", "friend", "assistant"];
        let l = text.to_ascii_lowercase();

        // your name is X
        if let Some(c) = Regex::new(r"(?i)^\s*your\s+name\s+is\s+(.+?)\s*[.!]?\s*$").unwrap().captures(text) {
            let cand = c[1].trim().to_string();
            if !cand.is_empty() { return Some(cand); }
        }

        // you are / you're X
        if let Some(c) = Regex::new(r"(?i)^\s*you\s+(?:are|\'re)\s+(.+?)\s*[.!]?\s*$").unwrap().captures(text) {
            let cand = c[1].trim().to_string();
            if !cand.is_empty() && !role_words.iter().any(|w| l.contains(w)) { return Some(cand); }
        }
        None
    }

    // ─────────────────────────── embedders
    struct ModelEmb<B: Backend> { model: Arc<Mutex<HippocampusModel<B>>> }
    impl<B: Backend> ModelEmb<B> { fn new(model: Arc<Mutex<HippocampusModel<B>>>) -> Self { Self { model } } }
    impl<B: Backend> Embeddings for ModelEmb<B> {
        fn embed(&self, text: &str)
                 -> Result<Embedding, Box<dyn std::error::Error + Send + Sync>>
        {
            // 1) Heavy work OUTSIDE the lock
            let pre = URIELV1::Hippocampus::infer::embed_text(text);

            // 2) Tiny matmuls INSIDE the lock
            let z = {
                let m = self.model.lock().map_err(|_| "model lock poisoned")?;
                m.forward_from_vec(&pre)
            };
            let mut v = z.to_data().as_slice::<f32>().map_err(|e| format!("{e:?}"))?.to_vec();

            // Normalize
            let n = (v.iter().map(|x| x*x).sum::<f32>()).sqrt().max(1e-6);
            for x in &mut v { *x /= n; }
            Ok(v)
        }
    }


    struct SmieHippEmbed<B: Backend> {
        model: Arc<Mutex<HippocampusModel<B>>>,
        dim: usize,
        // learned diagonal weights + last modified time
        oep_w: RwLock<Option<Vec<f32>>>,
        oep_path: String,
        oep_mtime: RwLock<Option<SystemTime>>,
    }
    impl<B: Backend> SmieHippEmbed<B> {
        fn new(model: Arc<Mutex<HippocampusModel<B>>>) -> anyhow::Result<Self> {
            // Heavy hashing outside the lock:
            let vprobe = URIELV1::Hippocampus::infer::embed_text("[dim-probe]");

            // Lock only for the tiny MLP forward:
            let m = model.lock().map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
            let z = m.forward_from_vec(&vprobe);
            let dim = z
                .to_data()
                .as_slice::<f32>()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?
                .len();
            drop(m);


            let path = std::env::var("OEP_WEIGHTS_PATH").unwrap_or_else(|_| "data/oep_w.json".to_string());
            let mut w: Option<Vec<f32>> = None;
            let mut mt: Option<SystemTime> = None;
            if let Ok(md) = fs::metadata(&path) {
                if let Ok(txt) = fs::read_to_string(&path) {
                    if let Ok(wf) = serde_json::from_str::<OepWeightsFile>(&txt) {
                        if wf.dim == dim && wf.w.len() == dim {
                            w = Some(wf.w);
                            mt = md.modified().ok();
                            eprintln!("[oep] loaded {}", path);
                        }
                    }
                }
            } else {
                eprintln!("[oep] no weights loaded (expected at {})", path);
            }

            Ok(Self {
                model,
                dim,
                oep_w: RwLock::new(w),
                oep_path: path,
                oep_mtime: RwLock::new(mt),
            })
        }
        fn maybe_hot_reload(&self) {
            if let Ok(md) = fs::metadata(&self.oep_path) {
                let now_mt = md.modified().ok();
                let last_mt = *self.oep_mtime.read().unwrap();
                if now_mt.is_some() && now_mt != last_mt {
                    if let Ok(txt) = fs::read_to_string(&self.oep_path) {
                        if let Ok(wf) = serde_json::from_str::<OepWeightsFile>(&txt) {
                            if wf.dim == self.dim && wf.w.len() == self.dim {
                                *self.oep_w.write().unwrap() = Some(wf.w);
                                *self.oep_mtime.write().unwrap() = now_mt;
                                eprintln!("[oep] hot-reloaded {}", self.oep_path);
                            }
                        }
                    }
                }
            }
        }
        fn set_oep_weights(&self, w: Vec<f32>) {
            if w.len() == self.dim {
                *self.oep_w.write().unwrap() = Some(w);
                // also bump mtime cache so we won’t auto-overwrite it unless the file changes
                *self.oep_mtime.write().unwrap() = None;
            }
        }
        fn save_oep(&self) {
            if let Some(w) = self.oep_w.read().unwrap().as_ref() {
                let obj = OepWeightsFile {
                    dim: self.dim,
                    w: w.clone(),
                    updated_at: now_sec() as i64,
                };
                if let Ok(txt) = serde_json::to_string_pretty(&obj) {
                    let _ = fs::create_dir_all("data");
                    let _ = fs::write(&self.oep_path, txt);
                    if let Ok(md) = fs::metadata(&self.oep_path) {
                        *self.oep_mtime.write().unwrap() = md.modified().ok();
                    }
                }
            }
        }
    }

    impl<B: Backend> Embedder for SmieHippEmbed<B> {
        fn dim(&self) -> usize { self.dim }
        fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
            self.maybe_hot_reload();

            // 1) Heavy work OUTSIDE the lock
            let pre = URIELV1::Hippocampus::infer::embed_text(text);

            // 2) Tiny matmuls INSIDE the lock
            let z = {
                let m = self.model.lock().map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
                m.forward_from_vec(&pre)
            };
            let mut v = z.to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?.to_vec();

            // base normalize
            let n0 = (v.iter().map(|x| x*x).sum::<f32>()).sqrt().max(1e-6);
            for x in &mut v { *x /= n0; }

            // optional diagonal weights
            if let Some(w) = self.oep_w.read().unwrap().as_ref() {
                for i in 0..self.dim { v[i] *= w[i]; }
                let n1 = (v.iter().map(|x| x*x).sum::<f32>()).sqrt().max(1e-6);
                for x in &mut v { *x /= n1; }
            }
            Ok(v)
        }
    }



    // kill SQLite ANN scans
    struct NullVS;
    impl VectorStore for NullVS {
        fn search(&self, _q: &Embedding, _k: usize) -> std::result::Result<Vec<MemoryHit>, Box<dyn Error + Send + Sync>> {
            Ok(vec![])
        }
    }
    fn render_concept_answer(key: &str, val: &str) -> String {
        match key {
            "you are" | "name" => format!("I’m {}.", val),
            _ => realize_concept(key, val),
        }
    }
    // ---------- Canonical topic + domain learning (no hardcoding) ----------

    // normalize surface tokens for topics
    fn topic_surface_norm(s: &str) -> String {
        ascii_norm(s).to_ascii_lowercase()
            .replace(['–','—'], "-")
            .trim()
            .to_string()
    }

    // light plural→singular heuristic (generic, not domain-specific)
    fn maybe_singular(s: &str) -> String {
        let mut t = topic_surface_norm(s);
        if t.ends_with("ies") && t.len() > 3 {
            t.truncate(t.len() - 3);
            t.push('y');
        } else if t.ends_with('s') && !t.ends_with("ss") && !t.ends_with("us") {
            t.truncate(t.len() - 1);
        }
        t
    }

    // canonicalize a topic surface form (generic normalization)
    // NOTE: not a hardcoded table; learns aliases below.
    fn canonicalize_surface(s: &str) -> String {
        let mut t = topic_surface_norm(s);
        // strip simple determiners
        for pre in ["the ", "a ", "an "] {
            if t.starts_with(pre) {
                t = t[pre.len()..].to_string();
                break;
            }
        }
        // drop one parenthetical descriptor, keep lead headword
        t = strip_topic_parens(&t);
        // singularize
        maybe_singular(&t)
    }

    // open-ended domain normalization (no fixed vocab)
    fn norm_domain(s: &str) -> String {
        ascii_norm(s).to_ascii_lowercase().replace([' ', '\t'], "")
    }

    // Persist/read a learned alias mapping: alias:<surface_norm> -> <canonical>
    fn read_alias(surface_norm: &str) -> Option<String> {
        let k = format!("alias:{}", surface_norm);
        read_concept_fact(&k).ok().flatten()
    }
    fn write_alias(surface_norm: &str, canonical: &str, ts: u64) {
        let k = format!("alias:{}", surface_norm);
        let _ = upsert_concept_fact(&k, canonical, ts);
    }
    fn read_alias_dom(domain: Option<&str>, surface_norm: &str) -> Option<String> {
        if let Some(d) = domain {
            let k = format!("alias:{}:{}", norm_domain(d), surface_norm);
            if let Ok(Some(v)) = read_concept_fact(&k).map_err(to_any) { return Some(v); }
        }
        read_alias(surface_norm)
    }

    fn write_alias_dom(domain: Option<&str>, surface_norm: &str, canonical: &str, ts: u64) {
        if let Some(d) = domain {
            let k = format!("alias:{}:{}", norm_domain(d), surface_norm);
            let _ = upsert_concept_fact(&k, canonical, ts);
        } else {
            write_alias(surface_norm, canonical, ts);
        }
    }

    // Build the key used for defs: def:<topic>  or  def:<domain>:<topic>
    // (domain is open-ended; any noun phrase users provide)
    fn def_key(domain: Option<&str>, topic: &str) -> String {
        match domain {
            Some(d) if !d.is_empty() => format!("def:{}:{}", norm_domain(d), canonicalize_surface(topic)),
            _ => format!("def:{}", canonicalize_surface(topic)),
        }
    }

    // Parse domain-bearing definitions, open-ended domain:
    //  - "In <domain>, <Topic> is <Desc>"
    //  - "<Topic> is <Desc> in <domain>"
    fn parse_domained_def(line: &str) -> Option<(String, String, String)> {
        let l = ascii_norm(line);
        let cop = r"(?:is|are|refers\s+to|means|=)";

        // In <domain>, Topic cop Desc
        if let Ok(re) = Regex::new(&format!(
            r"(?i)^\s*in\s+([a-z][a-z0-9 _\-]{{2,32}})\s*,\s*([A-Za-z0-9+./#\-\s'’]{{1,64}}?)\s+{}\s+(.+?)\s*$", cop
        )) {
            if let Some(c) = re.captures(&l) {
                let dom   = norm_domain(c.get(1).unwrap().as_str());
                let topic = strip_quotes(c.get(2).unwrap().as_str()).to_string();
                let desc  = c.get(3).unwrap().as_str().trim().to_string();
                if !dom.is_empty() && !topic.is_empty() && !desc.is_empty() {
                    return Some((dom, topic, desc));
                }
            }
        }

        // Topic cop Desc in <domain>
        if let Ok(re) = Regex::new(&format!(
            r"(?i)^\s*([A-Za-z0-9+./#\-\s'’]{{1,64}}?)\s+{}\s+(.+?)\s+in\s+([a-z][a-z0-9 _\-]{{2,32}})\s*$", cop
        )) {
            if let Some(c) = re.captures(&l) {
                let topic = strip_quotes(c.get(1).unwrap().as_str()).to_string();
                let desc  = c.get(2).unwrap().as_str().trim().to_string();
                let dom   = norm_domain(c.get(3).unwrap().as_str());
                if !dom.is_empty() && !topic.is_empty() && !desc.is_empty() {
                    return Some((dom, topic, desc));
                }
            }
        }

        None
    }


    // Optional: explicit domain hint in question, e.g. "... in <domain> ?"
    fn infer_domain_from_question(q: &str) -> Option<String> {
        let l = ascii_norm(q);

        // Try several ways users hint a topic/domain
        let pats = [
            r"(?i)\bin\s+([a-z][a-z0-9 _\-]{2,32})\b",                      // in art
            r"(?i)\b(?:about|on|regarding)\s+([a-z][a-z0-9 _\-]{2,32})\b",  // about art / on art / regarding art
            r"(?i)^\s*(?:let'?s|we)\s+(?:talk|chat|speak)\s+about\s+([a-z][a-z0-9 _\-]{2,32})\b", // let's talk about art
            r"(?i)\btopic\s*:\s*([a-z][a-z0-9 _\-]{2,32})\b",               // topic: art
        ];

        for p in pats {
            if let Ok(re) = Regex::new(p) {
                if let Some(c) = re.captures(&l) {
                    return Some(norm_domain(c.get(1).unwrap().as_str()));
                }
            }
        }
        None
    }


    // Infer domain from stored evidence via SMIE (learned; no keyword lists)
    fn infer_domain_for_topic_from_smie(smie: &Smie, topic: &str, mask: u64) -> Option<String> {
        let qs = [
            format!("{topic}:"),
            format!("{topic} is"),
            topic.to_string(),
        ];
        for q in qs.iter() {
            if let Ok(hits) = smie.recall(q, 6, mask) {
                for (_id, _s, _t, txt) in hits {
                    if let Some((dom, _t2, _desc)) = parse_domained_def(&txt) {
                        return Some(dom);
                    }
                }
            }
        }
        None
    }

    // ─────────────────────────── MicroThalamus + goal helpers
    fn oep_save_every() -> u64 {
        std::env::var("OEP_SAVE_EVERY")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(50) // default: save every 50 micro-updates
    }

    fn drives_as_weights(drives: &HashMap<String, SyntheticDrive>) -> HashMap<String, f32> {
        drives.iter().map(|(k, d)| (k.clone(), d.weight)).collect()
    }
    fn push_goal(goals: &mut Vec<Goal>, mut g: Goal, drives: &HashMap<String, SyntheticDrive>, ctx: &Ctx) {
        g.score(&drives_as_weights(drives), ctx); goals.push(g);
    }
    fn rescore_goals(goals: &mut Vec<Goal>, drives: &HashMap<String, SyntheticDrive>, ctx: &Ctx) {
        let w = drives_as_weights(drives);
        for g in goals.iter_mut() { g.score(&w, ctx); }
        goals.sort_by(|a,b| b.last_score.partial_cmp(&a.last_score).unwrap_or(Ordering::Equal));
        goals.retain(|g| !matches!(g.status, URIELV1::Microthalamus::goal::GoalStatus::Succeeded | URIELV1::Microthalamus::goal::GoalStatus::Failed | URIELV1::Microthalamus::goal::GoalStatus::Aborted));
    }

    // P1.2 helpers
    fn mine_negatives_from_smie(smie: &Smie, query: &str, top_k: usize) -> Vec<(String, f32)> {
        let mut out = Vec::new();
        if let Ok(hits) = smie.recall(query, top_k, tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER) {
            for (_id, sim, _tags, txt) in hits { out.push((txt, sim)); }
        }
        out
    }
    fn attach_negs(conn: &mut rusqlite::Connection, qid: i64, negs: &[(String, f32)]) {
        if let Ok(tx) = conn.transaction() {
            for (neg, sim) in negs {
                let _ = tx.execute(
                    "INSERT INTO learn_negs(qid,negative,sim) VALUES(?1,?2,?3)",
                    rusqlite::params![qid, neg, *sim],
                );
            }
            let _ = tx.commit();
        }
    }

    // utils
    fn now_sec() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
    }

    // Persist a single line into SQLite + SMIE, with concept upsert and durable dedupe.
    fn auto_persist_line<B: Backend>(
        conn: &rusqlite::Connection,
        model: &HippocampusModel<B>,
        smie: &Smie,
        ctx: &str,
        text_raw: &str,
        concepts: &Registry,
        _learner: &mut ConceptLearner,
        seen_hashes: &mut HashSet<String>,
    ) -> anyhow::Result<bool> {
        let mut clean = sanitize_input(text_raw);
        if clean.is_empty() { return Ok(false); }

        let mut force_store = false;
        let lower = clean.to_ascii_lowercase();
        if lower.starts_with("store this:") {
            if let Some(rest) = clean.splitn(2, ':').nth(1) {
                clean = rest.trim().to_string();
                force_store = true;
            }
        }

        // Treat anything that looks like a question or “request” as a prompt,
        // but not when we’ve forced store via “store this:”.
        let mut is_prompt = (is_question(&clean) || is_prompt_like(&clean)) && !force_store;
        let allow_store = auto_store_allowed();
        eprintln!("[persist] allow_store={} is_prompt={} text='{}'", allow_store, is_prompt, clean);

        let mut did_store = false;

        // ---- durable + in-process dedupe ----
        let n   = normalize_for_hash(&clean);
        let key = sha256_hex(&n);

        // NEW: skip de-dupe for explicit stores or concept-bearing statements,
        // so we can update the fact even if the same line was seen before.
        let looks_like_concept = concept_of(&clean).is_some()
            || parse_domained_def(&clean).is_some()
            || extract_definition(&clean).is_some();

        let skip_dedupe = force_store || looks_like_concept;

        if !skip_dedupe && !seen_hashes.insert(key.clone()) {
            let _ = smie.ingest(&clean, tags::USER | tags::SHORT)?;
            return Ok(false);
        }
        let ts = now_sec();
        if !skip_dedupe && has_hash(ctx, &key).map_err(to_any)? {
            let _ = smie.ingest(&clean, tags::USER | tags::SHORT)?;
            return Ok(false);
        }
        if !skip_dedupe {
            // only record the de-dupe hash on the “ingest-governed” path
            record_hash(ctx, &key, ts).map_err(to_any)?;
        }


        // ---- style preference (“say ‘I am’ / contractions”) — only when storing & not a prompt ----
        if allow_store && !is_prompt {
            if let Some(long) = belief_implies_copula_long(&clean) {
                URIELV1::concepts_and_contexts::speech_concepts::set_copula_style_long(long);
                let _ = upsert_concept_fact("speech.copula", if long { "long" } else { "short" }, ts);

                let turn_id = format!("style-{}-{}", ts, rand::random::<u32>());
                let step = MetaStep {
                    turn_id: turn_id.clone(),
                    goal: "style_change".into(),
                    hypothesis: format!("copula={}", if long { "long" } else { "short" }),
                    plan: vec!["apply".into()],
                    needed: vec![],
                    evidence: vec![],
                    confidence: 0.0,
                    depth: 0,
                    outcome: "accepted".into(),
                    bandit: None,
                    sources: Some(SourcesInfo { answered_from: "user".into(), db_hit: false }),
                };
                record_meta_trace(conn, &turn_id, "style_change", &step, ts as i64);
            }
        }

        if allow_store && !is_prompt {
            // 2a) Domain-aware defs: "In <domain>, <Topic> is <Desc>" (open-ended domain)
            if let Some((dom, topic_d, desc_d)) = parse_domained_def(&clean) {
                let surf = canonicalize_surface(&topic_d);
                let canon = read_alias(&surf).unwrap_or_else(|| surf.clone());
                if canon != surf {
                    write_alias(&surf, &canon, ts);
                }
                let key_cf = def_key(Some(&dom), &canon);
                safe_upsert_concept_fact(conn, "runtime", &key_cf, &desc_d, ts, Some("def.domained"), None).map_err(to_any)?;
                did_store = true;

                // Ingest a compact form that preserves domain as text (learned signal for SMIE)
                let short = format!("{} ({}): {}", canon, dom, desc_d);
                let _ = smie.ingest(&short, tags::SYSTEM | tags::SHORT)?;
            }

            // 2b) Undomained defs: "Topic is Desc" OR "Topic: Desc" etc (generic)
            if let Some((topic, desc)) = extract_definition(&clean) {
                let surf = canonicalize_surface(&topic);
                let canon = read_alias(&surf).unwrap_or_else(|| surf.clone());
                if canon != surf {
                    write_alias(&surf, &canon, ts);
                }
                let key_cf = def_key(None, &canon);
                safe_upsert_concept_fact(conn, "runtime", &key_cf, &desc, ts, Some("def.generic"), None).map_err(to_any)?;
                did_store = true;

                let short = format!("{}: {}", canon, desc);
                let _ = smie.ingest(&short, tags::SYSTEM | tags::SHORT)?;
            }
        }


        // ---- belief capture (identity/purpose) — only when storing & not a prompt ----
        if allow_store && !is_prompt {
            if let Some(mut belief) = detect_beliefish(&clean) {
                belief = belief.trim().trim_end_matches(|c: char| c.is_ascii_punctuation()).to_string();
                if !belief.is_empty() {
                    if insert_belief(conn, &belief, "user", 1.0).is_ok() {
                        did_store = true;

                        if let Some(long) = belief_implies_copula_long(&belief) {
                            URIELV1::concepts_and_contexts::speech_concepts::set_copula_style_long(long);
                        }
                        let turn_id = format!("belief-{}-{}", ts, rand::random::<u32>());
                        let step = MetaStep {
                            turn_id: turn_id.clone(),
                            goal: "belief_insert".into(),
                            hypothesis: format!("B: {}", belief),
                            plan: vec!["capture".into()],
                            needed: vec![],
                            evidence: vec![],
                            confidence: 0.0,
                            depth: 0,
                            outcome: "accepted".into(),
                            bandit: None,
                            sources: Some(SourcesInfo { answered_from: "user".into(), db_hit: false }),
                        };
                        record_meta_trace(conn, &turn_id, "belief_insert", &step, ts as i64);
                    }
                }
            }
        }

        // ---- concept extraction / copula name — only when storing & not a prompt ----
        if allow_store && !is_prompt {
            let hits = concepts.extract_all_from_statement(&clean);
            if !hits.is_empty() {
                for h in &hits {
                    safe_upsert_concept_fact(conn, "runtime", h.key, &h.value, ts, Some("concept.extract"), None).map_err(to_any)?;
                    if cognition_writes_enabled() {
                        let clean_v = canon_value_for(h.key, &h.value);
                        let _ = kops::upsert_fact(conn, ktypes::FactWrite {
                            subj: None,
                            predicate: h.key.to_string(),
                            obj_text: clean_v.clone(),
                            obj_entity: None,
                            confidence: 0.66,               // seed confidence for manual store
                            ts: ts as i64,
                            method: "pattern".to_string(),  // provenance for “store this”
                            evidence: vec!["store:this".to_string()],
                        });
                        let _ = kops::prefer_latest(conn, None, h.key);
                    }

                    did_store = true;

                    // Tombstone older conflicting SMIE lines
                    let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
                    if let Ok(old) = smie.recall(h.key, 64, mask) {
                        for (id, _s, _t, txt) in old {
                            if txt.to_ascii_lowercase().contains(h.key) {
                                let _ = smie.tombstone(id);
                            }
                        }
                    }
                }
            } else if let Some(name) = extract_copula_name_for_agent(&clean) {
                safe_upsert_concept_fact(conn, "runtime", "name",   &name, ts, Some("concept.name"), None).map_err(to_any)?;
                safe_upsert_concept_fact(conn, "runtime", "you are",&name, ts, Some("concept.name"), None).map_err(to_any)?;

                did_store = true;

                for k in ["name", "you are"] {
                    let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
                    if let Ok(old) = smie.recall(k, 64, mask) {
                        for (id, _s, _t, txt) in old {
                            if txt.to_ascii_lowercase().contains(k) {
                                let _ = smie.tombstone(id);
                            }
                        }
                    }
                }
            } else if let Some(caps) = Regex::new(
                r"(?i)^\s*i\s+am\s+your\s+([a-z\s]+?),\s*([A-Za-z][A-Za-z .'\-]+)\s*[.!]?\s*$"
            )?.captures(&clean) {
                let roles = caps[1].to_ascii_lowercase();
                let name  = caps[2].trim().to_string();

                if roles.contains("creator") {
                    safe_upsert_concept_fact(conn, "runtime", "creator", &name, ts, Some("concept.role"), None).map_err(to_any)?;
                    did_store = true;

                    let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
                    if let Ok(old) = smie.recall("creator", 64, mask) {
                        for (id, _s, _t, txt) in old {
                            if txt.to_ascii_lowercase().contains("creator") {
                                let _ = smie.tombstone(id);
                            }
                        }
                    }
                }
                if roles.contains("mentor") {
                    safe_upsert_concept_fact(conn, "runtime", "mentor", &name, ts, Some("concept.role"), None).map_err(to_any)?;
                    did_store = true;

                    let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
                    if let Ok(old) = smie.recall("mentor", 64, mask) {
                        for (id, _s, _t, txt) in old {
                            if txt.to_ascii_lowercase().contains("mentor") {
                                let _ = smie.tombstone(id);
                            }
                        }
                    }
                }
            }
        }

        // ---- legacy strict fallback (“your name is … / the launch code is …”) — only when storing & not a prompt ----
        if allow_store && !is_prompt {
            if let Some(con) = concept_of(&clean) {

                let qn = if con == "name" { "your name is" } else { "the launch code is" };
                if let Some(value) = extract_answer(qn, &clean) {
                    safe_upsert_concept_fact(conn, "runtime", con, &value, ts, Some("concept.strict"), None).map_err(to_any)?;
                    did_store = true;
                    if con == "launch code" {
                        // twin predicate to catch extractor/normalizer mismatches
                        let twin = "launch_code";
                        let _ = safe_upsert_concept_fact(conn, "runtime", twin, &value, ts, Some("concept.twin"), None);
                        if cognition_writes_enabled() {
                            let clean_v = canon_value_for(twin, &value);
                            let _ = kops::upsert_fact(conn, ktypes::FactWrite {
                                subj: None,
                                predicate: twin.to_string(),
                                obj_text: clean_v.clone(),
                                obj_entity: None,
                                confidence: 0.66,
                                ts: ts as i64,
                                method: "pattern".to_string(),
                                evidence: vec!["store:this".to_string()],
                            });
                            let _ = kops::prefer_latest(conn, None, twin);
                        }
                    }
                    let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
                    if let Ok(old) = smie.recall(con, 64, mask) {
                        for (id, _s, _t, txt) in old {
                            if concept_of(&txt) == Some(con) {
                                let _ = smie.tombstone(id);


                            }
                        }
                    }
                    if cognition_writes_enabled() {
                        let clean_v = canon_value_for(con, &value);
                        let _ = kops::upsert_fact(conn, ktypes::FactWrite {
                            subj: None,
                            predicate: con.to_string(),
                            obj_text: clean_v.clone(),
                            obj_entity: None,
                            confidence: 0.66,
                            ts: ts as i64,
                            method: "pattern".to_string(),
                            evidence: vec!["store:this".to_string()],
                        });
                        let _ = kops::prefer_latest(conn, None, con);
                    }

                }
            }

        }

        // ---- always ingest raw text for retrieval, and mirror to rows (safe in both modes) ----
        let _ = smie.ingest(&clean, tags::USER | tags::SHORT)?;
        pipeline::ingest(model, ctx, &clean, 7 * 24 * 3600, "runtime")
            .map_err(|e| anyhow::anyhow!("ingest failed: {e}"))?;

        Ok(did_store)
    }




    // ─────────── P1.2 / P1.3: stubs (safe) — wire to trainer/checkpointer when ready
    fn maybe_stage_anchor_for_training(
        conn: &rusqlite::Connection,
        question: &str,
        positive: &str,
        reward: f32,
    ) {
        let _ = conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS learn_queue(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts INTEGER NOT NULL,
              question TEXT NOT NULL,
              positive TEXT NOT NULL,
              hardness REAL NOT NULL,
              used INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS learn_negs(
              qid INTEGER NOT NULL,
              negative TEXT NOT NULL,
              sim REAL NOT NULL,
              FOREIGN KEY(qid) REFERENCES learn_queue(id) ON DELETE CASCADE
            );
            "#,
        );
        if reward >= 0.9 {
            let mut rng = rand::thread_rng();
            if rng.gen::<f32>() <= 0.05 {
                let _ = conn.execute(
                    "INSERT INTO learn_queue(ts,question,positive,hardness,used)
                     VALUES(strftime('%s','now'),?1,?2,?3,0)",
                    rusqlite::params![question, positive, 0.05_f32],
                );
            }
        }
    }

    fn enqueue_hard_example(
        conn: &rusqlite::Connection,
        question: &str,
        positive: &str,
        hardness: f32,
    ) -> Option<i64> {
        let ok = conn.execute(
            "INSERT INTO learn_queue(ts,question,positive,hardness,used)
             VALUES(strftime('%s','now'),?1,?2,?3,0)",
            rusqlite::params![question, positive, hardness],
        ).ok()?;
        if ok == 1 { Some(conn.last_insert_rowid()) } else { None }
    }

    fn maybe_checkpoint_on_high_conf(_reward: f32) {
        // throttleable save hook for encoder when reward≥0.9
    }

    // ---------- Confidence helpers (Phase 1) ----------

    fn clamp01(x: f32) -> f32 { x.max(0.0).min(1.0) }

    fn to_base_from_source(src: &str) -> f32 {
        match src {
            "db"         => 0.95,
            "normalized" => 0.85,
            "keyword"    => 0.70,
            "original"   => 0.60,
            "rows"       => 0.55,
            "pfc"        => 0.50,
            _            => 0.20,
        }
    }

    // map a cosine-like score ∈[0,1] to [0,1] with hinge at 0.50→0, 0.80→1
    fn retrieval_from_cos(max_cos: Option<f32>) -> f32 {
        match max_cos {
            Some(s) => clamp01((s - 0.50) / 0.30),
            None    => 0.0,
        }
    }

    // very light pattern checks
    fn is_valid_for_key(key: &str, val: &str) -> bool {
        let v = val.trim();
        match key {
            "name" | "creator" | "mentor" => {
                let re = Regex::new(r"^[A-Za-z0-9 .'\-]{2,60}$").unwrap();
                re.is_match(v)
            }
            "launch code" => {
                let re = Regex::new(r"^\d{3}[-\s]?\d{3}[-\s]?\d{3}$").unwrap();
                re.is_match(v)
            }
            "you are" => {
                let re = Regex::new(r"^[A-Za-z0-9 .'\-]{2,60}$").unwrap();
                re.is_match(v)
            }
            "purpose" => {
                // accept short human sentences
                let re = Regex::new(r"^[A-Za-z0-9 ,.'\-()/:;?!]{3,200}$").unwrap();
                re.is_match(v)
            }

            _ => true
        }
    }

    // compare to canonical DB (case/space/punct loose equality)
    fn loose_eq(a: &str, b: &str) -> bool {
        fn norm(s: &str) -> String {
            s.chars()
                .filter(|c| !c.is_ascii_punctuation() && !c.is_whitespace())
                .flat_map(|c| c.to_lowercase())
                .collect()
        }
        norm(a) == norm(b)
    }

    // parse rows:default:<ts> from evidence for recency
    fn rows_recency_from_evidence(evidence: &[String]) -> f32 {
        let now = now_sec() as i64;
        for e in evidence {
            if let Some(ts_s) = e.strip_prefix("rows:default:") {
                if let Ok(ts) = ts_s.parse::<i64>() {
                    let age_days = ((now - ts).max(0) as f32) / 86_400.0;
                    return (-age_days / 7.0).exp(); // 1.0 at fresh, ~0.24 at 1 week
                }
            }
        }
        0.0
    }

    // combine features into a single confidence
    fn combine_confidence(
        is_concept: bool,
        base: f32,
        retrieval: f32,
        extract_ok: bool,
        consistency: Option<f32>, // Some(1.0 match, 0.5 unknown, 0.0 mismatch)
        recency: f32,
        arm_prior: Option<f32>,   // if you have one later; else None
    ) -> f32 {
        let extract_ok_f = if extract_ok { 1.0 } else { 0.0 };
        let cons = consistency.unwrap_or(0.5);
        let arm  = arm_prior.unwrap_or(0.0);

        let score = if is_concept {
            0.40*base + 0.25*retrieval + 0.15*extract_ok_f + 0.15*cons + 0.05*recency + 0.05*arm
        } else {
            0.35*base + 0.35*retrieval + 0.10*extract_ok_f + 0.10*cons + 0.05*recency + 0.05*arm
        };
        clamp01(score)
    }

    fn canon_value_for(key: &str, raw: &str) -> String {
        let s = raw.trim();
        if key == "name" || key == "you are" {
            // Prefer the bare token if the string contains an answer phrase
            if let Some(c) = Regex::new(r"(?i)\b(?:i\s+am|i['’]m|you\s+are|you['’]re|your\s+name\s+is)\s+([A-Za-z][A-Za-z .'\-]{1,59})")
                .unwrap()
                .captures(s)
            {
                return c.get(1).unwrap().as_str().trim()
                    .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
                    .to_string();
            }
        }
        // For defs accidentally stored as "Topic: Desc", keep the Desc
        if key.starts_with("def:") {
            if let Some(c) = Regex::new(r"^[^:]{1,64}:\s*(.+)$").unwrap().captures(s) {
                return c.get(1).unwrap().as_str().trim().to_string();
            }
        }
        s.to_string()
    }

    // ─────────────────────────── runtime
    fn main() -> Result<()> {
        assert_primordial_invariants();
        eprintln!(
            "[paths] cwd={} SMIE_DB_PATH={} OEP_WEIGHTS_PATH={}",
            std::env::current_dir()?.display(),
            std::env::var("SMIE_DB_PATH").unwrap_or_else(|_| "data/semantic_memory.db".into()),
            std::env::var("OEP_WEIGHTS_PATH").unwrap_or_else(|_| "data/oep_w.json".into()),
        );

        let (route_T, route_thr) = load_routercopy_calib("data/runs/router_v4/calib.json")
            .unwrap_or((1.0, Thresholds{promote:0.60, refuse:0.40}));

        // Bandits
        let concept_arms: &[Arm] = &[
            Arm { name: "db" }, Arm { name: "normalized" }, Arm { name: "keyword" }, Arm { name: "original" },
        ];
        let general_arms: &[Arm] = &[Arm { name: "normalized" }, Arm { name: "original" }];
        let concept_bandit = Bandit { intent: "concept_q", arms: concept_arms, epsilon: 0.15 };
        let general_bandit = Bandit { intent: "general_q", arms: general_arms, epsilon: 0.15 };
        // -------- DB open (respects SMIE_DB_PATH) + schema ensure --------
        let db_path = std::env::var("SMIE_DB_PATH")
            .unwrap_or_else(|_| "data/semantic_memory.db".to_string());
        eprintln!("[db] opening {}", db_path);
        let mut conn = rusqlite::Connection::open(&db_path).expect("open db");
        ensure_runtime_schema(&conn)?;   // <-- create tables/views if missing
        // -----------------------------------------------------------------
        // ---- Callosum bus + metacog subscriber ----
        let bus = Arc::new(CorpusCallosum::new());
        // ---- Mesh config (file first, then env fallback) ----
        let mesh_cfg = load_mesh_config();

        let riftnet_dll  = mesh_cfg.as_ref().and_then(|c| c.riftnet.as_ref())
            .and_then(|r| r.dll.clone())
            .or_else(|| std::env::var("RIFTNET_DLL").ok());

        let riftnet_host = mesh_cfg.as_ref().and_then(|c| c.riftnet.as_ref())
            .and_then(|r| r.host.clone())
            .unwrap_or_else(|| std::env::var("RIFTNET_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()));

        let riftnet_port: u16 = mesh_cfg.as_ref().and_then(|c| c.riftnet.as_ref())
            .and_then(|r| r.port)
            .or_else(|| std::env::var("RIFTNET_PORT").ok().and_then(|p| p.parse().ok()))
            .unwrap_or(42080);

        let mesh_forward         = mesh_cfg.as_ref().and_then(|c| c.mesh.as_ref()).and_then(|m| m.forward)
            .unwrap_or(env_bool("MESH_FORWARD", false));
        let mesh_broadcast_ep    = mesh_cfg.as_ref().and_then(|c| c.mesh.as_ref()).and_then(|m| m.broadcast_episodes)
            .unwrap_or(env_bool("MESH_BROADCAST_EPISODES", false));
        let mesh_broadcast_facts = mesh_cfg.as_ref().and_then(|c| c.mesh.as_ref()).and_then(|m| m.broadcast_facts)
            .unwrap_or(env_bool("MESH_BROADCAST_FACTS", false));
        let mesh_accept_ep       = mesh_cfg.as_ref().and_then(|c| c.mesh.as_ref()).and_then(|m| m.accept_episodes)
            .unwrap_or(env_bool("MESH_ACCEPT_EPISODES", true));
        let mesh_accept_facts    = mesh_cfg.as_ref().and_then(|c| c.mesh.as_ref()).and_then(|m| m.accept_facts)
            .unwrap_or(env_bool("MESH_ACCEPT_FACTS", true));
        let mesh_token           = mesh_cfg.as_ref().and_then(|c| c.mesh.as_ref()).and_then(|m| m.token.clone())
            .or_else(|| std::env::var("MESH_TOKEN").ok());

        // ---- RiftNet client (optional) ----
        let riftnet = riftnet_dll.and_then(|dll| {
            match RiftClientDyn::load(&dll) {
                Ok(mut c) => {
                    match c.start(&riftnet_host, riftnet_port) {
                        Ok(rx) => {
                            let bus_c = bus.clone();
                            let accept_ep   = mesh_accept_ep;
                            let accept_fact = mesh_accept_facts;
                            let token       = mesh_token.clone();

                            std::thread::spawn(move || {
                                for ev in rx.iter() {
                                    if ev.kind == URIELV1::riftnet::RiftEventType::PacketReceived {
                                        if let Some(bytes) = ev.bytes {
                                            if let Ok(text) = String::from_utf8(bytes) {
                                                // Optional token check
                                                let mut allow = true;
                                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                                                    if let Some(t) = token.as_deref() {
                                                        if v.get("token").and_then(|x| x.as_str()) != Some(t) {
                                                            allow = false;
                                                        }
                                                    }
                                                    if allow {
                                                        match v.get("type").and_then(|x| x.as_str()) {
                                                            Some("episode") if accept_ep => {
                                                                let role = v.get("role").and_then(|x| x.as_str()).unwrap_or("agent");
                                                                let txt  = v.get("text").and_then(|x| x.as_str()).unwrap_or("");
                                                                bus_c.publish_dynamic("riftnet", None, EpisodeEvent {
                                                                    role: role.to_string(), text: txt.to_string(), turn_id: None,
                                                                });
                                                            }
                                                            Some("fact") if accept_fact => {
                                                                let pred = v.get("predicate").and_then(|x| x.as_str()).unwrap_or("unknown");
                                                                let val  = v.get("value").and_then(|x| x.as_str()).unwrap_or("");
                                                                bus_c.publish_dynamic("riftnet", None, FactUpsertEvent {
                                                                    predicate: pred.to_string(), value: val.to_string(), ts: now_sec() as i64,
                                                                });
                                                            }
                                                            _ => { /* ignore */ }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            });
                            Some(c)
                        }
                        Err(e) => { eprintln!("[riftnet] start failed: {e}"); None }
                    }
                }
                Err(e) => { eprintln!("[riftnet] load failed: {e}"); None }
            }
        });

        // subscriber uses its own connection (do NOT share the same handle across threads)
        let mut conn_for_mc = rusqlite::Connection::open(&db_path).expect("open db (metacog)");
        ensure_runtime_schema(&conn_for_mc)?;   // safe: IF NOT EXISTS
        conn_for_mc.busy_timeout(std::time::Duration::from_millis(500)).ok();
        let _cc_handle = spawn_metacog_subscriber(bus.clone(), conn_for_mc);

        conn.busy_timeout(std::time::Duration::from_secs(2)).ok();
        kschema::apply_cognition_migration(&conn)?;
        kops::ensure_global_entity(&conn)?;
        // Teach the compare operator layout (facets are data, not code)
        let _ = tms_upsert_support(&conn, "op:compare:name", "compare", "seed", 1.0, &[]);
        let _ = tms_upsert_support(&conn, "op:compare:signature", "topic,topic->comparison", "seed", 1.0, &[]);

        // Edit this JSON anytime without code changes (add/remove facets or per-domain later)
        let compare_facets = r#"["definition","category","pros","cons","use_cases","when_use","when_avoid","performance"]"#;
        let _ = tms_upsert_support(&conn, "op:compare:facets", compare_facets, "seed", 1.0, &[]);

        let meta2 = TraceSink::new("data/meta/meta_events.jsonl");
        let mut cof = CofWriter::new("data/logs")?; // creates data/logs/trace-*.jsonl
        // ── Dataset emitter: turns meta events → datasets/router/*.jsonl (and encoder if present)
        std::fs::create_dir_all("datasets").ok(); // base folder

        // A stop channel so we can shut the thread down cleanly
        let (emit_stop_tx, emit_stop_rx) = xchan::bounded::<()>(1);

        // Spawn the background tick loop
        let emit_handle = std::thread::spawn(move || {
            // Construct inside the thread so we don’t move anything heavy
            let emitter = DatasetEmitter::new("data/meta/meta_events.jsonl", "datasets");

            loop {
                if emit_stop_rx.try_recv().is_ok() {
                    break;
                }
                if let Err(e) = emitter.tick() {
                    eprintln!("[ingest] tick error: {e}");
                    // brief backoff so a parse error doesn't spin hot
                    std::thread::sleep(std::time::Duration::from_millis(250));
                    continue;
                }
                // throttle; adjust if you want faster shard writes
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            eprintln!("[ingest] emitter thread stopped.");
        });

        // Router bias preload
        let router_bias_path = "data/router_bias.json".to_string();
        let router_bias = Arc::new(RwLock::new(load_router_bias(&router_bias_path)));
        let mut router_bias_updates: u64 = 0;
        let mut oep_updates: u64 = 0; // NEW

        let _ = URIELV1::lexicon::init_schema(&conn);
        let _ = URIELV1::lexicon::load_jsonl(&conn, "data/lex/seed.jsonl");

        // in main(), after opening `conn`
        fn web_ingest_enabled() -> bool {
            std::env::var("URIEL_WEB_INGEST")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on"))
                .unwrap_or(false) // DEFAULT OFF
        }

        let cfg = URIELV1::tools::web_ingest::IngestCfg {
            whitelist_domains: &["example.com", "wikipedia.org", "github.com", "sciencedirect.com", "jstor.org"],
            max_bytes_per_page: 300_000,
            max_chunks: 80,
        };

        if web_ingest_enabled() {
            let _ = URIELV1::tools::web_ingest::ingest_urls(&conn, &[
                "https://en.wikipedia.org/wiki/URIEL"
            ], &cfg);
        }

        if let Ok(Some(style)) = read_concept_fact("speech.copula").map_err(to_any) {
            URIELV1::concepts_and_contexts::speech_concepts::set_copula_style_long(
                style.eq_ignore_ascii_case("long")
            );
        }

        // concepts registry + online learner (learner reserved; not used for ranking yet)
        let concepts = Registry::new()
            .register(Box::new(NameConcept))
            .register(Box::new(LaunchCodeConcept))
            .register(Box::new(CreatorConcept))
            .register(Box::new(MentorConcept))
            .register(Box::new(PurposeConcept))
            .register(Box::new(IdentityConcept));

        let mut learner = ConceptLearner::new("data/learners");

        type B = NdArray<f32>;
        let device = <B as Backend>::Device::default();

        // PFC model
        let model_runtime =
            HippocampusModel::<B>::load_or_init(&device, "R:/uriel/models/checkpoints/uriel_encoder_step008500.bin");

        // shared encoder (PFC + SMIE)
        let model_for_emb = std::sync::Arc::new(std::sync::Mutex::new(
            HippocampusModel::<B>::load_or_init(&device, "R:/uriel/models/checkpoints/uriel_encoder_step008500.bin"),
        ));

        const RC_VOCAB: usize = 95;
        const RC_ARGK: usize  = 6;
        let mut router: Option<RouterCopySystem<B>> = None;
        if let Ok(rec) = CompactRecorder::new()
            .load("data/runs/router_v1/routercopy_step010000.bin".into(), &device)
        {
            let sys_rc = RouterCopySystem::<B>::new(RC_VOCAB, RC_ARGK, &device).load_record(rec);
            router = Some(sys_rc);
        }


        let smie_embedder = Arc::new(SmieHippEmbed::new(model_for_emb.clone())?);
        let smie = Arc::new(Smie::new(
            SmieConfig { dim: smie_embedder.dim(), metric: Metric::Cosine },
            smie_embedder.clone(),     // <-- clone so we retain our typed handle
        )?);

        std::fs::create_dir_all("data").ok();
        let _ = smie.load("data/uriel");
        // === Router + Policy + Clarifier (global to the REPL loop) ===
        // They must live on the main thread, outside any inner closures/threads.
        type RB = burn_autodiff::Autodiff<burn_cuda::Cuda>;
        let rdev = <RB as burn::tensor::backend::Backend>::Device::new(0);

        let router_ckpt = "ckpts/router-v1/router_head_step000200.bin";
        let router_infer = URIELV1::policy::router_infer::RouterInfer::<RB>::load(&rdev, router_ckpt)
            .unwrap_or_else(|_| {
                eprintln!("[router] falling back to noop ({} not found)", router_ckpt);
                URIELV1::policy::router_infer::RouterInfer::<RB>::noop(&rdev)
            });

        let policy    = URIELV1::policy::policy_infer::RtPolicy::load_or_default("data/policy/router_policy.json")?;
        let clarifier = URIELV1::policy::clarification::Clarifier::default();
        // let guardrails = URIELV1::policy::guardrail::default_guardrails(); // (optional, later)

        // ─── Metacognition Loop (background) ───
        let (mc_stop_tx, mc_stop_rx) = xchan::bounded::<()>(1);
        let (quit_tx, quit_rx) = xchan::bounded::<()>(1);

        let mc_cfg = MetacogCfg {
            // run hourly in production; for testing set to 10–30 sec
            interval_secs: 1,
            // how many facts/beliefs to sweep per tick
            batch_size: 8,
            // commit replacement when new evidence ≥ 0.60
            promote_threshold: 0.60,
            // treat ≤ 0.30 as stale/dubious (consider replacement)
            demote_threshold: 0.30,
            // ambiguous band triggers a web check
            ambiguous_low: 0.31,
            ambiguous_high: 0.59,
            // domains the loop is allowed to consult
            peer_domains: vec![
                "wikipedia.org",
                "nih.gov", "nature.com", "science.org",
                "stanford.edu", "mit.edu", "harvard.edu",
                "arxiv.org"
            ],
            max_web_bytes_per_page: 300_000,
            max_web_chunks: 80,
            // audit file
            metacog_log_path: "data/meta/metacog_trace.jsonl".to_string(),
        };

        // pass only the path; the loop opens its own thread-local connection
        let mc_handle = spawn_metacog_loop(
            "data/semantic_memory.db".to_string(),
            mc_cfg,
            Some(mc_stop_rx),
        );

        let emb = Arc::new(ModelEmb::new(model_for_emb.clone()));
        let ts  = Arc::new(SmieTextSearch::new(smie.clone()));
        let rcfg = RetrieverCfg { top_k_default: 8, top_k_research: 16, min_score: 0.45, use_cache: true };
        let retriever = ContextRetriever::new_with_textsearch(
            Arc::new(NullVS), // keep placeholder ANN for later
            emb,
            None,
            rcfg,
            Some(ts),
        );
        // ─── Soarus background worker ───
        let (soarus_tx, soarus_rx) = start_worker(8);

        // Example default config (tweak if you like)
        let soarus_cfg = SoarusIcpConfig {
            mode: URIELV1::soarus::SoarusIcpMode::PointToPlane,
            max_corr: 0.02,
            iters: 30,
            trim: 0.20,
            pyramid: Some(vec![0.01, 0.005, 0.002]),
            normals_r_scale: 1.5,
            max_plane_dist: 0.016,
            huber_delta: 0.01,
        };


        // ─── routing / task context ───
        let ctx = "default";
        // --- Session context with short half-life (~2 minutes) ---
        let mut sess = SessionCtx::new(120);

        // ─── MicroThalamus + goals ───
        let (goal_tx, goal_rx): (Sender<Goal>, Receiver<Goal>) = unbounded();
        let mut thalamus = MicroThalamus::new(goal_tx.clone());
        thalamus.register_drive(SyntheticDrive::new(
            "Curiosity", 0.25, 0.01, Duration::from_secs(3), 0.5,
            "drive:curiosity", vec![0.1, 0.2, 0.3, 0.0, 0.1, 0.4, 0.1, 0.2],
        ));
        thalamus.register_drive(SyntheticDrive::new(
            "ResolveConflict", 0.35, 0.01, Duration::from_secs(5), 1.0,
            "drive:resolveconflict", vec![0.1, 0.2, 0.3, 0.0, 0.1, 0.4, 0.1, 0.2],
        ));
        // initial drives snapshot
        let mut drives: HashMap<String, SyntheticDrive> = thalamus.drives.clone();

        let mut goals: Vec<Goal> = Vec::new();

        // frontal
        let mut wm = WorkingMemory::new(50);
        let inhib_state = InhibitionState::default();
        let inhib_cfg = InhibitionConfig::default();
        let mut last_out: Option<CortexOutput> = None;

        // dedupe for auto-persist
        let mut seen_hashes: HashSet<String> = HashSet::with_capacity(4096);

        // autosave
        {
            let smie_c = smie.clone();
            let mc_stop_tx_c = mc_stop_tx.clone();
            let router_bias_c = router_bias.clone();
            let router_bias_path_c = router_bias_path.clone();
            let smie_embedder_c = smie_embedder.clone();
            let quit_tx_c = quit_tx.clone();

            ctrlc::set_handler(move || {
                // ask the bg loops to stop and tell REPL to break
                let _ = mc_stop_tx_c.send(());
                let _ = quit_tx_c.send(());

                // NOTE: no riftnet, no process::exit here.
            })?;
        }

        println!("URIEL runtime. Type lines. Prefix with 'store this:' to save, ask 'what … ?' to retrieve.");
        println!("URIEL runtime (async). Sources: stdin (opt), tail file (opt).");

        // === unified input (CLI + tail) ===
        let (in_tx, in_rx) = unbounded::<String>();

        // Synchronous UX via a stdin feeder thread (prompt + flush done inside the feeder)
        if std::env::var("URIEL_ENABLE_CLI")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false") && !v.eq_ignore_ascii_case("off"))
            .unwrap_or(true)
        {
            spawn_stdin_source(in_tx.clone());
        }

        // Optional file tail: set URIEL_INPUT_FILE="path". Add URIEL_TAIL_FROM_START=1 to read from top.
        if let Ok(path) = std::env::var("URIEL_INPUT_FILE") {
            let from_start = std::env::var("URIEL_TAIL_FROM_START")
                .map(|v| v=="1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on"))
                .unwrap_or(false);
            spawn_tail_source(path, from_start, in_tx.clone());
        }

        // Optional tiny inbox de-dupe (seconds). Set INBOX_DEDUPE_SECS=2..5 to avoid echo loops.
        let inbox_dedupe_secs: u64 = std::env::var("INBOX_DEDUPE_SECS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let mut inbox_seen: HashMap<String, u64> = HashMap::with_capacity(4096);
        // repl
        let stdin = std::io::stdin();
        let mut line = String::new();
        loop {
            if quit_rx.try_recv().is_ok() { break; }
            // ---- Soarus: poll background ICP results (non-blocking) ----
            while let Ok(msg) = soarus_rx.try_recv() {
                match msg {
                    SoarusMsg::Done(o) => {
                        // o.pose: nalgebra::Isometry3<f32>, o.rmse, o.p99, o.inliers
                        println!(
                            "(soarus) ICP pose ok  rmse={:.6}  p99={:.6}  inliers={}",
                            o.rmse, o.p99, o.inliers
                        );

                        // TODO: fuse pose into URIEL’s world state/map here …
                        // e.g., store to SQLite/SMIE:
                        // upsert pose + metrics into your table / memory system
                        // (example helper below)
                        if let Err(e) = persist_soarus_pose(&mut conn, &o) {
                            eprintln!("(soarus) persist error: {e}");
                        }

                        // Optionally publish on the bus
                        bus.publish_dynamic("soarus", None, URIELV1::callosum::metacog_subscriber::FactUpsertEvent {
                            predicate: "soarus.pose".into(),
                            value: format!(
                                "{{qx:{:.6},qy:{:.6},qz:{:.6},qw:{:.6},tx:{:.6},ty:{:.6},tz:{:.6},rmse:{:.6},p99:{:.6},inliers:{}}}",
                                o.pose.rotation.i, o.pose.rotation.j, o.pose.rotation.k, o.pose.rotation.w,
                                o.pose.translation.x, o.pose.translation.y, o.pose.translation.z,
                                o.rmse, o.p99, o.inliers
                            ),

                            ts: now_sec() as i64,
                        });
                    }
                    SoarusMsg::Err(e) => {
                        eprintln!("(soarus) ICP error: {e}");
                    }
                }
            }

            print!("> ");
            io::stdout().flush()?;
            line.clear();
            if stdin.read_line(&mut line)? == 0 { break; }

            // sanitize
            let raw = line.as_str();
            let input = sanitize_input(raw);
            // ---- Soarus: enqueue ICP job via command "align <src> <tgt>" ----
            if let Some(rest) = input.strip_prefix("align ") {
                let mut parts = rest.split_whitespace();
                if let (Some(src), Some(tgt)) = (parts.next(), parts.next()) {
                    let _ = soarus_tx.send(SoarusJob::RegisterPaths {
                        src_path: src.to_string(),
                        tgt_path: tgt.to_string(),
                        cfg: soarus_cfg.clone(),
                    });
                    println!("(soarus) queued ICP: src='{}' → tgt='{}'", src, tgt);
                    continue; // go to next loop tick
                } else {
                    println!("usage: align <src.ply> <tgt.ply>");
                    continue;
                }
            }

            if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") { break; }

            // Split “lead context. question” so lead can seed the domain
            let (lead_ctx, input_main) = split_lead_context(&input);

            // Age + bump from lead and current line
            sess.decay();
            if let Some(ref lead) = lead_ctx {
                if let Some(d) = infer_domain_from_question(lead) {
                    sess.bump_domain(&d, 1.0);
                }
            }
            if let Some(d) = infer_domain_from_question(&input_main) {
                sess.bump_domain(&d, 1.0);
            }

            // Intent is computed from the main clause
            let intent = infer_intent(&input_main);



            // Phase 1: per-turn meta trace id + shallow step
            let turn_id = format!("{}-{}", now_sec(), rand::random::<u32>());
            let _ = meta2.sense("run-1", &turn_id, "chat", Some(input.clone()));
            // Router per-turn bookkeeping
            let mut router_pred_class: Option<usize> = None; // 0=Respond,1=Retrieve,2=Store,3=Defer
            let mut did_router_store: bool = false;

            let _ = kops::record_episode(
                &conn,
                ktypes::EpisodeWrite {
                    ts: now_sec() as i64,
                    role: "user",
                    text: &input,
                    turn_id: Some(&turn_id),
                    smie_id: None,
                    ctx_json: None,
                },
            );

            // publish to Callosum
            let corr = Some(turn_hash_u64(&turn_id));
            bus.publish_dynamic("runtime", corr, EpisodeEvent {
                role: "user".to_string(),
                text: input.clone(),
                turn_id: Some(turn_id.clone()),
            });
            if mesh_forward {
                if let Some(cli) = riftnet.as_ref() {
                    let ask = serde_json::json!({
                "type": "ask",
                "turn_id": turn_id,
                "question": input_main
            });
                    let ask = add_token(ask, mesh_token.as_deref());
                    if let Err(e) = cli.send_str(&ask.to_string()) {
                        eprintln!("[riftnet] send ask: {e}");
                    }
                }
            }

            if mesh_broadcast_ep {
                if let Some(cli) = riftnet.as_ref() {
                    let ep = serde_json::json!({
                "type": "episode",
                "role": "user",
                "text": input,
                "turn_id": turn_id
            });
                    let ep = add_token(ep, mesh_token.as_deref());
                    if let Err(e) = cli.send_str(&ep.to_string()) {
                        eprintln!("[riftnet] send episode: {e}");
                    }
                }
            }

            let shallow = MetaStep {
                turn_id: turn_id.clone(),
                goal: "answer user question".into(),
                hypothesis: format!("H: {}", input),
                plan: vec!["check_db".into(), "probe_smie_norm".into(), "probe_smie_orig".into()],
                needed: vec![],
                evidence: vec![],
                confidence: 0.0,
                depth: 0,
                outcome: "revise".into(),
                bandit: None,
                sources: None,
            };
            record_meta_trace(&conn, &turn_id, "shallow", &shallow, now_sec() as i64);

            // auto-persist non-questions
            match intent {
                IntentKind::DefineAssert | IntentKind::Correct | IntentKind::Style => {
                    let _ = auto_persist_line(
                        &conn, &model_runtime, &smie, ctx, &input, &concepts, &mut learner, &mut seen_hashes
                    );
                }
                IntentKind::Ask | IntentKind::Compare | IntentKind::Instruct | IntentKind::Chat => {
                    // don’t store yet; these trigger retrieval/answering later
                }
            }

            // MicroThalamus tick
            thalamus.inject_stimulus(ctx, &input);
            thalamus.tick();
            drives = thalamus.drives.clone();

            // drain goals
            let mut new_goals = Vec::new();
            while let Ok(g) = goal_rx.try_recv() { new_goals.push(g); }
            if !new_goals.is_empty() {
                let ctx_sig = Ctx { signals: [("weight".into(), 0.35)].into_iter().collect() };
                for g in new_goals { push_goal(&mut goals, g, &drives, &ctx_sig); }
                rescore_goals(&mut goals, &drives, &ctx_sig);
            }

            // explicit STORE branch
            // if lower.starts_with("store this:") {
            //     let rest = input.splitn(2, ':').nth(1).unwrap_or("").trim();
            //
            //     // Use the same unified pipeline: durable dedupe → beliefs → concepts → ingest → SMIE
            //     let _ = auto_persist_line(
            //         &conn,               // <- the same rusqlite::Connection you already opened
            //         &model_runtime,
            //         &smie,
            //         ctx,
            //         rest,
            //         &concepts,
            //         &mut learner,
            //         &mut seen_hashes,
            //     );
            //
            //     wm.set_focus(rest);
            //     println!("(stored)");
            //
            //     // (Optional) meta trace "final" for store-only, if you want to keep the record
            //     let final_step = MetaStep {
            //         turn_id: turn_id.clone(),
            //         goal: "answer user question".into(),
            //         hypothesis: format!("H: {}", input),
            //         plan: vec!["store".into()],
            //         needed: vec![],
            //         evidence: vec![],
            //         confidence: 0.0,
            //         depth: 0,
            //         outcome: "accepted".into(),
            //         bandit: None,
            //         sources: Some(SourcesInfo { answered_from: "none".into(), db_hit: false }),
            //     };
            //     record_meta_trace(&conn, &turn_id, "final", &final_step, now_sec() as i64);
            //     continue;
            // }


            // SMIE-first answering
            let mut answer = String::new();
            let mut last_hit_score: Option<f32> = None;     // track SMIE best.score when we use it
            let mut final_kv: Option<(String, String)> = None; // (concept_key, value) if we answered a concept

            // for meta final
            let mut meta_arm: Option<String> = None;
            let mut meta_reward: f32 = 0.0;
            let mut answered_from: String = "none".into();
            let mut evidence: Vec<String> = Vec::new();
            let mut concept_key_for_belief: Option<String> = None;

            if matches!(intent, IntentKind::Ask | IntentKind::Compare | IntentKind::Instruct)
                && !looks_like_url(&input_main)
                && !input_main.trim_start().to_ascii_lowercase().starts_with("from ")
            {
                let q_norm = normalize_question(&input_main);
                eprintln!("[q] q_norm='{}'", q_norm);
                // --- DB fast-path for normalized Q → "the launch code is ..." ---
                // Runs before SMIE/definitions. Answers from concept_facts/knowledge.
                if answer.is_empty() && q_norm.starts_with("the launch code is") {
                    if let Some((v, src)) = read_launch_code_any(&conn) {
                        eprintln!("[answer/launch] src={} value={}", src, v);
                        answer = v.clone();
                        final_kv = Some(("launch code".to_string(), v.clone())); // outward normalized key
                        answered_from = "db".into();
                        evidence.push(format!("{}:{}", src, "launch_code"));
                    }
                }

                let mask: u64 = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
                if answer.is_empty() {
                    // Launch code
                    if answer.is_empty() && q_norm.starts_with("the launch code is") {
                        if let Some((v, src)) = read_launch_code_any(&conn) {
                            eprintln!("[answer/launch] src={} value={}", src, v);
                            answer = v.clone();
                            // normalize outward predicate to the spaced form for UI/logs
                            final_kv = Some(("launch code".to_string(), v.clone()));
                            answered_from = "db".into();
                            evidence.push(format!("{}:launch_code", src));
                        }
                    }
                }

                // logits: Tensor<B,2> [B,4]
                // --- Unconditional definition DB fast-path (before router/SMIE) ---
                // --- Learning-based definition fast-path (canonical + domain inference) ---
                if answer.is_empty() {
                    // Allow explicit definitional asks *and* “about … (in …)” forms
                    let mut topic_raw_opt = parse_definition_query(&input_main);
                    let mut dom_hint_q = infer_domain_from_question(&input_main);

                    if topic_raw_opt.is_none() {
                        if let Some((topic_about, dom_about)) = parse_about_query(&input_main) {
                            topic_raw_opt = Some(topic_about);
                            if dom_hint_q.is_none() { dom_hint_q = dom_about; }
                        }
                    }

                    if let Some(topic_raw) = topic_raw_opt {
                        let surf = canonicalize_surface(&topic_raw);
                        let topic_canon = read_alias(&surf).unwrap_or_else(|| surf.clone());
                        let dom_smie = infer_domain_for_topic_from_smie(&smie, &topic_canon, mask); // Option<String>
                        let dom_ctx  = sess.current_domain(0.35);                                   // Option<String>

                        // Build a selected domain **without** moving the originals:
                        let dom_sel = dom_hint_q.clone()
                            .or(dom_ctx.clone())
                            .or(dom_smie.clone());

                        if answer.is_empty() && dom_hint_q.is_none() {
                            if let (Some(a), Some(b)) = (dom_smie.as_ref(), dom_ctx.as_ref()) {
                                if a != b {
                                    println!("Do you mean {} in '{}' or '{}'?", topic_canon, a, b);
                                    continue; // wait for the user’s clarification
                                }
                            }
                        }

                        if let Some(ref ds) = dom_sel { sess.bump_domain(ds, 0.30); }
                        sess.observe_topic(&topic_canon);

                        // 1) Try domained definition (if any), else 2) generic definition, else 3) salvage via SMIE text
                        let mut keys_to_try = vec![
                            def_key(dom_sel.as_deref(), &topic_canon),
                            def_key(None, &topic_canon),
                        ];

                        if let Some(k) = nearest_def_key_for(&smie, &topic_canon, dom_sel.as_deref(), 12) {
                            keys_to_try.push(k);
                        }
                        if let Some(k) = nearest_def_key_for(&smie, &topic_canon, None, 12) {
                            keys_to_try.push(k);
                        }
                        keys_to_try.sort();
                        keys_to_try.dedup();

                        let mut found: Option<(String, String, &'static str)> = None;
                        for key in keys_to_try {
                            if cognition_reads_enabled() {
                                if let Ok(Some(def)) = kops::read_best_fact_text(&conn, None, &key) {
                                    found = Some((key.clone(), def, "db"));
                                    break;
                                }
                            }
                            if found.is_none() {
                                if let Ok(Some(def)) = read_concept_fact(&key).map_err(to_any) {
                                    found = Some((key.clone(), def, "db"));
                                    break;
                                }
                            }
                        }

                        if let Some((key, def, src)) = found {
                            answer = def.clone();
                            answered_from = src.into();
                            evidence.push(key.clone());

                            let _ = cof.log_call(&turn_id, Op::ReadFact, None, "rule");
                            let ev = Evidence {
                                id: key.clone(),
                                kind: "doc".into(),
                                source_id: key.clone(),
                                url: None,
                                sha256: None,
                                fetched_at: now_sec() as i64,
                                excerpt: def.clone(),
                                quality: EvidenceQuality { trust_w: 0.85, recency_w: 0.7, method_w: 0.35 },
                            };
                            let _ = meta2.retrieve("run-1", &turn_id, "chat", vec![ev.clone()], Some(src.to_string()));
                            let _ = cof.log_obs_evidence(&turn_id, src, vec![ev]);
                            meta_reward = 0.9;
                        } else {
                            // Salvage from SMIE: use the best definitional snippet we *learned* earlier
                            let qvars = [format!("{}:", topic_canon), format!("{} is", topic_canon), topic_canon.clone()];
                            'salv: for qv in qvars.iter() {
                                if let Ok(hits) = smie.recall(qv, 8, mask) {
                                    for (_id, _sim, _tg, txt) in hits {
                                        if let Some((_t, desc)) = extract_definition(&txt) {
                                            answer = desc.clone();
                                            answered_from = "smie".into();
                                            meta_reward = 0.7;
                                            break 'salv;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // --- "X vs Y" comparison using your DB (fallback to SMIE) ---
                // --- Comparison (X vs Y / compare/contrast) ---
                if answer.is_empty() {
                    if let Some((lhs_raw, rhs_raw)) =
                        parse_vs_query(&input_main).or_else(|| parse_compare_query(&input_main))
                    {
                        let dom_hint = infer_domain_from_question(&input_main);
                        if let Some(out) = act_compare(&conn, &smie, &lhs_raw, &rhs_raw, dom_hint.as_deref()) {
                            answer = format!("{}\n\n{}", out.table, out.summary);
                            answered_from = "compare".into();
                            evidence.extend(out.evidence);
                        }
                    }
                }


                // If `answer` is set, skip routing entirely

                if answer.is_empty() {

                    // ----------------- NEW DECISION PIPELINE -----------------

                    // Cheap per-turn features.
                    // If you don’t have a light retrieval probe yet, leave top1 as None for now.
                    let top1_score_opt: Option<f32> = None;
                    let evid_count = 0usize;                // pre-route
                    let tool_code_avail = true;             // wire from your tool registry when ready
                    let tool_sql_avail  = false;
                    let session_retr_ratio = 0.0;
                    let observed_latency_ms = 0u32;

                    let f = extract_features(
                        &input_main.to_ascii_lowercase(),
                        top1_score_opt,
                        evid_count,
                        tool_code_avail,
                        tool_sql_avail,
                        session_retr_ratio,
                        observed_latency_ms
                    );

                    // 1) (optional) guardrails here

                    // 2) learned router head → probs
                    let rp = router_infer.score(&input_main);

                    // 3) clarifier: ask when underspecified / low-confidence
                    if let ClarifyDecision::Ask { prompt, reason, .. } = clarifier.decide(&input_main, &rp, &f) {
                        let _ = meta2.plan("run-1", &turn_id, "chat", "clarify", &reason, None);
                        println!("{}", prompt);
                        continue; // skip routing/dispatch this turn
                    }

                    // 4) policy: blend router_probs + features → action
                    let final_route = policy.decide(&rp, &f);

                    // 5) map action to your existing flags
                    let mut force_retrieve_only = matches!(final_route, Route::Retrieve);
                    let mut force_store         = matches!(final_route, Route::ToolCode | Route::ToolSql);
                    let mut _defer_asked = false;

                    // ----------------- END NEW PIPELINE -----------------


                    if force_store {
                        let _ = auto_persist_line(
                            &conn, &model_runtime, &smie, ctx, &input, &concepts, &mut learner, &mut seen_hashes
                        );
                        did_router_store = true;

                    }

                    // --- ONLY route if still unanswered ---
                    if answer.is_empty() {
                        // -------------------- CONCEPT BRANCH --------------------
                        if let Some((key, norm)) = concepts.normalize_question(&input_main) {
                            concept_key_for_belief = Some(key.to_string());
                            let _ = cof.log_call(&turn_id, Op::ReadFact, Some(Arg::ConceptKey { value: key.to_string() }), "rule");

                            // quick identity special-case
                            // quick identity special-case
                            if key == "you are" {
                                let mut hit = None;

                                if cognition_reads_enabled() {
                                    if let Ok(Some(v)) = kops::read_best_fact_text(&conn, None, "you are") { hit = Some(("you are".to_string(), v)); }
                                    else if let Ok(Some(v)) = kops::read_best_fact_text(&conn, None, "name") { hit = Some(("you are".to_string(), v)); }
                                }

                                if hit.is_none() {
                                    if let Ok(Some(v)) = read_concept_fact("you are").map_err(to_any) {
                                        hit = Some(("you are".to_string(), v));
                                    } else if let Ok(Some(v)) = read_concept_fact("name").map_err(to_any) {
                                        hit = Some(("you are".to_string(), v));
                                    }
                                }

                                if let Some((kk, v)) = hit {
                                    answer = render_concept_answer(&kk, &v);
                                    final_kv = Some((kk.clone(), v.clone()));
                                    answered_from = "db".into();
                                    evidence.push(format!("cf:{kk}"));
                                    meta_reward = 1.0;
                                    let ev = Evidence {
                                        id: format!("cf:{kk}"),
                                        kind: "doc".into(),
                                        source_id: kk.clone(),
                                        url: None,
                                        sha256: None,
                                        fetched_at: now_sec() as i64,
                                        excerpt: v.clone(),
                                        quality: EvidenceQuality { trust_w: 0.8, recency_w: 0.7, method_w: 0.3 },
                                    };
                                    let _ = meta2.retrieve("run-1", &turn_id, "chat", vec![ev.clone()], Some("db".to_string()));
                                    let _ = cof.log_obs_evidence(&turn_id, "db", vec![ev]);
                                }
                            }


                            // bandit probe order for concept answers
                            let chosen = concept_bandit.choose(&conn);
                            meta_arm = Some(chosen.to_string());
                            let _ = meta2.plan("run-1", &turn_id, "chat", "concept_bandit", chosen, None);

                            let arms = ["db", "normalized", "keyword", "original"];
                            let mut order: Vec<&str> = Vec::with_capacity(4);
                            order.push(chosen);
                            for a in arms.iter() { if *a != chosen { order.push(*a); } }

                            for arm in order {
                                match arm {
                                    "db" => {
                                        let mut got: Option<String> = None;
                                        if cognition_reads_enabled() {
                                            if let Ok(Some(v)) = kops::read_best_fact_text(&conn, None, key) {
                                                got = Some(v);
                                            }
                                        }
                                        if got.is_none() {
                                            if let Ok(Some(v)) = read_concept_fact(key).map_err(to_any) {
                                                got = Some(v);
                                            }
                                        }
                                        if let Some(v) = got {
                                            answer = render_concept_answer(key, &v);
                                            final_kv = Some((key.to_string(), v.clone()));
                                            answered_from = "db".into();
                                            evidence.push(format!("cf:{key}"));

                                            let ev = Evidence {
                                                id: format!("cf:{key}"),
                                                kind: "doc".into(),
                                                source_id: key.to_string(),
                                                url: None,
                                                sha256: None,
                                                fetched_at: now_sec() as i64,
                                                excerpt: v.clone(),
                                                quality: EvidenceQuality { trust_w: 0.8, recency_w: 0.7, method_w: 0.3 },
                                            };
                                            let _ = meta2.retrieve("run-1", &turn_id, "chat", vec![ev.clone()], Some("db".to_string()));
                                            let _ = cof.log_obs_evidence(&turn_id, "db", vec![ev]);
                                            meta_reward = 1.0;
                                            break;
                                        }
                                    }

                                    "normalized" => {
                                        if let Some(v) = answer_concept(&smie, key, &norm, &input_main)? {
                                            let val = extract_answer(&norm, &v).unwrap_or_else(|| v.clone());
                                            answer = render_concept_answer(key, &val);
                                            final_kv = Some((key.to_string(), val.clone()));
                                            answered_from = "normalized".into();

                                            let ev = Evidence {
                                                id: "smie:norm".into(),
                                                kind: "log".into(),
                                                source_id: "smie".into(),
                                                url: None,
                                                sha256: None,
                                                fetched_at: now_sec() as i64,
                                                excerpt: v.clone(),
                                                quality: EvidenceQuality { trust_w: 0.8, recency_w: 0.7, method_w: 0.3 },
                                            };
                                            let _ = meta2.retrieve("run-1", &turn_id, "chat", vec![ev.clone()], Some("normalized".to_string()));
                                            let _ = cof.log_obs_evidence(&turn_id, "smie", vec![ev]);

                                            meta_reward = 0.9;
                                            break;
                                        }
                                    }
                                    "keyword" => {
                                        if let Ok(hits) = smie.recall(key, 8, mask) {
                                            if let Some(best) = choose_best_for(&norm, &hits) {
                                                if let Some(val) = extract_answer(&norm, &best.3) {
                                                    last_hit_score = Some(best.1);
                                                    answer = realize_concept(key, &val);
                                                    final_kv = Some((key.to_string(), val.clone()));
                                                    answered_from = "keyword".into();
                                                    evidence.push(format!("smid:{}", best.0));

                                                    let ev = Evidence {
                                                        id: format!("smid:{}", best.0),
                                                        kind: "log".into(),
                                                        source_id: "smie".into(),
                                                        url: None,
                                                        sha256: None,
                                                        fetched_at: now_sec() as i64,
                                                        excerpt: best.3.clone(),
                                                        quality: EvidenceQuality { trust_w: 0.8, recency_w: 0.7, method_w: 0.3 },
                                                    };
                                                    let _ = meta2.retrieve("run-1", &turn_id, "chat", vec![ev.clone()], Some("keyword".to_string()));
                                                    let _ = cof.log_obs_evidence(&turn_id, "smie", vec![ev]);

                                                    meta_reward = 0.7;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    "original" => {
                                        if let Ok(hits) = smie.recall(&input_main, 8, mask) {
                                            if let Some(best) = choose_best_for(&norm, &hits) {
                                                if let Some(val) = extract_answer(&norm, &best.3) {
                                                    last_hit_score = Some(best.1);
                                                    answer = realize_concept(key, &val);
                                                    final_kv = Some((key.to_string(), val.clone()));
                                                    answered_from = "original".into();
                                                    evidence.push(format!("smid:{}", best.0));

                                                    let ev = Evidence {
                                                        id: format!("smid:{}", best.0),
                                                        kind: "log".into(),
                                                        source_id: "smie".into(),
                                                        url: None,
                                                        sha256: None,
                                                        fetched_at: now_sec() as i64,
                                                        excerpt: best.3.clone(),
                                                        quality: EvidenceQuality { trust_w: 0.8, recency_w: 0.7, method_w: 0.3 },
                                                    };
                                                    let _ = meta2.retrieve("run-1", &turn_id, "chat", vec![ev.clone()], Some("original".to_string()));
                                                    let _ = cof.log_obs_evidence(&turn_id, "smie", vec![ev]);

                                                    meta_reward = 0.6;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }

                        // -------------------- NON-CONCEPT BRANCH --------------------
                        } else {
                            let q_base = normalize_question(&input_main);
                            let mask2: u64 = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;

                            let _ = cof.log_call(&turn_id, Op::Recall, None, "rule");
                            let chosen = general_bandit.choose(&conn);
                            let _ = meta2.plan("run-1", &turn_id, "chat", "general_bandit", chosen, None);

                            let mut order = vec![chosen];
                            for a in ["normalized", "original"].iter() { if *a != chosen { order.push(*a); } }

                            // query variants (normalized + original + definitional synonyms)
                            let mut q_variants: Vec<String> = vec![q_base.clone(), input_main.clone()];
                            if let Some(top_raw) = parse_definition_query(&input_main) {
                                match canonical_topic(&top_raw).as_str() {
                                    "cpp" => {
                                        q_variants.push("what is cpp".into());
                                        q_variants.push("what is c plus plus".into());
                                    }
                                    "rust" => {
                                        q_variants.push("what is rust".into());
                                        q_variants.push("what is rust lang".into());
                                    }
                                    _ => {}
                                }
                            }
                            // --- Definition salvage from SMIE when DB misses ---
                            if answer.is_empty() {
                                if let Some(topic_raw) = parse_definition_query(&input_main) {
                                    let topic = canonical_topic(&topic_raw);
                                    let qdefs = [
                                        format!("{topic}:"),
                                        format!("{topic} is"),
                                        format!("{topic} -"),
                                        format!("{topic} —"),
                                        topic.clone(),
                                    ];
                                    let mask_smie: u64 = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
                                    'seek_def: for qd in qdefs.iter() {
                                        if let Ok(hits) = smie.recall(qd, 8, mask_smie) {
                                            for (_id, _sim, _tg, txt) in hits {
                                                if let Some((_t, desc)) = extract_definition(&txt) {
                                                    // commit to kg_fact + prefer
                                                    let key = format!("def:{}", canonical_topic(&topic));
                                                    let desc_clean = desc.trim().to_string();
                                                    let _ = kops::upsert_fact(&conn, ktypes::FactWrite {
                                                        subj: None,
                                                        predicate: key.clone(),
                                                        obj_text: desc_clean.clone(),
                                                        obj_entity: None,
                                                        confidence: 0.75,
                                                        ts: now_sec() as i64,
                                                        method: "smie".to_string(),
                                                        evidence: vec![format!("smie:{qd}")],
                                                    });
                                                    let _ = kops::prefer_latest(&conn, None, &key);

                                                    // answer
                                                    answer = desc_clean.clone();
                                                    answered_from = "smie".into();
                                                    meta_reward = 0.7;
                                                    break 'seek_def;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            let mut answered_nc = false;
                            'arm_loop: for arm in order {
                                for q in &q_variants {
                                    if let Ok(hits) = smie.recall(q, 8, mask2) {
                                        if let Some(best) = choose_best_for(&q_base, &hits) {
                                            if let Some(val) = extract_answer(&q_base, &best.3) {
                                                answer = val;
                                            } else {
                                                answer = best.3.clone();
                                            }

                                            let ev = Evidence {
                                                id: format!("smid:{}", best.0),
                                                kind: "log".into(),
                                                source_id: "smie".into(),
                                                url: None,
                                                sha256: None,
                                                fetched_at: now_sec() as i64,
                                                excerpt: best.3.clone(),
                                                quality: EvidenceQuality { trust_w: 0.8, recency_w: 0.7, method_w: 0.3 },
                                            };
                                            let _ = meta2.retrieve("run-1", &turn_id, "chat", vec![ev.clone()], Some(arm.to_string()));
                                            let _ = cof.log_obs_evidence(&turn_id, "smie", vec![ev]);

                                            answered_nc = true;
                                            answered_from = arm.into();
                                            evidence.push(format!("smid:{}", best.0));
                                            break 'arm_loop;
                                        }
                                    }
                                }
                            }

                            let reward = if answered_nc { 0.8 } else { 0.0 };
                            let _ = meta2.learn("run-1", &turn_id, "chat", chosen, reward as f64, None);
                            general_bandit.update(&conn, chosen, reward);

                            // if reward >= 0.35 && reward < 0.9 {
                            //     if let Some(qid) = enqueue_hard_example(&conn, &input, &answer, 1.0 - reward) {
                            //         let mut negs = mine_negatives_from_smie(&smie, &input, 6);
                            //         negs.retain(|(t,_s)| t != &answer);
                            //         negs.truncate(5);
                            //         attach_negs(&mut conn, qid, &negs);
                            //     }
                            // } else if reward >= 0.9 {
                            //     maybe_stage_anchor_for_training(&conn, &input, &answer, reward);
                            //     maybe_checkpoint_on_high_conf(reward);
                            // }
                        }
                    }
                }
            }

            // If SMIE didn’t answer, try PFC (Thought only for concepts)
            if answer.trim().is_empty()
                && matches!(intent, IntentKind::Ask | IntentKind::Compare | IntentKind::Instruct)
            {            let score = SalienceScore { urgency: 0.3, familiarity: 0.2, understanding: 0.5, drive_alignment: 0.0, final_weight: 0.35 };
                let frame = CortexInputFrame { raw_input: &input_main, score, drives: &drives };
                let out = execute_cortex(frame, &retriever);

                if concepts.normalize_question(&input_main).is_some() || normalize_question(&input_main).starts_with("you are") {
                    match out {
                        CortexOutput::Thought(_t) => { /* keep quiet during seeding */ answered_from = "pfc".into(); }
                        CortexOutput::Multi(list) => { for item in &list {
                            if let CortexOutput::Thought(_t) = item {
                                /* keep quiet during seeding */
                                answered_from = "pfc".into();
                            }
                        } }
                        CortexOutput::Action(a) => println!("(action) {:?}", a),
                        CortexOutput::VerbalResponse(_) => { /* ignore PFC verbal for concepts */ }
                    }
                } else {
                    match out {
                        CortexOutput::VerbalResponse(text) => { answer = text; answered_from = "pfc".into(); }
                        CortexOutput::Multi(list) => {
                            for item in &list { if let CortexOutput::Thought(t) = item { println!("{t}"); answered_from = "pfc".into(); } }
                            if let Some(CortexOutput::VerbalResponse(t)) =
                                list.into_iter().find(|o| matches!(o, CortexOutput::VerbalResponse(_))) { answer = t; }
                        }
                        CortexOutput::Thought(t) => { println!("{t}"); answered_from = "pfc".into(); }
                        CortexOutput::Action(a) => println!("(action) {:?}", a),
                    }
                }
            }

            // Final fallback: newest rows scan
            if answer.trim().is_empty()
                && matches!(intent, IntentKind::Ask | IntentKind::Compare | IntentKind::Instruct)
            {
                let qn = normalize_question(&input_main);
                if let Ok(rows) = recall_latest("default", 128) {
                    for r in rows {
                        if let Some(val) = extract_answer(&qn, &r.data) {
                            // If this was a concept question, render via realize_concept; else use raw text.
                            if let Some((k, _norm)) = concepts.normalize_question(&input_main) {
                                answer = render_concept_answer(k, &val);
                                final_kv = Some((k.to_string(), val.clone()));
                            } else {
                                answer = val.clone();
                            }

                            answered_from = "rows".into();
                            evidence.push(format!("rows:default:{}", r.timestamp));

                            // log op + evidence for rows scan
                            let _ = cof.log_call(&turn_id, Op::RowsScan, None, "rule");

                            // Optional: record a v2 evidence entry (note the i64 cast!)
                            let ev = Evidence {
                                id: format!("rows:default:{}", r.timestamp),
                                kind: "log".into(),
                                source_id: "rows".into(),
                                url: None,
                                sha256: None,
                                fetched_at: r.timestamp as i64,  // <-- fix: cast u64 -> i64
                                excerpt: r.data.clone(),
                                quality: EvidenceQuality { trust_w: 0.8_f32, recency_w: 0.7_f32, method_w: 0.3_f32 },
                            };
                            let _ = meta2.retrieve("run-1", &turn_id, "chat", vec![ev.clone()], Some("rows".to_string()));
                            let _ = cof.log_obs_evidence(&turn_id, "rows", vec![ev]);

                            break;
                        }
                    }
                }
            }


            // Frontal gating
            let active_goal_labels: Vec<String> = goals.iter().map(|g| g.label.clone()).collect();
            let fctx = FrontalContext {
                active_goals: active_goal_labels,
                current_drive_weights: &drives,
                last_response: last_out.as_ref(),
                working_memory: std::cell::RefCell::new(wm.clone()),
                inhib_state: std::cell::RefCell::new(inhib_state.clone()),
                inhib_cfg: inhib_cfg.clone(),
            };

            let candidate = CortexOutput::VerbalResponse(answer.clone());
            match evaluate_frontal_control(&candidate, &fctx) {
                FrontalDecision::ExecuteImmediately(out) => {
                    match out.clone() {
                        CortexOutput::VerbalResponse(text) => println!("{text}"),
                        other => println!("{other:?}"),
                    }
                    last_out = Some(out);
                }
                FrontalDecision::QueueForLater(out) => { eprintln!("(queued) {out:?}"); last_out = Some(out); }
                FrontalDecision::SuppressResponse(msg) => { eprintln!("{msg}"); }
                FrontalDecision::ReplaceWithGoal(msg) => { println!("{msg}"); last_out = Some(CortexOutput::Thought(msg)); }
            }
            // -------- Phase 1: compute confidence just before recording meta --------
            let is_concept = concepts.normalize_question(&input_main).is_some()
                || normalize_question(&input_main).starts_with("you are");

            // base from source
            let base = to_base_from_source(&answered_from);

            // retrieval from cosine-like score (if we used a SMIE hit)
            let retrieval = retrieval_from_cos(last_hit_score);

            // extraction validity (only if we know the concept key + value)
            let extract_ok = match &final_kv {
                Some((k,v)) => is_valid_for_key(k, v),
                None        => true,
            };

            // consistency vs canonical DB (for concept answers)
            let consistency = match &final_kv {
                Some((k,v)) => {
                    match read_concept_fact(k).map_err(to_any) {
                        Ok(Some(dbv)) => Some(if loose_eq(&dbv, v) { 1.0 } else { 0.0 }),
                        Ok(None)      => Some(0.5), // unknown
                        Err(_)        => None,
                    }
                }
                None => None,
            };

            // recency if we used rows fallback
            let recency = rows_recency_from_evidence(&evidence);

            // (optional) arm prior — not wired yet; use None for now
            let arm_prior = None;

            let final_conf = combine_confidence(
                is_concept, base, retrieval, extract_ok, consistency, recency, arm_prior,
            );
            let mut final_conf = final_conf; // from your combiner
            if let Some(ref kfb) = concept_key_for_belief {
                let now_i = now_sec() as i64;
                if let Ok(aggs) = URIELV1::Hippocampus::sqlite_backend::beliefs_agg(&conn, 50) {
                    let key_l = kfb.to_ascii_lowercase();
                    if let Some(best) = aggs.iter().find(|a| a.key_norm.starts_with(&key_l)) {
                        let s = URIELV1::Hippocampus::sqlite_backend::belief_stability(best, now_i);
                        final_conf = clamp01(final_conf + 0.15 * s);
                    }
                }
            }
            if !answer.is_empty() {
                let _ = kops::record_episode(
                    &conn,
                    ktypes::EpisodeWrite {
                        ts: now_sec() as i64,
                        role: "agent",
                        text: &answer,
                        turn_id: Some(&turn_id),
                        smie_id: None,
                        ctx_json: None,
                    },
                );
            }
            if !answer.is_empty() {
                let corr = Some(turn_hash_u64(&turn_id));
                bus.publish_dynamic("runtime", corr, EpisodeEvent {
                    role: "agent".to_string(),
                    text: answer.clone(),
                    turn_id: Some(turn_id.clone()),
                });
            }
            if mesh_broadcast_ep && !answer.is_empty() {
                if let Some(cli) = riftnet.as_ref() {
                    let ep = serde_json::json!({
                "type": "episode",
                "role": "agent",
                "text": answer,
                "turn_id": turn_id
            });
                    let ep = add_token(ep, mesh_token.as_deref());
                    if let Err(e) = cli.send_str(&ep.to_string()) {
                        eprintln!("[riftnet] send episode: {e}");
                    }
                }
            }
            // -------- end confidence block --------
            // ---- Critic: score & enqueue learning item ----
            {
                // Build CriticInput from the features you just computed
                let ci = kops::CriticInput {
                    is_concept,
                    answered_from: answered_from_static(&answered_from),
                    extract_ok,
                    consistency,
                    recency,
                    retrieval_cos: last_hit_score,
                    plan_success: None, // set Some(x) later if you compute plan success
                };
                let co = kops::critic_justify(ci);
                // record per-arm reward for routing policy
                let _ = kops::router_stats_update(&conn, answered_from_static(&answered_from), co.reward);

                // Only generate a training item for questions that produced some answer
                if is_question(&input_main) {
                    let q_for_train = normalize_question(&input_main);

                    // Prefer concept key/value when available; else use the free-form answer text
                        if let Some((ref k, ref v)) = final_kv {
                            let pos = canon_value_for(k, v);
                            if !looks_like_web_error(&pos) {
                                if let Ok(qid) = kops::critic_enqueue_learning(&conn, &q_for_train, &pos, co.reward) {
                                // attach up to 5 hard negatives mined from SMIE
                                let mut negs = mine_negatives_from_smie(&smie, &q_for_train, 6);
                                negs.retain(|(t,_s)| t.trim() != pos);
                                for (neg, sim) in negs.into_iter().take(5) {
                                    let _ = kops::critic_attach_negative(&conn, qid, &neg, sim);
                                }
                            }
                        }
                    } else if !answer.is_empty() {
                        let pos = answer.trim().to_string();
                        if !looks_like_web_error(&pos) {
                            if let Ok(qid) = kops::critic_enqueue_learning(&conn, &q_for_train, &pos, co.reward) {
                                let mut negs = mine_negatives_from_smie(&smie, &q_for_train, 6);
                                negs.retain(|(t,_s)| t.trim() != pos);
                                for (neg, sim) in negs.into_iter().take(5) {
                                    let _ = kops::critic_attach_negative(&conn, qid, &neg, sim);
                                }
                            }
                        }
                    }
                }
                // ---- Micro-OEP: 1 diagonal update step (live) ----
                {
                    // only if we have a question and a positive answer text
                    if is_question(&input_main) {
                        let q_for_train = normalize_question(&input_main);

                        // pick the positive text
                        let pos_opt = if let Some((ref k, ref v)) = final_kv {
                            Some(canon_value_for(k, v))
                        } else if !answer.is_empty() {
                            Some(answer.trim().to_string())
                        } else { None };

                        if let Some(pos) = pos_opt {
                            // mine one hard negative from SMIE
                            let mut cands = mine_negatives_from_smie(&smie, &q_for_train, 4);
                            cands.retain(|(t,_s)| t.trim() != pos);
                            if let Some((neg, _)) = cands.first() {
                                // embed q, pos, neg (normalize)
                                // 1) Heavy hashing OUTSIDE the lock
                                let vq = URIELV1::Hippocampus::infer::embed_text(&q_for_train);
                                let vp = URIELV1::Hippocampus::infer::embed_text(&pos);
                                let vn = URIELV1::Hippocampus::infer::embed_text(neg);

                                // 2) Tiny MLP forwards INSIDE the lock
                                let m = model_for_emb.lock().map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
                                let eq0 = m.forward_from_vec(&vq)
                                    .to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?.to_vec();
                                let ep0 = m.forward_from_vec(&vp)
                                    .to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?.to_vec();
                                let en0 = m.forward_from_vec(&vn)
                                    .to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?.to_vec();
                                drop(m);

                                if !eq0.is_empty() && eq0.len() == ep0.len() && ep0.len() == en0.len() {
                                    let dim = eq0.len();
                                    let norm = |mut v: Vec<f32>| {
                                        let n = (v.iter().map(|x| x*x).sum::<f32>()).sqrt().max(1e-6);
                                        for x in &mut v { *x /= n; }
                                        v
                                    };
                                    let eq = norm(eq0);
                                    let ep = norm(ep0);
                                    let en = norm(en0);

                                    // current weights (init if absent)
                                    let mut w = smie_embedder.oep_w.read().unwrap().clone().unwrap_or_else(|| vec![1.0; dim]);

                                    // scores s_pos and s_neg under current w
                                    let mut s_pos = 0.0f32;
                                    let mut s_neg = 0.0f32;
                                    for i in 0..dim {
                                        let qh = w[i] * eq[i];
                                        s_pos += qh * (w[i] * ep[i]);
                                        s_neg += qh * (w[i] * en[i]);
                                    }

                                    // update if margin not met
                                    let margin: f32 = 0.20;
                                    if s_pos - s_neg < margin {
                                        let eta: f32 = 0.05 * co.reward.max(0.1); // scale by critic reward
                                        for i in 0..dim {
                                            w[i] += eta * (eq[i] * (ep[i] - en[i]));
                                        }
                                        // clamp + renormalize mean to 1.0
                                        let mut sum = 0.0f32;
                                        for wi in &mut w {

                                            if *wi < 0.05 { *wi = 0.05; }
                                            sum += *wi;
                                        }
                                        let mean = (sum / dim as f32).max(1e-6);
                                        for wi in &mut w { *wi /= mean; }

                                        // write back (live)
                                        smie_embedder.set_oep_weights(w);
                                        oep_updates += 1;
                                        if oep_updates % oep_save_every() == 0 {
                                            smie_embedder.save_oep();
                                            eprintln!("[oep] saved weights to {}", "data/oep_w.json");
                                        }

                                    }
                                }
                            }
                        }
                    }
                }
                // ---- Router bias online update (light, perceptron-like) ----
                // Class labels: 0=Respond, 1=Retrieve, 2=Store, 3=Defer
                {
                    // derive the label from what actually happened
                    let label_class = if answer.is_empty() {
                        3 // we effectively deferred
                    } else if did_router_store {
                        2 // we stored due to router forcing store
                    } else {
                        match answered_from_static(&answered_from) {
                            "db" | "normalized" | "keyword" | "original" => 1, // retrieval path
                            _ => 0, // responded directly (PFC/verbal)
                        }
                    };

                    if let Some(chosen) = router_pred_class {
                        if label_class != chosen {
                            // small step scaled by critic reward
                            let eta: f32 = 0.02 * co.reward.max(0.1);
                            let mut b = router_bias.write().unwrap();
                            b[label_class] += eta;
                            b[chosen]      -= eta;
                            router_bias_updates += 1;
                            if router_bias_updates % 100 == 0 {
                                save_router_bias(&router_bias_path, *b);
                                eprintln!("[router] bias saved to {}", &router_bias_path);
                            }
                        } else {
                            // even when correct, you can gently reinforce the chosen class if you like:
                            // let eta: f32 = 0.005 * co.reward.max(0.1);
                            // let mut b = router_bias.write().unwrap();
                            // b[chosen] += eta;
                        }
                    }
                }

            }


            if cognition_writes_enabled() {
                if let Some((ref k, ref v)) = final_kv {
                    // 1) canonicalize the value you’re storing
                    let clean_v = canon_value_for(k, v);

                    // 2) record this as a justification (source, confidence, evidence)
                    let _ = tms_upsert_support(
                        &conn,
                        k,
                        &clean_v,
                        map_source_to_method(&answered_from),
                        final_conf,
                        &evidence,
                    );

                    // 3) (optional) see what the currently preferred value is
                    if let Ok(Some((pref_v, support, _last))) = tms_preferred(&conn, k) {
                        if pref_v != clean_v {
                            // You can log/trace that the winner stayed pref_v because its support is higher.
                            // eprintln!("kept '{}' over '{}' (support {:.3})", pref_v, clean_v, support);
                        }
                    }

                    // 4) keep your existing bus/fanout so the rest of the system sees the update
                    //    (same as before — unchanged)
                    let corr = Some(turn_hash_u64(&turn_id));
                    bus.publish_dynamic("runtime", corr, FactUpsertEvent {
                        predicate: k.clone(),
                        value: clean_v.clone(),
                        ts: now_sec() as i64,
                    });
                    if mesh_broadcast_facts {
                        if let Some(cli) = riftnet.as_ref() {
                            let fact = serde_json::json!({
                        "type": "fact",
                        "predicate": k,
                        "value": clean_v
                    });
                            let fact = add_token(fact, mesh_token.as_deref());
                            let _ = cli.send_str(&fact.to_string());
                        }
                    }
                }
            }


            // Phase 2: record final meta trace
            let final_step = MetaStep {
                turn_id: turn_id.clone(),
                goal: "answer user question".into(),
                hypothesis: format!("H: {}", input),
                plan: vec!["check_db".into(), "probe_smie_norm".into(), "probe_smie_orig".into()],
                needed: vec![],
                evidence,
                confidence: final_conf,
                depth: 0,
                outcome: if answer.is_empty() { "fail".into() } else { "accepted".into() },
                bandit: meta_arm.as_ref().map(|a| BanditInfo { arm: a.clone(), reward: meta_reward }),
                sources: Some(SourcesInfo { db_hit: answered_from == "db", answered_from }),
            };
            let _ = meta2.report("run-1", &turn_id, "chat", Some(answer.clone()), final_conf, None);
            let label = if answer.is_empty() { FinalLabel::IDK } else { FinalLabel::OK };
            let _ = cof.log_final(&turn_id, label, &answer, &[], Some(final_conf));
            record_meta_trace(&conn, &turn_id, "final", &final_step, now_sec() as i64);
        }
        let _ = mc_stop_tx.send(());
        let _ = mc_handle.join();
        let _ = emit_stop_tx.send(());
        let _ = emit_handle.join();
        let _ = smie.save("data/uriel");

        save_router_bias(&router_bias_path, *router_bias.read().unwrap());
        smie_embedder.save_oep();
        // Shutdown RiftNet on the owning/main thread (safe)
        if let Some(mut cli) = riftnet {
            cli.shutdown();
        }

        Ok(())
    }

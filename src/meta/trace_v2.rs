    use serde::{Serialize, Deserialize};
    use std::{fs, io::Write};
    use anyhow::Result;

    #[derive(Serialize, Deserialize, Clone)]
    pub enum MetaPhase {
        Sense, RetrieveEvidence, Hypothesize, Plan, Act, Observe, Critique,
        AskPermission, Learn, BeliefUpdate, Report
    }

    #[derive(Serialize, Deserialize, Clone, Default)]
    pub struct EvidenceQuality { pub trust_w: f32, pub recency_w: f32, pub method_w: f32 }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct Evidence {
        pub id: String,
        pub kind: String,           // "peerreview" | "preprint" | "doc" | "code" | "log"
        pub source_id: String,      // vector id or local id
        pub url: Option<String>,
        pub sha256: Option<String>,
        pub fetched_at: i64,
        pub excerpt: String,
        pub quality: EvidenceQuality,
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct BeliefUpdate {
        pub claim_id: String,
        pub claim_text: String,
        pub prior_log_odds: f64,
        pub delta_log_odds: f64,
        pub posterior_log_odds: f64,
        pub posterior_p: f64,
        pub rationale: String,
        pub evidence_ids: Vec<String>,
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct MetaEvent {
        pub ts: i64,
        pub run_id: String,
        pub task_id: String,
        pub domain: String,
        pub phase: MetaPhase,
        pub policy: Option<String>,
        pub arm: Option<String>,
        pub expected_value: Option<f64>,
        pub observed_value: Option<f64>,
        pub confidence: Option<(f32, f32, f32)>,   // p, ci_low, ci_high
        pub cost: Option<(u32, u32, f32, f32)>,    // ms, tokens, cpu_s, $
        pub inputs_ref: Option<String>,
        pub outputs_ref: Option<String>,
        pub evidence: Vec<Evidence>,
        pub belief_update: Option<BeliefUpdate>,
        pub notes: Option<String>,
    }

    #[inline] fn now_sec() -> i64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64
    }

    pub struct TraceSink {
        path: String
    }
    impl TraceSink {
        pub fn new(path: impl Into<String>) -> Self { Self { path: path.into() } }
        pub fn emit(&self, ev: &MetaEvent) -> Result<()> {
            if let Some(dir) = std::path::Path::new(&self.path).parent() {
                let _ = fs::create_dir_all(dir);
            }
            let mut f = fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
            let line = serde_json::to_string(ev)?;
            writeln!(f, "{line}")?;
            Ok(())
        }

        // convenience shims you can call from runtime:
        pub fn sense(&self, run: &str, task: &str, domain: &str, inputs_ref: Option<String>) -> Result<()> {
            self.emit(&MetaEvent {
                ts: now_sec(), run_id: run.into(), task_id: task.into(), domain: domain.into(),
                phase: MetaPhase::Sense, policy: None, arm: None,
                expected_value: None, observed_value: None, confidence: None,
                cost: None, inputs_ref, outputs_ref: None, evidence: vec![],
                belief_update: None, notes: None,
            })
        }
        pub fn plan(&self, run: &str, task: &str, domain: &str, policy: &str, arm: &str, notes: Option<String>) -> Result<()> {
            self.emit(&MetaEvent {
                ts: now_sec(), run_id: run.into(), task_id: task.into(), domain: domain.into(),
                phase: MetaPhase::Plan, policy: Some(policy.into()), arm: Some(arm.into()),
                expected_value: None, observed_value: None, confidence: None,
                cost: None, inputs_ref: None, outputs_ref: None, evidence: vec![],
                belief_update: None, notes,
            })
        }
        pub fn retrieve(&self, run: &str, task: &str, domain: &str, sources: Vec<Evidence>, notes: Option<String>) -> Result<()> {
            self.emit(&MetaEvent {
                ts: now_sec(), run_id: run.into(), task_id: task.into(), domain: domain.into(),
                phase: MetaPhase::RetrieveEvidence, policy: None, arm: None,
                expected_value: None, observed_value: None, confidence: None,
                cost: None, inputs_ref: None, outputs_ref: None, evidence: sources,
                belief_update: None, notes,
            })
        }
        pub fn report(&self, run: &str, task: &str, domain: &str, outputs_ref: Option<String>, p: f32, ci: Option<(f32,f32)>) -> Result<()> {
            self.emit(&MetaEvent {
                ts: now_sec(), run_id: run.into(), task_id: task.into(), domain: domain.into(),
                phase: MetaPhase::Report, policy: None, arm: None,
                expected_value: None, observed_value: None, confidence: Some((p, ci.map(|x|x.0).unwrap_or(p), ci.map(|x|x.1).unwrap_or(p))),
                cost: None, inputs_ref: None, outputs_ref, evidence: vec![],
                belief_update: None, notes: None,
            })
        }
        pub fn learn(&self, run: &str, task: &str, domain: &str, arm: &str, reward: f64, notes: Option<String>) -> Result<()> {
            self.emit(&MetaEvent {
                ts: now_sec(), run_id: run.into(), task_id: task.into(), domain: domain.into(),
                phase: MetaPhase::Learn, policy: None, arm: Some(arm.into()),
                expected_value: None, observed_value: Some(reward),
                confidence: None, cost: None, inputs_ref: None, outputs_ref: None,
                evidence: vec![], belief_update: None, notes,
            })
        }
    }

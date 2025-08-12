// src/Inference/inference.rs
use std::cmp::Ordering;
use std::fmt::Write as _;

use crate::core::core::{
    Action, Evidence, InferenceDecision, InferenceEngine, InferenceInput,
};

/// Tunable knobs for the baseline engine.
#[derive(Clone, Debug)]
pub struct RuleCfg {
    /// Respond only if we have at least this many strong hits.
    pub min_hits_for_respond: usize,
    /// A "strong" hit must exceed this cosine similarity.
    pub min_cosine: f32,
    /// And at least this trust/salience.
    pub min_trust: f32,
    /// If most strong hits are older than this, prefer SearchInternet.
    pub stale_after_s: f32,
    /// How many hits to cite when responding.
    pub evidence_k: usize,
    /// Curiosity gate to ask a follow-up question.
    pub ask_curiosity_gate: f32,
}

impl Default for RuleCfg {
    fn default() -> Self {
        Self {
            min_hits_for_respond: 2,
            min_cosine: 0.55,
            min_trust: 0.40,
            stale_after_s: 86_400.0, // 24h
            evidence_k: 3,
            ask_curiosity_gate: 0.60,
        }
    }
}

/// A deterministic, fast inference engine.
/// Decisions are made from the hit set + basic meta; no heavy model here.
pub struct RuleEngine {
    cfg: RuleCfg,
}

impl RuleEngine {
    pub fn new(cfg: RuleCfg) -> Self {
        Self { cfg }
    }

    fn strong_hits<'a>(&self, hits: &'a [Evidence]) -> Vec<&'a Evidence> {
        hits.iter()
            .filter(|e| e.cosine >= self.cfg.min_cosine && e.trust >= self.cfg.min_trust)
            .collect()
    }

    fn pick_evidence_ids(&self, hits: &[&Evidence]) -> Vec<String> {
        let mut v: Vec<&Evidence> = hits.to_vec();
        // Prefer freshest, then highest cosine
        v.sort_by(|a, b| {
            let ord_time = a.time_delta_s.partial_cmp(&b.time_delta_s).unwrap_or(Ordering::Equal);
            match ord_time {
                Ordering::Less => Ordering::Less, // a is fresher (smaller delta)
                Ordering::Greater => Ordering::Greater,
                Ordering::Equal => b.cosine.partial_cmp(&a.cosine).unwrap_or(Ordering::Equal),
            }
        });
        v.into_iter()
            .take(self.cfg.evidence_k)
            .map(|e| e.id.clone())
            .collect()
    }

    fn summarize(&self, inp: &InferenceInput, strong: &[&Evidence]) -> (f32, f32, f32, f32, f32) {
        let n = inp.hits.len().max(1) as f32;
        let max_sim = inp.hits.iter().map(|e| e.cosine).fold(0.0, f32::max);
        let mean_sim = inp.hits.iter().map(|e| e.cosine).sum::<f32>() / n;
        let strong_ratio = (strong.len() as f32) / n;
        // in RuleEngine::summarize()
        let is_stale = |e: &Evidence| {
            let by_age = e.time_delta_s > self.cfg.stale_after_s;
            let by_ttl = e.ttl_s > 0.0 && e.ttl_s <= 1.0; // count TTL only if set
            by_age || by_ttl
        };
        let stale_ratio = if strong.is_empty() { 1.0 } else {
            strong.iter().filter(|e| is_stale(e)).count() as f32 / strong.len() as f32
        };

        let mean_trust = inp.hits.iter().map(|e| e.trust).sum::<f32>() / n;
        (max_sim, mean_sim, strong_ratio, stale_ratio, mean_trust)
    }

    fn confidence(&self, max_sim: f32, strong_ratio: f32, stale_ratio: f32) -> f32 {
        // simple, monotone mapping; clamp to [0,1]
        let mut c = 0.6 * max_sim + 0.5 * strong_ratio - 0.4 * stale_ratio;
        if c.is_nan() { c = 0.0; }
        c.clamp(0.0, 1.0)
    }
}

impl InferenceEngine for RuleEngine {
    fn infer(&self, inp: &InferenceInput) -> InferenceDecision {
        // 1) Sort hits by cosine desc for stable behavior
        let mut hits = inp.hits.clone();
        hits.sort_by(|a, b| b.cosine.partial_cmp(&a.cosine).unwrap_or(Ordering::Equal));

        let strong = self.strong_hits(&hits);
        let (max_sim, mean_sim, strong_ratio, stale_ratio, mean_trust) =
            self.summarize(inp, &strong);

        // Feature: curiosity drive
        let curiosity = *inp.drives.get("Curiosity").unwrap_or(&0.0);

        // 2) Decide action
        let action = if strong.len() >= self.cfg.min_hits_for_respond && stale_ratio < 0.5 {
            Action::Respond
        } else if stale_ratio >= 0.5 && max_sim >= 0.35 {
            // We have *some* signal but it's old/stale → fetch fresh context.
            Action::SearchInternet(inp.raw_input.to_string())
        } else if curiosity >= self.cfg.ask_curiosity_gate && max_sim < self.cfg.min_cosine {
            // Low similarity + high curiosity → ask clarifying Q
            Action::Ask
        } else if strong.is_empty() && max_sim < 0.35 {
            // Almost nothing matches: store for later; don't hallucinate
            Action::StoreOnly
        } else {
            // Default: ask rather than over-claim
            Action::Ask
        };

        // 3) Choose evidence IDs (even for Ask/Search we keep these for logging)
        let chosen_ids = self.pick_evidence_ids(&strong);

        // 4) Confidence + rationale
        let conf = self.confidence(max_sim, strong_ratio, stale_ratio);

        let mut why = String::new();
        let _ = write!(
            &mut why,
            "max_sim={:.2} mean_sim={:.2} strong_ratio={:.2} stale_ratio={:.2} mean_trust={:.2} curiosity={:.2}; ",
            max_sim, mean_sim, strong_ratio, stale_ratio, mean_trust, curiosity
        );
        match &action {
            Action::Respond => {
                let _ = write!(
                    &mut why,
                    "Respond: {} strong hits (≥{:.2} cosine & ≥{:.2} trust), fresh enough.",
                    strong.len(),
                    self.cfg.min_cosine,
                    self.cfg.min_trust
                );
            }
            Action::Ask => {
                let _ = write!(
                    &mut why,
                    "Ask: similarity insufficient (max {:.2}), or ambiguity detected.",
                    max_sim
                );
            }
            Action::StoreOnly => {
                let _ = write!(
                    &mut why,
                    "StoreOnly: no strong evidence (max {:.2}); keep for future context.",
                    max_sim
                );
            }
            Action::SearchInternet(q) => {
                let _ = write!(
                    &mut why,
                    "SearchInternet: evidence stale (>{:.0}s) despite some signal; q='{}'",
                    self.cfg.stale_after_s, truncate(q, 80)
                );
            }
            Action::RunSubmodule(name) => {
                let _ = write!(&mut why, "RunSubmodule('{}') by rule trigger.", name);
            }
        }

        InferenceDecision {
            action,
            chosen_ids,
            confidence: conf,
            rationale: why,
        }
    }
}

// --- helpers ---
fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n { s.to_string() } else { format!("{}…", &s[..n]) }
}

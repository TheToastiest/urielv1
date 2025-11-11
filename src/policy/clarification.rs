use super::types::{RtFeatures, RtProbs, ClarifyDecision};

pub struct Clarifier {
    pub max_prob_thresh: f32,
    pub retr_top1_thresh: f32,
    pub min_tokens: u16,
}
impl Default for Clarifier {
    fn default() -> Self {
        Self { max_prob_thresh: 0.58, retr_top1_thresh: 0.35, min_tokens: 3 }
    }
}

impl Clarifier {
    pub fn decide(&self, q_norm: &str, p: &RtProbs, f: &RtFeatures) -> ClarifyDecision {
        let l = q_norm.to_ascii_lowercase();
        let maxp = p_max(p);

        if token_guess(q_norm) < self.min_tokens && !l.starts_with("help") {
            return ClarifyDecision::Ask {
                prompt: "Could you add a bit more detail about what you want to know?".into(),
                reason: "too_short".into(),
                fields: Vec::<String>::new(),
            };
        }

        if contains_ambiguous_pronoun(&l) && f.session_retr_ratio < 0.2 {
            return ClarifyDecision::Ask {
                prompt: "What does “it/this/they” refer to here?".into(),
                reason: "pronoun_ambiguous".into(),
                fields: vec!["referent".to_string()],
            };
        }

        if looks_def_overview(&l) && f.evid_count == 0 && f.retr_top1 < self.retr_top1_thresh {
            return ClarifyDecision::Ask {
                prompt: "Do you mean 1NF in databases, or in another context?".into(),
                reason: "domain_unspecified".into(),
                fields: vec!["domain".to_string()],
            };
        }

        if looks_sql(&l) && f.tool_code_avail && missing_sql_args(&l) {
            return ClarifyDecision::Ask {
                prompt: "Which table/columns should I query, and any filters or limits?".into(),
                reason: "sql_missing_args".into(),
                fields: vec![
                    "table".to_string(), "columns".to_string(),
                    "filters".to_string(), "limit".to_string()
                ],
            };
        }

        if maxp < self.max_prob_thresh && f.retr_top1 < self.retr_top1_thresh && f.evid_count == 0 {
            return ClarifyDecision::Ask {
                prompt: "I can look this up or answer directly; what specifically are you trying to find out?".into(),
                reason: "low_conf".into(),
                fields: Vec::<String>::new(),
            };
        }

        ClarifyDecision::None
    }
}

/* helpers (unchanged) */
fn p_max(p: &RtProbs) -> f32 {
    p.p_direct.max(p.p_retrieve.max(p.p_tool_code.max(p.p_tool_sql)))
}
fn token_guess(s: &str) -> u16 { s.split_whitespace().count().min(u16::MAX as usize) as u16 }
fn contains_ambiguous_pronoun(l: &str) -> bool {
    l.contains(" it ") || l.contains(" this ") || l.contains(" that ") || l.contains(" they ")
}
fn looks_def_overview(l: &str) -> bool {
    l.starts_with("what is ")
        || l.starts_with("what's ")
        || l.starts_with("define ")
        || l.starts_with("explain ")
        || l.starts_with("tell me about ")
        || l.starts_with("give me an overview of ")
        || l.contains(" difference ")
        || l.contains(" vs ")
}
fn looks_sql(l: &str) -> bool {
    l.contains("select ") || l.contains(" join ") || l.contains(" where ")
}
fn missing_sql_args(l: &str) -> bool {
    if let Some(i) = l.find(" from ") {
        let rest = &l[i + 6..];
        rest.split_whitespace().next().map(|t| t.len() < 2).unwrap_or(true)
    } else { true }
}

use super::types::{Route, RtFeatures};

pub trait Guardrail {
    fn name(&self) -> &'static str;
    /// Return Some(route) to force a route; None = no decision.
    fn check(&self, q_norm: &str, f: &RtFeatures) -> Option<Route>;
}

// 1) Always route comparisons to retrieve
pub struct GrCompare;
impl Guardrail for GrCompare {
    fn name(&self) -> &'static str { "compare_vs" }
    fn check(&self, q: &str, _f: &RtFeatures) -> Option<Route> {
        let l = q.to_ascii_lowercase();
        if l.contains(" vs ") || l.contains(" difference ") || l.contains(" instead of ") { Some(Route::Retrieve) } else { None }
    }
}

// 2) Explicit tool invocation (if you support it)
pub struct GrToolCodeExplicit;
impl Guardrail for GrToolCodeExplicit {
    fn name(&self) -> &'static str { "tool_code_explicit" }
    fn check(&self, q: &str, f: &RtFeatures) -> Option<Route> {
        let l = q.to_ascii_lowercase();
        if f.tool_code_avail && (l.contains("SELECT ") || l.contains(" from ") || l.contains("sql ")) { Some(Route::ToolCode) } else { None }
    }
}

pub fn default_guardrails() -> Vec<Box<dyn Guardrail + Send + Sync>> {
    vec![ Box::new(GrCompare), Box::new(GrToolCodeExplicit) ]
}

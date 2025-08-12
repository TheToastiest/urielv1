// parietal/route_cfg.rs
#[derive(Clone, Debug)]
pub struct RouteCfg {
    /// If true, explicit retrieval intent (“what/when…?”, “recall”, “lookup”) always routes to Retrieve.
    pub retrieval_override: bool,
    /// Extra substrings that should trigger retrieval.
    pub extra_retrieval_terms: &'static [&'static str],
}

impl Default for RouteCfg {
    fn default() -> Self {
        Self {
            retrieval_override: true,
            extra_retrieval_terms: &[],
        }
    }
}

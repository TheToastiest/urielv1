#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route { Direct, Retrieve, ToolCode, ToolSql }

#[derive(Clone, Debug, Default)]
pub struct RtFeatures {
    pub has_qmark: bool,
    pub len_tokens: u16,
    pub retr_top1: f32,      // single-shot retrieval top1 score (0..1)
    pub evid_count: u8,      // RetrieveEvidence count seen in this turn so far
    pub latency_ms: u32,     // optional
    pub session_retr_ratio: f32, // recent window (0..1)
    pub tool_code_avail: bool,
    pub tool_sql_avail: bool,
}

#[derive(Clone, Debug)]
pub struct RtProbs { // router head output
    pub p_direct: f32,
    pub p_retrieve: f32,
    pub p_tool_code: f32,
    pub p_tool_sql: f32,
}
#[derive(Clone, Debug)]
pub enum ClarifyDecision {
    None,
    Ask {
        prompt: String,     // the clarifying question to ask
        reason: String,     // short tag for analytics
        fields: Vec<String> // optional named fields we’re asking for (e.g., ["table","columns"])
    }
}

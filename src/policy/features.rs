use super::types::RtFeatures;

pub fn extract_features(
    query_norm: &str,
    retr_top1: Option<f32>,
    evid_count: usize,
    tool_code_avail: bool,
    tool_sql_avail: bool,
    session_retr_ratio: f32,
    latency_ms: u32,
) -> RtFeatures {
    RtFeatures {
        has_qmark: query_norm.ends_with('?'),
        len_tokens: query_norm.split_whitespace().count().min(u16::MAX as usize) as u16,
        retr_top1: retr_top1.unwrap_or(0.0).clamp(0.0, 1.0),
        evid_count: (evid_count as u8),
        latency_ms,
        session_retr_ratio: session_retr_ratio.clamp(0.0, 1.0),
        tool_code_avail,
        tool_sql_avail,
    }
}

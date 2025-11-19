// src/Hippocampus/ids.rs
pub fn make_mem_id(context: &str, ts_sec: u64) -> String {
    format!("sm:{context}:{ts_sec}")
}

pub fn parse_mem_id(id: &str) -> Option<(String, u64)> {
    let parts: Vec<_> = id.split(':').collect();
    if parts.len() != 3 || parts[0] != "sm" {
        return None;
    }
    let ts = parts[2].parse::<u64>().ok()?;
    Some((parts[1].to_string(), ts))
}

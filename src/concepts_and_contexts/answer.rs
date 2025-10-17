use smie_core::{Smie, tags};
use crate::Hippocampus::sqlite_backend as db;
use crate::Hippocampus::sqlite_backend::recall_latest;

/// DB → SMIE → newest-row fallback
pub fn answer_concept(smie:&Smie, key:&str, normalized:&str, _original:&str) -> anyhow::Result<Option<String>> {
    if let Some(v) = db::read_concept_fact(key).map_err(|e| anyhow::anyhow!(e.to_string()))? {
        return Ok(Some(v));
    }
    let mask = tags::SHORT | tags::LONG | tags::SYSTEM | tags::USER;
    if let Ok(mut hits) = smie.recall(normalized, 16, mask) {
        hits.retain(|(_,_,_,t)| t.to_ascii_lowercase().contains(key));
        hits.sort_by(|a,b| b.0.cmp(&a.0).then_with(|| b.1.total_cmp(&a.1)));
        if let Some((_,_,_,text)) = hits.first() { return Ok(Some(text.clone())); }
    }
    if let Ok(rows) = recall_latest("default", 128) {
        for r in rows {
            if r.data.to_ascii_lowercase().contains(key) { return Ok(Some(r.data)); }
        }
    }
    Ok(None)
}

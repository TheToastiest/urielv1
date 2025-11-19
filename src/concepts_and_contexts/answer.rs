use crate::Hippocampus::sqlite_backend as db;
use crate::Hippocampus::sqlite_backend::recall_latest;

/// DB → newest-row fallback only (no SMIE)
pub fn answer_concept(
    key: &str,
    normalized: &str,
    _original: &str,
) -> anyhow::Result<Option<String>> {
    // 1) direct concept_facts lookup
    if let Some(v) = db::read_concept_fact(key).map_err(|e| anyhow::anyhow!(e.to_string()))? {
        return Ok(Some(v));
    }

    // 2) loose text search over latest rows as a salvage path
    let norm_lc = normalized.to_ascii_lowercase();
    let key_lc  = key.to_ascii_lowercase();

    if let Ok(rows) = recall_latest("default", 128) {
        for r in rows {
            let ld = r.data.to_ascii_lowercase();
            if ld.contains(&norm_lc) || ld.contains(&key_lc) {
                return Ok(Some(r.data));
            }
        }
    }

    Ok(None)
}

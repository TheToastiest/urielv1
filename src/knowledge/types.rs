#[derive(Clone)]
pub struct FactWrite {
    pub subj: Option<i64>,          // None => GLOBAL (1)
    pub predicate: String,
    pub obj_text: String,
    pub obj_entity: Option<i64>,
    pub confidence: f32,            // 0..1
    pub ts: i64,                    // unix seconds
    pub method: String,             // 'db','smie','rows','pfc','unknown'
    pub evidence: Vec<String>,      // 'cf:name','smid:123','rows:..','url:..'
}

#[derive(Clone)]
pub struct EpisodeWrite<'a> {
    pub ts: i64,
    pub role: &'a str,              // 'user','agent','system','tool'
    pub text: &'a str,
    pub turn_id: Option<&'a str>,
    pub smie_id: Option<&'a str>,
    pub ctx_json: Option<&'a str>,
}

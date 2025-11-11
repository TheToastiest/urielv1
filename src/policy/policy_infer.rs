// src/policy/policy_infer.rs
use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::types::{Route, RtFeatures};
use crate::policy::types::RtProbs;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RtPolicyWeights {
    pub w_retrieve: [f32; 8],
    pub w_direct:   [f32; 8],
    pub w_toolcode: [f32; 8],
    pub w_toolsql:  [f32; 8],
}

pub struct RtPolicy {
    w: RtPolicyWeights,
}

impl RtPolicy {
    pub fn load_or_default(path: &str) -> Result<Self> {
        if let Ok(txt) = std::fs::read_to_string(path) {
            if let Ok(w) = serde_json::from_str::<RtPolicyWeights>(&txt) {
                return Ok(Self { w });
            }
        }
        // sane defaults: prefer retrieve when router+probe agree; else direct
        Ok(Self {
            w: RtPolicyWeights {
                // [p_retr, p_direct, top1, has_q, len_norm, evid, sess_retr, bias]
                w_retrieve: [ 1.2,  -0.5,  1.0,  0.2, -0.1,  0.3,  0.1,  0.0 ],
                w_direct:   [ -0.4,  1.0, -0.6, -0.1,  0.1, -0.2, -0.2,  0.0 ],
                w_toolcode: [ 0.2,  -0.3,  0.1,  0.0,  0.0,  0.1,  0.0, -0.5 ],
                w_toolsql:  [ 0.2,  -0.3,  0.1,  0.0,  0.0,  0.1,  0.0, -0.5 ],
            }
        })
    }

    pub fn decide(&self, p: &RtProbs, f: &RtFeatures) -> Route {
        let x = [
            p.p_retrieve,
            p.p_direct,
            f.retr_top1,
            if f.has_qmark { 1.0 } else { 0.0 },
            (f.len_tokens as f32).min(64.0) / 64.0,
            f.evid_count as f32 / 4.0,
            f.session_retr_ratio,
            1.0, // bias
        ];
        let dot = |w: &[f32; 8]| -> f32 { w.iter().zip(x.iter()).map(|(a, b)| a * b).sum() };

        let scores = [
            (Route::Direct,   dot(&self.w.w_direct)),
            (Route::Retrieve, dot(&self.w.w_retrieve)),
            (Route::ToolCode, dot(&self.w.w_toolcode)),
            (Route::ToolSql,  dot(&self.w.w_toolsql)),
        ];
        scores.into_iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0
    }
}

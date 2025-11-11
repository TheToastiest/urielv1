// src/Learning/routercopy/dataset.rs
use crate::tokenizer::tokenizer::Vocab;
use anyhow::Result;
use rand::{seq::SliceRandom, thread_rng};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Deserialize, Clone)]
pub struct ObsText {
    pub snippet_id: String,
    pub text: String,
}

#[derive(Deserialize, Clone)]
pub struct RouterCopyRow {
    pub user_text: String,
    pub obs_text: Vec<ObsText>,
    pub final_text: String,

    pub op_id: usize,                         // {NONE, READ_FACT, RECALL, ROWS_SCAN} → 0..3
    #[serde(default)] pub arg_class_id: Option<usize>, // concept key id if applicable
    #[serde(default)] pub arg_span: Option<[usize; 2]>,

    // optional (exporter may omit for v0); we will length-fix on load
    #[serde(default)] pub copy_mask: Vec<u8>,  // len = final_tokens
    #[serde(default)] pub ptr_index: Vec<i32>, // len = final_tokens, -1 = not copied
}

pub struct RouterCopyDataset {
    rows: Vec<RouterCopyRow>,
    vocab: Vocab,
    pub max_user: usize,
    pub max_obs: usize,
    pub max_final: usize,

}

pub struct RouterCopyBatch {
    pub user_tokens: Vec<Vec<usize>>,   // [B, max_user]
    pub obs_tokens: Vec<Vec<usize>>,    // [B, max_obs] (concatenated OBS)
    pub final_tokens: Vec<Vec<usize>>,  // [B, max_final]

    pub op_targets: Vec<usize>,         // [B]
    pub arg_class_targets: Vec<i32>,    // [B]  (-1 if N/A)
    pub arg_span_targets: Vec<[i32; 2]>,// [B,2](-1 if N/A)

    pub copy_mask: Vec<Vec<u8>>,        // [B, max_final]
    pub ptr_index: Vec<Vec<i32>>,       // [B, max_final]
}

impl RouterCopyDataset {
    pub fn load(path: &str, vocab: Vocab, max_user: usize, max_obs: usize, max_final: usize) -> Result<Self> {
        let f = File::open(path)?;
        let rdr = BufReader::new(f).lines();
        let mut rows = Vec::new();
        for line in rdr {
            let s = line?;
            if s.trim().is_empty() { continue; }
            rows.push(serde_json::from_str::<RouterCopyRow>(&s)?);
        }
        Ok(Self { rows, vocab, max_user, max_obs, max_final })
    }

    #[inline]
    fn enc(&self, s: &str, max_len: usize) -> Vec<usize> {
        self.vocab.encode(s, max_len)
    }

    fn enc_obs_concat(&self, snippets: &[ObsText], max_len: usize) -> Vec<usize> {
        // Simple concat; no explicit [SEP] to keep pointer indexing contiguous.
        let mut out = Vec::<usize>::new();
        for sn in snippets {
            if out.len() >= max_len { break; }
            let mut t = self.vocab.encode(&sn.text, max_len.saturating_sub(out.len()));
            out.append(&mut t);
            if out.len() >= max_len { break; }
        }
        if out.len() < max_len {
            out.resize(max_len, 0);
        }
        out
    }

    #[inline]
    fn fix_len_i32(mut v: Vec<i32>, want: usize, fill: i32) -> Vec<i32> {
        if v.len() > want { v.truncate(want); }
        if v.len() < want { v.resize(want, fill); }
        v
    }
    #[inline]
    fn fix_len_u8(mut v: Vec<u8>, want: usize, fill: u8) -> Vec<u8> {
        if v.len() > want { v.truncate(want); }
        if v.len() < want { v.resize(want, fill); }
        v
    }

    pub fn sample_batch(&self, bsz: usize) -> RouterCopyBatch {
        let mut rng = thread_rng();
        let batch = self.rows.choose_multiple(&mut rng, bsz);

        let mut user_tokens = Vec::with_capacity(bsz);
        let mut obs_tokens = Vec::with_capacity(bsz);
        let mut final_tokens = Vec::with_capacity(bsz);

        let mut op_targets = Vec::with_capacity(bsz);
        let mut arg_class_targets = Vec::with_capacity(bsz);
        let mut arg_span_targets = Vec::with_capacity(bsz);

        let mut copy_mask = Vec::with_capacity(bsz);
        let mut ptr_index = Vec::with_capacity(bsz);

        for r in batch {
            let u = self.enc(&r.user_text, self.max_user);
            let o = self.enc_obs_concat(&r.obs_text, self.max_obs);
            let f = self.enc(&r.final_text, self.max_final);

            // Targets
            op_targets.push(r.op_id);
            arg_class_targets.push(r.arg_class_id.map(|x| x as i32).unwrap_or(-1));
            arg_span_targets.push(r.arg_span.map(|ab| [ab[0] as i32, ab[1] as i32]).unwrap_or([-1, -1]));

            // Copy supervision (length-fix to final_tokens)
            let cm = Self::fix_len_u8(r.copy_mask.clone(), self.max_final, 0);
            let pi = Self::fix_len_i32(r.ptr_index.clone(), self.max_final, -1);

            user_tokens.push(u);
            obs_tokens.push(o);
            final_tokens.push(f);
            copy_mask.push(cm);
            ptr_index.push(pi);
        }

        RouterCopyBatch {
            user_tokens,
            obs_tokens,
            final_tokens,
            op_targets,
            arg_class_targets,
            arg_span_targets,
            copy_mask,
            ptr_index,
        }
    }
    pub fn vocab_size(&self) -> usize { self.vocab.size() }

}

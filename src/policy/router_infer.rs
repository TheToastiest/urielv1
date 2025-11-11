// src/policy/router_infer.rs
use anyhow::Result;
use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};
use burn::record::{CompactRecorder, Recorder};
use crate::training::model_router::RouterHead;
use crate::policy::types::RtProbs;
// simple char-95 one-hot
fn one_hot_from_text<B: Backend>(text: &str, vocab: usize, max_t: usize, dev: &B::Device) -> Tensor<B, 3> {
    let mut toks = Vec::with_capacity(max_t);
    for ch in text.chars().take(max_t) {
        let idx = (ch as i32 - 32).clamp(0, (vocab as i32) - 1) as usize;
        toks.push(idx);
    }
    while toks.len() < max_t { toks.push(0); }
    let mut data = vec![0.0f32; 1 * max_t * vocab];
    for (t, ix) in toks.iter().enumerate() { data[t * vocab + *ix] = 1.0; }
    Tensor::<B, 1>::from_floats(data.as_slice(), dev).reshape([1, max_t, vocab])
}


pub struct RouterInfer<B: Backend> {
    model: RouterHead<B>,
    device: B::Device,
    vocab: usize,
    max_t: usize,
}

impl<B: Backend> RouterInfer<B> {
    pub fn load(device: &B::Device, ckpt_path: &str) -> Result<Self> {
        // model shape must match your training one (95 -> 256 -> 4)
        let mut model = RouterHead::<B>::new(95, 256, 4, device);
        let rec = CompactRecorder::new().load(ckpt_path.into(), device)?;
        model = model.load_record(rec);
        Ok(Self { model, device: device.clone(), vocab: 95, max_t: 512 })
    }

    pub fn noop(device: &B::Device) -> Self {
        // a neutral model (won’t crash if no checkpoint yet)
        let model = RouterHead::<B>::new(95, 256, 4, device);
        Self { model, device: device.clone(), vocab: 95, max_t: 512 }
    }

    pub fn score(&self, query_norm: &str) -> RtProbs {
        let x = one_hot_from_text::<B>(query_norm, self.vocab, self.max_t, &self.device); // [1,T,V]
        let logits = self.model.forward(x);                                               // [1,4]
        let probs = burn::tensor::activation::softmax(logits, 1);
        let v = probs.to_data().as_slice::<f32>().unwrap().to_vec();
        RtProbs {
            p_direct:   v[0],
            p_retrieve: v[1],
            p_tool_code:v[2],
            p_tool_sql: v[3],
        }
    }
}

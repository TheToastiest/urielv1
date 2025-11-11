#![allow(clippy::needless_return)]

use burn::{
    module::Module,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{CompactRecorder, Recorder},
    tensor::{backend::Backend, Tensor},
};
use burn::tensor::backend::AutodiffBackend;

use crate::embedder::embedder::FixedWeightEmbedder;
use crate::Learning::routercopy::dataset::{RouterCopyBatch, RouterCopyDataset};
use crate::Learning::routercopy::loss::{
    total_routercopy_loss, Weights as LossWeights,
};
use crate::Learning::routercopy::model::RouterCopyModel;

/* ---------------------- Config ---------------------- */

#[derive(Clone, Debug)]
pub struct TrainCfg {
    pub max_steps: usize,
    pub batch_size: usize,

    // sequence caps (must match dataset encoder choices)
    pub max_user: usize,
    pub max_obs: usize,
    pub max_final: usize,

    pub vocab_size: usize,
    pub arg_classes: usize,   // 0 if you want to skip arg head

    pub lr: f32,
    pub weight_decay: f32,
    pub log_every: usize,
    pub ckpt_every: usize,
    pub ckpt_path: Option<String>,

    pub loss_w: LossWeights,
}

impl Default for TrainCfg {
    fn default() -> Self {
        Self {
            max_steps: 200_000,
            batch_size: 32,
            max_user: 128,
            max_obs: 512,
            max_final: 64,
            vocab_size: 95,
            arg_classes: 6, // name, creator, mentor, launch code, you are, purpose (example)
            lr: 3e-4,
            weight_decay: 0.01,
            log_every: 100,
            ckpt_every: 5_000,
            ckpt_path: None,
            loss_w: LossWeights::default(),
        }
    }
}

/* ----------------- System (Model + Embedder) -------- */

#[derive(Module, Debug)]
pub struct RouterCopySystem<B: Backend> {
    pub embed: FixedWeightEmbedder<B>, // [V -> 384], trainable
    pub model: RouterCopyModel<B>,     // heads
}

impl<B: Backend> RouterCopySystem<B> {
    pub fn new(vocab_size: usize, arg_classes: usize, device: &B::Device) -> Self {
        Self {
            embed: FixedWeightEmbedder::new(vocab_size, 384, device),
            model: RouterCopyModel::new(vocab_size, arg_classes, device),
        }
    }
}

/* ------------------ Token utils --------------------- */

/// tokens [B,T] -> one-hot [B,T,V]
fn tokens_to_one_hot<B: Backend>(
    tokens: &[Vec<usize>],
    vocab_size: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let b = tokens.len();
    let t = tokens.first().map(|x| x.len()).unwrap_or(0);
    let mut data = vec![0.0f32; b * t * vocab_size];

    for (bi, row) in tokens.iter().enumerate() {
        for (ti, &idx) in row.iter().enumerate() {
            if idx < vocab_size {
                data[bi * t * vocab_size + ti * vocab_size + idx] = 1.0;
            }
        }
    }

    let flat = Tensor::<B, 1>::from_floats(data.as_slice(), device);
    flat.reshape([b, t, vocab_size])
}

/// One-hot [B,T,V] -> embedded seq [B,T,384] using the trainable embedder.
fn embed_seq<B: Backend>(one_hot: Tensor<B, 3>, embedder: &FixedWeightEmbedder<B>) -> Tensor<B, 3> {
    let [b, t, v] = {
        let d = one_hot.dims();
        [d[0], d[1], d[2]]
    };
    let flat = one_hot.reshape([b * t, v]);           // [B*T, V]
    let proj = embedder.forward(flat);                // [B*T, 384]
    proj.reshape([b, t, 384])                         // [B, T, 384]
}

/// Mean pool over time: [B,T,384] -> [B,384]
fn mean_over_time<B: Backend>(seq: Tensor<B, 3>) -> Tensor<B, 2> {
    seq.mean_dim(1).squeeze(1)
}
fn token_support_targets<B: Backend>(
    ptr_index: &Vec<Vec<i32>>,
    to: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let b = ptr_index.len();
    let mut data = vec![0.0f32; b * to];

    for (bi, idxs) in ptr_index.iter().enumerate() {
        if idxs.is_empty() { continue; }
        for &t_i32 in idxs {
            if t_i32 >= 0 {
                let t = t_i32 as usize;
                if t < to { data[bi * to + t] += 1.0; }
            }
        }
    }
    for bi in 0..b {
        let row = &mut data[bi * to .. (bi+1) * to];
        let s: f32 = row.iter().sum();
        if s > 0.0 { for v in row.iter_mut() { *v /= s; } }
    }
    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([b, to])
}

fn kl_attn_loss<B: Backend>(mem_logits: Tensor<B, 2>, gold_dist: Tensor<B, 2>) -> Tensor<B, 1> {
    let logp = burn::tensor::activation::log_softmax(mem_logits, 1); // [B,To]
    let ce = -(gold_dist * logp).sum_dim(1).squeeze(1);              // [B]
    ce
}

/* -------------------- Train step --------------------- */

fn forward_and_loss<B: AutodiffBackend>(
    sys: &RouterCopySystem<B>,
    batch: RouterCopyBatch,
    cfg: &TrainCfg,
    device: &B::Device,
) -> Tensor<B, 1> {
    // ---- embed sequences
    let u_oh = tokens_to_one_hot::<B>(&batch.user_tokens, cfg.vocab_size, device);
    let o_oh = tokens_to_one_hot::<B>(&batch.obs_tokens,  cfg.vocab_size, device);
    let f_oh = tokens_to_one_hot::<B>(&batch.final_tokens, cfg.vocab_size, device);

    let u_emb = embed_seq::<B>(u_oh, &sys.embed); // [B,Tu,384]
    let o_emb = embed_seq::<B>(o_oh, &sys.embed); // [B,To,384]
    let f_emb = embed_seq::<B>(f_oh, &sys.embed); // [B,Tf,384]

    // ---- route heads from pooled user
    let u_pool = mean_over_time::<B>(u_emb.clone()); // [B,384]
    let (op_logits, arg_logits_opt) = sys.model.route_logits(u_pool.clone()); // [B,4], [B,K]?

    // ---- pointer + LM heads (BASE)
    let ptr_logits_base = sys.model.pointer_logits(f_emb.clone(), o_emb.clone()); // [B,Tf,To]
    let final_logits    = sys.model.final_logits(f_emb.clone());                  // [B,Tf,V]

    // ---- memory attention → bias pointer logits
    let mem_logits = sys.model.memory_attn_logits(u_pool.clone(), o_emb.clone()); // [B,To]
    let tf = ptr_logits_base.dims()[1];
    let mem_bias = mem_logits.clone().unsqueeze_dim(1).repeat_dim(1, tf);                 // [B,Tf,To]
    let ptr_logits = ptr_logits_base + mem_bias;                                  // [B,Tf,To]

    // ---- (optional) attention alignment aux loss
    let to = o_emb.dims()[1];
    let gold_attn = token_support_targets::<B>(&batch.ptr_index, to, device);     // [B,To]
    let l_attn    = kl_attn_loss::<B>(mem_logits.clone(), gold_attn);             // [B]
    let w_attn: f32 = 0.10;

    // ---- total loss
    let arg_pack = if cfg.arg_classes > 0 {
        Some((arg_logits_opt.expect("arg head missing"), batch.arg_class_targets.clone(), cfg.arg_classes))
    } else { None };

    let main = total_routercopy_loss::<B>(
        cfg.loss_w,
        op_logits, &batch.op_targets, 4,
        arg_pack,
        ptr_logits, &batch.ptr_index,                // <-- use biased logits
        final_logits, &batch.final_tokens, &batch.copy_mask, cfg.vocab_size,
    ); // [1]

    main + l_attn.mean().mul_scalar(w_attn)
}



/* -------------------- Train loop --------------------- */

pub fn train_routercopy<B: AutodiffBackend>(
    ds: &RouterCopyDataset,
    device: &B::Device,
    cfg: TrainCfg,
) -> anyhow::Result<RouterCopySystem<B>> {
    let mut sys = RouterCopySystem::<B>::new(cfg.vocab_size, cfg.arg_classes, device);
    let mut opt = AdamWConfig::new()
        .with_weight_decay(cfg.weight_decay)
        .init::<B, RouterCopySystem<B>>();

    <B as Backend>::seed(42);

    for step in 1..=cfg.max_steps {
        let batch = ds.sample_batch(cfg.batch_size);
        let loss = forward_and_loss::<B>(&sys, batch, &cfg, device); // [1]

        if step % cfg.log_every == 0 {
            let v = loss.to_data().as_slice::<f32>().unwrap()[0];
            println!("[routercopy step {step}] loss={v:.6}");
        }

        let grads_all = loss.backward();
        let grads = GradientsParams::from_grads(grads_all, &sys);
        sys = opt.step(cfg.lr as f64, sys, grads);

        if cfg.ckpt_path.is_some() && step % cfg.ckpt_every == 0 {
            save_checkpoint(&sys, &cfg, step)?;
        }
    }

    Ok(sys)
}

/* --------------------- Checkpoint -------------------- */

fn save_checkpoint<B: Backend>(
    sys: &RouterCopySystem<B>,
    cfg: &TrainCfg,
    step: usize,
) -> anyhow::Result<()> {
    if let Some(dir) = &cfg.ckpt_path {
        std::fs::create_dir_all(dir)?;
        let path = std::path::PathBuf::from(format!("{}/routercopy_step{:06}.bin", dir, step));
        CompactRecorder::new().record(sys.clone().into_record(), path.clone())?;
        println!("Saved checkpoint: {}", path.display());
    }
    Ok(())
}

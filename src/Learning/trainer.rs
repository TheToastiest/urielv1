// src/Learning/trainer.rs
#![allow(clippy::needless_return)]

use std::path::PathBuf;

use burn::{
    module::Module,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{CompactRecorder, Recorder},
    tensor::{backend::Backend, Tensor},
};
use burn::tensor::backend::AutodiffBackend;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::Learning::dataset::JsonlDataset;
use crate::Learning::loss::{default_weights, total_loss, LossWeights};
use crate::Learning::model::UrielModel;
// 👇 bring your embedder (make sure it #[derive(Module)] and uses Param<Tensor<…>>)
use crate::embedder::embedder::FixedWeightEmbedder;

/* ---------------------- Config ---------------------- */

#[derive(Clone, Debug)]
pub struct TrainCfg {
    pub max_steps: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
    pub num_route_classes: usize,
    pub tau: f32,
    pub lr: f32,
    pub weight_decay: f32,
    pub log_every: usize,
    pub ckpt_every: usize,
    pub ckpt_path: Option<String>,
    pub loss_weights: LossWeights,
}

impl Default for TrainCfg {
    fn default() -> Self {
        Self {
            max_steps: 500_000,
            batch_size: 64,
            seq_len: 256,
            vocab_size: 95,
            num_route_classes: 4,
            tau: 0.07,
            lr: 3e-4,
            weight_decay: 0.01,
            log_every: 100,
            ckpt_every: 5_000,
            ckpt_path: None,
            loss_weights: default_weights(),
        }
    }
}

/* ----------------- System (Model + Embedder) -------- */

// Bundle both so the optimizer can step them together and we can checkpoint once.
#[derive(Module, Debug)]
pub struct UrielSystem<B: Backend> {
    pub embed: FixedWeightEmbedder<B>, // [V -> 384], trainable
    pub model: UrielModel<B>,          // heads on top of [B,384]
}

impl<B: Backend> UrielSystem<B> {
    pub fn new(vocab_size: usize, route_classes: usize, device: &B::Device) -> Self {
        Self {
            embed: FixedWeightEmbedder::new(vocab_size, 384, device),
            model: UrielModel::new(route_classes, device),
        }
    }
}

/* ------------------ Batch utilities ------------------ */

/// tokens [B,T] -> one-hot [B,T,V]  (kept for simplicity; later swap to ID gather)
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

/// one_hot [B,T,V] -> project with trainable embedder -> [B,384]
fn mean_pool_with_embedder<B: Backend>(
    one_hot: Tensor<B, 3>,          // [B,T,V]
    embedder: &FixedWeightEmbedder<B>,
) -> Tensor<B, 2> {
    let [b, t, v] = {
        let d = one_hot.dims();
        [d[0], d[1], d[2]]
    };
    let flat = one_hot.reshape([b * t, v]);      // [B*T, V]
    let proj = embedder.forward(flat);           // [B*T, 384]
    proj.reshape([b, t, 384]).mean_dim(1).squeeze(1) // [B, 384]
}

/* -------------------- Train step --------------------- */

/// One training step -> scalar loss [1]
fn train_step<B: AutodiffBackend>(
    sys: &UrielSystem<B>,
    ds: &JsonlDataset,
    cfg: &TrainCfg,
    device: &B::Device,
) -> Tensor<B, 1> {
    // 1) sample batch
    let (q_tok, p_tok, sals, routes, cavs) = ds.sample_batch(cfg.batch_size);
    let b = routes.len();

    // Basic guards
    let max_q = q_tok.iter().flatten().copied().max().unwrap_or(0);
    let max_p = p_tok.iter().flatten().copied().max().unwrap_or(0);
    assert!(
        max_q < cfg.vocab_size && max_p < cfg.vocab_size,
        "Token id out of range: max_q={max_q}, max_p={max_p}, vocab_size={}",
        cfg.vocab_size
    );
    if b == 0 {
        return Tensor::<B, 1>::from_floats([0.0f32], device);
    }

    // 2) encode tokens via trainable embedder
    let q_oh = tokens_to_one_hot::<B>(&q_tok, cfg.vocab_size, device); // [B,T,V]
    let p_oh = tokens_to_one_hot::<B>(&p_tok, cfg.vocab_size, device); // [B,T,V]
    let x_q = mean_pool_with_embedder::<B>(q_oh, &sys.embed);          // [B,384]
    let x_p = mean_pool_with_embedder::<B>(p_oh, &sys.embed);          // [B,384]

    // 3) forward (paired) through heads
    let (z_q, sal_pred, route_logits, cav_pred) = sys.model.forward(x_q);
    let (z_p, _sal2, _rlg2, _cav2) = sys.model.forward(x_p);

    // 4) targets -> tensors (create 1D then reshape using actual b)
    let sal_vec: Vec<f32> = sals.iter().flat_map(|a| a.iter().copied()).collect();
    let sal_tgt = Tensor::<B, 1>::from_floats(sal_vec.as_slice(), device).reshape([b, 4]);

    let cav_vec: Vec<f32> = cavs.iter().flat_map(|a| a.iter().copied()).collect();
    let cav_tgt = Tensor::<B, 1>::from_floats(cav_vec.as_slice(), device).reshape([b, 11]);

    // 5) total multi-task loss
    total_loss(
        z_q,
        z_p,
        cfg.tau,
        sal_pred,
        sal_tgt,
        route_logits,
        &routes,
        cfg.num_route_classes,
        cav_pred,
        cav_tgt,
        cfg.loss_weights,
    )
}

/* -------------------- Train loop --------------------- */

/// Backend-agnostic (CPU/GPU) trainer. Choose backend in your harness:
///   type B = burn::backend::Autodiff<burn::backend::Wgpu>; // GPU (WebGPU/Vulkan/Metal)
///   // or: type B = burn::backend::Autodiff<burn::backend::Cuda>; // NVIDIA CUDA
pub fn train_pilot<B: AutodiffBackend>(
    ds: &JsonlDataset,
    device: &B::Device,
    cfg: TrainCfg,
) -> anyhow::Result<UrielSystem<B>> {
    // 1) model system (trainable embedder + heads)
    let mut sys = UrielSystem::<B>::new(cfg.vocab_size, cfg.num_route_classes, device);

    // 2) optimizer (steps both embedder and model because sys is a Module)
    let mut opt = AdamWConfig::new()
        .with_weight_decay(cfg.weight_decay)
        .init::<B, UrielSystem<B>>();

    // (Optional) seed for reproducibility across backends
    <B as Backend>::seed(42);

    // 3) loop
    for step in 1..=cfg.max_steps {
        let loss = train_step::<B>(&sys, ds, &cfg, device); // [1]

        if step % cfg.log_every == 0 {
            let v = loss.to_data().as_slice::<f32>().unwrap()[0];
            println!("[step {step}] loss={v:.6}");
        }

        // Backprop → grads → map to module params → step with LR
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
    sys: &UrielSystem<B>,
    cfg: &TrainCfg,
    step: usize,
) -> anyhow::Result<()> {
    if let Some(dir) = &cfg.ckpt_path {
        std::fs::create_dir_all(dir)?;
        let path = PathBuf::from(format!("{}/uriel_system_step{:06}.bin", dir, step));
        CompactRecorder::new().record(sys.clone().into_record(), path.clone())?;
        println!("Saved checkpoint: {}", path.display());
    }
    Ok(())
}

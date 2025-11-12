// training/trainer.rs
use anyhow::Result;
use burn::{
    module::Module,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::CompactRecorder,
    tensor::{backend::Backend, Int, Tensor},
};
use burn::tensor::backend::AutodiffBackend;
use burn_autodiff::Autodiff;
use burn::record::Recorder;

use crate::training::config::TrainConfig;
use crate::training::dataset::{load_router_samples, one_hot_batch, BandMode};
use crate::training::model_router::RouterHead;

pub fn train_router(cfg: &TrainConfig) -> Result<()> {
    // ---- backend device (CUDA Autodiff) ----
    type B = burn_autodiff::Autodiff<burn_cuda::Cuda>;
    let dev = <B as Backend>::Device::new(0);
    <B as Backend>::seed(cfg.seed as u64);

    // ---- data ----
    // Train with HEURISTIC band (deploy-legal), Val with NO band (no leakage).
    let train = load_router_samples(
        &cfg.data.train_glob,
        cfg.train.seq_len,
        cfg.data.min_conf,
        BandMode::Heuristic, // train band source
        0.25,                // band_len_frac (shorter; avoids overdominance)
        0.70,                // band_prob (not every sample)
        cfg.seed,
    )?;
    let val = load_router_samples(
        &cfg.data.val_glob,
        cfg.train.seq_len,
        cfg.data.min_conf,
        BandMode::None, // no band at eval
        0.0,
        0.0,
        cfg.seed,
    )?;

    // label counts
    let (mut tn0, mut tn1, mut tn2, mut tn3) = (0, 0, 0, 0);
    for s in &train {
        match s.label {
            0 => tn0 += 1,
            1 => tn1 += 1,
            2 => tn2 += 1,
            3 => tn3 += 1,
            _ => {}
        }
    }
    let (mut n0, mut n1, mut n2, mut n3) = (0, 0, 0, 0);
    for s in &val {
        match s.label {
            0 => n0 += 1,
            1 => n1 += 1,
            2 => n2 += 1,
            3 => n3 += 1,
            _ => {}
        }
    }
    let maj = *[n0, n1, n2, n3].iter().max().unwrap() as f32 / val.len().max(1) as f32;
    eprintln!(
        "[data] loaded train={}  val={} | train_counts=[{} {} {} {}]  val_counts=[{} {} {} {}] maj≈{:.3}",
        train.len(), val.len(), tn0, tn1, tn2, tn3, n0, n1, n2, n3, maj
    );
    assert!(!train.is_empty(), "No router training samples found.");

    // ---- model + optimizer ----
    let vocab = 96usize;       // 0=PAD, 1..=95 printable ASCII
    let d_in = vocab;
    let d_hidden = 256usize;
    let d_out = 4usize;

    let mut model = RouterHead::<B>::new(d_in, d_hidden, d_out, &dev);

    // track best validation accuracy
    let mut best_acc: f32 = -1.0;
    let mut best_step: usize = 0;

    let mut opt = AdamWConfig::new()
        .with_weight_decay(cfg.train.weight_decay as f32)
        .init::<B, RouterHead<B>>();

    // ---- loop ----
    let mut step = 0usize;
    let mb = cfg.train.micro_batch.max(1);
    let accum = cfg.train.grad_accum.max(1);

    'outer: while step < cfg.train.max_steps {
        // stratified lists for 0/1 (ignore 2/3 if absent)
        let mut idx_dir = Vec::new();
        let mut idx_retr = Vec::new();
        for (i, s) in train.iter().enumerate() {
            match s.label {
                1 => idx_retr.push(i), // retrieve_then_answer
                0 => idx_dir.push(i),  // direct
                _ => {}                // ignore 2/3 in this run
            }
        }
        fastrand::shuffle(&mut idx_dir);
        fastrand::shuffle(&mut idx_retr);

        let mut accum_count = 0usize;
        let mut loss_acc: Option<Tensor<B, 1>> = None;
        let mut running_loss_sum = 0.0f32;
        let mut running_loss_cnt = 0usize;

        let mut pd = 0usize;
        let mut pr = 0usize;
        let retr_per_mb = (mb / 2).max(1);
        let dir_per_mb = mb - retr_per_mb;

        loop {
            if pd >= idx_dir.len() && pr >= idx_retr.len() {
                break;
            }

            // build balanced batch indices
            let mut batch_idx: Vec<usize> = Vec::with_capacity(mb);
            for _ in 0..dir_per_mb {
                if pd < idx_dir.len() {
                    batch_idx.push(idx_dir[pd]);
                    pd += 1;
                }
            }
            for _ in 0..retr_per_mb {
                if pr < idx_retr.len() {
                    batch_idx.push(idx_retr[pr]);
                    pr += 1;
                }
            }
            if batch_idx.is_empty() {
                break;
            }
            fastrand::shuffle(&mut batch_idx);

            // ---- micro-batch ----
            let batch: Vec<_> = batch_idx.iter().map(|&i| train[i].clone()).collect();

            // X: [B,T,V]  Y: [B] (Int i32)
            let x = one_hot_batch::<B>(&batch, vocab, &dev);
            let labels: Vec<i32> = batch.iter().map(|s| s.label as i32).collect();
            let y = Tensor::<B, 1, Int>::from_ints(labels.as_slice(), &dev);

            // optional: quick per-batch label histogram (first few steps)
            if step < 3 {
                let mut c = [0usize; 4];
                for &z in &labels {
                    c[z as usize] += 1;
                }
                eprintln!("[train-batch] labels={:?}", c);
            }

            let logits = model.forward(x); // [B,4]
            debug_assert_eq!(logits.dims()[1], 4, "RouterHead must output [B,4]");

            // ---------- Unweighted CE (balanced minibatches already handle class skew) ----------
            let ll = burn::tensor::activation::log_softmax(logits.clone(), 1); // [B,4]

            // build one-hot labels [B,4]
            let bsz = labels.len();
            let mut yoh_host = vec![0.0f32; bsz * 4];
            for (i, &c) in labels.iter().enumerate() {
                yoh_host[i * 4 + (c as usize)] = 1.0;
            }
            let y_oh = Tensor::<B, 1>::from_floats(yoh_host.as_slice(), &dev).reshape([bsz, 4]); // [B,4]

            // negative log-likelihood per sample
            let nll_b1 = (-(ll * y_oh)).sum_dim(1); // [B,1]
            let nll_b = nll_b1.squeeze::<1>(1);     // [B]

            // mean loss (NO class weights)
            let loss = nll_b.mean(); // [1]
            // ------------------------------------------------------------------------------------

            // logging
            let loss_val = *loss
                .clone()
                .to_data()
                .as_slice::<f32>()
                .unwrap()
                .first()
                .unwrap();
            running_loss_sum += loss_val;
            running_loss_cnt += 1;

            // grad accumulation
            loss_acc = Some(match loss_acc {
                None => loss,
                Some(prev) => prev + loss,
            });
            accum_count += 1;

            if accum_count == accum {
                // scale accumulated loss, backward, step
                let scale = 1.0f32 / (accum as f32);
                let loss_scaled =
                    loss_acc.take().unwrap() * Tensor::<B, 1>::from_floats([scale], &dev);

                let grads_all = loss_scaled.backward();
                let grads = GradientsParams::from_grads(grads_all, &model);

                // tiny LR schedule to reduce late drift
                let mut lr = cfg.train.lr as f64;
                if step >= 250 {
                    lr *= 0.5;
                }

                model = opt.step(lr, model, grads);

                accum_count = 0;
                step += 1;

                if step % cfg.train.log_every == 0 {
                    let avg = running_loss_sum / (running_loss_cnt as f32);
                    eprintln!("[step {}] loss={:.4}", step, avg);
                    running_loss_sum = 0.0;
                    running_loss_cnt = 0;
                }
                if step % cfg.train.eval_every == 0 {
                    let accv = eval_router::<B>(&model, &val, vocab, cfg.train.seq_len, &dev);

                    if accv > best_acc {
                        best_acc = accv;
                        best_step = step;
                        // save a checkpoint whenever we get a new best
                        let _ = save_ckpt(&model, &cfg.checkpoints.dir, step);
                        eprintln!("[best] acc={:.3} at step {} (ckpt saved)", best_acc, best_step);
                    }

                    if accv >= 0.999 {
                        eprintln!("[early-stop] perfect acc reached at step {}", step);
                        break 'outer;
                    }
                }

                if step % cfg.train.checkpoint_every.max(usize::MAX / 2) == 0 {
                    save_ckpt(&model, &cfg.checkpoints.dir, step)?;
                }
                if step >= cfg.train.max_steps {
                    break 'outer;
                }
            }
        }
    }

    eprintln!(
        "[final] best acc={:.3} at step {} (ckpt saved there)",
        best_acc, best_step
    );
    Ok(())
}

fn eval_router<B: Backend>(
    model: &crate::training::model_router::RouterHead<B>,
    val: &Vec<crate::training::dataset::RouterSample>,
    vocab: usize,
    _seq_len: usize,
    dev: &B::Device,
) -> f32 {
    if val.is_empty() {
        eprintln!("[eval] (skipped) val=0");
        return 0.0;
    }
    let mut conf = [[0usize; 4]; 4]; // [gt][pred]
    let mut total = 0usize;

    for chunk in val.chunks(64) {
        let x = crate::training::dataset::one_hot_batch::<B>(chunk, vocab, dev);
        let logits = model.forward(x);
        let preds: Vec<i32> = logits
            .argmax(1)
            .to_data()
            .as_slice::<i32>()
            .expect("preds i32")
            .to_vec();

        if total == 0 {
            let mut pc = [0usize; 4];
            for &p in &preds { pc[p as usize] += 1; }
            eprintln!("[eval] pred_hist={:?}", pc);
        }

        for (i, &p) in preds.iter().enumerate() {
            let gt = chunk[i].label;
            let pr = (p as usize).min(3);
            conf[gt][pr] += 1;
            total += 1;
        }
    }
    let correct = conf[0][0] + conf[1][1] + conf[2][2] + conf[3][3];
    let acc = if total > 0 { correct as f32 / total as f32 } else { 0.0 };

    eprintln!(
        "[eval] acc={:.3} n={}  conf=[[{} {} {} {}],[{} {} {} {}],[{} {} {} {}],[{} {} {} {}]]",
        acc, total,
        conf[0][0], conf[0][1], conf[0][2], conf[0][3],
        conf[1][0], conf[1][1], conf[1][2], conf[1][3],
        conf[2][0], conf[2][1], conf[2][2], conf[2][3],
        conf[3][0], conf[3][1], conf[3][2], conf[3][3],
    );

    acc
}

fn save_ckpt<B: Backend>(model: &RouterHead<B>, dir: &str, step: usize) -> Result<()> {
    if dir.is_empty() { return Ok(()); }
    std::fs::create_dir_all(dir)?;
    let path = std::path::PathBuf::from(format!("{}/router_head_step{:06}.bin", dir, step));
    CompactRecorder::new().record(model.clone().into_record(), path.clone())?;
    eprintln!("[ckpt] saved {}", path.display());
    Ok(())
}

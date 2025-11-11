use anyhow::Result;
use clap::Parser;
use burn::{
    module::Module,                     // <-- brings load_record/into_record into scope
    record::{CompactRecorder, Recorder},// <-- Recorder trait for .load(...)
    tensor::backend::Backend,
};
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

use URIELV1::Learning::routercopy::dataset::{RouterCopyBatch, RouterCopyDataset};
use URIELV1::Learning::routercopy::trainer::RouterCopySystem;
use URIELV1::tokenizer::tokenizer::{Vocab, EncodingMode};

#[derive(Parser, Debug)]
struct Args {
    /// Dataset folder containing dev.jsonl
    #[arg(long, default_value = "data/datasets/router_v1")]
    data: String,
    /// Checkpoint to load (from train)
    #[arg(long)]
    ckpt: String,
    /// Output calib file
    #[arg(long, default_value = "data/runs/router_v1/calib.json")]
    out: String,

    /// Eval samples (will sample repeatedly if fewer rows)
    #[arg(long, default_value_t = 4096)]
    samples: usize,

    /// Batch size for eval
    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    /// Max tokens – must match training
    #[arg(long, default_value_t = 128)]
    max_user: usize,
    #[arg(long, default_value_t = 512)]
    max_obs: usize,
    #[arg(long, default_value_t = 64)]
    max_final: usize,
}

type AD = Autodiff<NdArray<f32>>;

fn build_vocab() -> Vocab {
    Vocab::new(EncodingMode::Readable)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let vocab = build_vocab();

    // dataset (dev)
    let dev_path = format!("{}/dev.jsonl", &args.data);
    let ds = RouterCopyDataset::load(&dev_path, vocab, args.max_user, args.max_obs, args.max_final)?;

    // load system weights into a fresh instance (shapes must match training)
    let device = <AD as Backend>::Device::default();
    let record = CompactRecorder::new().load(std::path::PathBuf::from(&args.ckpt), &device)?;
    let arg_classes = 6; // 🔧 keep in sync with training cfg
    let mut sys = RouterCopySystem::<AD>::new(ds.vocab_size(), arg_classes, &device);
    sys = sys.load_record(record);

    // collect logits/labels
    let mut op_logits_all: Vec<f32> = Vec::new(); // flattened [N*4]
    let mut op_labels: Vec<usize> = Vec::new();

    let mut seen = 0usize;
    while seen < args.samples {
        let batch = ds.sample_batch(args.batch_size);
        let (op_logits, labels) = forward_op_logits::<AD>(&sys, &batch, &device)?;
        op_logits_all.extend(op_logits);
        op_labels.extend(labels.clone());
        seen += labels.len();
    }

    // temperature scaling for op
    let t_best = grid_search_temperature(&op_logits_all, &op_labels, 4, 0.5, 5.0, 0.05);
    let (ece, brier) = ece_brier(&op_logits_all, &op_labels, 4, t_best, 15);

    // write calib.json
    let calib = serde_json::json!({
        "op": { "T": t_best, "ece": ece, "brier": brier },
        "thresholds": { "promote": 0.60, "refuse": 0.40 }
    });
    if let Some(parent) = std::path::Path::new(&args.out).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&calib)?)?;
    println!("calibration written to {}", &args.out);
    println!("T_op={:.3}  ECE={:.4}  Brier={:.4}", t_best, ece, brier);
    Ok(())
}

/* ---------- local helpers (avoid private imports) ---------- */

use burn::tensor::Tensor;

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

fn embed_seq<B: Backend>(
    one_hot: Tensor<B, 3>,
    embedder: &URIELV1::embedder::embedder::FixedWeightEmbedder<B>,
) -> Tensor<B, 3> {
    let [b, t, v] = {
        let d = one_hot.dims();
        [d[0], d[1], d[2]]
    };
    let flat = one_hot.reshape([b * t, v]); // [B*T, V]
    let proj = embedder.forward(flat);      // [B*T, 384]
    proj.reshape([b, t, 384])               // [B, T, 384]
}

fn mean_over_time<B: Backend>(seq: Tensor<B, 3>) -> Tensor<B, 2> {
    seq.mean_dim(1).squeeze(1)
}

fn forward_op_logits<B: Backend>(
    sys: &RouterCopySystem<B>,
    batch: &RouterCopyBatch,
    device: &B::Device,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let vocab_size = sys.model.vocab_size();
    let u_oh = tokens_to_one_hot::<B>(&batch.user_tokens, vocab_size, device);
    let u_emb = embed_seq::<B>(u_oh, &sys.embed);
    let u_pool = mean_over_time::<B>(u_emb);
    let (op_logits, _arg) = sys.model.route_logits(u_pool); // [B,4]
    let slice = op_logits.to_data().as_slice::<f32>().unwrap().to_vec();
    Ok((slice, batch.op_targets.clone()))
}

/* ---------- temp scaling + metrics ---------- */

fn grid_search_temperature(
    flat_logits: &Vec<f32>,
    labels: &Vec<usize>,
    classes: usize,
    t_min: f32,
    t_max: f32,
    step: f32,
) -> f32 {
    let mut best_t = 1.0;
    let mut best_nll = f32::INFINITY;
    let mut t = t_min;
    while t <= t_max + 1e-6 {
        let nll = avg_nll(flat_logits, labels, classes, t);
        if nll < best_nll {
            best_nll = nll;
            best_t = t;
        }
        t += step;
    }
    best_t
}

fn avg_nll(flat_logits: &Vec<f32>, labels: &Vec<usize>, classes: usize, t: f32) -> f32 {
    let b = labels.len();
    let mut sum = 0.0f32;
    for i in 0..b {
        let row = &flat_logits[i * classes..(i + 1) * classes];
        let mut maxv = f32::NEG_INFINITY;
        for &v in row {
            if v / t > maxv {
                maxv = v / t;
            }
        }
        let mut den = 0.0f32;
        for &v in row {
            den += (v / t - maxv).exp();
        }
        let logp = (row[labels[i]] / t - maxv).exp().ln() - den.ln();
        sum += -logp;
    }
    sum / b as f32
}

fn ece_brier(
    flat_logits: &Vec<f32>,
    labels: &Vec<usize>,
    classes: usize,
    t: f32,
    bins: usize,
) -> (f32, f32) {
    let b = labels.len();
    let mut ece_sum = 0.0f32;
    let mut brier_sum = 0.0f32;

    let mut counts = vec![0usize; bins];
    let mut conf_sum = vec![0.0f32; bins];
    let mut acc_sum = vec![0.0f32; bins];

    for i in 0..b {
        let row = &flat_logits[i * classes..(i + 1) * classes];

        // softmax/T
        let mut maxv = f32::NEG_INFINITY;
        for &v in row {
            if v / t > maxv {
                maxv = v / t;
            }
        }
        let mut den = 0.0f32;
        let mut probs = vec![0.0f32; classes];
        for (j, &v) in row.iter().enumerate() {
            let e = (v / t - maxv).exp();
            probs[j] = e;
            den += e;
        }
        for j in 0..classes {
            probs[j] /= den;
        }

        let (pred, conf) = probs.iter().enumerate().fold((0usize, -1.0f32), |acc, (j, p)| {
            if *p > acc.1 { (j, *p) } else { acc }
        });
        let acc = if pred == labels[i] { 1.0 } else { 0.0 };

        let bin = ((conf * bins as f32).floor() as usize).min(bins - 1);
        counts[bin] += 1;
        conf_sum[bin] += conf;
        acc_sum[bin] += acc;

        // Brier
        let mut brier = 0.0f32;
        for j in 0..classes {
            let y = if j == labels[i] { 1.0 } else { 0.0 };
            brier += (probs[j] - y) * (probs[j] - y);
        }
        brier_sum += brier;
    }

    for k in 0..bins {
        if counts[k] > 0 {
            let avg_conf = conf_sum[k] / counts[k] as f32;
            let avg_acc = acc_sum[k] / counts[k] as f32;
            ece_sum += (counts[k] as f32 / b as f32) * (avg_conf - avg_acc).abs();
        }
    }
    (ece_sum, brier_sum / b as f32)
}

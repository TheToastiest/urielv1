use anyhow::Result;
use clap::Parser;
use burn::tensor::backend::Backend;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

use URIELV1::Learning::routercopy::dataset::RouterCopyDataset;
use URIELV1::Learning::routercopy::trainer::{TrainCfg, train_routercopy};
use URIELV1::Learning::routercopy::trainer::RouterCopySystem;
use URIELV1::tokenizer::tokenizer::{Vocab, EncodingMode}; // uses your existing Vocab type
fn build_vocab() -> Vocab {
    Vocab::new(EncodingMode::Readable)
}
#[derive(Parser, Debug)]
struct Args {
    /// Dataset folder containing train.jsonl
    #[arg(long, default_value="data/datasets/router_v1")]
    data: String,
    /// Runs/checkpoints folder
    #[arg(long, default_value="data/runs/router_v1")]
    runs: String,

    /// Max tokens (caps) – must match exporter expectations
    #[arg(long, default_value_t=128)]
    max_user: usize,
    #[arg(long, default_value_t=512)]
    max_obs: usize,
    #[arg(long, default_value_t=64)]
    max_final: usize,

    /// Training steps and batch size
    #[arg(long, default_value_t=20000)]
    steps: usize,
    #[arg(long, default_value_t=32)]
    batch_size: usize,

    /// Learning rate / weight decay
    #[arg(long, default_value_t=3e-4)]
    lr: f32,
    #[arg(long, default_value_t=0.01)]
    wd: f32,
}

type AD = Autodiff<NdArray<f32>>;

fn main() -> Result<()> {
    let args = Args::parse();

    // --- vocab: use your existing default constructor or adjust as needed
    let vocab = build_vocab();

    // --- dataset
    let train_path = format!("{}/train.jsonl", &args.data);
    let ds = RouterCopyDataset::load(&train_path, vocab, args.max_user, args.max_obs, args.max_final)?;

    // --- config
    let mut cfg = TrainCfg::default();
    cfg.max_steps = args.steps;
    cfg.batch_size = args.batch_size;
    cfg.max_user = args.max_user;
    cfg.max_obs = args.max_obs;
    cfg.max_final = args.max_final;
    cfg.lr = args.lr;
    cfg.weight_decay = args.wd;
    cfg.ckpt_path = Some(args.runs.clone());
    // If you don't have arg labels yet, set to 0 to disable arg head:
    // cfg.arg_classes = 0;

    // --- device
    let device = <AD as Backend>::Device::default();

    // --- train
    let _sys: RouterCopySystem<AD> = train_routercopy::<AD>(&ds, &device, cfg)?;

    Ok(())
}

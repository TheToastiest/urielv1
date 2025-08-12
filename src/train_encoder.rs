use anyhow::Result;
// Use the decorator type from the crate directly:
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

type TB = Autodiff<NdArray<f32>>; // CPU autodiff backend

use burn::tensor::backend::Backend;


// NOTE: crate names are lowercase in Rust. If your Cargo.toml says `name = "urielv1"`,
// then the correct path is `urielv1::...` (not URIELV1).
use URIELV1::Learning::trainer::{train_pilot, TrainCfg};
use URIELV1::Learning::dataset::JsonlDataset;
use URIELV1::tokenizer::tokenizer::{EncodingMode, Vocab};

fn main() -> Result<()> {
    // Concrete device from the backend
    let device = <TB as Backend>::Device::default();

    // Vocab with explicit mode
    let vocab = Vocab::new(EncodingMode::Hybrid);

    // max_len must match cfg.seq_len
    let ds = JsonlDataset::load("./training_data.jsonl", 256, vocab)?;

    let mut cfg = TrainCfg::default();
    cfg.ckpt_path = Some("R:/uriel/models/checkpoints".into());
    cfg.vocab_size = 95;        // your charset size
    cfg.num_route_classes = 4;  // Respond/Search/Store/Defer etc.
    cfg.seq_len = 256;

    // Train with NdArray backend (which implements AutodiffBackend in your build)
    let _model = train_pilot::<TB>(&ds, &device, cfg)?;
    Ok(())
}
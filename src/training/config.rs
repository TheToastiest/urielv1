// training/config.rs
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct TrainConfig {
    pub task: String,            // "router" | "encoder" | "sft"
    pub run_id: String,
    pub device: String,          // "cuda:0"
    pub precision: String,       // "bf16" | "fp16"
    pub seed: u64,

    pub model: ModelCfg,
    pub train: TrainKnobs,
    pub data: DataCfg,
    pub checkpoints: CkptCfg,
    pub export: ExportCfg,
}
#[derive(Deserialize, Debug, Clone)]
pub struct ModelCfg { pub name: String }

#[derive(Deserialize, Debug, Clone)]
pub struct TrainKnobs {
    pub seq_len: usize,
    pub micro_batch: usize,
    pub grad_accum: usize,
    pub lr: f64,
    pub warmup_steps: usize,
    pub scheduler: String,
    pub weight_decay: f64,
    pub max_steps: usize,
    pub log_every: usize,
    pub eval_every: usize,
    pub checkpoint_every: usize,
}

#[derive(Deserialize, Debug, Clone)]
pub struct DataCfg {
    pub train_glob: String,
    pub val_glob: String,
    pub bucket_boundaries: Vec<usize>,
    pub pack_sequences: bool,
    pub num_workers: usize,
    pub dedup: bool,
    pub min_conf: f32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct CkptCfg { pub dir: String, pub keep_last: usize, pub keep_best: usize }
#[derive(Deserialize, Debug, Clone)]
pub struct ExportCfg { pub onnx: bool, pub quantize: String }

pub fn load_config(path: &str) -> anyhow::Result<TrainConfig> {
    let txt = std::fs::read_to_string(path)?;
    Ok(toml::from_str::<TrainConfig>(&txt)?)
}

// training/manager.rs
use anyhow::Result;
use crate::training::config::load_config;

pub fn run(config_path: &str) -> Result<()> {
    let cfg = load_config(config_path)?;
    match cfg.task.as_str() {
        "router" => crate::training::trainer::train_router(&cfg),
        other => anyhow::bail!("Task '{}' not implemented yet", other),
    }
}

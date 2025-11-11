// src/bin/uriel-train.rs
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let cfg_path = args.next().unwrap_or_else(|| "configs/router-dev.toml".into());
    URIELV1::training::manager::run(&cfg_path)
}

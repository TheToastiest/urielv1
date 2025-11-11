// URIELV1/src/soarus/mod.rs
use anyhow::Result;
use crossbeam_channel::{Sender, Receiver};
use nalgebra::Isometry3;
use serde::{Deserialize, Serialize};
use soarus_bridge::{IcpConfig as BrCfg, IcpMode as BrMode, IcpResult};
use soarus_core::Cloud;
use soarus_io::read_auto;
use std::fmt;
// ---------- public URIEL-facing types ----------

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum SoarusIcpMode { PointToPoint, PointToPlane }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoarusIcpConfig {
    pub mode: SoarusIcpMode,
    pub max_corr: f32,            // meters
    pub iters: usize,             // per level
    pub trim: f32,                // 0..0.5
    pub pyramid: Option<Vec<f32>>,// e.g., [0.01, 0.005, 0.002]
    // pt2plane-only
    pub normals_r_scale: f32,     // radius = scale * voxel
    pub max_plane_dist: f32,      // |n·(ps-q)| cutoff
    pub huber_delta: f32,         // robust kernel delta
}

impl Default for SoarusIcpConfig {
    fn default() -> Self {
        Self {
            mode: SoarusIcpMode::PointToPlane,
            max_corr: 0.02, iters: 30, trim: 0.20,
            pyramid: Some(vec![0.01, 0.005, 0.002]),
            normals_r_scale: 1.5,
            max_plane_dist: 0.016,
            huber_delta: 0.01,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoarusIcpOutput {
    pub pose: Isometry3<f32>,     // src -> tgt
    pub rmse: f32,
    pub p99: f32,
    pub inliers: usize,
}

// ---------- sync API (simple call sites) ----------

pub fn icp_register_paths(src_path: &str, tgt_path: &str, cfg: &SoarusIcpConfig) -> Result<SoarusIcpOutput> {
    let src = read_auto(src_path)?;
    let tgt = read_auto(tgt_path)?;
    icp_register_clouds(&src, &tgt, cfg)
}

pub fn icp_register_clouds(src: &Cloud, tgt: &Cloud, cfg: &SoarusIcpConfig) -> Result<SoarusIcpOutput> {
    let out: IcpResult = soarus_bridge::icp_register(src, tgt, map_cfg(cfg))?;
    Ok(SoarusIcpOutput {
        pose: out.pose,
        rmse: out.metrics.rmse,
        p99:  out.metrics.p99,
        inliers: out.metrics.inliers,
    })
}

fn map_cfg(c: &SoarusIcpConfig) -> BrCfg {
    BrCfg {
        mode: match c.mode {
            SoarusIcpMode::PointToPoint => BrMode::PointToPoint,
            SoarusIcpMode::PointToPlane => BrMode::PointToPlane,
        },
        max_corr: c.max_corr,
        iters: c.iters,
        trim: c.trim,
        pyramid: c.pyramid.clone(),
        normals_r_scale: c.normals_r_scale,
        max_plane_dist: c.max_plane_dist,
        huber_delta: c.huber_delta,
    }
}

// ---------- background worker (non-blocking) ----------

#[derive(Clone)]
pub enum SoarusJob {
    RegisterPaths { src_path: String, tgt_path: String, cfg: SoarusIcpConfig },
    RegisterClouds { src: soarus_core::Cloud, tgt: soarus_core::Cloud, cfg: SoarusIcpConfig },
}

impl fmt::Debug for SoarusJob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SoarusJob::RegisterPaths { src_path, tgt_path, .. } => {
                f.debug_struct("RegisterPaths")
                    .field("src_path", src_path)
                    .field("tgt_path", tgt_path)
                    .finish()
            }
            SoarusJob::RegisterClouds { src, tgt, .. } => {
                f.debug_struct("RegisterClouds")
                    .field("src_len", &src.len())
                    .field("tgt_len", &tgt.len())
                    .finish()
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum SoarusMsg {
    Done(SoarusIcpOutput),
    Err(String),
}

/// Spawn a Soarus worker thread. Returns (tx_jobs, rx_results).
pub fn start_worker(buffer: usize) -> (Sender<SoarusJob>, Receiver<SoarusMsg>) {
    let (tx_job, rx_job) = crossbeam_channel::bounded::<SoarusJob>(buffer);
    let (tx_out, rx_out) = crossbeam_channel::bounded::<SoarusMsg>(buffer);

    std::thread::Builder::new()
        .name("soarus_worker".into())
        .spawn({
            let tx_out = tx_out.clone();
            move || {
                while let Ok(job) = rx_job.recv() {
                    let res = match job {
                        SoarusJob::RegisterPaths { src_path, tgt_path, cfg } =>
                            icp_register_paths(&src_path, &tgt_path, &cfg),
                        SoarusJob::RegisterClouds { src, tgt, cfg } =>
                            icp_register_clouds(&src, &tgt, &cfg),
                    };
                    let _ = match res {
                        Ok(out) => tx_out.send(SoarusMsg::Done(out)),
                        Err(e)  => tx_out.send(SoarusMsg::Err(format!("{e:#}"))),
                    };
                }
            }
        })
        .expect("spawn soarus_worker");

    (tx_job, rx_out)
}

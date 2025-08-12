pub mod parietal;
pub mod infer_route;
pub mod route_cfg;
pub mod parietal_adapter;

// Re-export so other modules can just use `crate::parietal::{...}`
pub use parietal::{SalienceScore, ParietalRoute};

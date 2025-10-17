pub mod parietal;
pub mod Hippocampus;
pub mod Microthalamus;
pub mod callosum;
pub mod Temporal;
pub mod PFC;
pub mod tokenizer;
pub mod embedder;
pub mod core;
pub mod Inference;
pub mod Learning;
pub mod Frontal;
pub mod policy;

pub mod concepts_and_contexts;

pub use crate::parietal::{SalienceScore, ParietalRoute};
//pub use Hippocampus::pipeline::{store_embedding_sqlite,load_embedding_sqlite};
pub use Hippocampus::{
    cache_semantic_memory,
    generate_embedding_from_text,
    ingest,
    recall_recent_entries,
    start_scanner,
    stop_scanner,
    SmieError,
};

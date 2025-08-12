pub mod Hippocampus;
pub mod Microthalamus;
pub mod parietal;
pub mod Temporal;
pub mod PFC;
pub mod tokenizer;
pub mod embedder;
pub mod core;
pub mod Inference;
pub mod Learning;
pub mod callosum;

pub mod Frontal;

pub use Hippocampus::{
    cache_semantic_memory,
    generate_embedding_from_text,
    ingest_text,
    load_embedding_for_key,
    recall_recent_entries,
    start_scanner,
    stop_scanner,
    store_embedding_for_key,
    SmieError,
};

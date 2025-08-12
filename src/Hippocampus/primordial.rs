// src/Hippocampus/primordial.rs
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Primordial {
    pub version: u32,
    pub axioms: Vec<String>,
    pub prohibitions: Vec<String>,
    pub drives: Vec<DriveDef>,
    pub strategies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriveDef {
    pub name: String,
    pub weight: f32,
    pub description: String,
}

// 1) Canonical blob compiled into the binary (immutable at runtime)
pub static PRIMORDIAL_BLOB: &[u8] = include_bytes!("primordial.json"); // keep committed to repo

// 2) Canonical SHA-256 of the blob, also compiled into the binary
pub const PRIMORDIAL_SHA256: &str = "REPLACE_WITH_ACTUAL_SHA256_OF_JSON";

// 3) Parsed, validated, ready-to-use singleton
pub static PRIMORDIAL: Lazy<Primordial> = Lazy::new(|| {
    use sha2::{Digest, Sha256};

    // integrity check
    let mut hasher = Sha256::new();
    hasher.update(PRIMORDIAL_BLOB);
    let digest = format!("{:x}", hasher.finalize());
    if digest != PRIMORDIAL_SHA256 {
        // This is your lobotomy guard:
        panic!(
            "Primordial integrity failure. Expected {}, got {}. System halting.",
            PRIMORDIAL_SHA256, digest
        );
    }

    // parse
    serde_json::from_slice::<Primordial>(PRIMORDIAL_BLOB)
        .expect("Primordial parse failed. System halting.")
});

// Public accessor (never expose mutable access)
pub fn primordial() -> &'static Primordial {
    &PRIMORDIAL
}

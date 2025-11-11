// src/concepts_and_contexts/speech_concepts.rs
use once_cell::sync::OnceCell;

/// Public entry: keep the same signature so you don’t have to touch callers.
pub fn realize_concept(key: &str, val: &str) -> String {
    let style = copula_style_from_env(); // "I’m" by default; set URIEL_COPULA=long for "I am"

    match key {
        // Identity: keep it crisp; no “This means…”
        "name" | "you are" => {
            let who = clean_tail(val);
            format!("{} {}.", copula(style), who)
        }

        "creator"     => format!("My creator is {}.", clean_tail(val)),
        "mentor"      => format!("My mentor is {}.", clean_tail(val)),

        // Codes should remain raw, no punctuation enforcement
        "launch code" => val.trim().to_string(),

        // Add a little variety without randomness (stable per value)
        "purpose"     => realize_purpose(style, val),

        // Fallback
        _             => clean_tail(val),
    }
}

/* ── helpers ─────────────────────────────────────────────────────────────── */

#[derive(Copy, Clone)]
enum CopulaStyle { Long, Contracted }

static COPULA_OVERRIDE: OnceCell<CopulaStyle> = OnceCell::new();

pub fn set_copula_style_long(is_long: bool) {
    let _ = COPULA_OVERRIDE.set(if is_long { CopulaStyle::Long } else { CopulaStyle::Contracted });
}

#[inline]
fn copula(style: CopulaStyle) -> &'static str {
    match style { CopulaStyle::Long => "I am", CopulaStyle::Contracted => "I’m" }
}

#[inline]
fn copula_style_from_env() -> CopulaStyle {
    if let Some(s) = COPULA_OVERRIDE.get() { return *s; }
    match std::env::var("URIEL_COPULA").ok().as_deref() {
        Some("long") => CopulaStyle::Long,
        _            => CopulaStyle::Contracted,
    }
}

/// Trim trailing punctuation; keep spacing clean.
#[inline]
fn clean_tail(v: &str) -> String {
    v.trim().trim_end_matches(|c: char| c.is_ascii_punctuation()).to_string()
}

/// Deterministic “purpose” phrasing + micro synonym spin.
fn realize_purpose(style: CopulaStyle, val: &str) -> String {
    // If the value already starts with "to …", don’t duplicate "to".
    let v_full = clean_tail(val);
    let bare   = v_full.strip_prefix("to ").map(|s| s.trim()).unwrap_or(v_full.as_str());

    match det_pick(&v_full, 3) {
        0 => format!("My purpose is {}.", v_full),                  // “My purpose is to …”
        1 => format!("{} here to {}.", copula(style), spin_verbs(bare)),
        _ => format!("I exist to {}.",            spin_verbs(bare)),
    }
}

/// Tiny, deterministic lexical spin so repeated answers don’t sound identical.
fn spin_verbs(s: &str) -> String {
    let lower = s.to_ascii_lowercase();

    if let Some(rest) = lower.strip_prefix("help ") {
        let alts = ["assist ", "support ", "aid "];
        let pick = det_pick(s, alts.len());
        return format!("{}{}", alts[pick], &s[5..].trim());
    }
    if let Some(rest) = lower.strip_prefix("build ") {
        let alts = ["create ", "construct ", "forge ", "develop "];
        let pick = det_pick(s, alts.len());
        return format!("{}{}", alts[pick], &s[6..].trim());
    }
    // You can add more light spins over time (e.g., “make”, “improve”, …)
    s.to_string()
}

/// Stable template choice (no RNG) so the same input maps to the same surface.
fn det_pick(seed: &str, n: usize) -> usize {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut h);
    (h.finish() % (n as u64)) as usize
}

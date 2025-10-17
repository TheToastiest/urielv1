use std::{collections::HashMap, fs, io::{Read, Write}, path::Path};

const FEAT_DIM: usize = 1 << 16; // 65,536 hashed features
const LR: f32 = 0.1;
const EPS: f32 = 1e-6;

// --- hashing (no external deps) ---
fn hash_bytes(bytes: &[u8]) -> u64 {
    // simple FNV-1a
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
fn hash_u64(mut x: u64) -> usize {
    // splitmix64-ish
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    ((x ^ (x >> 31)) as usize) & (FEAT_DIM - 1)
}
fn feats_char_ngrams(text: &str, nmin: usize, nmax: usize) -> Vec<usize> {
    let s = text.as_bytes();
    let mut out = Vec::new();
    for n in nmin..=nmax {
        if s.len() < n { break; }
        for i in 0..=s.len() - n {
            let h64 = hash_bytes(&s[i..i+n]);
            out.push(hash_u64(h64));
        }
    }
    out
}

// --- model ---
#[derive(Default, Clone)]
pub struct OnlineLogit {
    pub w:  Vec<f32>,
    pub g2: Vec<f32>,
}
impl OnlineLogit {
    pub fn new() -> Self { Self { w: vec![0.0; FEAT_DIM], g2: vec![0.0; FEAT_DIM] } }
    pub fn predict(&self, x: &[usize]) -> f32 {
        let mut z = 0f32; for &i in x { z += self.w[i]; }
        1.0 / (1.0 + (-z).exp())
    }
    pub fn update(&mut self, x: &[usize], y: f32) {
        let p = self.predict(x);
        let err = p - y;
        for &i in x {
            let g = err;
            self.g2[i] += g*g;
            let step = LR / (self.g2[i].sqrt() + EPS);
            self.w[i] -= step * g;
        }
    }
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut f = fs::File::create(path)?;
        f.write_all(bytemuck::cast_slice(&self.w))?;
        f.write_all(bytemuck::cast_slice(&self.g2))?;
        Ok(())
    }
    pub fn load_or_new(path: &str) -> std::io::Result<Self> {
        if !Path::new(path).exists() { return Ok(Self::new()); }
        let mut f = fs::File::open(path)?;
        let mut buf = vec![0u8; FEAT_DIM * 4 * 2];
        f.read_exact(&mut buf)?;
        let (w_bytes, g_bytes) = buf.split_at(FEAT_DIM * 4);
        Ok(Self {
            w:  bytemuck::cast_slice(w_bytes).to_vec(),
            g2: bytemuck::cast_slice(g_bytes).to_vec(),
        })
    }
}

// --- concept learner (keys are owned Strings) ---
pub struct ConceptLearner {
    pub models: HashMap<String, OnlineLogit>, // OWN the key
    dir: String,
}
impl ConceptLearner {
    pub fn new(dir: impl Into<String>) -> Self {
        let dir = dir.into();
        let _ = fs::create_dir_all(&dir);
        Self { models: HashMap::new(), dir }
    }
    fn path(&self, key: &str) -> String {
        format!("{}/{}.bin", self.dir, key.replace(' ', "_"))
    }
    fn ensure_loaded(&mut self, key: &str) {
        if !self.models.contains_key(key) {
            let p = self.path(key);
            let mdl = OnlineLogit::load_or_new(&p).unwrap_or_else(|_| OnlineLogit::new());
            self.models.insert(key.to_string(), mdl);
        }
    }
    pub fn predict(&mut self, key: &str, text: &str) -> f32 {
        self.ensure_loaded(key);
        let x = feats_char_ngrams(text, 3, 5);
        self.models.get(key).unwrap().predict(&x)
    }
    pub fn update(&mut self, key: &str, text: &str, y: f32) {
        self.ensure_loaded(key);

        // Compute anything that needs an immutable &self BEFORE the mutable borrow.
        let path = self.path(key);
        let x = feats_char_ngrams(text, 3, 5);

        // Mutable borrow ends at the end of this block.
        if let Some(m) = self.models.get_mut(key) {
            m.update(&x, y);
            // This uses `m`, not `self`, so it's fine to call here.
            // (We already computed `path` above.)
            let _ = m.save(&path);
        }
    }

}

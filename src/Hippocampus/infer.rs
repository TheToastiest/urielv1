// src/Hippocampus/infer.rs
use burn::{module::Module, nn::{Linear, LinearConfig}, tensor::{activation, backend::Backend, Tensor}};
use burn::record::{CompactRecorder, Recorder};
use std::path::Path;

// ---- Shapes (keep honest) ----
pub const IN_DIM: usize = 384;
pub const OUT_DIM: usize = 256;
impl<B: Backend> HippocampusModel<B> {
    /// Forward pass from a precomputed IN_DIM vector (no embed work in the lock).
    pub fn forward_from_vec(&self, v: &[f32]) -> Tensor<B, 2> {
        assert_eq!(v.len(), IN_DIM, "expected {}, got {}", IN_DIM, v.len());
        let x1 = Tensor::<B, 1>::from_floats(v, &self.device);
        let x2 = x1.reshape([1, IN_DIM]);
        self.model.forward(x2)
    }
}

// ---- Tiny MLP encoder ----
#[derive(Module, Debug)]
pub struct MyModel<B: Backend> {
    fc1: Linear<B>, // 384 -> 512
    fc2: Linear<B>, // 512 -> 256
}

impl<B: Backend> MyModel<B> {
    pub fn new(input: usize, output: usize, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(input, 512).init(device),
            fc2: LinearConfig::new(512, output).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.fc1.forward(x));
        self.fc2.forward(x)
    }
}

    // ---- Stub embedder (returns 384 floats). Swap with your real one later. ----
    // ---- Hashing embedder with caps/strides to avoid stalls ---------------------
    pub fn embed_text(text: &str) -> Vec<f32> {
        const D: usize = IN_DIM;
        const MAX_BYTES: usize = 4096;   // hard cap
        const BIG_BYTES: usize = 1024;   // start striding after this

        #[inline]
        fn fnv1a64(bytes: &[u8]) -> u64 {
            let mut h: u64 = 0xcbf29ce484222325;
            for &b in bytes { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
            h
        }
        #[inline]
        fn bucket(h: u64) -> (usize, f32) {
            let idx = (h % D as u64) as usize;
            let sign = if (h & 1) == 0 { 1.0 } else { -1.0 };
            (idx, sign)
        }

        // Normalize + bound input
        let s = text.to_ascii_lowercase();
        let mut bytes = s.as_bytes();
        if bytes.len() > MAX_BYTES { bytes = &bytes[..MAX_BYTES]; }

        let mut v = vec![0.0f32; D];

        // Choose stride: 1 for short, 2 for medium, 4 for long
        let stride = if bytes.len() > 2*BIG_BYTES { 4 } else if bytes.len() > BIG_BYTES { 2 } else { 1 };

        // 1-grams (always)
        let mut i = 0;
        while i < bytes.len() {
            let (bkt, sgn) = bucket(fnv1a64(&bytes[i..i+1]));
            v[bkt] += sgn;
            i += stride;
        }

        // 2- and 3-grams only for shorter inputs (keeps cost bounded)
        if bytes.len() <= BIG_BYTES {
            let mut i = 0;
            while i + 1 < bytes.len() {
                let (bkt, sgn) = bucket(fnv1a64(&bytes[i..i+2]));
                v[bkt] += 0.5 * sgn;
                i += stride;
            }
            let mut i = 0;
            while i + 2 < bytes.len() {
                let (bkt, sgn) = bucket(fnv1a64(&bytes[i..i+3]));
                v[bkt] += 0.25 * sgn;
                i += stride;
            }
        }

        // L2 normalize
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n = if norm > 1e-6 { norm } else { 1e-6 };
        for x in &mut v { *x /= n; }
        v
    }


// ---- High-level wrapper: load/save + inference ----
#[derive(Debug, Clone)]
pub struct HippocampusModel<B: Backend> {
    pub model: MyModel<B>,
    pub device: B::Device,
}

impl<B: Backend> HippocampusModel<B> {
    /// Build model; if `ckpt` exists, load weights.
    pub fn load_or_init(device: &B::Device, ckpt_path: &str) -> Self {
        let mut model = MyModel::<B>::new(IN_DIM, OUT_DIM, device);

        if Path::new(ckpt_path).exists() {
            // Explicit record type helps type inference on some setups.
            let record: <MyModel<B> as Module<B>>::Record = CompactRecorder::new()
                .load(ckpt_path.into(), device)
                .expect("Failed to load model record");
            model = model.load_record(record);
        } else {
            eprintln!("(hint) No checkpoint at {ckpt_path}; using randomly-initialized weights.");
        }

        Self { model, device: device.clone() }
    }

    /// Save weights to a checkpoint.
    pub fn save(&self, ckpt_path: &str) -> anyhow::Result<()> {
        let record = self.model.clone().into_record();
        CompactRecorder::new().record(record, ckpt_path.into())?;
        Ok(())
    }

    /// Text -> [1,384] -> MLP -> [1,256] tensor.
    pub fn run_inference(&self, text: &str) -> Tensor<B, 2> {
        let v = embed_text(text);
        assert_eq!(v.len(), IN_DIM, "embed_text returned {} dims, expected {}", v.len(), IN_DIM);

        let x1 = Tensor::<B, 1>::from_floats(v.as_slice(), &self.device);
        let x2 = x1.reshape([1, IN_DIM]);
        self.model.forward(x2)
    }

    /// Convenience: returns plain Vec<f32> for downstream code.
    pub fn run_inference_vec(&self, text: &str) -> Vec<f32> {
        self.run_inference(text)
            .to_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }
}

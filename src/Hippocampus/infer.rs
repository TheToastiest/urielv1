// src/Hippocampus/model.rs
use burn::{module::Module, nn::{Linear, LinearConfig}, tensor::{activation, backend::Backend, Tensor}};
use burn::record::{CompactRecorder, Recorder};
use std::path::Path;

// ---- Shapes (keep honest) ----
pub const IN_DIM: usize = 384;
pub const OUT_DIM: usize = 256;

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
pub fn embed_text(_text: &str) -> Vec<f32> {
    vec![0.42; IN_DIM]
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

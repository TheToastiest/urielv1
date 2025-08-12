// src/Hippocampus/infer.rs (or model_infer.rs)
use burn::module::Module;
use std::path::Path;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::{backend::Backend, Tensor};

use crate::Hippocampus::burn_model::MyModel; // your 384->256 MLP
use crate::Hippocampus::burn_model::embed_text; // returns Vec<f32> length 384

// Keep constants explicit so shapes stay honest.
const IN_DIM: usize = 384;   // matches embed_text() length
const OUT_DIM: usize = 256;  // MyModel output

#[derive(Debug, Clone)]
pub struct HippocampusModel<B: Backend> {
    pub model: MyModel<B>,
    pub device: B::Device,
}

impl<B: Backend> HippocampusModel<B> {
    pub fn load(device: &B::Device, model_path: &str) -> Self {
        // 1) Build a fresh model with the right shapes
        let mut model = MyModel::<B>::new(IN_DIM, OUT_DIM, device);

        // 2) If a checkpoint exists, load the record and apply it to the model
        if Path::new(model_path).exists() {
            // CompactRecorder::load returns the *record*, not the model.
            let record = CompactRecorder::new()
                .load(model_path.into(), device)
                .expect("Failed to load model record");
            model = model.load_record(record);
        } else {
            eprintln!("(hint) No checkpoint at {model_path}; using randomly-initialized weights.");
        }

        Self {
            model,
            device: device.clone(),
        }
    }

    /// Run one forward pass: take text -> 384 vec -> [1,384] tensor -> MyModel -> [1,256]
    pub fn run_inference(&self, text: &str) -> Tensor<B, 2> {
        // Produce a 384-d float feature vector (your stub for now)
        let v = embed_text(text);
        assert_eq!(
            v.len(),
            IN_DIM,
            "embed_text produced {} dims, expected {}",
            v.len(),
            IN_DIM
        );

        // Make a 1D tensor then reshape to [1, IN_DIM]
        let x1d = Tensor::<B, 1>::from_floats(v.as_slice(), &self.device);
        let x2d = x1d.reshape([1, IN_DIM]);

        // Forward through your tiny MLP: [1,384] -> [1,256]
        self.model.forward(x2d)
    }
}

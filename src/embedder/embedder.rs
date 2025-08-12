use burn::prelude::*;
use burn::tensor::{Tensor, TensorData, Shape as BurnShape};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Simple linear embedder: y = x W + b
/// Shapes:
///   x: [batch, in_dim]
///   W: [in_dim, out_dim]
///   b: [out_dim]
#[derive(Debug, Module)]
pub struct FixedWeightEmbedder<B: Backend> {
    w: Tensor<B, 2>,
    b: Tensor<B, 1>,
}

impl<B: Backend> FixedWeightEmbedder<B> {
    /// Default constructor with a fixed seed for reproducibility.
    pub fn new(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        Self::new_with_seed(in_dim, out_dim, 42, device)
    }

    /// Seeded constructor. Use this in tests to keep outputs stable.
    pub fn new_with_seed(in_dim: usize, out_dim: usize, seed: u64, device: &B::Device) -> Self {
        // Xavier/Glorot uniform init range
        let limit = (6.0f32 / ((in_dim + out_dim) as f32)).sqrt();

        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // Weights [in_dim, out_dim]
        let weights: Vec<f32> = (0..in_dim * out_dim)
            .map(|_| rng.gen_range(-limit..limit))
            .collect();
        let w_data = TensorData::new(weights, BurnShape::new([in_dim, out_dim]));
        let w = Tensor::<B, 2>::from_data(w_data, device);

        // Bias [out_dim] (start at zero; you can randomize if you want)
        let b_vals = vec![0.0f32; out_dim];
        let b_data = TensorData::new(b_vals, BurnShape::new([out_dim]));
        let b = Tensor::<B, 1>::from_data(b_data, device);

        Self { w, b }
    }

    /// Forward: [batch, in_dim] · [in_dim, out_dim] + [out_dim] -> [batch, out_dim]
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let y = x.matmul(self.w.clone());         // owns tensors; clone is cheap (Arc)
        y + self.b.clone().unsqueeze()           // broadcast bias across batch
    }
}

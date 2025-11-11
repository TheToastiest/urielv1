// training/model_router.rs
use burn::{
    module::Module,
    nn,
    tensor::{backend::Backend, Int, Tensor},
};
use burn::tensor::activation;

#[derive(Module, Debug)]
pub struct RouterHead<B: Backend> {
    lin1: nn::Linear<B>,
    lin2: nn::Linear<B>,
}

impl<B: Backend> RouterHead<B> {
    /// d_in == vocab size (keep 96: 0=PAD, 1..=95 real chars).
    pub fn new(d_in: usize, d_hidden: usize, d_out: usize, dev: &B::Device) -> Self {
        Self {
            // We keep feature size V; we mask over time instead of dropping the PAD channel.
            lin1: nn::LinearConfig::new(d_in, d_hidden).init(dev),
            lin2: nn::LinearConfig::new(d_hidden, d_out).init(dev),
        }
    }

    /// x_oh: [B, T, V], one-hot; channel 0 is PAD.
    /// x_oh: [B, T, V] one-hot; channel 0 is PAD.
    pub fn forward(&self, x_oh: Tensor<B, 3>) -> Tensor<B, 2> {
        let dev = &self.lin1.weight.device();

        // Get runtime dims
        let dims = x_oh.dims();      // e.g., [B, T, V]
        let b = dims[0];
        let t = dims[1];
        // let v = dims[2];  // not needed below, but here if you want

        // PAD selector (index 0) as Int tensor
        let idx0: Tensor<B, 1, Int> = Tensor::<B, 1, Int>::from_ints([0i32], dev); // [1]

        // PAD mask over time: pad_bt in [B,T]
        let pad_bt1: Tensor<B, 3> = x_oh.clone().select(2, idx0);   // [B,T,1]
        let pad_bt:  Tensor<B, 2> = pad_bt1.squeeze::<2>(2);        // [B,T]

        // mask 1 for real tokens, 0 for PAD (per timestep)
        let ones_bt: Tensor<B, 2> = pad_bt.ones_like();             // [B,T]
        let mask_bt: Tensor<B, 2> = ones_bt - pad_bt;               // [B,T]

        // Explicit reshape to [B,T,1] to match x_oh [B,T,V]
        let mask_bt1: Tensor<B, 3> = mask_bt.clone().reshape([b, t, 1]);    // [B,T,1]
        let masked:   Tensor<B, 3> = x_oh * mask_bt1;               // [B,T,V] (PAD rows zeroed)

        // Sum over time → [B,1,V], then squeeze → [B,V]
        let sum_b1v: Tensor<B, 3> = masked.sum_dim(1);              // [B,1,V]
        let sum_bv:  Tensor<B, 2> = sum_b1v.squeeze::<2>(1);        // [B,V]

        // Normalize by count of non-PAD tokens: [B,1]
        let denom_b1: Tensor<B, 2> = mask_bt.sum_dim(1).clamp_min(1.0); // [B,1]
        let x:        Tensor<B, 2> = sum_bv / denom_b1;                  // [B,V]

        // MLP head
        let h = activation::relu(self.lin1.forward(x)); // [B,H]
        self.lin2.forward(h)                             // [B,4]
    }

}

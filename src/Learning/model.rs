use burn::{module::Module, nn::{Linear, LinearConfig}, tensor::{activation, backend::Backend, Tensor}};

#[derive(Module, Debug)]
pub struct UrielModel<B: Backend> {
    // sentence encoder MLP (you had this)
    fc1: Linear<B>,            // 384 -> 512
    fc2: Linear<B>,            // 512 -> 256  (z-pre)

    // heads
    proj: Linear<B>,           // 256 -> 256  (z, then L2 normalize)
    salience: Linear<B>,       // 256 -> 4
    route: Linear<B>,          // 256 -> C
    cav: Linear<B>,            // 256 -> 11
}

impl<B: Backend> UrielModel<B> {
    pub fn new(output_classes: usize, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(384, 512).init(device),
            fc2: LinearConfig::new(512, 256).init(device),

            proj:     LinearConfig::new(256, 256).init(device),
            salience: LinearConfig::new(256, 4).init(device),
            route:    LinearConfig::new(256, output_classes).init(device),
            cav:      LinearConfig::new(256, 11).init(device),
        }
    }

    /// x: [batch, 384] (mean-pooled token embeddings)
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B,2>, Tensor<B,2>, Tensor<B,2>, Tensor<B,2>) {
        let h = activation::relu(self.fc1.forward(x));
        let h = self.fc2.forward(h);

        let mut z = self.proj.forward(h.clone());                    // [B,256]
        z = l2_normalize(z);                                         // unit vectors

        let sal = self.salience.forward(h.clone());                  // [B,4]
        let rlg = self.route.forward(h.clone());                     // [B,C]
        let cav = self.cav.forward(h);                               // [B,11]
        (z, sal, rlg, cav)
    }
}

// tiny helper
fn l2_normalize<B: Backend>(x: Tensor<B,2>) -> Tensor<B,2> {
    let norms = (x.clone() * x.clone())
        .sum_dim(1)
        .sqrt()
        .add_scalar(1e-12);          // ← scalar epsilon, no rank issues
    x / norms.unsqueeze()             // broadcast over last dim
}

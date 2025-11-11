use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation::relu, activation, backend::Backend, Tensor},
};

/// Router+Copy model:
/// - route heads (op / arg) from pooled user features
/// - pointer head (final ↔ obs) via scaled dot-product
/// - LM head over final tokens for glue text
#[derive(Module, Debug)]
pub struct RouterCopyModel<B: Backend> {
    // ---- route features from pooled user embedding (384 -> 256)
    user_fc1: Linear<B>,       // 384 -> 512
    user_fc2: Linear<B>,       // 512 -> 256

    // ---- heads
    op_head: Linear<B>,        // 256 -> 4  (NONE, READ_FACT, RECALL, ROWS_SCAN)
    arg_head: Option<Linear<B>>, // 256 -> K (concept keys), optional

    // ---- pointer projections for scaled dot-product attention
    q_proj: Linear<B>,         // 384 -> 256 (final)
    k_proj: Linear<B>,         // 384 -> 256 (obs)

    // ---- LM head for glue tokens (not copied)
    lm: Linear<B>,             // 384 -> vocab_size

    // NEW: lightweight memory attention projections
    mem_q: Linear<B>,   // 384 -> 256
    mem_k: Linear<B>,   // 384 -> 256

    vocab_size: usize,
    arg_classes: usize,
}

impl<B: Backend> RouterCopyModel<B> {
    pub fn new(vocab_size: usize, arg_classes: usize, device: &B::Device) -> Self {
        Self {
            user_fc1: LinearConfig::new(384, 512).init(device),
            user_fc2: LinearConfig::new(512, 256).init(device),

            op_head: LinearConfig::new(256, 4).init(device),
            arg_head: if arg_classes > 0 {
                Some(LinearConfig::new(256, arg_classes).init(device))
            } else {
                None
            },

            q_proj: LinearConfig::new(384, 256).init(device),
            k_proj: LinearConfig::new(384, 256).init(device),

            lm: LinearConfig::new(384, vocab_size).init(device),

            mem_q: LinearConfig::new(384, 256).init(device),
            mem_k: LinearConfig::new(384, 256).init(device),

            vocab_size,
            arg_classes,
        }
    }
    pub fn memory_attn_logits(&self, u_pool: Tensor<B, 2>, obs_seq: Tensor<B, 3>) -> Tensor<B, 2> {
        let q = relu(self.mem_q.forward(u_pool));             // [B,256]
        let k = relu(self.mem_k.forward(obs_seq));            // [B,To,256]
        let q3 = q.unsqueeze_dim(1);                          // [B,1,256]
        let kt = k.swap_dims(1, 2);                           // [B,256,To]
        let logits = q3.matmul(kt).squeeze(1);                // [B,To]
        logits.div_scalar((256.0f32).sqrt())
    }

    /// Compute route logits (op + optional arg) from pooled user features [B,384].
    pub fn route_logits(
        &self,
        user_pooled: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Option<Tensor<B, 2>>) {
        let h = activation::relu(self.user_fc1.forward(user_pooled));
        let h = activation::relu(self.user_fc2.forward(h));
        let op = self.op_head.forward(h.clone());
        let arg = self.arg_head.as_ref().map(|head| head.forward(h));
        (op, arg)
    }

    /// Pointer logits between final sequence [B,Tf,384] and obs sequence [B,To,384].
    pub fn pointer_logits(&self, final_seq: Tensor<B, 3>, obs_seq: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = relu_linear_3d(&self.q_proj, final_seq); // [B, Tf, 256]
        let k = relu_linear_3d(&self.k_proj, obs_seq);   // [B, To, 256]
        let kt = k.swap_dims(1, 2);                      // [B, 256, To]  (Burn: swap_dims)
        // Batched matmul: [B, Tf, 256] x [B, 256, To] -> [B, Tf, To]
        q.matmul(kt).div_scalar((256.0f32).sqrt())
    }


    /// Final-token LM logits: [B,Tf,384] -> [B,Tf,V]
    pub fn final_logits(&self, final_seq: Tensor<B, 3>) -> Tensor<B, 3> {
        linear_3d(&self.lm, final_seq)
    }

    pub fn vocab_size(&self) -> usize { self.vocab_size }
    pub fn arg_classes(&self) -> usize { self.arg_classes }
}

/* ---------- small helpers for Linear on rank-3 ---------- */

fn linear_3d<B: Backend>(lin: &Linear<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [b, t, e] = {
        let d = x.dims();
        [d[0], d[1], d[2]]
    };
    let y2 = lin.forward(x.reshape([b * t, e])); // [B*T, out]
    let out_dim = y2.dims()[1];
    y2.reshape([b, t, out_dim])
}
fn relu_linear_3d<B: Backend>(lin: &Linear<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    activation::relu(linear_3d(lin, x))
}

// src/Learning/loss.rs

use burn::tensor::{activation, backend::Backend, Tensor};

/// L2-normalize along last dim of [B, D] -> [B, D]
pub fn l2_normalize<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    // (x^2).sum_dim(1) -> [B,1]; add eps as scalar; divide with broadcast via unsqueeze.
    let norms = (x.clone() * x.clone())
        .sum_dim(1)
        .sqrt()
        .add_scalar(1e-12);
    x / norms.unsqueeze()
}

/// Cosine sim matrix between batches: [B, D] x [B, D] -> [B, B]
pub fn cosine_sim<B: Backend>(a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
    let a_n = l2_normalize(a);
    let b_n = l2_normalize(b);
    a_n.matmul(b_n.transpose())
}

/// Build one-hot: labels len B, classes C -> [B, C]
pub fn one_hot<B: Backend>(
    labels: &[usize],
    num_classes: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let b = labels.len();
    let mut data = vec![0.0f32; b * num_classes];
    for (i, &cls) in labels.iter().enumerate() {
        if cls < num_classes {
            data[i * num_classes + cls] = 1.0;
        }
    }
    // Create rank-1 then reshape to [B, C]
    let flat = Tensor::<B, 1>::from_floats(data.as_slice(), device);
    flat.reshape([b, num_classes])
}

/// Cross-entropy from logits [B,C] and integer labels -> per-sample loss [B]
pub fn cross_entropy_from_logits<B: Backend>(
    logits: Tensor<B, 2>,   // [B, C]
    labels: &[usize],       // len B
    num_classes: usize,
) -> Tensor<B, 1> {
    let device = logits.device();
    let oh = one_hot::<B>(labels, num_classes, &device);   // [B, C]
    let log_probs = activation::log_softmax(logits, 1);    // [B, C]
    // sum over classes -> [B,1], then squeeze to [B]
    let per_row = (oh * log_probs).sum_dim(1).squeeze(1);
    -per_row
}

/// Mean squared error per sample: [B]
pub fn mse_loss<B: Backend>(pred: Tensor<B, 2>, tgt: Tensor<B, 2>) -> Tensor<B, 1> {
    let diff = pred - tgt;
    let sq = diff.clone() * diff;
    sq.mean_dim(1).squeeze(1)
}

/// InfoNCE with in-batch negatives (diagonal positives) -> [B]
pub fn info_nce_inbatch<B: Backend>(z_q: Tensor<B, 2>, z_p: Tensor<B, 2>, tau: f32) -> Tensor<B, 1> {
    let sim = cosine_sim(z_q, z_p);     // [B, B]
    let logits = sim.div_scalar(tau);   // divide by scalar tau
    let b = logits.dims()[0];
    let labels: Vec<usize> = (0..b).collect(); // diag = positives
    cross_entropy_from_logits(logits, &labels, b)
}

/// Weights for multi-task loss
#[derive(Clone, Copy, Debug)]
pub struct LossWeights {
    pub w_contrastive: f32,
    pub w_salience: f32,
    pub w_route: f32,
    pub w_cav: f32,
}

/// Total multi-task loss -> scalar [1]
pub fn total_loss<B: Backend>(
    z_q: Tensor<B, 2>,
    z_p: Tensor<B, 2>,
    tau: f32,
    sal_pred: Tensor<B, 2>,          // [B,4]
    sal_tgt: Tensor<B, 2>,           // [B,4]
    route_logits: Tensor<B, 2>,      // [B,C]
    route_labels: &[usize],          // len B
    num_route_classes: usize,
    cav_pred: Tensor<B, 2>,          // [B,11]
    cav_tgt: Tensor<B, 2>,           // [B,11]
    w: LossWeights,
) -> Tensor<B, 1> {
    let l_con   = info_nce_inbatch(z_q, z_p, tau);                            // [B]
    let l_sal   = mse_loss(sal_pred, sal_tgt);                                 // [B]
    let l_route = cross_entropy_from_logits(route_logits, route_labels, num_route_classes); // [B]
    let l_cav   = mse_loss(cav_pred, cav_tgt);                                 // [B]

    // Use scalar multipliers to avoid rank issues
    let total = l_con.mul_scalar(w.w_contrastive)
        + l_sal.mul_scalar(w.w_salience)
        + l_route.mul_scalar(w.w_route)
        + l_cav.mul_scalar(w.w_cav);                                           // [B]

    total.mean() // [1]
}

pub fn default_weights() -> LossWeights {
    LossWeights {
        w_contrastive: 1.0,
        w_salience: 0.5,
        w_route: 0.5,
        w_cav: 0.5,
    }
}

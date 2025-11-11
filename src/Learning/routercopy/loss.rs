// src/Learning/routercopy/loss.rs
use burn::tensor::{activation, backend::Backend, Tensor};
use crate::Learning::loss::{cross_entropy_from_logits}; // reuse your CE

#[derive(Clone, Copy, Debug)]
pub struct Weights {
    pub w_op: f32,       // tool selection
    pub w_arg: f32,      // concept key
    pub w_copy: f32,     // pointer copy
    pub w_final: f32,    // LM glue text
}
impl Default for Weights {
    fn default() -> Self {
        Self { w_op: 1.0, w_arg: 0.5, w_copy: 1.0, w_final: 0.3 }
    }
}

/// Cross-entropy for op head: logits [B,4], targets len B
pub fn op_loss<B: Backend>(op_logits: Tensor<B, 2>, op_targets: &[usize], num_classes: usize) -> Tensor<B, 1> {
    cross_entropy_from_logits::<B>(op_logits, op_targets, num_classes).mean()
}

/// Cross-entropy for arg head: logits [B,K], targets len B with -1 for N/A (masked)
pub fn arg_loss<B: Backend>(arg_logits: Tensor<B, 2>, arg_targets: &[i32], num_classes: usize) -> Tensor<B, 1> {
    let device = arg_logits.device();
    let b = arg_logits.dims()[0];

    // Flatten labels and build a mask (1.0 where valid, else 0.0)
    let labels: Vec<usize> = arg_targets.iter().map(|&t| if t >= 0 { t as usize } else { 0 }).collect();
    let mask_f: Vec<f32> = arg_targets.iter().map(|&t| if t >= 0 { 1.0 } else { 0.0 }).collect();
    let ce = cross_entropy_from_logits::<B>(arg_logits, &labels, num_classes); // [B]
    let mask = Tensor::<B, 1>::from_floats(mask_f.as_slice(), &device);       // [B]
    let masked = ce * mask.clone();
    let denom = mask.sum().add_scalar(1e-6);
    masked.sum() / denom
}

/// Masked pointer CE over OBS tokens.
/// ptr_logits: [B, Tfinal, Tobs]
/// ptr_index:  [B, Tfinal] with -1 where not copied
pub fn copy_pointer_loss<B: Backend>(
    ptr_logits: Tensor<B, 3>,                 // [B, Tfinal, Tobs]
    ptr_index: &Vec<Vec<i32>>,                // [B, Tfinal], -1 where not copied
) -> Tensor<B, 1> {
    let device = ptr_logits.device();
    let (b, t_final, t_obs) = {
        let d = ptr_logits.dims();
        (d[0], d[1], d[2])
    };

    // [B*Tfinal, Tobs]
    let flat = ptr_logits.reshape([b * t_final, t_obs]);

    // Flatten labels + mask
    let mut labels = Vec::<usize>::with_capacity(b * t_final);
    let mut mask_f = Vec::<f32>::with_capacity(b * t_final);
    for row in ptr_index {
        for &idx in row {
            if idx >= 0 {
                labels.push(idx as usize);
                mask_f.push(1.0);
            } else {
                labels.push(0);
                mask_f.push(0.0);
            }
        }
    }

    let ce   = cross_entropy_from_logits::<B>(flat, &labels, t_obs); // [B*Tfinal]
    let mask = Tensor::<B, 1>::from_floats(mask_f.as_slice(), &device);
    let masked = ce * mask.clone();
    let denom  = mask.sum().add_scalar(1e-6);
    masked.sum() / denom
}


/// Masked LM loss on final tokens (only where NOT copied).
/// final_logits: [B, Tfinal, V]; final_tokens: [B, Tfinal]; copy_mask: [B, Tfinal] (1=copied)
pub fn final_lm_loss<B: Backend>(
    final_logits: Tensor<B, 3>,
    final_tokens: &Vec<Vec<usize>>,
    copy_mask: &Vec<Vec<u8>>,
    vocab_size: usize,
) -> Tensor<B, 1> {
    let device = final_logits.device();
    let [b, t_final, v] = {
        let d = final_logits.dims();
        [d[0], d[1], d[2]]
    };
    assert_eq!(v, vocab_size, "final head vocab mismatch");

    // [B*Tfinal, V]
    let flat = final_logits.reshape([b * t_final, v]);

    // Flatten labels + inverse mask (only glue tokens: copy_mask==0)
    let mut labels = Vec::<usize>::with_capacity(b * t_final);
    let mut mask_f = Vec::<f32>::with_capacity(b * t_final);
    for (row_y, row_m) in final_tokens.iter().zip(copy_mask.iter()) {
        for (&y, &m) in row_y.iter().zip(row_m.iter()) {
            labels.push(y);
            mask_f.push(if m == 0 { 1.0 } else { 0.0 });
        }
    }

    let ce   = cross_entropy_from_logits::<B>(flat, &labels, vocab_size);   // [B*Tfinal]
    let mask = Tensor::<B, 1>::from_floats(mask_f.as_slice(), &device);     // [B*Tfinal]
    let masked = ce * mask.clone();                                         // keep mask for denom
    let denom  = mask.sum().add_scalar(1e-6);                               // avoid div-by-0
    masked.sum() / denom
}


/// Total objective
pub fn total_routercopy_loss<B: Backend>(
    w: Weights,
    // op/arg
    op_logits: Tensor<B, 2>, op_targets: &[usize], num_ops: usize,
    arg_logits: Option<(Tensor<B, 2>, Vec<i32>, usize)>, // (logits, labels, K)
    // pointer copy
    ptr_logits: Tensor<B, 3>, ptr_index: &Vec<Vec<i32>>,
    // lm glue
    final_logits: Tensor<B, 3>, final_tokens: &Vec<Vec<usize>>, copy_mask: &Vec<Vec<u8>>, vocab_size: usize,
) -> Tensor<B, 1> {
    let device = op_logits.device();

    let l_op   = op_loss::<B>(op_logits, op_targets, num_ops);
    let l_arg  = match arg_logits {
        Some((logits, labels, k)) => arg_loss::<B>(logits, &labels, k),
        None => Tensor::<B, 1>::from_floats([0.0], &device),
    };
    let l_copy = copy_pointer_loss::<B>(ptr_logits, ptr_index);
    let l_final= final_lm_loss::<B>(final_logits, final_tokens, copy_mask, vocab_size);

    l_op.mul_scalar(w.w_op)
        + l_arg.mul_scalar(w.w_arg)
        + l_copy.mul_scalar(w.w_copy)
        + l_final.mul_scalar(w.w_final)
}

use URIELV1::tokenizer::tokenizer::{EncodingMode, Vocab};
use URIELV1::embedder::embedder::{FixedWeightEmbedder};
use burn::prelude::*;
use burn::tensor::{ElementConversion, TensorData, Shape};
use URIELV1::Hippocampus::burn_model::MyBackend;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { return 0.0; }
    dot / (na * nb)
}

/// Encode -> [1,32] tensor -> model.forward -> [1,64] -> squeeze -> Vec<f32>
fn run_embedding_pipeline(
    input: &str,
    device: &<MyBackend as Backend>::Device,
    model: &FixedWeightEmbedder<MyBackend>,
) -> Vec<f32> {
    // 1) Encode to exactly 32 tokens (pad/truncate).
    let vocab = Vocab::new(EncodingMode::Readable);
    let mut tokens = vocab.encode(input, 32);
    if tokens.len() > 32 {
        tokens.truncate(32);
    } else if tokens.len() < 32 {
        tokens.resize(32, 0);
    }

    // 2) Build [1,32] tensor.
    let arr32: [f32; 32] = tokens.iter().map(|&t| t as f32).collect::<Vec<_>>()
        .try_into().expect("tokenizer must yield exactly 32 ids");
    let data = TensorData::from([arr32]); // shape [1,32]
    let input_tensor: Tensor<MyBackend, 2> = Tensor::from_data(data, device);
    debug_assert_eq!(input_tensor.shape(), Shape::new([1, 32]));

    // 3) Forward -> [1,64]
    let out_2d: Tensor<MyBackend, 2> = model.forward(input_tensor);
    debug_assert_eq!(out_2d.shape(), Shape::new([1, 64]));

    // 4) Squeeze batch -> [64] (rank-1), then to Vec<f32>.
    let out_1d: Tensor<MyBackend, 1> = out_2d.squeeze(0);
    debug_assert_eq!(out_1d.shape(), Shape::new([64]));

    let data = out_1d.into_data().convert::<f32>();
    data.to_vec().expect("Tensor to_vec failed")
}

fn main() {
    let device = &Default::default();
    let model = FixedWeightEmbedder::new(32, 64, device); // build once, reuse

    let input = "example input";
    let flat_vec = run_embedding_pipeline(input, device, &model);

    println!("🔢 Embedding Output: {:?}", flat_vec);
}


#[test]
fn test_tokenizer_and_embedding_pipeline() {
    let device = &Default::default();
    let model = FixedWeightEmbedder::new_with_seed(32, 64, 1337, device);

    let near_a = "dogs and cats play in the yard";
    let near_b = "cats and dogs are playing outside"; // near
    let mid    = "a dog quickly runs around a field";  // mid-ish
    let far1   = "quantum field operators commute under constraints";
    let far2   = "oil futures roll yield affects contango arbitrage";

    let va = run_embedding_pipeline(near_a, device, &model);
    let vb = run_embedding_pipeline(near_b, device, &model);
    let vm = run_embedding_pipeline(mid,    device, &model);
    let vf1 = run_embedding_pipeline(far1,  device, &model);
    let vf2 = run_embedding_pipeline(far2,  device, &model);

    let s_self = cosine_similarity(&va, &va);
    let s_near = cosine_similarity(&va, &vb);
    let s_mid  = cosine_similarity(&va, &vm);
    let s_f1   = cosine_similarity(&va, &vf1);
    let s_f2   = cosine_similarity(&va, &vf2);
    let s_far_avg = (s_f1 + s_f2) / 2.0;

    // Identity ~1
    assert!(s_self > 0.999, "cos(x,x) ~ 1; got {s_self}");

    // Ranking: near > mid > far (average)
    assert!(s_near > s_mid, "near must beat mid: near={s_near}, mid={s_mid}");
    assert!(s_mid  > s_far_avg, "mid must beat far avg: mid={s_mid}, far_avg={s_far_avg}");

    // Margins (tighter than before)
    assert!(s_near >= 0.74, "near too low: {s_near}");
    assert!(s_mid  <= 0.71, "mid too high: {s_mid}");
    assert!(s_far_avg <= 0.48, "far avg too high: {s_far_avg}");
    assert!(s_near >= s_far_avg + 0.28, "insufficient separation: near={s_near}, far_avg={s_far_avg}");

    // Collapse guard: different inputs should not be ~identical
    let s_near_far_max = s_f1.max(s_f2);
    assert!(s_near - s_near_far_max >= 0.18, "near too close to far: near={s_near}, far_max={s_near_far_max}");

    // Vector diversity (L1 distance not tiny)
    let l1_near_far: f32 = va.iter().zip(vf1.iter()).map(|(a,b)| (a-b).abs()).sum();
    assert!(l1_near_far > 1e-3, "vectors too similar (collapse), L1={l1_near_far}");
}

fn var(xs: &[f32]) -> f32 {
    let m = xs.iter().sum::<f32>() / xs.len() as f32;
    xs.iter().map(|x| (x - m)*(x - m)).sum::<f32>() / xs.len() as f32
}

#[test]
fn embedding_has_variance() {
    let device = &Default::default();
    let model = FixedWeightEmbedder::new_with_seed(32, 64, 1337, device);
    let v = run_embedding_pipeline("some input", device, &model);
    assert_eq!(v.len(), 64);
    assert!(var(&v) > 1e-6, "embedding collapsed (near-constant vector)");
}

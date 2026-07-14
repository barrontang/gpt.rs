// microgpt.rs — port of karpathy's microgpt.py (gpt.py from ng-video-lecture)
// https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
//
// Trains a character-level GPT on input.txt (tinyshakespeare by default).
//
// Usage:  cargo run --release
//
// Hyperparameters (matching the Python original):
//   batch_size  = 64
//   block_size  = 256
//   max_iters   = 5000
//   n_embd      = 384   (d_model)
//   n_head      = 6
//   n_layer     = 6
//   dropout     = 0.2
//   lr          = 3e-4  (AdamW)

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, ops, AdamW, Embedding, LayerNorm, Linear,
    Optimizer, ParamsAdamW, VarBuilder, VarMap,
};
use rand::Rng;
use std::collections::HashMap;

// ─── Hyperparameters ─────────────────────────────────────────────────────────

const BATCH_SIZE: usize = 64;
const BLOCK_SIZE: usize = 256;
const MAX_ITERS: usize = 5000;
const EVAL_INTERVAL: usize = 500;
const LEARNING_RATE: f64 = 3e-4;
const EVAL_ITERS: usize = 200;
const N_EMBD: usize = 384;
const N_HEAD: usize = 6;
const N_LAYER: usize = 6;
const DROPOUT: f32 = 0.2;

// ─── Tokeniser ────────────────────────────────────────────────────────────────

struct Tokeniser {
    stoi: HashMap<char, u32>,
    itos: Vec<char>,
}

impl Tokeniser {
    fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>().into_iter().collect();
        chars.sort();
        let stoi: HashMap<char, u32> = chars.iter().enumerate().map(|(i, &c)| (c, i as u32)).collect();
        Self { stoi, itos: chars }
    }

    fn vocab_size(&self) -> usize {
        self.itos.len()
    }

    fn encode(&self, s: &str) -> Vec<u32> {
        s.chars().map(|c| self.stoi[&c]).collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter().map(|&i| self.itos[i as usize]).collect()
    }
}

// ─── Data helpers ─────────────────────────────────────────────────────────────

fn get_batch(
    data: &[u32],
    device: &Device,
    rng: &mut impl Rng,
) -> candle_core::Result<(Tensor, Tensor)> {
    let ix: Vec<usize> = (0..BATCH_SIZE)
        .map(|_| rng.gen_range(0..data.len() - BLOCK_SIZE))
        .collect();

    let xs: Vec<u32> = ix.iter().flat_map(|&i| data[i..i + BLOCK_SIZE].iter().copied()).collect();
    let ys: Vec<u32> = ix.iter().flat_map(|&i| data[i + 1..i + 1 + BLOCK_SIZE].iter().copied()).collect();

    let x = Tensor::from_vec(xs, (BATCH_SIZE, BLOCK_SIZE), device)?;
    let y = Tensor::from_vec(ys, (BATCH_SIZE, BLOCK_SIZE), device)?;
    Ok((x, y))
}

// Build the causal (lower-triangular) mask of shape (t, t).
// Returns a u8 tensor: 1 = keep, 0 = mask out.
fn causal_mask(t: usize, device: &Device) -> candle_core::Result<Tensor> {
    let data: Vec<u8> = (0..t)
        .flat_map(|i| (0..t).map(move |j| u8::from(j <= i)))
        .collect();
    Tensor::from_vec(data, (t, t), device)
}

// ─── Model components ─────────────────────────────────────────────────────────

/// One head of causal self-attention.
struct Head {
    key: Linear,
    query: Linear,
    value: Linear,
    head_size: usize,
}

impl Head {
    fn new(head_size: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let key = linear_no_bias(N_EMBD, head_size, vb.pp("key"))?;
        let query = linear_no_bias(N_EMBD, head_size, vb.pp("query"))?;
        let value = linear_no_bias(N_EMBD, head_size, vb.pp("value"))?;
        Ok(Self { key, query, value, head_size })
    }

    fn forward(&self, x: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let (_b, t, _c) = x.dims3()?;
        let k = self.key.forward(x)?;   // (B, T, hs)
        let q = self.query.forward(x)?; // (B, T, hs)

        // Scaled dot-product attention scores
        let scale = (self.head_size as f64).powf(-0.5);
        let wei = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?; // (B, T, T)

        // Causal mask: replace upper-triangle with -inf
        let mask = causal_mask(t, x.device())?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?.broadcast_as(wei.shape())?;
        let mask_b = mask.broadcast_as(wei.shape())?;
        let wei = mask_b.where_cond(&wei, &neg_inf)?; // (B, T, T)

        let wei = ops::softmax(&wei, D::Minus1)?; // (B, T, T)
        let wei = if train { ops::dropout(&wei, DROPOUT)? } else { wei };

        let v = self.value.forward(x)?; // (B, T, hs)
        wei.matmul(&v)                   // (B, T, hs)
    }
}

/// Multiple heads of self-attention running in parallel.
struct MultiHeadAttention {
    heads: Vec<Head>,
    proj: Linear,
}

impl MultiHeadAttention {
    fn new(num_heads: usize, head_size: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let heads: candle_core::Result<Vec<Head>> = (0..num_heads)
            .map(|i| Head::new(head_size, vb.pp(format!("h{i}"))))
            .collect();
        let proj = linear(head_size * num_heads, N_EMBD, vb.pp("proj"))?;
        Ok(Self { heads: heads?, proj })
    }

    fn forward(&self, x: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let outs: candle_core::Result<Vec<Tensor>> =
            self.heads.iter().map(|h| h.forward(x, train)).collect();
        let out = Tensor::cat(&outs?, D::Minus1)?; // (B, T, n_embd)
        let out = self.proj.forward(&out)?;
        if train { ops::dropout(&out, DROPOUT) } else { Ok(out) }
    }
}

/// Simple two-layer feed-forward network with ReLU.
struct FeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl FeedForward {
    fn new(n_embd: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            fc1: linear(n_embd, 4 * n_embd, vb.pp("fc1"))?,
            fc2: linear(4 * n_embd, n_embd, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let x = self.fc1.forward(x)?.relu()?;
        let x = self.fc2.forward(&x)?;
        if train { ops::dropout(&x, DROPOUT) } else { Ok(x) }
    }
}

/// Transformer block: communication (self-attention) + computation (FFN).
struct Block {
    sa: MultiHeadAttention,
    ffwd: FeedForward,
    ln1: LayerNorm,
    ln2: LayerNorm,
}

impl Block {
    fn new(n_embd: usize, n_head: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let head_size = n_embd / n_head;
        Ok(Self {
            sa: MultiHeadAttention::new(n_head, head_size, vb.pp("sa"))?,
            ffwd: FeedForward::new(n_embd, vb.pp("ffwd"))?,
            ln1: layer_norm(n_embd, 1e-5, vb.pp("ln1"))?,
            ln2: layer_norm(n_embd, 1e-5, vb.pp("ln2"))?,
        })
    }

    fn forward(&self, x: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let x = (x + self.sa.forward(&self.ln1.forward(x)?, train)?)?;
        let x = (&x + self.ffwd.forward(&self.ln2.forward(&x)?, train)?)?;
        Ok(x)
    }
}

/// Full GPT language model.
struct GPTLanguageModel {
    token_embedding_table: Embedding,
    position_embedding_table: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
    vocab_size: usize,
}

impl GPTLanguageModel {
    fn new(vocab_size: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let blocks: candle_core::Result<Vec<Block>> = (0..N_LAYER)
            .map(|i| Block::new(N_EMBD, N_HEAD, vb.pp(format!("b{i}"))))
            .collect();
        Ok(Self {
            token_embedding_table: embedding(vocab_size, N_EMBD, vb.pp("tok_emb"))?,
            position_embedding_table: embedding(BLOCK_SIZE, N_EMBD, vb.pp("pos_emb"))?,
            blocks: blocks?,
            ln_f: layer_norm(N_EMBD, 1e-5, vb.pp("ln_f"))?,
            lm_head: linear(N_EMBD, vocab_size, vb.pp("lm_head"))?,
            vocab_size,
        })
    }

    /// Forward pass.  Returns (logits, loss).
    /// `targets` is `None` at generation time.
    fn forward(
        &self,
        idx: &Tensor,        // (B, T)  u32
        targets: Option<&Tensor>, // (B, T) u32
        train: bool,
    ) -> candle_core::Result<(Tensor, Option<Tensor>)> {
        let (b, t) = idx.dims2()?;

        let tok_emb = self.token_embedding_table.forward(idx)?; // (B, T, C)
        let positions = Tensor::arange(0u32, t as u32, idx.device())?;
        let pos_emb = self.position_embedding_table.forward(&positions)?; // (T, C)
        let mut x = tok_emb.broadcast_add(&pos_emb)?; // (B, T, C)

        for block in &self.blocks {
            x = block.forward(&x, train)?;
        }
        x = self.ln_f.forward(&x)?;                  // (B, T, C)
        let logits = self.lm_head.forward(&x)?;      // (B, T, vocab_size)

        let loss = match targets {
            None => None,
            Some(tgt) => {
                // Flatten to (B*T, vocab_size) / (B*T,) for cross-entropy
                let logits_2d = logits.reshape((b * t, self.vocab_size))?;
                let tgt_1d = tgt.reshape(b * t)?;
                Some(candle_nn::loss::cross_entropy(&logits_2d, &tgt_1d)?)
            }
        };
        Ok((logits, loss))
    }

    /// Autoregressive generation: append `max_new_tokens` tokens to `idx`.
    fn generate(
        &self,
        idx: &Tensor, // (B, T) u32
        max_new_tokens: usize,
        rng: &mut impl Rng,
    ) -> candle_core::Result<Tensor> {
        let mut idx = idx.clone();
        for _ in 0..max_new_tokens {
            // Crop to block_size
            let t = idx.dim(1)?;
            let idx_cond = if t > BLOCK_SIZE {
                idx.narrow(1, t - BLOCK_SIZE, BLOCK_SIZE)?
            } else {
                idx.clone()
            };

            let (logits, _) = self.forward(&idx_cond, None, false)?; // (B, T, V)
            // Focus on last time step
            let t2 = logits.dim(1)?;
            let logits_last = logits.narrow(1, t2 - 1, 1)?.squeeze(1)?; // (B, V)

            // Softmax → probabilities
            let probs = ops::softmax(&logits_last, D::Minus1)?; // (B, V)

            // Sample one token per batch element
            let b = probs.dim(0)?;
            let probs_vec: Vec<f32> = probs.reshape(b * self.vocab_size)?.to_vec1()?;
            let next_ids: Vec<u32> = (0..b)
                .map(|bi| {
                    let slice = &probs_vec[bi * self.vocab_size..(bi + 1) * self.vocab_size];
                    sample_from_probs(slice, rng)
                })
                .collect();
            let next = Tensor::from_vec(next_ids, (b, 1), idx.device())?;
            idx = Tensor::cat(&[&idx, &next], 1)?;
        }
        Ok(idx)
    }
}

// ─── Sampling helper ──────────────────────────────────────────────────────────

fn sample_from_probs(probs: &[f32], rng: &mut impl Rng) -> u32 {
    let r: f32 = rng.gen();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

// ─── Loss estimation ──────────────────────────────────────────────────────────

fn estimate_loss(
    model: &GPTLanguageModel,
    train_data: &[u32],
    val_data: &[u32],
    device: &Device,
    rng: &mut impl Rng,
) -> candle_core::Result<(f32, f32)> {
    let mut results = [0.0f32; 2];
    for (si, data) in [train_data, val_data].iter().enumerate() {
        let mut total = 0.0f32;
        for _ in 0..EVAL_ITERS {
            let (xb, yb) = get_batch(data, device, rng)?;
            let (_, loss) = model.forward(&xb, Some(&yb), false)?;
            total += loss.unwrap().to_scalar::<f32>()?;
        }
        results[si] = total / EVAL_ITERS as f32;
    }
    Ok((results[0], results[1]))
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() -> candle_core::Result<()> {
    let mut rng = rand::thread_rng();
    let device = Device::Cpu;

    // ── Read and tokenise the corpus ──
    let text = std::fs::read_to_string("input.txt").expect("input.txt not found");
    let tok = Tokeniser::new(&text);
    let vocab_size = tok.vocab_size();
    println!("vocab size: {vocab_size}");

    let data: Vec<u32> = tok.encode(&text);
    let n = (data.len() as f64 * 0.9) as usize;
    let train_data = &data[..n];
    let val_data = &data[n..];

    // ── Build model ──
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
    let model = GPTLanguageModel::new(vocab_size, vb)?;

    let n_params: usize = vm.all_vars().iter().map(|v| v.elem_count()).sum();
    println!("{:.2}M parameters", n_params as f64 / 1e6);

    // ── Optimiser ──
    let params = ParamsAdamW { lr: LEARNING_RATE, ..Default::default() };
    let mut opt = AdamW::new(vm.all_vars(), params)?;

    // ── Training loop ──
    for iter in 0..MAX_ITERS {
        if iter % EVAL_INTERVAL == 0 || iter == MAX_ITERS - 1 {
            let (train_loss, val_loss) =
                estimate_loss(&model, train_data, val_data, &device, &mut rng)?;
            println!("step {iter}: train loss {train_loss:.4}, val loss {val_loss:.4}");
        }

        let (xb, yb) = get_batch(train_data, &device, &mut rng)?;
        let (_, loss) = model.forward(&xb, Some(&yb), true)?;
        opt.backward_step(&loss.unwrap())?;
    }

    // ── Generate ──
    let context = Tensor::zeros((1, 1), DType::U32, &device)?;
    let generated = model.generate(&context, 500, &mut rng)?;
    let ids: Vec<u32> = generated.squeeze(0)?.to_vec1()?;
    println!("\n{}", tok.decode(&ids));

    Ok(())
}

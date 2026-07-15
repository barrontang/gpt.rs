# gpt.rs 🦀

> A faithful Rust translation of Andrej Karpathy's character-level GPT — powered by HuggingFace [candle](https://github.com/huggingface/candle), no PyTorch required.

[![Rust](https://img.shields.io/badge/Rust-2021-orange?logo=rust)](https://www.rust-lang.org/)
[![candle](https://img.shields.io/badge/candle-0.11-blue?logo=huggingface)](https://github.com/huggingface/candle)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is this?

**gpt.rs** is a clean, minimal Rust port of [`ng-video-lecture/gpt.py`](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py) by Andrej Karpathy — the same model famously explained in his [*Let's build GPT from scratch*](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture.

It trains a character-level language model on any plain-text file and generates new text one character at a time, from scratch — no pretrained weights, no Python, no libtorch.

---

## ✨ Features

- 🦀 **Pure Rust** — safe, fast, and dependency-light
- ⚡ **No PyTorch / libtorch** — uses HuggingFace's [`candle`](https://github.com/huggingface/candle) ML framework
- 🧠 **Full GPT architecture** — multi-head self-attention, feed-forward blocks, layer norm, learned positional embeddings
- 📄 **Train on any text** — just drop in an `input.txt`
- 🎓 **Educational** — direct line-by-line correspondence to Karpathy's original Python

---

## 🏗️ Architecture

The model implements the classic decoder-only Transformer:

```
input tokens
     │
Token Embedding + Positional Embedding
     │
┌────▼────────────────────────┐
│  Transformer Block × N      │
│  ┌──────────────────────┐   │
│  │  LayerNorm            │   │
│  │  Multi-Head Attention │   │
│  │  (causal / masked)   │   │
│  └──────────────────────┘   │
│  ┌──────────────────────┐   │
│  │  LayerNorm            │   │
│  │  Feed-Forward (MLP)  │   │
│  └──────────────────────┘   │
└─────────────────────────────┘
     │
  LayerNorm
     │
  Linear → logits
     │
  Softmax → next character
```

All hyperparameters (embedding size, heads, layers, context length, etc.) are configurable at the top of `src/main.rs`.

---

## 🚀 Getting Started

### Prerequisites

- [Rust](https://rustup.rs/) (2021 edition or later)

### Build

```bash
git clone https://github.com/barrontang/gpt.rs
cd gpt.rs
cargo build --release
```

### Train & Generate

1. Place your training text in `input.txt` in the project root. Any plain-text corpus works — Shakespeare, code, novels, song lyrics.

2. Run:

```bash
cargo run --release
```

The model will train on your text and then generate a sample — watch the loss drop and new text appear.

---

## 📖 Usage Example

```
# Training on Shakespeare's works
$ echo "To be, or not to be, that is the question..." > input.txt
$ cargo run --release

step 0: loss 4.17
step 100: loss 2.93
step 500: loss 2.21
...

--- Generated Text ---
KING RICHARD:
What light is yonder window breaks?
...
```

---

## 🙏 Acknowledgements

- **Andrej Karpathy** — for the brilliant [*Let's build GPT from scratch*](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture and the [original gpt.py](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py)
- **HuggingFace candle** — for making pure-Rust ML a reality

---

## 📄 License

MIT

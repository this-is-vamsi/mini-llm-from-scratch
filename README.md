# mini-llm-from-scratch

> A hands-on implementation of a decoder-only transformer built entirely from scratch in TensorFlow/Keras, followed by GPT-2 fine-tuning with Hugging Face — demonstrating the full spectrum from architecture internals to production-grade transfer learning.

---

## What This Project Covers

This project has two distinct parts with different goals:

**Part 1 — Build from Scratch (Educational)**
Implement every component of a transformer decoder from the ground up: character-level tokenization, token + positional embeddings, causal self-attention with masking, feed-forward layers, residual connections, and layer normalisation. Train a 120K parameter model on the TinyStories dataset and observe the fundamental limitations of training at small scale.

**Part 2 — Transfer Learning with GPT-2 (Practical)**
Load a pre-trained GPT-2 model (124M parameters — 1,000× larger) via Hugging Face Transformers and fine-tune it on TinyStories in just 3 epochs. Compare the generation quality directly against the scratch-trained model.

The core insight: training from scratch teaches you *how* transformers work. Transfer learning teaches you *how they're actually used*.

---

## Architecture (Part 1 — Scratch Model)

```
Input Token IDs
      │
      ▼
TokenAndPositionEmbedding
  ├── Token Embedding  (vocab_size → d_model)
  └── Positional Embedding  (block_size → d_model)
      │
      ▼
TransformerBlock × N
  ├── CausalSelfAttention  (multi-head, causal mask)
  ├── Add & Norm
  ├── FeedForward  (d_model → 4×d_model → d_model)
  └── Add & Norm
      │
      ▼
Linear + Softmax  (d_model → vocab_size)
      │
      ▼
Next Token Prediction
```

### Model Configuration

| Parameter | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | ~120K |
| Embedding dim (`d_model`) | 128 |
| Attention heads | 4 |
| Context length (`block_size`) | 64 tokens |
| Tokenization | Character-level |
| Dataset | TinyStories |

---

## Tech Stack

| Component | Library |
|---|---|
| Framework | TensorFlow 2.x + Keras |
| Pre-trained LLM | GPT-2 via Hugging Face Transformers |
| Dataset | TinyStories (`roneneldan/TinyStories`) |
| Data loading | HuggingFace `datasets` |
| Visualisation | Matplotlib |
| Runtime | Google Colab (GPU recommended) |

---

## Project Structure

```
mini-llm-from-scratch/
├── mini_llm_training.ipynb   # Full notebook: scratch build + GPT-2 fine-tuning
└── README.md
```

---

## Quickstart

### Run on Google Colab (Recommended)

1. Open `mini_llm_training.ipynb` in Google Colab
2. Go to **Runtime → Change runtime type → GPU**
3. Run all cells from top to bottom

### Run Locally

```bash
git clone https://github.com/this-is-vamsi/mini-llm-from-scratch.git
cd mini-llm-from-scratch
pip install tensorflow keras transformers datasets matplotlib
jupyter notebook mini_llm_training.ipynb
```

---

## What You Will See

### Part 1 — Scratch Model Generation (after training)

```
Prompt: "Once upon a time"
Output (greedy): "Once upon a time tee  a  i  the  a a a  l ..."
```
Limited quality — expected. A 120K parameter model trained on a fraction of TinyStories cannot generalise. This is the point: you see *why* scale matters.

### Part 2 — GPT-2 Fine-tuned Generation

```
Prompt: "Once upon a time"
Output: "Once upon a time there was a little girl named Lily.
         She lived in a small house near a big forest..."
```
Coherent, fluent, contextually consistent — because GPT-2 already learned language at scale.

### Sampling Strategies Demonstrated

| Strategy | Temperature | top-k | Effect |
|---|---|---|---|
| Greedy | ~0 | — | Deterministic, repetitive |
| Focused | 0.5 | — | Coherent, less diverse |
| Creative | 1.2 | — | Varied, occasionally incoherent |
| Top-k | 0.8 | 10 | Balanced quality and diversity |

---

## Key Concepts Implemented

- **Causal masking** — prevents attention to future tokens during training
- **Learned positional embeddings** — position information encoded as trainable vectors
- **Multi-head self-attention** — parallel attention across subspaces
- **Residual connections + LayerNorm** — stable training of deep networks
- **Autoregressive generation** — token-by-token inference loop
- **Temperature scaling** — controlling output distribution sharpness
- **Top-k sampling** — restricting sampling to the k most probable tokens
- **Transfer learning** — fine-tuning a 124M parameter pre-trained model in minutes

---

## Training Details

### Part 1 — Scratch Model
- **Loss**: Sparse Categorical Cross-Entropy
- **Optimiser**: Adam with learning rate warmup
- **Regularisation**: Dropout (0.1), Early Stopping (patience=5)
- **Training time**: ~2–5 minutes on GPU (Colab T4)

### Part 2 — GPT-2 Fine-tuning
- **Base model**: `gpt2` (124M parameters)
- **Epochs**: 3
- **Training time**: ~5–10 minutes on GPU (Colab T4)

---

## Why This Project

Most LLM tutorials treat the model as a black box — you call an API and get a response. This project takes the opposite approach: build every layer yourself first so you understand what the API is actually doing. The fine-tuning comparison then shows why transfer learning is the industry standard: not because from-scratch training is wrong, but because it requires scale that a single GPU cannot provide.

---

## License

MIT

# mini-llm-from-scratch

> A hands-on implementation of a decoder-only Transformer built entirely from scratch in TensorFlow/Keras, followed by GPT-2 fine-tuning with Hugging Face — demonstrating the full spectrum from architecture internals to production-grade transfer learning.

---

## What This Project Covers

This project has two distinct parts with different goals:

**Part 1 — Build from Scratch (Architecture Internals)**
Implement every component of a Transformer decoder from the ground up: character-level tokenization, token + learned positional embeddings, causal self-attention with lower-triangular masking, position-wise feed-forward layers, residual connections, and layer normalisation. Train a custom decoder model (~356K parameters) on the TinyStories dataset and observe the fundamental scaling behaviour of small models.

**Part 2 — Transfer Learning with GPT-2 (Practical Fine-Tuning)**
Load a pre-trained GPT-2 model (124.4M parameters — ~350× larger) via Hugging Face Transformers and fine-tune it on TinyStories in 3 epochs using `Trainer` and `DataCollatorForLanguageModeling`. Compare the generation quality and convergence directly against the scratch-trained model.

The core insight: training from scratch teaches you *how* Transformers work internally; transfer learning demonstrates *how* pre-trained representations enable rapid domain adaptation.

---

## Architecture (Part 1 — Scratch Model)

```text
Input Token IDs (batch_size, 64)
      │
      ▼
TokenAndPositionEmbedding
  ├── Token Embedding       (vocab_size=67 → d_model=128)
  └── Positional Embedding  (block_size=64 → d_model=128)
      │
      ▼
TransformerBlock × 2
  ├── CausalSelfAttention   (4 heads, key_dim=32, lower-triangular mask)
  ├── Add & LayerNorm
  ├── FeedForward           (128 → 384 → 128, ReLU + Dropout 0.2)
  └── Add & LayerNorm
      │
      ▼
Linear Projection           (d_model=128 → vocab_size=67 logits)
      │
      ▼
Autoregressive Next-Token Sampling (Greedy / Temperature / Top-k)
```

### Model Configuration Comparison

| Parameter | Part 1: Scratch Mini-LLM | Part 2: Fine-Tuned GPT-2 |
|---|---|---|
| **Architecture** | 2-Layer Decoder-only Transformer | 12-Layer GPT-2 Decoder |
| **Total Parameters** | **356,163** (~356K) | **124,439,808** (124.4M) |
| **Embedding Dim (`d_model`)** | 128 (`dff=384`) | 768 (`dff=3072`) |
| **Attention Heads** | 4 | 12 |
| **Context Length (`block_size`)** | 64 characters | 512 subword tokens |
| **Tokenization** | Character-level (67 ASCII chars) | Byte-Pair Encoding (50,257 vocab) |
| **Dataset** | TinyStories (1,000 stories / ~942K chars) | TinyStories (2,000 train / 200 val stories) |
| **Final Training Loss** | `1.1468` (train) / `1.3554` (val, 62.1% acc) | `1.9053` (3 epochs in 543s on GPU) |

---

## Tech Stack

| Component | Library |
|---|---|
| Scratch Model Framework | TensorFlow 2.x + Keras |
| Pre-trained LLM & Fine-Tuning | GPT-2 via Hugging Face `transformers` (`Trainer`, `TrainingArguments`) |
| Dataset | TinyStories (`roneneldan/TinyStories`) via Hugging Face `datasets` |
| Visualisation | Matplotlib |

---

## Project Structure

```text
mini-llm-from-scratch/
├── mini_llm_training.ipynb   # Full notebook with executed training logs, loss curves & generation outputs
├── transformer_model.py      # Modular Keras implementation of all custom Transformer layers & sampling
├── requirements.txt          # Python dependencies
├── .gitignore                # Standard Python/Jupyter gitignore
└── README.md                 # Project documentation
```

---

## Quickstart

### Run Locally

```bash
git clone https://github.com/this-is-vamsi/mini-llm-from-scratch.git
cd mini-llm-from-scratch
pip install -r requirements.txt
jupyter notebook mini_llm_training.ipynb
```

### Run on Google Colab (GPU Recommended for Part 2 Fine-Tuning)

1. Open `mini_llm_training.ipynb` in Google Colab
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run all cells from top to bottom

---

## Results & Generation Comparison

### Part 1 — Scratch Model (Character-Level, 356K Params, 1 Epoch)
Reaches **63.7% training accuracy / 62.1% validation accuracy** (`val_loss = 1.3554`) over 67 characters, capturing character transition probabilities and punctuation structure, while illustrating why small character-level models struggle with long-range word coherence.

### Part 2 — GPT-2 Fine-Tuned on TinyStories (124.4M Params, 3 Epochs)
```text
Prompt: "Once upon a time"
Output: "Once upon a time, there was a girl named Lily. She was 3 years old and loved
         to explore the world around her. One day, she went for a walk in the woods
         with her mom..."
```
Before fine-tuning, base GPT-2 generates adult/news-style prose. After 3 epochs (~9 minutes on GPU, `training_loss = 1.9053`), the model adapts cleanly to the simple narrative structure and vocabulary of TinyStories.

### Sampling Strategies Implemented

| Strategy | Temperature | `top_k` | Effect |
|---|---|---|---|
| **Greedy** | `0.01` | `0` | Deterministic; picks argmax logit at each step |
| **Focused** | `0.5` | `0` | Sharpens logit distribution for more conservative transitions |
| **Creative** | `1.2` | `0` | Flattens logit distribution for higher entropy |
| **Top-$k$** | `0.8` | `10` | Truncates tail probabilities to the top 10 candidate tokens |

---

## Key Concepts Implemented

- **Causal masking (`tf.linalg.band_part`)** — prevents self-attention heads from attending to future positions
- **Learned positional embeddings** — encodes sequence order via trainable position vectors added to token embeddings
- **Multi-head self-attention** — parallel attention across 4 subspaces (`key_dim = 32`)
- **Residual connections + LayerNorm (`epsilon=1e-6`)** — stabilizes gradient flow across stacked decoder blocks
- **Autoregressive generation loop** — sliding-window token-by-token inference with temperature scaling and `tf.nn.top_k` filtering
- **Domain adaptation via Transfer Learning** — fine-tuning a 124M-parameter BPE causal language model with mixed precision (`fp16`) and cosine/linear warmup

---

## License

MIT

"""
Mini LLM: Decoder-Only Transformer from Scratch (TensorFlow / Keras)
====================================================================
Modular implementation of a character-level decoder-only Transformer:
- CharTokenizer
- TokenAndPositionEmbedding
- CausalSelfAttention (lower-triangular causal mask)
- FeedForward
- TransformerBlock (Residual + LayerNorm)
- Autoregressive text generation with temperature & top-k sampling
"""

from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CharTokenizer:
    """Character-level tokenizer mapping unique ASCII characters to integer IDs."""

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, token_ids: List[int]) -> str:
        return "".join([self.idx_to_char.get(idx, "") for idx in token_ids])


class TokenAndPositionEmbedding(layers.Layer):
    """Combines token embeddings with learned positional embeddings."""

    def __init__(self, vocab_size: int, block_size: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.token_emb = layers.Embedding(vocab_size, d_model)
        self.pos_emb = layers.Embedding(block_size, d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)
        tokens = self.token_emb(x)
        return tokens + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "d_model": self.d_model,
        })
        return config


class CausalSelfAttention(layers.Layer):
    """Multi-head self-attention with lower-triangular causal masking."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training: bool = False):
        seq_len = tf.shape(x)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.cast(causal_mask, dtype=tf.bool)
        attn_output = self.mha(
            query=x,
            key=x,
            value=x,
            attention_mask=causal_mask,
            training=training,
        )
        return self.dropout(attn_output, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
        })
        return config


class FeedForward(layers.Layer):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, dff: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout
        self.dense1 = layers.Dense(dff, activation="relu")
        self.dense2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training: bool = False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "dff": self.dff,
            "dropout": self.dropout_rate,
        })
        return config


class TransformerBlock(layers.Layer):
    """Single decoder Transformer block: CausalSelfAttention + FFN + Residual + LayerNorm."""

    def __init__(self, d_model: int, num_heads: int, dff: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout

        self.attention = CausalSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, dff, dropout)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training: bool = False):
        attn_output = self.attention(x, training=training)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(x, training=training)
        x = self.layernorm2(x + ffn_output)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout": self.dropout_rate,
        })
        return config


def create_mini_llm(config: Dict) -> keras.Model:
    """Assemble the decoder-only Transformer model from configuration."""
    inputs = layers.Input(shape=(config["block_size"],), dtype=tf.int32)
    x = TokenAndPositionEmbedding(
        config["vocab_size"],
        config["block_size"],
        config["d_model"],
    )(inputs)

    for _ in range(config["num_layers"]):
        x = TransformerBlock(
            config["d_model"],
            config["num_heads"],
            config["dff"],
            config["dropout"],
        )(x)

    outputs = layers.Dense(config["vocab_size"])(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="mini_llm")


def generate_text(
    model: keras.Model,
    tokenizer: CharTokenizer,
    prompt: str,
    block_size: int = 64,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
) -> str:
    """Generate text autoregressively using temperature scaling and optional top-k sampling."""
    tokens = tokenizer.encode(prompt)[-block_size:]

    for _ in range(max_new_tokens):
        input_tokens = tokens[-block_size:]
        if len(input_tokens) < block_size:
            input_tokens = [0] * (block_size - len(input_tokens)) + input_tokens

        input_tensor = tf.constant([input_tokens])
        logits = model(input_tensor, training=False)
        logits = logits[0, -1, :] / temperature

        if top_k > 0:
            top_k_logits, top_k_indices = tf.nn.top_k(logits, k=top_k)
            logits = tf.scatter_nd(
                tf.expand_dims(top_k_indices, 1),
                top_k_logits,
                logits.shape,
            )
            logits = tf.where(logits == 0, -1e10, logits)

        probs = tf.nn.softmax(logits)
        next_token = int(tf.random.categorical(tf.expand_dims(probs, 0), num_samples=1)[0, 0])
        tokens.append(next_token)

    return tokenizer.decode(tokens)

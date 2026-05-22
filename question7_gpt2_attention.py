"""
Assignment 2 - Question 7 (20 marks / 6.7%)
Theoretical Question: Masked Self-Attention in GPT-2

This file contains the comprehensive answer to Question 7.
It also includes a visualisation of the masked self-attention mechanism.

Requirements: pip install matplotlib numpy seaborn
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# ANSWER TO QUESTION 7
# ============================================================================

answer = """
================================================================================
QUESTION 7: The Masked Self-Attention Layer in GPT-2
================================================================================

1. DESCRIPTION OF THE MASKED SELF-ATTENTION MECHANISM
------------------------------------------------------------------------
GPT-2 (Generative Pre-trained Transformer 2) is an autoregressive language
model that generates text token by token, from left to right. At the heart
of GPT-2 is the masked (causal) self-attention mechanism.

In standard self-attention, each token can attend to every other token in the
sequence. However, in masked self-attention, each token can ONLY attend to
itself and all preceding (left-context) tokens. Future tokens are masked out.

Mathematically, for a sequence of tokens x_1, x_2, ..., x_n:

  (a) Each token x_i is projected into three vectors:
      - Query:  Q_i = x_i * W_Q
      - Key:    K_i = x_i * W_K
      - Value:  V_i = x_i * W_V

  (b) Attention scores are computed:
      score(i, j) = (Q_i · K_j) / sqrt(d_k)

  (c) A causal mask is applied:
      score(i, j) = -infinity   if j > i  (future positions)

  (d) Softmax is applied to get attention weights:
      alpha(i, j) = softmax(score(i, :))_j

  (e) The output is a weighted sum of values:
      output_i = sum_j alpha(i, j) * V_j

The mask ensures that when predicting token x_{t+1}, the model only sees
tokens x_1, ..., x_t. This is a lower-triangular attention pattern.

2. WHY IS MASKED SELF-ATTENTION POWERFUL FOR NLP?
------------------------------------------------------------------------
a) Autoregressive Consistency: The masking ensures that the model's
   predictions during training mirror the generation process at inference
   time. During generation, future tokens do not exist yet, so the mask
   simulates this constraint during training. This consistency between
   training and inference is critical for high-quality text generation.

b) Parallelised Training: Unlike RNNs that process tokens sequentially,
   masked self-attention processes all positions in parallel during training.
   The causal mask is applied as a matrix operation, enabling efficient GPU
   utilisation. This makes training much faster than sequential models.

c) Long-Range Dependencies: Self-attention can capture dependencies between
   any two positions in the sequence in O(1) operations (compared to O(n)
   for RNNs). Even with the causal mask, token i can directly attend to
   token 1, regardless of the sequence length. This ability to model
   long-range dependencies is crucial for understanding context in language.

d) Multi-Head Attention: GPT-2 uses multi-head masked self-attention,
   where multiple attention heads operate in parallel, each learning
   different types of relationships (syntactic, semantic, positional).
   This enriches the model's representational capacity.

e) Scalability: The transformer architecture underlying GPT-2 scales
   well with data and model size. Larger GPT-2 variants (up to 1.5B
   parameters) achieve progressively better performance, demonstrating
   the scalability advantage of masked self-attention.

3. PROS AND CONS OF MASKED SELF-ATTENTION IN GPT-2
------------------------------------------------------------------------
PROS:
  - Enables autoregressive text generation (token-by-token).
  - Parallel training on all positions simultaneously.
  - Captures long-range dependencies effectively.
  - Compatible with pre-training on massive corpora (unsupervised learning).
  - Multi-head variant allows learning diverse attention patterns.

CONS:
  - Unidirectional: Can only attend to left context. This is a limitation
    for tasks that benefit from bidirectional context (e.g., sentiment
    analysis, NER), where models like BERT excel.
  - Quadratic Complexity: Self-attention has O(n^2) time and memory
    complexity w.r.t. sequence length n. This limits the maximum context
    window (GPT-2 uses 1024 tokens).
  - Fixed Context Window: GPT-2 cannot attend beyond its maximum sequence
    length, unlike recurrent models that theoretically can process
    arbitrarily long sequences.
  - No Explicit Memory: Each forward pass is independent; the model
    cannot maintain state across separate sequences.

4. WHY THE MASK IS REQUIRED FOR LANGUAGE GENERATION
------------------------------------------------------------------------
Language generation is inherently a sequential process: when generating
the (t+1)-th token, only tokens 1 through t are available. If the model
were allowed to see future tokens during training:

  - It would "cheat" by simply copying the next token from the input.
  - The model would not learn to predict; it would learn to read.
  - At inference time, when future tokens are unavailable, the model
    would fail catastrophically (train-test mismatch).

The causal mask prevents information leakage from future positions,
forcing the model to genuinely learn the conditional distribution:
  P(x_t | x_1, x_2, ..., x_{t-1})

This is the foundation of autoregressive modelling and is essential
for coherent text generation.

================================================================================
"""
print(answer)

# ============================================================================
# VISUALISATION: Masked Self-Attention Pattern
# ============================================================================

# Create a visual example of the masked attention matrix
tokens = ["The", "cat", "sat", "on", "the", "mat"]
n = len(tokens)

# Create causal mask (lower triangular)
mask = np.tril(np.ones((n, n)))

# Simulate attention scores (random, then apply mask)
np.random.seed(42)
raw_scores = np.random.randn(n, n)
masked_scores = np.where(mask == 1, raw_scores, -1e9)

# Apply softmax row-wise
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

attention_weights = softmax(masked_scores, axis=-1)

# Plot 1: The causal mask
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Mask pattern
ax = axes[0]
sns.heatmap(mask, annot=True, fmt=".0f", cmap="RdYlGn",
            xticklabels=tokens, yticklabels=tokens, ax=ax,
            cbar_kws={"label": "1=Attend, 0=Masked"})
ax.set_xlabel("Key Position (j)")
ax.set_ylabel("Query Position (i)")
ax.set_title("(a) Causal Mask\n(Lower Triangular)", fontsize=13)

# Raw masked scores
ax = axes[1]
display_scores = np.where(mask == 1, raw_scores, np.nan)
sns.heatmap(display_scores, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=tokens, yticklabels=tokens, ax=ax,
            mask=~mask.astype(bool))
ax.set_xlabel("Key Position (j)")
ax.set_ylabel("Query Position (i)")
ax.set_title("(b) Masked Attention Scores\n(future positions = -∞)", fontsize=13)

# Attention weights after softmax
ax = axes[2]
sns.heatmap(attention_weights, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=tokens, yticklabels=tokens, ax=ax)
ax.set_xlabel("Key Position (j)")
ax.set_ylabel("Query Position (i)")
ax.set_title("(c) Attention Weights\n(after softmax)", fontsize=13)

plt.suptitle("Masked Self-Attention Mechanism in GPT-2", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("q7_masked_attention_visualisation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: q7_masked_attention_visualisation.png")

# Plot 2: Multi-head attention visualisation
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for head_idx in range(4):
    np.random.seed(head_idx * 10 + 1)
    raw = np.random.randn(n, n)
    masked = np.where(mask == 1, raw, -1e9)
    weights = softmax(masked, axis=-1)

    ax = axes[head_idx]
    sns.heatmap(weights, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=tokens, yticklabels=tokens, ax=ax)
    ax.set_title(f"Head {head_idx + 1}", fontsize=12)
    if head_idx == 0:
        ax.set_ylabel("Query Position")
    ax.set_xlabel("Key Position")

plt.suptitle("Multi-Head Masked Self-Attention (4 Heads)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("q7_multihead_attention.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: q7_multihead_attention.png")

# Save answer to text file
with open("q7_answer.txt", "w") as f:
    f.write(answer)
print("Saved: q7_answer.txt")
print("\nQuestion 7 complete.")

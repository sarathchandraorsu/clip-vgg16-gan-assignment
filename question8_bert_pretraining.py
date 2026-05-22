"""
Assignment 2 - Question 8 (20 marks / 6.7%)
Theoretical Question: BERT Pre-training Process

This file contains the comprehensive answer to Question 8.

Requirements: pip install matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================================
# ANSWER TO QUESTION 8
# ============================================================================

answer = """
================================================================================
QUESTION 8: The Pre-training Process in BERT
================================================================================

1. DESCRIPTION OF THE BERT PRETRAINING STEP
------------------------------------------------------------------------
BERT (Bidirectional Encoder Representations from Transformers) is pretrained
using two self-supervised tasks on a large corpus (BooksCorpus + Wikipedia,
~3.3 billion words):

  TASK 1: Masked Language Modelling (MLM)
  ----------------------------------------
  - 15% of input tokens are randomly selected for masking.
  - Of the selected tokens:
    * 80% are replaced with [MASK]
    * 10% are replaced with a random token
    * 10% remain unchanged
  - The model must predict the original token at each masked position.
  - Loss: Cross-entropy on the predicted vs. actual masked tokens.

  TASK 2: Next Sentence Prediction (NSP)
  ----------------------------------------
  - The model receives two sentences (A and B).
  - 50% of the time, B is the actual next sentence after A (label: IsNext).
  - 50% of the time, B is a random sentence from the corpus (label: NotNext).
  - The [CLS] token representation is used for binary classification.
  - Loss: Binary cross-entropy.

  Total pretraining loss = MLM loss + NSP loss

  BERT uses a bidirectional transformer encoder, meaning each token can
  attend to ALL other tokens (both left and right context), unlike GPT-2.

2. WHY DOES PRETRAINING IMPROVE BERT'S DOWNSTREAM PERFORMANCE?
------------------------------------------------------------------------
a) Rich Contextual Representations:
   Through MLM, BERT learns deep bidirectional representations where each
   token's embedding captures context from both directions. This is
   fundamentally more informative than unidirectional models for
   understanding tasks (classification, QA, NER).

b) Transfer of Linguistic Knowledge:
   Pretraining on billions of words forces BERT to learn:
   - Syntax (grammatical structure, agreement)
   - Semantics (word meanings, synonymy, polysemy)
   - World knowledge (common facts, relationships)
   - Discourse (sentence relationships via NSP)
   This knowledge transfers to downstream tasks, reducing the need
   for task-specific labelled data.

c) Feature Hierarchy:
   Different layers of BERT capture different levels of linguistic
   information. Lower layers capture syntax, while higher layers
   capture semantics. This hierarchical representation benefits
   diverse downstream tasks.

d) Data Efficiency for Fine-tuning:
   Because BERT has already learned general language understanding,
   fine-tuning on a small labelled dataset (even hundreds of examples)
   can achieve strong performance. Without pretraining, the same
   architecture would require orders of magnitude more labelled data.

e) Regularisation Effect:
   Pretraining provides a strong initialisation that acts as an
   implicit regulariser during fine-tuning, preventing overfitting
   on small downstream datasets.

3. ILLUSTRATION WITH A SIMPLE EXAMPLE
------------------------------------------------------------------------
Consider the sentence pair:

  Sentence A: "The cat sat on the [MASK]."
  Sentence B: "It was very comfortable."

  MLM Task:
  - The model sees: "[CLS] The cat sat on the [MASK] . [SEP] It was very comfortable . [SEP]"
  - It must predict [MASK] = "mat" (or "chair", "couch", etc.)
  - The model uses bidirectional context: "The cat sat on the ___" (left)
    and "It was very comfortable" (right) to predict "mat".

  NSP Task:
  - The model uses the [CLS] representation to predict whether
    Sentence B naturally follows Sentence A.
  - In this case: IsNext = True (it's a coherent continuation).

  Counter-example for NSP:
  - Sentence A: "The cat sat on the mat."
  - Sentence B: "The stock market closed at 5 PM."
  - IsNext = False (random sentence).

4. CHALLENGES OF THE BERT PRETRAINING PHASE IN REAL-WORLD
------------------------------------------------------------------------
a) Computational Cost:
   Pretraining BERT-Large (340M parameters) requires ~64 TPUs for ~4 days.
   This is extremely expensive and carbon-intensive. Most practitioners
   use pretrained checkpoints rather than training from scratch.

b) Pretraining-Fine-tuning Discrepancy:
   The [MASK] token appears during pretraining but never during fine-tuning
   or inference. This train-test mismatch can degrade performance. The
   80/10/10 masking strategy partially mitigates this but doesn't
   eliminate it entirely.

c) Fixed Vocabulary:
   BERT uses WordPiece tokenisation with a fixed vocabulary (30K tokens).
   Out-of-vocabulary words are split into subwords, which may lose
   semantic nuance, especially for domain-specific terminology.

d) Maximum Sequence Length:
   BERT has a maximum input length of 512 tokens. Documents longer than
   this must be truncated or split, losing long-range context.

e) Domain Mismatch:
   BERT is pretrained on general text (Wikipedia + books). When applied
   to specialised domains (medical, legal, scientific), the pretrained
   representations may not capture domain-specific language patterns,
   requiring domain-specific pretraining (e.g., BioBERT, SciBERT).

5. COMPARISON: MASKED LANGUAGE MODELLING (MLM) vs NEXT SENTENCE PREDICTION (NSP)
------------------------------------------------------------------------
  Aspect        | MLM                          | NSP
  ------------- | ---------------------------- | ----------------------------
  Objective     | Predict masked tokens        | Predict sentence relationship
  Granularity   | Token-level                  | Sentence-level
  Context       | Bidirectional within sequence | Cross-sentence
  Contribution  | Primary driver of BERT's     | Helps with sentence-pair
  to Performance| strong representations       | tasks (QA, NLI)
  Criticism     | [MASK] token mismatch        | Later work (RoBERTa) showed
                |                              | NSP may not be necessary
  Impact        | Essential; all BERT variants  | Controversial; RoBERTa
                | retain MLM                   | removes NSP with better results

  Analysis:
  MLM is the dominant pretraining objective. It forces the model to learn
  rich contextual representations by predicting missing words from
  bidirectional context. NSP was intended to help with sentence-pair tasks
  but has been shown by subsequent work (RoBERTa, ALBERT) to be less
  important. RoBERTa achieves better results by removing NSP and training
  with longer sequences. However, NSP does provide some benefit for tasks
  requiring understanding of sentence relationships (e.g., natural language
  inference, question answering).

================================================================================
"""

print(answer)

# ============================================================================
# VISUALISATION: BERT Pretraining Process
# ============================================================================

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis("off")

# Title
ax.text(8, 10.5, "BERT Pretraining Process", fontsize=18, fontweight="bold",
        ha="center", va="center")

# Input representation
ax.add_patch(patches.FancyBboxPatch((0.5, 8.5), 15, 1.2, boxstyle="round,pad=0.1",
             facecolor="#E8F4FD", edgecolor="black", linewidth=1.5))
ax.text(8, 9.4, "Input: [CLS] The cat sat on the [MASK] . [SEP] It was comfortable . [SEP]",
        fontsize=11, ha="center", va="center", fontfamily="monospace")
ax.text(8, 8.7, "Token Emb + Segment Emb + Position Emb", fontsize=9, ha="center",
        va="center", style="italic", color="gray")

# Arrow
ax.annotate("", xy=(8, 8.0), xytext=(8, 8.4),
            arrowprops=dict(arrowstyle="->", lw=2))

# Transformer Encoder
ax.add_patch(patches.FancyBboxPatch((2, 6.2), 12, 1.6, boxstyle="round,pad=0.1",
             facecolor="#FFF3E0", edgecolor="darkorange", linewidth=2))
ax.text(8, 7.3, "Bidirectional Transformer Encoder", fontsize=14, fontweight="bold",
        ha="center", va="center")
ax.text(8, 6.7, "12 layers × 12 heads × 768 hidden (BERT-Base)", fontsize=10,
        ha="center", va="center", color="gray")
ax.text(8, 6.4, "Each token attends to ALL other tokens (bidirectional)", fontsize=9,
        ha="center", va="center", style="italic", color="darkorange")

# Two branches
# MLM branch (left)
ax.annotate("", xy=(4, 5.2), xytext=(6, 6.1),
            arrowprops=dict(arrowstyle="->", lw=2, color="blue"))

ax.add_patch(patches.FancyBboxPatch((1, 3.5), 6, 1.5, boxstyle="round,pad=0.1",
             facecolor="#E3F2FD", edgecolor="blue", linewidth=1.5))
ax.text(4, 4.6, "Task 1: Masked Language\nModelling (MLM)", fontsize=12,
        fontweight="bold", ha="center", va="center", color="blue")
ax.text(4, 3.8, "Predict: [MASK] → 'mat'", fontsize=10, ha="center",
        va="center", fontfamily="monospace")

# NSP branch (right)
ax.annotate("", xy=(12, 5.2), xytext=(10, 6.1),
            arrowprops=dict(arrowstyle="->", lw=2, color="green"))

ax.add_patch(patches.FancyBboxPatch((9, 3.5), 6, 1.5, boxstyle="round,pad=0.1",
             facecolor="#E8F5E9", edgecolor="green", linewidth=1.5))
ax.text(12, 4.6, "Task 2: Next Sentence\nPrediction (NSP)", fontsize=12,
        fontweight="bold", ha="center", va="center", color="green")
ax.text(12, 3.8, "[CLS] → IsNext / NotNext", fontsize=10, ha="center",
        va="center", fontfamily="monospace")

# Loss
ax.annotate("", xy=(8, 2.2), xytext=(4, 3.4),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="blue"))
ax.annotate("", xy=(8, 2.2), xytext=(12, 3.4),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="green"))

ax.add_patch(patches.FancyBboxPatch((5, 1.2), 6, 1, boxstyle="round,pad=0.1",
             facecolor="#FCE4EC", edgecolor="red", linewidth=1.5))
ax.text(8, 1.7, "Total Loss = MLM Loss + NSP Loss", fontsize=12,
        fontweight="bold", ha="center", va="center", color="red")

# Footer
ax.text(8, 0.5, "Pretrained on: BooksCorpus + English Wikipedia (~3.3B words)",
        fontsize=10, ha="center", va="center", style="italic", color="gray")

plt.tight_layout()
plt.savefig("q8_bert_pretraining_diagram.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: q8_bert_pretraining_diagram.png")

# Save answer
with open("q8_answer.txt", "w") as f:
    f.write(answer)
print("Saved: q8_answer.txt")
print("\nQuestion 8 complete.")

"""
Assignment 2 - Question 9 (20 marks / 6.7%)
Theoretical Question: Fine-tuning Mechanism in CLIP

This file contains the comprehensive answer to Question 9.

Requirements: pip install matplotlib numpy
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================================
# ANSWER TO QUESTION 9
# ============================================================================

answer = """
================================================================================
QUESTION 9: Fine-tuning Mechanism in CLIP
================================================================================

1. EXPLANATION OF THE FINE-TUNING PHASE OF CLIP
------------------------------------------------------------------------
CLIP (Contrastive Language-Image Pre-training) is originally designed for
zero-shot prediction: given an image and a set of text descriptions, CLIP
computes similarity scores and selects the most matching text without any
task-specific training.

However, zero-shot performance may be insufficient for specialised
downstream tasks. The fine-tuning (transfer learning) approach adapts
CLIP to specific datasets by training additional components while
leveraging CLIP's pretrained representations.

The fine-tuning process typically involves:

  Step 1: Load the pretrained CLIP model (image encoder + text encoder).

  Step 2: Freeze the encoder parameters. The pretrained weights encode
          rich visual and linguistic knowledge that we want to preserve.

  Step 3: Add trainable adapter / projection layers on top of the frozen
          encoders. These lightweight modules map CLIP's features to a
          task-specific embedding space.

  Step 4: Train the adapters using contrastive loss (or task-specific loss)
          on the downstream dataset. The contrastive loss maximises the
          similarity between matched image-text pairs and minimises the
          similarity between unmatched pairs.

  Step 5: At inference time, compute image and text features through the
          frozen encoders and adapted projections, then select the class
          with the highest image-text similarity.

2. THE ROLE OF ADAPTERS IN FINE-TUNING CLIP
------------------------------------------------------------------------
Adapters are small, trainable neural network modules inserted between or
on top of frozen pretrained layers. They play several critical roles:

  a) Parameter Efficiency:
     Instead of fine-tuning the entire CLIP model (which may have hundreds
     of millions of parameters), adapters add only a small number of
     trainable parameters (typically <1% of the total model). This makes
     fine-tuning feasible with limited compute and data.

  b) Knowledge Preservation:
     By keeping the encoder parameters frozen and only training adapters,
     the rich general-purpose representations learned during pretraining
     are preserved. This prevents catastrophic forgetting of the pretrained
     knowledge.

  c) Domain Adaptation:
     Adapters learn to transform CLIP's general features into
     task/domain-specific representations. For example, adapting CLIP
     features from natural images to medical images or satellite imagery.

  d) Modular Design:
     Different adapters can be trained for different tasks while sharing
     the same frozen backbone. This enables multi-task learning with
     minimal additional parameters per task.

  Common adapter architectures include:
  - Linear projection layers (simplest)
  - Bottleneck adapters (down-project → nonlinearity → up-project)
  - Residual adapters (adapter output + skip connection)

3. ADVANTAGES OF FINE-TUNING OVER ZERO-SHOT LEARNING
------------------------------------------------------------------------
  a) Higher Task-Specific Accuracy:
     Fine-tuning adapts CLIP's representations to the specific data
     distribution and class structure of the downstream task, typically
     achieving 10-30% higher accuracy than zero-shot on specialised
     datasets.

  b) Custom Class Definitions:
     Zero-shot relies on textual class descriptions that may not capture
     fine-grained distinctions. Fine-tuning learns to distinguish classes
     based on actual visual/textual patterns in the data.

  c) Handling Domain Shift:
     When the downstream domain differs significantly from CLIP's
     pretraining data (web-scraped image-text pairs), fine-tuning
     bridges this gap.

  d) Learning Task-Specific Features:
     Fine-tuning can learn features that are important for the specific
     task but may not be well-represented in CLIP's zero-shot prompts.

  e) Improved Calibration:
     Fine-tuned models tend to have better-calibrated confidence scores
     for the target task distribution.

4. COMPLEXITY OF THE FINE-TUNING PHASE
------------------------------------------------------------------------
  a) Computational Complexity:
     - Forward pass through frozen encoders: O(n) per sample
       (same as zero-shot, but required for training)
     - Adapter training: O(d_adapter × d_clip) per sample
       Much cheaper than full fine-tuning
     - Total: significantly less than training from scratch

  b) Data Requirements:
     - Needs a labelled downstream dataset (unlike zero-shot)
     - Typically requires hundreds to thousands of samples
     - More data generally improves fine-tuning quality

  c) Hyperparameter Tuning:
     - Learning rate (critical: too high destroys features, too low
       leads to slow convergence)
     - Adapter architecture (depth, width, activation)
     - Training epochs (risk of overfitting with small datasets)
     - Loss function (contrastive, cross-entropy, or hybrid)

  d) Memory Requirements:
     - Must store frozen encoder parameters in memory (inference only)
     - Gradient computation only for adapter parameters
     - Significantly less GPU memory than full fine-tuning

5. OTHER FINE-TUNING STRATEGIES FOR CLIP
------------------------------------------------------------------------
  a) LoRA (Low-Rank Adaptation):
     LoRA decomposes the weight update matrix into two low-rank matrices:
     ΔW = B × A, where B ∈ R^{d×r} and A ∈ R^{r×d} with r << d.
     This is injected into specific attention layers of the frozen encoder.
     LoRA is extremely parameter-efficient (often <0.1% of total params)
     and avoids additional inference latency since ΔW can be merged with
     the original weights.

  b) Prompt Tuning / CoOp (Context Optimization):
     Instead of modifying the model, prompt tuning learns continuous
     "soft prompts" that are prepended to the text input. For example,
     instead of "a photo of a [class]", learnable vectors are optimised
     to provide optimal context for the downstream task. This is
     extremely lightweight (only a few hundred parameters).

  c) CLIP-Adapter:
     Adds lightweight residual-style adapters after the encoder features
     with a gating mechanism. The adapter output is combined with the
     original features via a learnable blend ratio.

  d) Tip-Adapter:
     A training-free adaptation method that constructs a key-value cache
     from the support set and performs retrieval-based adaptation.
     Can be further refined with a small amount of training (Tip-Adapter-F).

  e) Visual Prompt Tuning (VPT):
     Similar to text prompt tuning but applied to the visual encoder.
     Learnable tokens are prepended to the input visual tokens of the
     Vision Transformer, allowing task-specific visual adaptation.

  f) Full Fine-tuning:
     All parameters (including encoder weights) are updated. This is the
     most expressive but risks catastrophic forgetting of pretrained
     knowledge and requires more data and compute.

================================================================================
"""

print(answer)

# ============================================================================
# VISUALISATION: CLIP Fine-tuning Architecture
# ============================================================================

fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 13)
ax.axis("off")

# Title
ax.text(9, 12.5, "CLIP Fine-tuning Architecture with Adapters", fontsize=18,
        fontweight="bold", ha="center")

# --- Image Branch (Left) ---
# Image input
ax.add_patch(patches.FancyBboxPatch((1, 10), 3.5, 1, boxstyle="round,pad=0.1",
             facecolor="#E8F4FD", edgecolor="black", linewidth=1.5))
ax.text(2.75, 10.5, "Input Image", fontsize=11, ha="center", va="center", fontweight="bold")

ax.annotate("", xy=(2.75, 8.8), xytext=(2.75, 9.9),
            arrowprops=dict(arrowstyle="->", lw=2))

# Image Encoder (Frozen)
ax.add_patch(patches.FancyBboxPatch((0.5, 7.2), 4.5, 1.5, boxstyle="round,pad=0.1",
             facecolor="#FFECB3", edgecolor="darkorange", linewidth=2))
ax.text(2.75, 8.2, "Image Encoder", fontsize=12, ha="center", va="center", fontweight="bold")
ax.text(2.75, 7.7, "(ViT / ResNet)", fontsize=10, ha="center", va="center", color="gray")
ax.text(2.75, 7.4, "🔒 FROZEN", fontsize=9, ha="center", va="center", color="red",
        fontweight="bold")

ax.annotate("", xy=(2.75, 5.8), xytext=(2.75, 7.1),
            arrowprops=dict(arrowstyle="->", lw=2))

# Image Adapter (Trainable)
ax.add_patch(patches.FancyBboxPatch((0.5, 4.5), 4.5, 1.2, boxstyle="round,pad=0.1",
             facecolor="#C8E6C9", edgecolor="green", linewidth=2))
ax.text(2.75, 5.4, "Image Adapter", fontsize=12, ha="center", va="center", fontweight="bold")
ax.text(2.75, 4.9, "Linear → GELU → Linear", fontsize=9, ha="center", va="center", color="gray")
ax.text(2.75, 4.65, "🔓 TRAINABLE", fontsize=9, ha="center", va="center", color="green",
        fontweight="bold")

ax.annotate("", xy=(2.75, 3.2), xytext=(2.75, 4.4),
            arrowprops=dict(arrowstyle="->", lw=2, color="blue"))

# Image Features
ax.add_patch(patches.FancyBboxPatch((1, 2.5), 3.5, 0.6, boxstyle="round,pad=0.1",
             facecolor="#BBDEFB", edgecolor="blue", linewidth=1.5))
ax.text(2.75, 2.8, "Image Features (f_img)", fontsize=10, ha="center", va="center",
        fontfamily="monospace")

# --- Text Branch (Right) ---
# Text input
ax.add_patch(patches.FancyBboxPatch((13, 10), 3.5, 1, boxstyle="round,pad=0.1",
             facecolor="#F3E5F5", edgecolor="black", linewidth=1.5))
ax.text(14.75, 10.5, "Text Prompt", fontsize=11, ha="center", va="center", fontweight="bold")

ax.annotate("", xy=(14.75, 8.8), xytext=(14.75, 9.9),
            arrowprops=dict(arrowstyle="->", lw=2))

# Text Encoder (Frozen)
ax.add_patch(patches.FancyBboxPatch((13, 7.2), 4.5, 1.5, boxstyle="round,pad=0.1",
             facecolor="#FFECB3", edgecolor="darkorange", linewidth=2))
ax.text(14.75, 8.2, "Text Encoder", fontsize=12, ha="center", va="center", fontweight="bold")
ax.text(14.75, 7.7, "(Transformer)", fontsize=10, ha="center", va="center", color="gray")
ax.text(14.75, 7.4, "🔒 FROZEN", fontsize=9, ha="center", va="center", color="red",
        fontweight="bold")

ax.annotate("", xy=(14.75, 5.8), xytext=(14.75, 7.1),
            arrowprops=dict(arrowstyle="->", lw=2))

# Text Adapter (Trainable)
ax.add_patch(patches.FancyBboxPatch((13, 4.5), 4.5, 1.2, boxstyle="round,pad=0.1",
             facecolor="#C8E6C9", edgecolor="green", linewidth=2))
ax.text(14.75, 5.4, "Text Adapter", fontsize=12, ha="center", va="center", fontweight="bold")
ax.text(14.75, 4.9, "Linear → GELU → Linear", fontsize=9, ha="center", va="center", color="gray")
ax.text(14.75, 4.65, "🔓 TRAINABLE", fontsize=9, ha="center", va="center", color="green",
        fontweight="bold")

ax.annotate("", xy=(14.75, 3.2), xytext=(14.75, 4.4),
            arrowprops=dict(arrowstyle="->", lw=2, color="purple"))

# Text Features
ax.add_patch(patches.FancyBboxPatch((13.2, 2.5), 3.5, 0.6, boxstyle="round,pad=0.1",
             facecolor="#E1BEE7", edgecolor="purple", linewidth=1.5))
ax.text(14.95, 2.8, "Text Features (f_txt)", fontsize=10, ha="center", va="center",
        fontfamily="monospace")

# --- Similarity / Loss (Center) ---
ax.annotate("", xy=(8.75, 1.5), xytext=(4.6, 2.7),
            arrowprops=dict(arrowstyle="->", lw=2, color="red"))
ax.annotate("", xy=(8.75, 1.5), xytext=(13.1, 2.7),
            arrowprops=dict(arrowstyle="->", lw=2, color="red"))

ax.add_patch(patches.FancyBboxPatch((6, 0.5), 5.5, 1.2, boxstyle="round,pad=0.1",
             facecolor="#FFCDD2", edgecolor="red", linewidth=2))
ax.text(8.75, 1.4, "Contrastive Loss", fontsize=13, ha="center", va="center",
        fontweight="bold", color="red")
ax.text(8.75, 0.8, "cosine_sim(f_img, f_txt) → maximize matched pairs",
        fontsize=9, ha="center", va="center", fontfamily="monospace")

# Legend
ax.add_patch(patches.Rectangle((6.5, 9.5), 0.4, 0.3, facecolor="#FFECB3", edgecolor="darkorange"))
ax.text(7.1, 9.65, "= Frozen (pretrained)", fontsize=9, va="center")
ax.add_patch(patches.Rectangle((6.5, 9.0), 0.4, 0.3, facecolor="#C8E6C9", edgecolor="green"))
ax.text(7.1, 9.15, "= Trainable (adapters)", fontsize=9, va="center")

plt.tight_layout()
plt.savefig("q9_clip_finetuning_diagram.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: q9_clip_finetuning_diagram.png")

# Save answer
with open("q9_answer.txt", "w") as f:
    f.write(answer)
print("Saved: q9_answer.txt")
print("\nQuestion 9 complete.")

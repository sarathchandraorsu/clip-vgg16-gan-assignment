"""
Assignment 2 - Question 10 (20 marks / 6.7%)
Theoretical Question: Why Pretrained Models Outperform Training from Scratch

This file contains the comprehensive answer to Question 10.

Requirements: pip install matplotlib numpy
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ============================================================================
# ANSWER TO QUESTION 10
# ============================================================================

answer = """
================================================================================
QUESTION 10: Why Pretrained Models Usually Perform Better Than Models
             Trained from Scratch
================================================================================

1. WHY PRETRAINED MODELS LEAD TO BETTER RESULTS
------------------------------------------------------------------------
a) Superior Weight Initialisation:
   Pretrained models start with weights that already encode meaningful
   visual features learned from millions of images. Random initialisation
   places the model at an arbitrary point in the loss landscape, whereas
   pretraining places it near a region of good solutions. This results
   in faster convergence and often a better final solution.

b) Hierarchical Feature Transfer:
   Deep neural networks learn hierarchical features:
   - Layer 1-2: Edges, corners, colour gradients (universal)
   - Layer 3-5: Textures, patterns, simple shapes (mostly universal)
   - Layer 6-10: Object parts, complex patterns (somewhat task-specific)
   - Final layers: Task-specific features
   The early and middle layers learn features that are highly transferable
   across visual tasks. Pretraining captures these universal features,
   which would otherwise need to be learned from scratch.

c) Data Efficiency:
   Training deep networks from scratch requires large amounts of data to
   learn good representations. Pretrained models leverage knowledge from
   massive datasets (e.g., ImageNet with 1.2M images), enabling good
   performance even on small downstream datasets with hundreds of images.

d) Regularisation Effect:
   Pretrained weights constrain the model's parameter space, acting as
   a strong prior. This implicit regularisation reduces overfitting,
   especially beneficial when the downstream dataset is small.

e) Reduced Training Time:
   Since the feature extractor is already trained, only the classifier
   needs to be adapted. This dramatically reduces computation:
   - From scratch: Train all ~138M parameters (VGG16) for many epochs
   - Transfer: Train only ~4M classifier parameters for few epochs

f) Better Generalisation:
   Models trained from scratch on limited data tend to memorise training
   examples rather than learning generalisable patterns. Pretrained
   features, having been validated across millions of diverse images,
   generalise better to unseen data.

2. THE COMMONLY USED DATASET FOR PRETRAINING IN COMPUTER VISION
------------------------------------------------------------------------
The most widely used dataset for pretraining computer vision models is
ImageNet (specifically, ImageNet-1K or ILSVRC):

  ImageNet-1K (ILSVRC) Statistics:
  - 1,281,167 training images
  - 50,000 validation images
  - 1,000 object categories
  - Categories span animals, vehicles, food, instruments, etc.
  - Images are natural, real-world photographs
  - Resolution varies (typically resized to 224×224 or 256×256)

  Other pretraining datasets include:
  - ImageNet-21K: 14 million images, 21,841 categories
  - JFT-300M (Google): 300 million images, 18,291 categories (proprietary)
  - LAION-5B: 5 billion image-text pairs (for CLIP-style pretraining)
  - COCO: 330K images with 80 categories (used for detection/segmentation)
  - Places365: 1.8 million images for scene recognition

3. WHY IMAGENET IS COMMONLY USED FOR PRETRAINING
------------------------------------------------------------------------
a) Diversity: 1,000 categories cover a broad range of visual concepts,
   ensuring the model learns diverse features applicable to many domains.

b) Scale: 1.28M images provide sufficient data to train deep networks
   without severe overfitting.

c) Standardisation: ImageNet has been the de facto benchmark since the
   ILSVRC competition (2010-2017). All major architectures (AlexNet,
   VGG, ResNet, Inception, EfficientNet) provide ImageNet-pretrained
   weights, making it the standard transfer learning source.

d) Quality: Images are curated and labelled by human annotators,
   ensuring reliable ground truth.

e) Proven Effectiveness: Decades of research have demonstrated that
   ImageNet-pretrained features transfer well to diverse downstream
   tasks including medical imaging, satellite imagery, fine-grained
   recognition, and action recognition.

f) Accessibility: ImageNet is freely available for research, and all
   major deep learning frameworks (PyTorch, TensorFlow) provide
   pretrained model checkpoints.

4. STEPS TO APPLY A PRETRAINED MODEL TO DOWNSTREAM TASKS
------------------------------------------------------------------------
  Step 1: Select a pretrained model architecture (e.g., VGG16, ResNet50,
          EfficientNet) with weights pretrained on ImageNet.

  Step 2: Load the pretrained weights into the model.
          Example: model = models.resnet50(weights='IMAGENET1K_V1')

  Step 3: Freeze the feature extractor layers (convolutional layers).
          for param in model.features.parameters():
              param.requires_grad = False

  Step 4: Replace the final classification layer(s) to match the number
          of classes in the downstream task.
          model.fc = nn.Linear(2048, num_downstream_classes)

  Step 5: Prepare the downstream dataset with appropriate preprocessing
          (resize to the model's expected input size, apply the same
          normalisation used during pretraining).

  Step 6: Train only the new classification layers on the downstream
          dataset using a suitable optimizer and learning rate.

  Step 7: (Optional) Fine-tune the entire network with a very small
          learning rate to adapt all layers to the downstream task.

  Step 8: Evaluate on a held-out test set and iterate on hyperparameters.

5. LIMITATIONS OF PRETRAINED MODELS IN PRACTICE
------------------------------------------------------------------------
a) Domain Gap:
   If the downstream domain is very different from ImageNet (e.g., medical
   X-rays, microscopy, satellite imagery), the pretrained features may
   not transfer well. The visual statistics (textures, shapes, colours)
   may differ significantly.

b) Task Mismatch:
   ImageNet pretraining is for classification. If the downstream task is
   very different (e.g., object detection, segmentation, pose estimation),
   additional task-specific components and training are needed.

c) Bias Inheritance:
   Pretrained models inherit biases from the pretraining dataset. ImageNet
   has known biases (geographic, cultural, representational) that may
   propagate to downstream applications, leading to unfair outcomes.

d) Fixed Architecture:
   Using a pretrained model constrains the architecture choice. If the
   optimal architecture for the downstream task differs, the pretrained
   model may be suboptimal.

e) Computational Overhead:
   Large pretrained models (e.g., ViT-Large) require significant memory
   and compute even when only the classifier is trained, as forward
   passes through frozen layers are still needed.

f) Negative Transfer:
   In rare cases, pretrained features can hurt performance if the source
   and target domains are too dissimilar. The model may learn to rely on
   irrelevant features, leading to worse results than training from scratch.

g) License and Privacy Concerns:
   Some pretraining datasets may contain copyrighted or personally
   identifiable content, raising legal and ethical concerns.

================================================================================
"""

print(answer)

# ============================================================================
# VISUALISATION: Transfer Learning Pipeline
# ============================================================================

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis("off")

ax.text(8, 11.5, "Transfer Learning Pipeline for Vision Models", fontsize=18,
        fontweight="bold", ha="center")

# Step boxes
steps = [
    ("Step 1", "Load Pretrained\nModel (ImageNet)", "#E3F2FD", 1, 9.5),
    ("Step 2", "Freeze Feature\nExtractor Layers", "#FFF3E0", 5, 9.5),
    ("Step 3", "Replace Classifier\nfor N Classes", "#E8F5E9", 9, 9.5),
    ("Step 4", "Train Classifier\non Target Data", "#FCE4EC", 13, 9.5),
]

for step_name, desc, colour, x, y in steps:
    ax.add_patch(patches.FancyBboxPatch((x - 1.5, y - 0.8), 3, 1.6,
                 boxstyle="round,pad=0.1", facecolor=colour, edgecolor="black", linewidth=1.5))
    ax.text(x, y + 0.2, step_name, fontsize=11, fontweight="bold", ha="center", va="center")
    ax.text(x, y - 0.3, desc, fontsize=9, ha="center", va="center")

# Arrows between steps
for i in range(3):
    x_start = steps[i][3] + 1.5
    x_end = steps[i + 1][3] - 1.5
    ax.annotate("", xy=(x_end, 9.5), xytext=(x_start, 9.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="gray"))

# ImageNet feature hierarchy visualisation
feature_layers = [
    ("Layers 1-2", "Edges &\nCorners", "#BBDEFB", 2),
    ("Layers 3-5", "Textures &\nPatterns", "#90CAF9", 5.5),
    ("Layers 6-10", "Object\nParts", "#64B5F6", 9),
    ("FC Layers", "Task-Specific\nClassification", "#42A5F5", 12.5),
]

ax.text(8, 7.5, "Feature Hierarchy in Pretrained CNNs", fontsize=14,
        fontweight="bold", ha="center")

for name, desc, colour, x in feature_layers:
    ax.add_patch(patches.FancyBboxPatch((x - 1.2, 5.8), 2.4, 1.2,
                 boxstyle="round,pad=0.1", facecolor=colour, edgecolor="black", linewidth=1))
    ax.text(x, 6.8, name, fontsize=10, fontweight="bold", ha="center", va="center")
    ax.text(x, 6.1, desc, fontsize=8, ha="center", va="center")

# Bracket for frozen / trainable
ax.annotate("", xy=(0.8, 5.5), xytext=(10.2, 5.5),
            arrowprops=dict(arrowstyle="-", lw=2, color="red"))
ax.text(5.5, 5.2, "FROZEN (pretrained features)", fontsize=11, ha="center",
        color="red", fontweight="bold")

ax.annotate("", xy=(11.3, 5.5), xytext=(13.7, 5.5),
            arrowprops=dict(arrowstyle="-", lw=2, color="green"))
ax.text(12.5, 5.2, "TRAINABLE", fontsize=11, ha="center",
        color="green", fontweight="bold")

# Comparison table
ax.text(8, 4.2, "Why Pretrained > From Scratch", fontsize=14,
        fontweight="bold", ha="center")

table_data = [
    ("Aspect", "Pretrained", "From Scratch"),
    ("Initialisation", "Meaningful features", "Random weights"),
    ("Convergence", "Fast (few epochs)", "Slow (many epochs)"),
    ("Data needed", "Small (100s)", "Large (10,000s+)"),
    ("Generalisation", "Strong", "Risk of overfitting"),
    ("Compute cost", "Low", "High"),
]

for i, (col1, col2, col3) in enumerate(table_data):
    y = 3.7 - i * 0.45
    weight = "bold" if i == 0 else "normal"
    ax.text(3, y, col1, fontsize=9, ha="center", fontweight=weight)
    ax.text(8, y, col2, fontsize=9, ha="center", fontweight=weight, color="blue")
    ax.text(13, y, col3, fontsize=9, ha="center", fontweight=weight, color="red")

# Horizontal lines for table
for i in range(len(table_data) + 1):
    y = 3.9 - i * 0.45
    ax.plot([1, 15], [y, y], "k-", lw=0.5, alpha=0.3)

plt.tight_layout()
plt.savefig("q10_transfer_learning_diagram.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: q10_transfer_learning_diagram.png")

# Save answer
with open("q10_answer.txt", "w") as f:
    f.write(answer)
print("Saved: q10_answer.txt")
print("\nQuestion 10 complete.")

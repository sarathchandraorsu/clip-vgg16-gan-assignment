"""
Assignment 2 - Question 11 (20 marks / 6.8%)
Theoretical Question: Generative Adversarial Network (GAN)

This file contains the comprehensive answer to Question 11.

Requirements: pip install matplotlib numpy
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ============================================================================
# ANSWER TO QUESTION 11
# ============================================================================

answer = """
================================================================================
QUESTION 11: Generative Adversarial Network (GAN)
================================================================================

1. ROLE OF THE GENERATOR AND DISCRIMINATOR
------------------------------------------------------------------------
A GAN consists of two neural networks that are trained simultaneously
in an adversarial (competitive) framework:

  GENERATOR (G):
  - Input: A random noise vector z ~ N(0, I) from a latent space.
  - Output: A synthetic (fake) image G(z).
  - Objective: Generate images that are indistinguishable from real images.
  - Role: Acts as a "counterfeiter" trying to produce convincing fakes.
  - Architecture: Typically uses transposed convolutions (deconvolutions)
    to upsample from the low-dimensional latent space to the image space.
    Batch normalisation and ReLU activations are commonly used, with
    Tanh at the output to produce pixel values in [-1, 1].

  DISCRIMINATOR (D):
  - Input: An image (either real from the dataset or fake from G).
  - Output: A scalar probability D(x) ∈ [0, 1] indicating whether the
    input is real (1) or fake (0).
  - Objective: Correctly classify images as real or fake.
  - Role: Acts as a "detective" trying to identify counterfeits.
  - Architecture: Typically uses strided convolutions for downsampling.
    LeakyReLU activations and batch normalisation are common, with
    Sigmoid at the output for binary classification.

  The two networks play a minimax game: the Generator tries to maximise
  the probability that the Discriminator misclassifies fakes as real,
  while the Discriminator tries to maximise its classification accuracy.

2. LOSS FUNCTIONS
------------------------------------------------------------------------
The original GAN objective is a minimax game:

  min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]

  DISCRIMINATOR LOSS:
  L_D = -E_{x~p_data}[log D(x)] - E_{z~p_z}[log(1 - D(G(z)))]

  This has two components:
  (a) -log D(x): Penalises the Discriminator for NOT assigning high
      probability to real images. We want D(x) → 1 for real images.
  (b) -log(1 - D(G(z))): Penalises the Discriminator for NOT assigning
      low probability to fake images. We want D(G(z)) → 0 for fakes.

  The Discriminator minimises this loss (equivalently, maximises V(D,G)).

  GENERATOR LOSS:
  L_G = -E_{z~p_z}[log D(G(z))]

  In practice, instead of minimising log(1 - D(G(z))) (which has weak
  gradients when D is confident), we maximise log D(G(z)). This is the
  "non-saturating" loss variant:
  - The Generator wants D(G(z)) → 1, i.e., wants the Discriminator
    to believe the fake images are real.

  TRAINING PROCESS:
  For each training iteration:
    1. Sample a mini-batch of real images x from the dataset.
    2. Sample a mini-batch of noise vectors z ~ N(0, I).
    3. Generate fake images: G(z).
    4. Update Discriminator:
       - Compute D(x) on real images (should be close to 1).
       - Compute D(G(z)) on fake images (should be close to 0).
       - Compute L_D and update D's parameters via gradient descent.
    5. Update Generator:
       - Generate new fake images G(z').
       - Compute D(G(z')) (should be close to 1 from G's perspective).
       - Compute L_G and update G's parameters via gradient descent.
    6. Repeat until convergence (Nash equilibrium).

  At the theoretical Nash equilibrium:
  - G generates images from the true data distribution: p_G = p_data.
  - D outputs 0.5 for all images (cannot distinguish real from fake).

3. NETWORK STRUCTURE
------------------------------------------------------------------------
  GENERATOR (for 28×28 Fashion MNIST):
  z (100,) → FC → Reshape (256, 7, 7) → ConvTranspose2d (128, 14, 14)
  → ConvTranspose2d (64, 28, 28) → Conv2d (1, 28, 28) → Tanh

  DISCRIMINATOR (for 28×28 Fashion MNIST):
  Image (1, 28, 28) → Conv2d (64, 14, 14) → Conv2d (128, 7, 7)
  → Conv2d (256, 4, 4) → Conv2d (1, 1, 1) → Sigmoid

4. REAL-WORLD APPLICATIONS OF GANS
------------------------------------------------------------------------
a) Image Synthesis: Generating photorealistic faces (StyleGAN),
   landscapes, and objects for creative and design applications.

b) Image-to-Image Translation: Pix2Pix and CycleGAN for tasks like
   day→night, sketch→photo, satellite→map conversions.

c) Super-Resolution: SRGAN enhances low-resolution images to
   high-resolution, useful in medical imaging and surveillance.

d) Data Augmentation: Generating synthetic training data for
   domains with limited labelled samples (medical imaging, rare events).

e) Inpainting: Filling in missing or corrupted regions of images.

f) Video Generation: Generating short video sequences from noise
   or conditioning information.

g) Drug Discovery: Generating molecular structures with desired
   chemical properties.

h) Text-to-Image: Generating images from text descriptions
   (StackGAN, AttnGAN).

5. CHALLENGES IN THE TRAINING PROCESS OF GANs
------------------------------------------------------------------------
a) Mode Collapse:
   The Generator learns to produce only a small subset of possible
   outputs that fool the Discriminator. Instead of covering the full
   data distribution, it "collapses" to a few modes. For example,
   generating only one type of digit in MNIST.

b) Training Instability:
   The adversarial training is inherently unstable. The Generator
   and Discriminator losses oscillate rather than converge smoothly.
   If one network becomes much stronger than the other, training
   fails (the Discriminator can easily distinguish all fakes, so
   the Generator receives uninformative gradients).

c) Vanishing Gradients:
   If the Discriminator becomes too strong, D(G(z)) → 0 for all
   generated images. The gradient of log(1 - D(G(z))) becomes very
   small, providing almost no learning signal to the Generator.

d) Evaluation Difficulty:
   There is no single, universally accepted metric for GAN quality.
   Common metrics include FID (Fréchet Inception Distance), IS
   (Inception Score), and visual inspection, but each has limitations.

e) Hyperparameter Sensitivity:
   GANs are notoriously sensitive to learning rates, architecture
   choices, batch sizes, and noise dimensions. Small changes can
   lead to training failure.

f) No Convergence Guarantee:
   Unlike standard optimisation, the minimax game does not guarantee
   convergence to the Nash equilibrium. Training may oscillate
   indefinitely.

6. GANs COMPARED TO OTHER GENERATIVE MODELS
------------------------------------------------------------------------
  Aspect         | GAN              | VAE               | Diffusion Model
  -------------- | ---------------- | ----------------- | ---------------
  Output Quality | Sharp, realistic | Blurry            | Very sharp
  Training       | Unstable         | Stable (ELBO)     | Stable
  Diversity      | Mode collapse    | Good diversity    | Excellent diversity
  Speed (gen)    | Fast (1 pass)    | Fast (1 pass)     | Slow (many steps)
  Likelihood     | No explicit      | Approximate ELBO  | Tractable
  Latent Space   | Unstructured     | Structured/smooth | N/A (noise schedule)
  Control        | Limited          | Good (latent manip)| Good (conditioning)

  Analysis:
  - GANs excel at generating sharp, high-quality images but suffer from
    training instability and mode collapse.
  - VAEs provide a principled probabilistic framework with smooth latent
    spaces but tend to produce blurrier outputs due to the KL divergence
    regularisation.
  - Diffusion models (DDPM, Stable Diffusion) have emerged as the current
    state-of-the-art, offering both high quality and diversity, but at
    the cost of slow sampling due to the iterative denoising process.
  - In practice, diffusion models have largely surpassed GANs for
    image generation quality, but GANs remain relevant for applications
    requiring real-time generation due to their single-pass inference.

================================================================================
"""

print(answer)

# ============================================================================
# VISUALISATION: GAN Architecture and Training Process
# ============================================================================

fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 13)
ax.axis("off")

ax.text(9, 12.5, "Generative Adversarial Network (GAN) Architecture", fontsize=18,
        fontweight="bold", ha="center")

# ---- Generator ----
# Noise input
ax.add_patch(patches.FancyBboxPatch((0.5, 7.5), 2, 1, boxstyle="round,pad=0.1",
             facecolor="#E8F4FD", edgecolor="black", linewidth=1.5))
ax.text(1.5, 8, "Noise z\n~N(0,I)", fontsize=10, ha="center", va="center",
        fontfamily="monospace")

ax.annotate("", xy=(3.5, 8), xytext=(2.6, 8),
            arrowprops=dict(arrowstyle="->", lw=2))

# Generator box
ax.add_patch(patches.FancyBboxPatch((3.5, 6.5), 3.5, 3, boxstyle="round,pad=0.2",
             facecolor="#C8E6C9", edgecolor="green", linewidth=2))
ax.text(5.25, 9, "Generator G", fontsize=14, fontweight="bold", ha="center",
        va="center", color="green")
ax.text(5.25, 8.3, "ConvTranspose2d\n↑ Upsample\nBatchNorm + ReLU\nTanh output",
        fontsize=8, ha="center", va="center", fontfamily="monospace")
ax.text(5.25, 6.8, "Goal: G(z) ≈ real data", fontsize=8, ha="center",
        va="center", style="italic", color="darkgreen")

# Fake image output
ax.annotate("", xy=(8, 8), xytext=(7.1, 8),
            arrowprops=dict(arrowstyle="->", lw=2, color="green"))
ax.add_patch(patches.FancyBboxPatch((8, 7.3), 2, 1.4, boxstyle="round,pad=0.1",
             facecolor="#FFECB3", edgecolor="darkorange", linewidth=1.5))
ax.text(9, 8.2, "Fake Image", fontsize=11, ha="center", va="center", fontweight="bold")
ax.text(9, 7.7, "G(z)", fontsize=10, ha="center", va="center", fontfamily="monospace")

# Real image
ax.add_patch(patches.FancyBboxPatch((8, 10), 2, 1.4, boxstyle="round,pad=0.1",
             facecolor="#BBDEFB", edgecolor="blue", linewidth=1.5))
ax.text(9, 10.9, "Real Image", fontsize=11, ha="center", va="center", fontweight="bold")
ax.text(9, 10.4, "x ~ p_data", fontsize=10, ha="center", va="center",
        fontfamily="monospace")

# Arrows to Discriminator
ax.annotate("", xy=(11, 9.5), xytext=(10.1, 10.5),
            arrowprops=dict(arrowstyle="->", lw=2, color="blue"))
ax.annotate("", xy=(11, 9.5), xytext=(10.1, 8),
            arrowprops=dict(arrowstyle="->", lw=2, color="darkorange"))

# Discriminator box
ax.add_patch(patches.FancyBboxPatch((11, 7.5), 3.5, 4, boxstyle="round,pad=0.2",
             facecolor="#FFCDD2", edgecolor="red", linewidth=2))
ax.text(12.75, 11, "Discriminator D", fontsize=14, fontweight="bold",
        ha="center", va="center", color="red")
ax.text(12.75, 10.2, "Conv2d\n↓ Downsample\nLeakyReLU\nSigmoid output",
        fontsize=8, ha="center", va="center", fontfamily="monospace")
ax.text(12.75, 8.5, "Goal: D(x)→1\n       D(G(z))→0", fontsize=9,
        ha="center", va="center", fontfamily="monospace", color="darkred")
ax.text(12.75, 7.8, "Output: Real or Fake?", fontsize=8, ha="center",
        va="center", style="italic", color="darkred")

# Output
ax.annotate("", xy=(15.5, 9.5), xytext=(14.6, 9.5),
            arrowprops=dict(arrowstyle="->", lw=2, color="red"))
ax.add_patch(patches.FancyBboxPatch((15.5, 8.8), 2, 1.4, boxstyle="round,pad=0.1",
             facecolor="#F3E5F5", edgecolor="purple", linewidth=1.5))
ax.text(16.5, 9.7, "P(real)", fontsize=12, ha="center", va="center", fontweight="bold")
ax.text(16.5, 9.2, "∈ [0, 1]", fontsize=10, ha="center", va="center",
        fontfamily="monospace")

# ---- Loss Functions ----
ax.text(9, 5.8, "Loss Functions", fontsize=16, fontweight="bold", ha="center")

# D loss
ax.add_patch(patches.FancyBboxPatch((1, 4), 7, 1.4, boxstyle="round,pad=0.1",
             facecolor="#FFCDD2", edgecolor="red", linewidth=1.5))
ax.text(4.5, 5, "Discriminator Loss:", fontsize=11, fontweight="bold",
        ha="center", va="center", color="red")
ax.text(4.5, 4.4, "L_D = -E[log D(x)] - E[log(1 - D(G(z)))]",
        fontsize=10, ha="center", va="center", fontfamily="monospace")

# G loss
ax.add_patch(patches.FancyBboxPatch((9, 4), 7, 1.4, boxstyle="round,pad=0.1",
             facecolor="#C8E6C9", edgecolor="green", linewidth=1.5))
ax.text(12.5, 5, "Generator Loss:", fontsize=11, fontweight="bold",
        ha="center", va="center", color="green")
ax.text(12.5, 4.4, "L_G = -E[log D(G(z))]",
        fontsize=10, ha="center", va="center", fontfamily="monospace")

# Training loop
ax.text(9, 3.2, "Training Loop (per iteration):", fontsize=12,
        fontweight="bold", ha="center")
steps_text = ("1. Sample real batch x    2. Sample noise z    "
              "3. Generate G(z)    4. Update D (maximize)    "
              "5. Update G (minimize)    6. Repeat")
ax.text(9, 2.6, steps_text, fontsize=9, ha="center", va="center",
        fontfamily="monospace", color="gray")

# Nash equilibrium note
ax.add_patch(patches.FancyBboxPatch((3, 1), 12, 1, boxstyle="round,pad=0.1",
             facecolor="#FFF9C4", edgecolor="goldenrod", linewidth=1.5))
ax.text(9, 1.7, "At Nash Equilibrium:", fontsize=11, fontweight="bold",
        ha="center", va="center")
ax.text(9, 1.25, "p_G = p_data  and  D(x) = 0.5 for all x  (Generator perfectly mimics data)",
        fontsize=9, ha="center", va="center", fontfamily="monospace")

plt.tight_layout()
plt.savefig("q11_gan_architecture_diagram.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: q11_gan_architecture_diagram.png")

# Save answer
with open("q11_answer.txt", "w") as f:
    f.write(answer)
print("Saved: q11_answer.txt")
print("\nQuestion 11 complete.")

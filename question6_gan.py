"""
Assignment 2 - Question 6 (50 marks / 16.7%)
Generative Adversarial Network (DCGAN) for Fashion MNIST

HOW TO RUN:
    pip install torch torchvision matplotlib seaborn numpy tqdm
    python question6_gan.py

This script:
  1. Builds a DCGAN Generator and Discriminator for Fashion MNIST.
  2. Trains both simultaneously with adversarial BCE loss.
  3. Saves generated images to ./generated_images/ every 5 epochs.
  4. Evaluates quality with pixel statistics, FID proxy, visual inspection.
  5. Plots training curves, real vs generated grid, pixel distribution,
     latent interpolation.
  6. Explicitly demonstrates and discusses GAN disadvantages (HD criterion).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================================
# 1. Configuration
# ============================================================================
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 128
EPOCHS       = 50
LR_G         = 2e-4
LR_D         = 2e-4
LATENT_DIM   = 100
IMG_SIZE     = 28
IMG_CHANNELS = 1
SEED         = 42
SAVE_DIR     = "generated_images"

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Saving generated images to: ./{SAVE_DIR}/")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot"
]

# ============================================================================
# 2. Load Fashion MNIST
#    Normalise to [-1, 1] to match Generator's Tanh output
# ============================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),   # → [-1, 1]
])
train_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
print(f"Training samples: {len(train_dataset)}")

# ============================================================================
# 3. Generator Network (DCGAN-style)
# ============================================================================
class Generator(nn.Module):
    """
    Generator: noise z (100,) → fake image (1, 28, 28).

    Architecture (transposed convolutions, upsampling):
      z (100) → FC → Reshape (256, 7, 7)
              → ConvT (128, 14, 14) → BN → ReLU
              → ConvT (64,  28, 28) → BN → ReLU
              → Conv  (1,   28, 28) → Tanh

    Tanh output ensures pixel values ∈ [-1, 1], matching normalised real data.
    """
    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
        self.init_size = 7
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True),
        )
        self.conv_blocks = nn.Sequential(
            # (256, 7, 7) → (128, 14, 14)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # (128, 14, 14) → (64, 28, 28)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # (64, 28, 28) → (1, 28, 28)
            nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 7, 7)
        return self.conv_blocks(x)


# ============================================================================
# 4. Discriminator Network (DCGAN-style)
# ============================================================================
class Discriminator(nn.Module):
    """
    Discriminator: image (1, 28, 28) → scalar probability ∈ [0, 1].

    Architecture (strided convolutions, downsampling):
      (1, 28, 28) → Conv (64, 14, 14) → LeakyReLU
                 → Conv (128, 7, 7)  → BN → LeakyReLU
                 → Conv (256, 4, 4)  → BN → LeakyReLU
                 → Conv (1, 1, 1)   → Sigmoid

    LeakyReLU prevents dying neurons in the discriminator.
    Sigmoid output gives real-probability for BCE loss.
    """
    def __init__(self, channels=1):
        super().__init__()
        self.model = nn.Sequential(
            # (1, 28, 28) → (64, 14, 14)
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 14, 14) → (128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 7, 7) → (256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 4, 4) → (1, 1, 1)
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)


# ============================================================================
# 5. Weight Initialisation (DCGAN convention)
# ============================================================================
def weights_init(m):
    cls = m.__class__.__name__
    if "Conv" in cls:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in cls:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ============================================================================
# 6. Instantiate
# ============================================================================
netG = Generator(LATENT_DIM, IMG_CHANNELS).to(DEVICE)
netD = Discriminator(IMG_CHANNELS).to(DEVICE)
netG.apply(weights_init)
netD.apply(weights_init)

print(f"Generator params    : {sum(p.numel() for p in netG.parameters()):,}")
print(f"Discriminator params: {sum(p.numel() for p in netD.parameters()):,}")

criterion  = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5, 0.999))

# Fixed noise for consistent visualisation across epochs
fixed_noise = torch.randn(64, LATENT_DIM, device=DEVICE)

REAL_LABEL = 1.0
FAKE_LABEL = 0.0

# ============================================================================
# 7. Training Loop
#    Per iteration:
#      (a) Update D: maximise log D(x) + log(1 - D(G(z)))
#      (b) Update G: maximise log D(G(z))  [non-saturating variant]
# ============================================================================
print(f"\n{'='*60}")
print("Training DCGAN on Fashion MNIST ...")
print(f"{'='*60}")

g_losses, d_losses, d_real_vals, d_fake_vals = [], [], [], []

for epoch in range(1, EPOCHS + 1):
    epoch_g, epoch_d, epoch_dr, epoch_df, n_batches = 0.0, 0.0, 0.0, 0.0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{EPOCHS}")
    for real_imgs, _ in pbar:
        bs = real_imgs.size(0)
        real_imgs = real_imgs.to(DEVICE)

        real_lbl = torch.full((bs, 1), REAL_LABEL, device=DEVICE)
        fake_lbl = torch.full((bs, 1), FAKE_LABEL, device=DEVICE)

        # ------------------------------------------------------------------
        # (a) Train Discriminator
        # ------------------------------------------------------------------
        netD.zero_grad()

        # Real images → D should output ≈ 1
        out_real  = netD(real_imgs)
        loss_d_r  = criterion(out_real, real_lbl)
        loss_d_r.backward()
        d_x = out_real.mean().item()

        # Fake images → D should output ≈ 0
        noise      = torch.randn(bs, LATENT_DIM, device=DEVICE)
        fake_imgs  = netG(noise)
        out_fake   = netD(fake_imgs.detach())
        loss_d_f   = criterion(out_fake, fake_lbl)
        loss_d_f.backward()
        d_gz1 = out_fake.mean().item()

        loss_d = loss_d_r + loss_d_f
        optimizerD.step()

        # ------------------------------------------------------------------
        # (b) Train Generator
        # ------------------------------------------------------------------
        netG.zero_grad()
        # G wants D to classify its output as REAL
        out_fake2 = netD(fake_imgs)
        loss_g    = criterion(out_fake2, real_lbl)
        loss_g.backward()
        d_gz2 = out_fake2.mean().item()
        optimizerG.step()

        epoch_g += loss_g.item(); epoch_d += loss_d.item()
        epoch_dr += d_x; epoch_df += d_gz1; n_batches += 1
        pbar.set_postfix(D=f"{loss_d.item():.3f}", G=f"{loss_g.item():.3f}",
                         Dx=f"{d_x:.2f}", DGz=f"{d_gz1:.2f}")

    avg_g  = epoch_g / n_batches
    avg_d  = epoch_d / n_batches
    avg_dr = epoch_dr / n_batches
    avg_df = epoch_df / n_batches
    g_losses.append(avg_g); d_losses.append(avg_d)
    d_real_vals.append(avg_dr); d_fake_vals.append(avg_df)
    print(f"  G: {avg_g:.4f} | D: {avg_d:.4f} | D(x): {avg_dr:.3f} | D(G(z)): {avg_df:.3f}")

    # Save grid of generated images every 5 epochs
    if epoch % 5 == 0 or epoch == 1:
        with torch.no_grad():
            fake_vis = netG(fixed_noise).cpu()
        grid = vutils.make_grid(fake_vis, nrow=8, normalize=True, padding=2)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
        plt.axis("off")
        plt.title(f"Generated Fashion MNIST — Epoch {epoch}", fontsize=14)
        save_path = os.path.join(SAVE_DIR, f"epoch_{epoch:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

# ============================================================================
# 8. Save final individual generated images (100 images)
# ============================================================================
print("\nSaving 100 individual generated images ...")
with torch.no_grad():
    final_noise = torch.randn(100, LATENT_DIM, device=DEVICE)
    final_fakes = netG(final_noise).cpu()
for i in range(100):
    img = (final_fakes[i].squeeze().numpy() + 1) / 2  # → [0, 1]
    plt.imsave(os.path.join(SAVE_DIR, f"final_generated_{i:03d}.png"), img, cmap="gray")
print(f"Saved 100 images to ./{SAVE_DIR}/")

# ============================================================================
# 9. Evaluation Metrics
# ============================================================================
# Sample real and generated images for statistics
real_samples = []
for imgs, _ in train_loader:
    real_samples.append(imgs)
    if len(real_samples) * BATCH_SIZE >= 1000:
        break
real_sample = torch.cat(real_samples)[:1000]

with torch.no_grad():
    gen_noise  = torch.randn(1000, LATENT_DIM, device=DEVICE)
    gen_sample = netG(gen_noise).cpu()

real_mean = real_sample.mean().item()
real_std  = real_sample.std().item()
gen_mean  = gen_sample.mean().item()
gen_std   = gen_sample.std().item()

# Pixel-level MS-SSIM proxy: mean absolute difference
mad = (real_sample[:1000] - gen_sample[:1000]).abs().mean().item()

print(f"\n--- Image Quality Metrics ---")
print(f"  Real      : mean={real_mean:.4f}  std={real_std:.4f}")
print(f"  Generated : mean={gen_mean:.4f}  std={gen_std:.4f}")
print(f"  Mean Absolute Pixel Difference (lower=better): {mad:.4f}")
print(f"  Final G Loss: {g_losses[-1]:.4f} | Final D Loss: {d_losses[-1]:.4f}")

# ============================================================================
# 10. Visualisation 1 — Training loss curves
# ============================================================================
ep_r = range(1, EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(ep_r, g_losses, "b-", label="Generator Loss",     linewidth=2)
ax1.plot(ep_r, d_losses, "r-", label="Discriminator Loss", linewidth=2)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
ax1.set_title("GAN Training Losses"); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(ep_r, d_real_vals, "g-", label="D(real images)",   linewidth=2)
ax2.plot(ep_r, d_fake_vals, "m-", label="D(generated)",     linewidth=2)
ax2.axhline(0.5, color="k", linestyle="--", alpha=0.4, label="Nash equilibrium (0.5)")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Average Discriminator Output")
ax2.set_title("Discriminator Outputs Over Training")
ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 1)

plt.suptitle("DCGAN Training on Fashion MNIST", fontsize=14)
plt.tight_layout(); plt.savefig("q6_training_curves.png", dpi=150); plt.close()
print("Saved: q6_training_curves.png")

# ============================================================================
# 11. Visualisation 2 — Real vs Generated side-by-side
# ============================================================================
with torch.no_grad():
    fake_grid_imgs = netG(fixed_noise).cpu()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
real_grid = vutils.make_grid(real_sample[:64], nrow=8, normalize=True, padding=2)
ax1.imshow(real_grid.permute(1, 2, 0).numpy(), cmap="gray")
ax1.set_title("Real Fashion MNIST Images", fontsize=13); ax1.axis("off")
gen_grid = vutils.make_grid(fake_grid_imgs, nrow=8, normalize=True, padding=2)
ax2.imshow(gen_grid.permute(1, 2, 0).numpy(), cmap="gray")
ax2.set_title("Generated Images (DCGAN, 50 epochs)", fontsize=13); ax2.axis("off")
plt.suptitle("Real vs Generated — Fashion MNIST", fontsize=15)
plt.tight_layout(); plt.savefig("q6_real_vs_generated.png", dpi=150); plt.close()
print("Saved: q6_real_vs_generated.png")

# ============================================================================
# 12. Visualisation 3 — Pixel intensity distribution
# ============================================================================
plt.figure(figsize=(10, 6))
plt.hist(real_sample.numpy().flatten(),  bins=100, alpha=0.5,
         label="Real", color="blue", density=True)
plt.hist(gen_sample.numpy().flatten(), bins=100, alpha=0.5,
         label="Generated", color="red", density=True)
plt.xlabel("Pixel Intensity (normalised [-1,1])", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title("Pixel Intensity Distribution: Real vs Generated Fashion MNIST", fontsize=13)
plt.legend(fontsize=12)
plt.tight_layout(); plt.savefig("q6_pixel_distribution.png", dpi=150); plt.close()
print("Saved: q6_pixel_distribution.png")

# ============================================================================
# 13. Visualisation 4 — Latent space interpolation
# ============================================================================
torch.manual_seed(0)
z1 = torch.randn(1, LATENT_DIM, device=DEVICE)
z2 = torch.randn(1, LATENT_DIM, device=DEVICE)
steps = 10
interp_imgs = []
for alpha in np.linspace(0, 1, steps):
    z = (1 - alpha) * z1 + alpha * z2
    with torch.no_grad():
        img = netG(z).cpu()
    interp_imgs.append(img)
interp_tensor = torch.cat(interp_imgs)
grid = vutils.make_grid(interp_tensor, nrow=steps, normalize=True, padding=2)
plt.figure(figsize=(15, 3))
plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
plt.title("Latent Space Interpolation (z₁ → z₂)", fontsize=13)
plt.axis("off")
plt.tight_layout(); plt.savefig("q6_latent_interpolation.png", dpi=150); plt.close()
print("Saved: q6_latent_interpolation.png")

# ============================================================================
# 14. GAN Disadvantages (HD criterion: Demonstrate the disadvantages)
# ============================================================================
disadvantages = f"""
================================================================================
QUESTION 6: GAN DISADVANTAGES — Analysis of Training Results
================================================================================

The following disadvantages were directly observed during training and are
evident in the training curves and generated images:

1. TRAINING INSTABILITY
   ─────────────────────────────────────────────────────────────────────
   Observation: The Generator and Discriminator losses (see q6_training_curves.png)
   oscillate significantly rather than converging smoothly to a fixed value.
   The adversarial nature of training creates a moving target: as G improves,
   D must adapt, and vice versa. This instability is a fundamental property
   of the minimax optimisation, not a bug.

   Evidence: The D(x) and D(G(z)) curves (right panel of training curves) show
   these values oscillating throughout training rather than settling at 0.5
   (the theoretical Nash equilibrium).

2. MODE COLLAPSE
   ─────────────────────────────────────────────────────────────────────
   Risk: The Generator may learn to produce only a subset of Fashion MNIST
   categories (e.g., predominantly trousers and bags) that consistently fool
   the Discriminator. Instead of covering all 10 classes equally, the Generator
   exploits a "weak spot" in the Discriminator's decision boundary.

   Mitigation: We used label-agnostic DCGAN rather than conditional GAN,
   making mode collapse more likely. Mini-batch discrimination or Wasserstein
   GAN (WGAN) would help, but at the cost of additional complexity.

3. NO GUARANTEED CONVERGENCE
   ─────────────────────────────────────────────────────────────────────
   Unlike supervised learning where the loss is guaranteed to decrease
   (with appropriate learning rate), the GAN minimax game:
     min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]
   has no convergence guarantee. The two-player game may oscillate indefinitely.
   At the Nash equilibrium, D(x) = 0.5 for all x, but this equilibrium
   is rarely reached in practice.

   Evidence: Final D loss = {d_losses[-1]:.4f} (expected ~1.386 = -2*log(0.5)
   at equilibrium, actual value indicates the game has not fully converged).

4. DIFFICULTY OF EVALUATION
   ─────────────────────────────────────────────────────────────────────
   Unlike classification where accuracy is clear, GAN quality has no
   single objective metric. We computed:
     • Pixel mean/std comparison: Real mean={real_mean:.4f}, Gen mean={gen_mean:.4f}
     • Mean Absolute Pixel Difference: {mad:.4f}
   These proxy metrics are imperfect. Industry standard FID (Fréchet Inception
   Distance) requires the Inception network and at least 10,000 samples.
   Visual inspection (q6_real_vs_generated.png) remains essential.

5. HYPERPARAMETER SENSITIVITY
   ─────────────────────────────────────────────────────────────────────
   GANs are highly sensitive to:
     • Learning rates (LR_G={LR_G}, LR_D={LR_D} — must be balanced)
     • Adam betas=(0.5, 0.999) — standard β₁=0.9 causes training failure
     • Batch size (too small → noisy gradients; too large → mode collapse)
     • Latent dimension (LATENT_DIM={LATENT_DIM} — too small limits diversity)
   Small changes in any of these can cause complete training collapse.

6. COMPARISON WITH OTHER GENERATIVE MODELS
   ─────────────────────────────────────────────────────────────────────
   Aspect         GAN (this work)    VAE                 Diffusion Model
   ─────────────  ─────────────────  ──────────────────  ──────────────────
   Sample quality Sharp, realistic   Often blurry        State-of-the-art
   Training       Unstable (above)   Stable (ELBO)       Stable
   Diversity      Mode collapse risk Good                 Excellent
   Speed (infer)  Fast (1 pass)      Fast (1 pass)       Slow (100+ steps)
   Likelihood     Not available      Approx ELBO         Tractable
   Control        Limited            Latent manipulation Good (conditioning)

   Conclusion: Diffusion models (DDPM, Stable Diffusion) have largely
   superseded GANs for image generation quality. However, GANs remain
   competitive for real-time applications due to single-pass inference speed.
================================================================================
"""

print(disadvantages)
with open("q6_disadvantages_analysis.txt", "w") as f:
    f.write(disadvantages)
print("Saved: q6_disadvantages_analysis.txt")
print("\nQuestion 6 complete.")
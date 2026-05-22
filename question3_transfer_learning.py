"""
Assignment 2 - Question 3 (20 marks / 6.7%)
Transfer Learning with Pretrained VGG16 on Fashion MNIST

HOW TO RUN:
    pip install torch torchvision matplotlib seaborn scikit-learn tqdm
    python question3_transfer_learning.py

This script:
  1. Loads pretrained VGG16 (ImageNet weights).
  2. FREEZES all convolutional feature extractor layers.
  3. Replaces classifier head for 10-class Fashion MNIST.
  4. Trains ONLY the new classifier (transfer learning).
  5. Evaluates: accuracy, precision, recall, F1 + visualisations.
  6. Compares VGG16 vs ResNet18 transfer learning (High Distinction).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from tqdm import tqdm

# ============================================================================
# 1. Configuration
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot"
]

# ============================================================================
# 2. Load Fashion MNIST
#    VGG16 expects 3-channel 224x224 images with ImageNet normalisation
# ============================================================================
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_ds = torchvision.datasets.FashionMNIST(
    root="./data", train=True,  download=True, transform=transform_train)
test_ds  = torchvision.datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = torch.utils.data.DataLoader(
    test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

# ============================================================================
# 3. Build VGG16 Transfer Learning Model
# ============================================================================
def build_vgg16_transfer(num_classes=10):
    """
    Load ImageNet-pretrained VGG16.
    Freeze all convolutional layers (features).
    Replace the 3-layer classifier with a new one for num_classes.
    Only the new classifier layers are trainable.
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    # --- Freeze feature extractor ---
    for param in model.features.parameters():
        param.requires_grad = False
    # --- Replace classifier ---
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )
    return model

model = build_vgg16_transfer(num_classes=10).to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"VGG16 Transfer — Trainable: {trainable:,} / Total: {total:,} params")
print(f"  ({100*trainable/total:.1f}% of parameters are trainable)")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ============================================================================
# 4. Training & Evaluation utilities
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, lbls in tqdm(loader, desc="  Train", leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(lbls).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, all_preds, all_lbls, total = 0.0, [], [], 0
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="  Eval ", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss_sum += loss.item() * imgs.size(0)
            all_preds.append(out.argmax(1).cpu())
            all_lbls.append(lbls.cpu())
            total += imgs.size(0)
    preds = torch.cat(all_preds).numpy()
    lbls  = torch.cat(all_lbls).numpy()
    return loss_sum / total, preds, lbls

# ============================================================================
# 5. Training loop
# ============================================================================
print(f"\n{'='*60}")
print("Training VGG16 (Transfer Learning) on Fashion MNIST ...")
print(f"{'='*60}")
tr_losses, tr_accs, te_losses, te_accs = [], [], [], []

for epoch in range(1, EPOCHS + 1):
    tr_l, tr_a = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    te_l, te_p, te_lb = eval_epoch(model, test_loader, criterion, DEVICE)
    te_a = accuracy_score(te_lb, te_p)
    scheduler.step()
    tr_losses.append(tr_l); tr_accs.append(tr_a)
    te_losses.append(te_l); te_accs.append(te_a)
    print(f"  Epoch {epoch:2d}: TrainLoss={tr_l:.4f} TrainAcc={tr_a:.4f} "
          f"| TestLoss={te_l:.4f} TestAcc={te_a:.4f}")

_, final_preds, final_lbls = eval_epoch(model, test_loader, criterion, DEVICE)

# ============================================================================
# 6. Metrics
# ============================================================================
acc  = accuracy_score(final_lbls, final_preds)
prec = precision_score(final_lbls, final_preds, average="macro", zero_division=0)
rec  = recall_score(final_lbls, final_preds, average="macro", zero_division=0)
f1   = f1_score(final_lbls, final_preds, average="macro", zero_division=0)
print(f"\n--- VGG16 Transfer Learning Final Results ---")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1 Score : {f1:.4f}")
print(f"\nClassification Report:")
print(classification_report(final_lbls, final_preds, target_names=CLASS_NAMES, zero_division=0))

# Save metrics for Q5
torch.save({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "train_losses": tr_losses, "test_losses": te_losses,
            "train_accs":   tr_accs,   "test_accs":  te_accs},
           "q3_vgg16_transfer_metrics.pt")

# ============================================================================
# 7. Visualisation 1 — Training curves
# ============================================================================
ep_r = range(1, EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(ep_r, tr_losses, "b-o", label="Train Loss")
ax1.plot(ep_r, te_losses, "r-o", label="Test Loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Loss Curves – VGG16 Transfer Learning"); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(ep_r, tr_accs, "b-o", label="Train Acc")
ax2.plot(ep_r, te_accs, "r-o", label="Test Acc")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy Curves – VGG16 Transfer Learning"); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.suptitle("VGG16 Transfer Learning on Fashion MNIST", fontsize=14)
plt.tight_layout(); plt.savefig("q3_training_curves.png", dpi=150); plt.close()
print("Saved: q3_training_curves.png")

# ============================================================================
# 8. Visualisation 2 — Confusion Matrix
# ============================================================================
cm = confusion_matrix(final_lbls, final_preds)
plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted", fontsize=12); plt.ylabel("True", fontsize=12)
plt.title("Confusion Matrix – VGG16 Transfer Learning on Fashion MNIST", fontsize=13)
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
plt.tight_layout(); plt.savefig("q3_confusion_matrix.png", dpi=150); plt.close()
print("Saved: q3_confusion_matrix.png")

# ============================================================================
# 9. Visualisation 3 — Per-class accuracy
# ============================================================================
pca = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(12, 5))
bars = plt.bar(CLASS_NAMES, pca, color="steelblue", edgecolor="black")
for b, a in zip(bars, pca):
    plt.text(b.get_x() + b.get_width()/2, b.get_height()+0.01,
             f"{a:.2f}", ha="center", fontsize=9)
plt.xlabel("Class"); plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy – VGG16 Transfer Learning"); plt.ylim(0, 1.15)
plt.xticks(rotation=45, ha="right")
plt.tight_layout(); plt.savefig("q3_per_class_accuracy.png", dpi=150); plt.close()
print("Saved: q3_per_class_accuracy.png")

# ============================================================================
# 10. HIGH DISTINCTION — Compare with ResNet18 transfer learning
# ============================================================================
print(f"\n{'='*60}")
print("HD Comparison: ResNet18 Transfer Learning ...")
print(f"{'='*60}")

resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for p in resnet18.parameters():
    p.requires_grad = False
resnet18.fc = nn.Sequential(
    nn.Linear(resnet18.fc.in_features, 256),
    nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 10)
)
resnet18 = resnet18.to(DEVICE)
opt_r18  = optim.Adam(resnet18.fc.parameters(), lr=LR)
sch_r18  = optim.lr_scheduler.StepLR(opt_r18, step_size=5, gamma=0.5)

r18_tr_l, r18_te_a = [], []
for epoch in range(1, EPOCHS + 1):
    tr_l2, _ = train_epoch(resnet18, train_loader, criterion, opt_r18, DEVICE)
    te_l2, r18_p, r18_lb = eval_epoch(resnet18, test_loader, criterion, DEVICE)
    ta = accuracy_score(r18_lb, r18_p)
    sch_r18.step()
    r18_tr_l.append(tr_l2); r18_te_a.append(ta)
    print(f"  [ResNet18] Epoch {epoch}: TestAcc={ta:.4f}")

_, r18_p, r18_lb = eval_epoch(resnet18, test_loader, criterion, DEVICE)
r18_acc  = accuracy_score(r18_lb, r18_p)
r18_prec = precision_score(r18_lb, r18_p, average="macro", zero_division=0)
r18_rec  = recall_score(r18_lb, r18_p, average="macro", zero_division=0)
r18_f1   = f1_score(r18_lb, r18_p, average="macro", zero_division=0)
print(f"\n--- ResNet18 Transfer Learning ---")
print(f"  Accuracy={r18_acc:.4f}  Precision={r18_prec:.4f}  "
      f"Recall={r18_rec:.4f}  F1={r18_f1:.4f}")

# Comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))
mn = ["Accuracy", "Precision", "Recall", "F1"]
vv = [acc, prec, rec, f1]
rv = [r18_acc, r18_prec, r18_rec, r18_f1]
x  = np.arange(4); w = 0.3
ax.bar(x - w/2, vv, w, label="VGG16 (pretrained)",    color="steelblue")
ax.bar(x + w/2, rv, w, label="ResNet18 (pretrained)",  color="darkorange")
for i in range(4):
    ax.text(x[i]-w/2, vv[i]+0.01, f"{vv[i]:.3f}", ha="center", fontsize=9)
    ax.text(x[i]+w/2, rv[i]+0.01, f"{rv[i]:.3f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(mn); ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Transfer Learning Backbone Comparison: VGG16 vs ResNet18")
ax.legend(); plt.tight_layout()
plt.savefig("q3_backbone_comparison.png", dpi=150); plt.close()
print("Saved: q3_backbone_comparison.png")
print("\nQuestion 3 complete.")
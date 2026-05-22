"""
Assignment 2 - Question 4 (20 marks / 6.7%)
Training VGG16 from Scratch on Fashion MNIST

HOW TO RUN:
    pip install torch torchvision matplotlib seaborn scikit-learn tqdm
    python question4_train_from_scratch.py

This script:
  1. Loads VGG16 architecture WITHOUT any pretrained weights (weights=None).
  2. Adapts the classifier for 10-class Fashion MNIST.
  3. Trains ALL parameters from scratch.
  4. Evaluates with full metrics and multiple visualisations.
  5. Compares VGG16-scratch vs a Simple CNN (High Distinction).
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
EPOCHS = 15
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
# ============================================================================
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
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
# 3. VGG16 from Scratch — NO pretrained weights
# ============================================================================
def build_vgg16_scratch(num_classes=10):
    """
    VGG16 architecture with weights=None (random initialisation).
    ALL parameters (convolutional + classifier) are trainable.
    Classifier adapted for Fashion MNIST (10 classes).
    """
    model = models.vgg16(weights=None)           # <-- key difference from Q3
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

model = build_vgg16_scratch(num_classes=10).to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"VGG16 from Scratch — Trainable: {trainable:,} / Total: {total:,}")
print(f"  (100% of parameters trained from random initialisation)")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)

# ============================================================================
# 4. Training & Evaluation
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
    return loss_sum/total, torch.cat(all_preds).numpy(), torch.cat(all_lbls).numpy()

# ============================================================================
# 5. Training loop
# ============================================================================
print(f"\n{'='*60}")
print("Training VGG16 from Scratch on Fashion MNIST ...")
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
print(f"\n--- VGG16 from Scratch Final Results ---")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1 Score : {f1:.4f}")
print(f"\nClassification Report:")
print(classification_report(final_lbls, final_preds, target_names=CLASS_NAMES, zero_division=0))

torch.save({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "train_losses": tr_losses, "test_losses": te_losses,
            "train_accs":   tr_accs,   "test_accs":  te_accs},
           "q4_vgg16_scratch_metrics.pt")

# ============================================================================
# 7. Visualisation 1 — Training curves
# ============================================================================
ep_r = range(1, EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(ep_r, tr_losses, "b-o", label="Train Loss")
ax1.plot(ep_r, te_losses, "r-o", label="Test Loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Loss Curves – VGG16 from Scratch"); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(ep_r, tr_accs, "b-o", label="Train Acc")
ax2.plot(ep_r, te_accs, "r-o", label="Test Acc")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy Curves – VGG16 from Scratch"); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.suptitle("VGG16 from Scratch on Fashion MNIST", fontsize=14)
plt.tight_layout(); plt.savefig("q4_training_curves_scratch.png", dpi=150); plt.close()
print("Saved: q4_training_curves_scratch.png")

# ============================================================================
# 8. Visualisation 2 — Confusion Matrix
# ============================================================================
cm = confusion_matrix(final_lbls, final_preds)
plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted", fontsize=12); plt.ylabel("True", fontsize=12)
plt.title("Confusion Matrix – VGG16 from Scratch on Fashion MNIST", fontsize=13)
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
plt.tight_layout(); plt.savefig("q4_confusion_matrix_scratch.png", dpi=150); plt.close()
print("Saved: q4_confusion_matrix_scratch.png")

# ============================================================================
# 9. Visualisation 3 — Per-class accuracy
# ============================================================================
pca = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(12, 5))
bars = plt.bar(CLASS_NAMES, pca, color="darkorange", edgecolor="black")
for b, a in zip(bars, pca):
    plt.text(b.get_x() + b.get_width()/2, b.get_height()+0.01,
             f"{a:.2f}", ha="center", fontsize=9)
plt.xlabel("Class"); plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy – VGG16 from Scratch"); plt.ylim(0, 1.15)
plt.xticks(rotation=45, ha="right")
plt.tight_layout(); plt.savefig("q4_per_class_accuracy_scratch.png", dpi=150); plt.close()
print("Saved: q4_per_class_accuracy_scratch.png")

# ============================================================================
# 10. HIGH DISTINCTION — Compare with a Simple CNN from scratch
# ============================================================================
print(f"\n{'='*60}")
print("HD Comparison: Simple CNN from Scratch ...")
print(f"{'='*60}")

class SimpleCNN(nn.Module):
    """Lightweight 4-conv CNN for comparison with VGG16."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                      # 112
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                      # 56
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                      # 28
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                         # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

cnn = SimpleCNN(10).to(DEVICE)
opt_cnn = optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-4)
sch_cnn = optim.lr_scheduler.StepLR(opt_cnn, step_size=7, gamma=0.3)

cnn_te_accs = []
for epoch in range(1, EPOCHS + 1):
    train_epoch(cnn, train_loader, criterion, opt_cnn, DEVICE)
    _, cp, cl = eval_epoch(cnn, test_loader, criterion, DEVICE)
    ca = accuracy_score(cl, cp)
    cnn_te_accs.append(ca)
    sch_cnn.step()
    print(f"  [SimpleCNN] Epoch {epoch}: TestAcc={ca:.4f}")

_, cnn_p, cnn_l = eval_epoch(cnn, test_loader, criterion, DEVICE)
cnn_acc  = accuracy_score(cnn_l, cnn_p)
cnn_prec = precision_score(cnn_l, cnn_p, average="macro", zero_division=0)
cnn_rec  = recall_score(cnn_l, cnn_p, average="macro", zero_division=0)
cnn_f1   = f1_score(cnn_l, cnn_p, average="macro", zero_division=0)
print(f"\n--- Simple CNN from Scratch ---")
print(f"  Accuracy={cnn_acc:.4f}  F1={cnn_f1:.4f}")

# Comparison
fig, ax = plt.subplots(figsize=(10, 6))
mn = ["Accuracy", "Precision", "Recall", "F1"]
vv = [acc, prec, rec, f1]
cv = [cnn_acc, cnn_prec, cnn_rec, cnn_f1]
x  = np.arange(4); w = 0.3
ax.bar(x - w/2, vv, w, label="VGG16 (scratch)",    color="darkorange")
ax.bar(x + w/2, cv, w, label="Simple CNN (scratch)", color="mediumseagreen")
for i in range(4):
    ax.text(x[i]-w/2, vv[i]+0.01, f"{vv[i]:.3f}", ha="center", fontsize=9)
    ax.text(x[i]+w/2, cv[i]+0.01, f"{cv[i]:.3f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(mn); ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("From Scratch: VGG16 vs Simple CNN on Fashion MNIST")
ax.legend(); plt.tight_layout()
plt.savefig("q4_model_comparison.png", dpi=150); plt.close()
print("Saved: q4_model_comparison.png")
print("\nQuestion 4 complete.")
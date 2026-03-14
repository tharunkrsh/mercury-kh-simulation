"""
Step 2: PyTorch MLP — Ion Trapping Classifier
==============================================
Run this locally:
    pip install torch pandas numpy matplotlib scikit-learn
    python train.py

Outputs:
    model.pth          — saved model weights
    training_curve.png — loss + accuracy over epochs
    phase_diagram.png  — trapping probability heatmap (R vs dist0)
    confusion.png      — confusion matrix
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

# ── 1. Load data ───────────────────────────────────────────────────────────────

df = pd.read_csv("../data/kh_trapping_dataset.csv")
print(f"Dataset: {len(df)} samples  |  Trapped: {df.trapped.sum()}  |  Escaped: {(df.trapped==0).sum()}")

FEATURES = ["R", "x0", "y0", "charge", "r_g", "R_over_rg", "dist0", "m_mode"]
X = df[FEATURES].values.astype(np.float32)
y = df["trapped"].values.astype(np.float32)

# Scale features — critical for neural nets
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

# Save scaler params so the web app can replicate them
np.save("scaler_mean.npy", scaler.mean_.astype(np.float32))
np.save("scaler_scale.npy", scaler.scale_.astype(np.float32))
print(f"Scaler saved. Feature means: {np.round(scaler.mean_, 2)}")

# ── 2. Build dataset + splits ──────────────────────────────────────────────────

X_tensor = torch.tensor(X_scaled)
y_tensor = torch.tensor(y).unsqueeze(1)   # shape (N, 1) for BCELoss

dataset   = TensorDataset(X_tensor, y_tensor)
n_train   = int(0.75 * len(dataset))
n_val     = int(0.15 * len(dataset))
n_test    = len(dataset) - n_train - n_val

train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
print(f"Train: {n_train}  Val: {n_val}  Test: {n_test}")

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64)
test_loader  = DataLoader(test_ds,  batch_size=64)

# ── 3. Define MLP ──────────────────────────────────────────────────────────────

class TrapNet(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = TrapNet(input_dim=len(FEATURES))
print(f"\nModel:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

# ── 4. Training setup ──────────────────────────────────────────────────────────

# Class imbalance: weight the positive (trapped) class higher
n_pos    = y.sum()
n_neg    = len(y) - n_pos
pos_weight = torch.tensor([n_neg / n_pos])
print(f"\nPos weight (trapped upweighting): {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Rebuild model without final sigmoid since BCEWithLogitsLoss expects raw logits
class TrapNetLogits(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)   # raw logit — sigmoid applied in BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))

model    = TrapNetLogits(input_dim=len(FEATURES))
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

# ── 5. Training loop ───────────────────────────────────────────────────────────

EPOCHS = 120
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_loss = float('inf')

print(f"\nTraining for {EPOCHS} epochs...\n{'─'*55}")

for epoch in range(EPOCHS):
    # Train
    model.train()
    t_loss, t_correct, t_total = 0.0, 0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss  = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        t_loss    += loss.item() * len(xb)
        t_correct += ((torch.sigmoid(preds) > 0.5).float() == yb).sum().item()
        t_total   += len(xb)

    # Validate
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds  = model(xb)
            loss   = criterion(preds, yb)
            v_loss    += loss.item() * len(xb)
            v_correct += ((torch.sigmoid(preds) > 0.5).float() == yb).sum().item()
            v_total   += len(xb)

    t_loss /= t_total;  v_loss /= v_total
    t_acc   = t_correct / t_total
    v_acc   = v_correct / v_total

    train_losses.append(t_loss);  val_losses.append(v_loss)
    train_accs.append(t_acc);     val_accs.append(v_acc)

    scheduler.step(v_loss)

    # Save best
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        torch.save(model.state_dict(), "model.pth")

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  "
              f"train_acc={t_acc:.3f}  val_acc={v_acc:.3f}")

print(f"\nBest val loss: {best_val_loss:.4f}")
print("Model saved → model.pth")

# ── 6. Test evaluation ─────────────────────────────────────────────────────────

model.load_state_dict(torch.load("model.pth"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        probs = model.predict_proba(xb)
        all_preds.append((probs > 0.5).float())
        all_labels.append(yb)

all_preds  = torch.cat(all_preds).numpy().flatten()
all_labels = torch.cat(all_labels).numpy().flatten()
test_acc   = (all_preds == all_labels).mean()

print(f"\nTest accuracy: {test_acc*100:.1f}%")
print("\nClassification report:")
print(classification_report(all_labels, all_preds, target_names=["Escaped", "Trapped"]))

# ── 7. Training curve plot ─────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#0a0a0f')
for ax in (ax1, ax2):
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

epochs_range = range(1, EPOCHS + 1)

ax1.plot(epochs_range, train_losses, color='#3d9eff', linewidth=2, label='Train loss')
ax1.plot(epochs_range, val_losses,   color='#ff6b6b', linewidth=2, label='Val loss',  linestyle='--')
ax1.set_xlabel('Epoch', color='#8b949e')
ax1.set_ylabel('BCE Loss', color='#8b949e')
ax1.set_title('Loss Curve', color='white', fontsize=12)
ax1.legend(facecolor='#161b22', labelcolor='white')
ax1.grid(True, color='#21262d', alpha=0.6)

ax2.plot(epochs_range, [a*100 for a in train_accs], color='#3d9eff', linewidth=2, label='Train acc')
ax2.plot(epochs_range, [a*100 for a in val_accs],   color='#ff6b6b', linewidth=2, label='Val acc',  linestyle='--')
ax2.axhline(test_acc*100, color='#34d399', linewidth=1.5, linestyle=':', label=f'Test acc {test_acc*100:.1f}%')
ax2.set_xlabel('Epoch', color='#8b949e')
ax2.set_ylabel('Accuracy (%)', color='#8b949e')
ax2.set_title('Accuracy Curve', color='white', fontsize=12)
ax2.legend(facecolor='#161b22', labelcolor='white')
ax2.grid(True, color='#21262d', alpha=0.6)
ax2.set_ylim(50, 101)

plt.suptitle('TrapNet — Training Curves', color='white', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("training_curve.png", dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
print("\nSaved: training_curve.png")

# ── 8. Confusion matrix ────────────────────────────────────────────────────────

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#0d1117')

im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Escaped', 'Trapped'], color='white')
ax.set_yticklabels(['Escaped', 'Trapped'], color='white')
ax.set_xlabel('Predicted', color='#8b949e')
ax.set_ylabel('Actual',    color='#8b949e')
ax.set_title('Confusion Matrix', color='white', fontsize=12)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white', fontsize=16, fontweight='bold')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("confusion.png", dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
print("Saved: confusion.png")

# ── 9. Phase diagram — trapping probability across R vs dist0 ──────────────────

model.eval()
R_range    = np.linspace(0.5, 20, 120)
dist_range = np.linspace(0, 35, 120)
RR, DD     = np.meshgrid(R_range, dist_range)

# Build feature grid: vary R and dist0, fix others at typical values
# charge=+1, r_g=1, R_over_rg=R, x0=dist (on-axis), y0=0, m_mode=0
grid_rows = []
for R_val, d_val in zip(RR.ravel(), DD.ravel()):
    grid_rows.append([R_val, d_val, 0.0, 1.0, 1.0, R_val, d_val, 0.0])

grid_arr = np.array(grid_rows, dtype=np.float32)
grid_scaled = scaler.transform(grid_arr).astype(np.float32)
grid_tensor = torch.tensor(grid_scaled)

with torch.no_grad():
    probs = model.predict_proba(grid_tensor).numpy().reshape(RR.shape)

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#0a0a0f')

contf = ax.contourf(RR, DD, probs, levels=50, cmap='plasma')
cont  = ax.contour(RR, DD, probs, levels=[0.5], colors='white', linewidths=1.5, linestyles='--')
ax.clabel(cont, fmt='p=0.5', colors='white', fontsize=10)

cbar = plt.colorbar(contf, ax=ax)
cbar.set_label('Trapping Probability', color='white', fontsize=11)
cbar.ax.tick_params(colors='white')

ax.set_xlabel('Vortex Radius R', color='white', fontsize=12)
ax.set_ylabel('Initial Distance from Vortex Centre', color='white', fontsize=12)
ax.set_title('Ion Trapping Phase Diagram\n(positive ion, dusk flank, charge=+1)',
             color='white', fontsize=13)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#333333')

plt.tight_layout()
plt.savefig("phase_diagram.png", dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
print("Saved: phase_diagram.png")

print("\n✓ All done. Files produced:")
print("  model.pth · scaler_mean.npy · scaler_scale.npy")
print("  training_curve.png · confusion.png · phase_diagram.png")

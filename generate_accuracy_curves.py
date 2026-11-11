"""Script to generate accuracy learning curves (Accuracy vs Boosting Round) for both scenarios."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

from src import config

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

# Load cached features and train labels
cached = np.load(config.CACHE_DIR / "hog_train.npz")
manifest = pd.read_csv(REPORTS_DIR / "train_test_split.csv")
train_df = manifest[manifest["split"] == "train"]
X_train = cached["X"]
y_train = train_df["label"].to_numpy(dtype=np.int64)

# 15% internal validation split for learning curve tracking (no test set leakage)
X_sub, X_val, y_sub, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.15,
    stratify=y_train,
    random_state=config.SEED,
)

# Skenario 1: Tanpa PCA
params_no_pca = dict(config.XGB_PARAMS)
params_no_pca["eval_metric"] = ["mlogloss", "merror"]
xgb_no_pca = XGBClassifier(**params_no_pca)
xgb_no_pca.fit(X_sub, y_sub, eval_set=[(X_sub, y_sub), (X_val, y_val)], verbose=False)

res_no_pca = xgb_no_pca.evals_result()
epochs = np.arange(1, len(res_no_pca["validation_0"]["merror"]) + 1)
acc_train_no_pca = 1.0 - np.array(res_no_pca["validation_0"]["merror"])
acc_val_no_pca = 1.0 - np.array(res_no_pca["validation_1"]["merror"])

df_acc_no_pca = pd.DataFrame({
    "epoch": epochs,
    "train_accuracy": acc_train_no_pca,
    "val_accuracy": acc_val_no_pca,
})
df_acc_no_pca.to_csv(REPORTS_DIR / "accuracy_curve_tanpa_pca.csv", index=False)

# Skenario 2: Dengan PCA (Scaler & PCA fit hanya pada X_sub)
scaler = StandardScaler()
pca = PCA(n_components=config.PCA_VARIANCE, svd_solver="full", random_state=config.SEED)
X_sub_pca = pca.fit_transform(scaler.fit_transform(X_sub))
X_val_pca = pca.transform(scaler.transform(X_val))

params_pca = dict(config.XGB_PARAMS)
params_pca["eval_metric"] = ["mlogloss", "merror"]
xgb_pca = XGBClassifier(**params_pca)
xgb_pca.fit(X_sub_pca, y_sub, eval_set=[(X_sub_pca, y_sub), (X_val_pca, y_val)], verbose=False)

res_pca = xgb_pca.evals_result()
acc_train_pca = 1.0 - np.array(res_pca["validation_0"]["merror"])
acc_val_pca = 1.0 - np.array(res_pca["validation_1"]["merror"])

df_acc_pca = pd.DataFrame({
    "epoch": epochs,
    "train_accuracy": acc_train_pca,
    "val_accuracy": acc_val_pca,
})
df_acc_pca.to_csv(REPORTS_DIR / "accuracy_curve_dengan_pca.csv", index=False)

# -------------------------------------------------------------
# Plot 1: Accuracy Curve Tanpa PCA
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(epochs, acc_train_no_pca, label="Train Accuracy", color="#2b6cb0", linewidth=2)
ax.plot(epochs, acc_val_no_pca, label="Validation Accuracy (Internal Train)", color="#e53e3e", linewidth=2)
ax.axhline(acc_val_no_pca[-1], color="#e53e3e", linestyle=":", alpha=0.7, label=f"Final Val Acc: {acc_val_no_pca[-1]:.4f}")
ax.set_title("Kurva Akurasi per Epoch - XGBoost Tanpa PCA (1.296 Fitur)", fontsize=12, fontweight="bold")
ax.set_xlabel("Boosting Round (Epoch)", fontsize=11)
ax.set_ylabel("Akurasi (0.0 - 1.0)", fontsize=11)
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="lower right", fontsize=10)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "accuracy_curve_tanpa_pca.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------
# Plot 2: Accuracy Curve Dengan PCA
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(epochs, acc_train_pca, label="Train Accuracy", color="#2b6cb0", linewidth=2)
ax.plot(epochs, acc_val_pca, label="Validation Accuracy (Internal Train)", color="#d69e2e", linewidth=2)
ax.axhline(acc_val_pca[-1], color="#d69e2e", linestyle=":", alpha=0.7, label=f"Final Val Acc: {acc_val_pca[-1]:.4f}")
ax.set_title("Kurva Akurasi per Epoch - XGBoost Dengan PCA (157 Komponen)", fontsize=12, fontweight="bold")
ax.set_xlabel("Boosting Round (Epoch)", fontsize=11)
ax.set_ylabel("Akurasi (0.0 - 1.0)", fontsize=11)
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="lower right", fontsize=10)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "accuracy_curve_dengan_pca.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------
# Plot 3: Comparison Grid (Side-by-Side & Validation Comparison)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Train vs Val Tanpa PCA
axes[0].plot(epochs, acc_train_no_pca, label="Train (Tanpa PCA)", color="#2b6cb0", linewidth=2)
axes[0].plot(epochs, acc_val_no_pca, label=f"Validasi (Tanpa PCA - Final: {acc_val_no_pca[-1]:.2%})", color="#e53e3e", linewidth=2)
axes[0].plot(epochs, acc_train_pca, label="Train (Dengan PCA)", color="#4a5568", linestyle="--", alpha=0.7)
axes[0].plot(epochs, acc_val_pca, label=f"Validasi (Dengan PCA - Final: {acc_val_pca[-1]:.2%})", color="#38a169", linewidth=2)
axes[0].set_title("Perbandingan Kurva Akurasi Train & Validasi", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Boosting Round (Epoch)", fontsize=10)
axes[0].set_ylabel("Akurasi", fontsize=10)
axes[0].set_ylim(0, 1.05)
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].legend(loc="lower right", fontsize=9)

# Subplot 2: Langsung Head-to-Head Validasi Saja
axes[1].plot(epochs, acc_val_no_pca, label=f"Validasi: Tanpa PCA (Peak: {acc_val_no_pca.max():.2%})", color="#e53e3e", linewidth=2.2)
axes[1].plot(epochs, acc_val_pca, label=f"Validasi: Dengan PCA (Peak: {acc_val_pca.max():.2%})", color="#38a169", linewidth=2.2)
axes[1].set_title("Head-to-Head Akurasi Validasi: Tanpa PCA vs Dengan PCA", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Boosting Round (Epoch)", fontsize=10)
axes[1].set_ylabel("Akurasi Validasi", fontsize=10)
axes[1].set_ylim(0, 1.0)
axes[1].grid(True, linestyle="--", alpha=0.4)
axes[1].legend(loc="lower right", fontsize=10)

fig.suptitle("Analisis Kurva Akurasi Pembelajaran (Accuracy Learning Curve)", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "accuracy_curve_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Generated all accuracy curves successfully!")

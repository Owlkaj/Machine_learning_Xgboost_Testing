"""Script to plot per_class_metrics for both Tanpa PCA and Dengan PCA."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

df_no_pca = pd.read_csv(REPORTS_DIR / "per_class_metrics_tanpa_pca.csv")
df_pca = pd.read_csv(REPORTS_DIR / "per_class_metrics_dengan_pca.csv")

# -------------------------------------------------------------
# Plot 1: Per-Class Precision, Recall, and F1-Score Comparison
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
digits = df_no_pca["digit"].astype(str)
x = np.arange(len(digits))
width = 0.26

for ax, df, title in zip(
    axes,
    [df_no_pca, df_pca],
    ["XGBoost Tanpa PCA (1.296 Fitur)", "XGBoost Dengan PCA (157 Komponen)"]
):
    rects1 = ax.bar(x - width, df["precision"], width, label="Precision", color="#3182ce")
    rects2 = ax.bar(x, df["recall"], width, label="Recall", color="#38a169")
    rects3 = ax.bar(x + width, df["f1_score"], width, label="F1-Score", color="#dd6b20")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Digit Kelas", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(digits, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)

axes[0].set_ylabel("Nilai Metrik (0.0 - 1.0)", fontsize=11)
fig.suptitle("Perbandingan Metrik per Kelas (Precision, Recall, F1-Score)", fontsize=14, fontweight="bold")
fig.tight_layout()
output_path1 = PLOTS_DIR / "per_class_metrics.png"
fig.savefig(output_path1, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {output_path1}")

# -------------------------------------------------------------
# Plot 2: Per-Class TP, FP, FN Breakdown
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, df, title in zip(
    axes,
    [df_no_pca, df_pca],
    ["XGBoost Tanpa PCA", "XGBoost Dengan PCA"]
):
    b1 = ax.bar(x - width, df["TP"], width, label="True Positive (TP - Benar)", color="#2b6cb0")
    b2 = ax.bar(x, df["FP"], width, label="False Positive (FP - Salah Klaim)", color="#e53e3e")
    b3 = ax.bar(x + width, df["FN"], width, label="False Negative (FN - Luput)", color="#d69e2e")

    for bars in [b1, b2, b3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{int(height)}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 2),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Digit Kelas", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(digits, fontsize=10)
    ax.set_ylim(0, 14)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)

axes[0].set_ylabel("Jumlah Sampel Citra", fontsize=11)
fig.suptitle("Diagnostik Error per Kelas (True Positive, False Positive, False Negative)", fontsize=14, fontweight="bold")
fig.tight_layout()
output_path2 = PLOTS_DIR / "per_class_confusion_breakdown.png"
fig.savefig(output_path2, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {output_path2}")


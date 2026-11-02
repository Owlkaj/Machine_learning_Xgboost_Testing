"""Evaluation metrics, confusion matrices, error analysis, and plots (Tahap 6)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from skimage.exposure import rescale_intensity

from . import config
from .preprocess import (
    apply_clahe,
    hog_visualization,
    load_grayscale_image,
    resize_image,
)
from .train import CVResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """All test-set artefacts for one scenario."""

    scenario: str
    y_true: np.ndarray
    y_pred: np.ndarray
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion: np.ndarray
    confusion_normalized: np.ndarray
    report: pd.DataFrame
    per_class: pd.DataFrame
    top_confusions: pd.DataFrame


def evaluate_model(
    scenario: str,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[int] = config.CLASSES,
) -> EvaluationResult:
    """Compute all classification metrics for the official test set."""
    labels = list(classes)
    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)

    cm = confusion_matrix(true, predicted, labels=labels)
    cm_normalized = confusion_matrix(true, predicted, labels=labels, normalize="true")
    report_dict = classification_report(
        true,
        predicted,
        labels=labels,
        target_names=[str(label) for label in labels],
        zero_division=0,
        output_dict=True,
    )
    report = pd.DataFrame(report_dict).transpose()
    report.index.name = "kelas"

    per_class_rows = []
    total = int(cm.sum())
    for index, digit in enumerate(labels):
        true_positive = int(cm[index, index])
        false_negative = int(cm[index, :].sum() - true_positive)
        false_positive = int(cm[:, index].sum() - true_positive)
        true_negative = total - true_positive - false_positive - false_negative
        support = true_positive + false_negative
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class_rows.append(
            {
                "digit": digit,
                "TP": true_positive,
                "FP": false_positive,
                "FN": false_negative,
                "TN": true_negative,
                "support": support,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1_score": round(f1, 6),
            }
        )
    per_class = pd.DataFrame(per_class_rows)

    top_confusions = _top_confused_pairs(cm, labels, top_n=5)

    result = EvaluationResult(
        scenario=scenario,
        y_true=true,
        y_pred=predicted,
        accuracy=float(accuracy_score(true, predicted)),
        macro_precision=float(
            precision_score(true, predicted, labels=labels, average="macro", zero_division=0)
        ),
        macro_recall=float(
            recall_score(true, predicted, labels=labels, average="macro", zero_division=0)
        ),
        macro_f1=float(
            f1_score(true, predicted, labels=labels, average="macro", zero_division=0)
        ),
        confusion=cm,
        confusion_normalized=cm_normalized,
        report=report,
        per_class=per_class,
        top_confusions=top_confusions,
    )
    logger.info(
        "%s -> accuracy=%.4f, macro-P=%.4f, macro-R=%.4f, macro-F1=%.4f",
        config.SCENARIO_LABELS[scenario],
        result.accuracy,
        result.macro_precision,
        result.macro_recall,
        result.macro_f1,
    )
    return result


def _top_confused_pairs(
    cm: np.ndarray,
    classes: Sequence[int],
    top_n: int = 5,
) -> pd.DataFrame:
    """Extract the most frequent off-diagonal confusions."""
    rows = []
    for true_index, true_digit in enumerate(classes):
        support = int(cm[true_index, :].sum())
        for pred_index, pred_digit in enumerate(classes):
            if true_index == pred_index:
                continue
            count = int(cm[true_index, pred_index])
            if count == 0:
                continue
            rows.append(
                {
                    "digit_asli": int(true_digit),
                    "digit_prediksi": int(pred_digit),
                    "jumlah_kesalahan": count,
                    "persentase_terhadap_kelas_asli": round(
                        100 * count / support, 3
                    )
                    if support
                    else 0.0,
                }
            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "digit_asli",
            "digit_prediksi",
            "jumlah_kesalahan",
            "persentase_terhadap_kelas_asli",
        ],
    )
    if frame.empty:
        return frame
    return (
        frame.sort_values(
            ["jumlah_kesalahan", "persentase_terhadap_kelas_asli"],
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def save_evaluation_artifacts(
    result: EvaluationResult,
    reports_dir: Path = config.REPORTS_DIR,
) -> None:
    """Save classification report, per-class TP/FP/FN/TN, errors, predictions."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    result.report.to_csv(
        reports_dir / f"classification_report_{result.scenario}.csv"
    )
    result.per_class.to_csv(
        reports_dir / f"per_class_metrics_{result.scenario}.csv", index=False
    )
    result.top_confusions.to_csv(
        reports_dir / f"top_confusions_{result.scenario}.csv", index=False
    )
    pd.DataFrame(
        {
            "index_test": np.arange(len(result.y_true)),
            "label_asli": result.y_true,
            "prediksi": result.y_pred,
        }
    ).to_csv(reports_dir / f"predictions_{result.scenario}.csv", index=False)


def plot_confusion_matrices(
    results: Sequence[EvaluationResult],
    plots_dir: Path = config.PLOTS_DIR,
) -> None:
    """Save raw and row-normalised 10x10 confusion matrices side by side."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")
    labels = [str(digit) for digit in config.CLASSES]

    for normalized, filename, fmt, title in (
        (False, "confusion_matrix_counts.png", "d", "Confusion Matrix - Raw Counts"),
        (
            True,
            "confusion_matrix_normalized.png",
            ".2f",
            "Confusion Matrix - Normalized (row)",
        ),
    ):
        fig, axes = plt.subplots(1, len(results), figsize=(9 * len(results), 7.5))
        if len(results) == 1:
            axes = [axes]
        for ax, result in zip(axes, results):
            matrix = result.confusion_normalized if normalized else result.confusion
            sns.heatmap(
                matrix,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                cbar=True,
                square=True,
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
                annot_kws={"fontsize": 8},
            )
            ax.set_title(config.SCENARIO_LABELS[result.scenario], fontsize=12)
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Label Asli")
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(plots_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def pca_statistics(pipeline: Pipeline) -> dict[str, float | int]:
    """Return PCA dimension-reduction statistics from a fitted pipeline."""
    steps = getattr(pipeline, "named_steps", {})
    pca = steps.get("pca")
    if pca is None:
        return {
            "n_features_awal": config.EXPECTED_HOG_FEATURES,
            "n_komponen_pca": config.EXPECTED_HOG_FEATURES,
            "reduksi_dimensi_persen": 0.0,
            "cumulative_explained_variance": 1.0,
        }
    n_components = int(pca.n_components_)
    initial = int(getattr(pca, "n_features_in_", config.EXPECTED_HOG_FEATURES))
    cumulative = float(np.sum(pca.explained_variance_ratio_))
    return {
        "n_features_awal": initial,
        "n_komponen_pca": n_components,
        "reduksi_dimensi_persen": round(100 * (1 - n_components / initial), 3),
        "cumulative_explained_variance": round(cumulative, 6),
    }


def plot_pca_variance(
    pipeline: Pipeline,
    plots_dir: Path = config.PLOTS_DIR,
    reports_dir: Path = config.REPORTS_DIR,
) -> dict[str, float | int]:
    """Plot cumulative explained variance and save the underlying table."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    pca = pipeline.named_steps["pca"]
    ratios = np.asarray(pca.explained_variance_ratio_)
    cumulative = np.cumsum(ratios)
    components = np.arange(1, len(ratios) + 1)

    pd.DataFrame(
        {
            "komponen": components,
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance": cumulative,
        }
    ).to_csv(reports_dir / "pca_variance.csv", index=False)

    stats = pca_statistics(pipeline)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(components, cumulative, color="#2b6cb0", linewidth=2)
    ax.axhline(
        config.PCA_VARIANCE,
        color="#c53030",
        linestyle="--",
        label=f"Target variansi {config.PCA_VARIANCE:.0%}",
    )
    ax.axvline(
        int(stats["n_komponen_pca"]),
        color="#38a169",
        linestyle=":",
        label=f"{stats['n_komponen_pca']} komponen",
    )
    ax.set_title("Cumulative Explained Variance PCA")
    ax.set_xlabel("Jumlah komponen")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.text(
        0.03,
        0.06,
        (
            f"Fitur awal: {stats['n_features_awal']} | "
            f"Komponen: {stats['n_komponen_pca']} | "
            f"Reduksi: {stats['reduksi_dimensi_persen']:.1f}%"
        ),
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "pca_cumulative_variance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return stats


def plot_pca_scatter(
    X_train: np.ndarray,
    y_train: Sequence[int],
    plots_dir: Path = config.PLOTS_DIR,
) -> None:
    """Optional 2-D PCA scatter plot, fitted on train data only."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    projection = PCA(n_components=2, svd_solver="full", random_state=config.SEED)
    points = projection.fit_transform(X_train)
    labels = np.asarray(y_train)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    cmap = plt.get_cmap("tab10")
    for digit in config.CLASSES:
        mask = labels == digit
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=22,
            alpha=0.8,
            color=cmap(digit),
            label=str(digit),
        )
    ax.set_title("PCA 2D - Data Train (warna = kelas)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(title="Digit", ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "pca_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_clahe_comparison(
    paths_by_class: dict[int, Path],
    plots_dir: Path = config.PLOTS_DIR,
) -> None:
    """Show resized grayscale images before and after CLAHE."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(config.CLASSES), figsize=(18, 4.2))
    for column, digit in enumerate(config.CLASSES):
        path = paths_by_class[digit]
        resized = resize_image(load_grayscale_image(path))
        enhanced = apply_clahe(resized)

        axes[0, column].imshow(resized, cmap="gray", vmin=0, vmax=255)
        axes[0, column].set_title(f"Digit {digit}", fontsize=9)
        axes[0, column].axis("off")

        axes[1, column].imshow(enhanced, cmap="gray", vmin=0, vmax=255)
        axes[1, column].axis("off")

    axes[0, 0].set_ylabel("Sebelum CLAHE", fontsize=10)
    axes[1, 0].set_ylabel("Sesudah CLAHE", fontsize=10)
    fig.suptitle("Perbandingan Citra Sebelum dan Sesudah CLAHE (28x28)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(plots_dir / "clahe_before_after.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hog_visualization(
    paths_by_class: dict[int, Path],
    plots_dir: Path = config.PLOTS_DIR,
) -> None:
    """Visualise HOG gradients for one example of every digit."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, len(config.CLASSES), figsize=(18, 6.2))
    for column, digit in enumerate(config.CLASSES):
        path = paths_by_class[digit]
        resized = resize_image(load_grayscale_image(path))
        enhanced = apply_clahe(resized)
        _, hog_image = hog_visualization(enhanced)
        hog_image = rescale_intensity(
            hog_image, in_range=(float(hog_image.min()), float(hog_image.max()))
        )

        axes[0, column].imshow(resized, cmap="gray", vmin=0, vmax=255)
        axes[0, column].set_title(f"Digit {digit}", fontsize=9)
        axes[0, column].axis("off")

        axes[1, column].imshow(enhanced, cmap="gray", vmin=0, vmax=255)
        axes[1, column].axis("off")

        axes[2, column].imshow(hog_image, cmap="magma")
        axes[2, column].axis("off")

    axes[0, 0].set_ylabel("Asli 28x28", fontsize=10)
    axes[1, 0].set_ylabel("CLAHE", fontsize=10)
    axes[2, 0].set_ylabel("Gradien HOG", fontsize=10)
    fig.suptitle("Visualisasi Gradien HOG per Digit", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(plots_dir / "hog_visualization.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cv_accuracy(
    cv_results: dict[str, CVResult],
    plots_dir: Path = config.PLOTS_DIR,
) -> None:
    """Bar chart of CV accuracy mean and standard deviation per scenario."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    names = [config.SCENARIO_LABELS[key] for key in cv_results]
    means = [result.mean_accuracy for result in cv_results.values()]
    stds = [result.std_accuracy for result in cv_results.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        names,
        means,
        yerr=stds,
        capsize=8,
        color=["#2b6cb0", "#38a169"],
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.02,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy CV")
    ax.set_title("Stratified 5-Fold CV Accuracy per Skenario")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "cv_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

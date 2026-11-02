"""Data loading, validation, auditing, and stratified splitting.

This module implements Tahap 1 and Tahap 2 of the PRD.  No transformation
that learns from the data happens here; the train/test split is created before
any scaler, PCA, or model is fitted.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from . import config

logger = logging.getLogger(__name__)

METADATA_COLUMNS = ("origin", "group", "label", "file")


@dataclass(frozen=True)
class DataAudit:
    """Summary of the dataset validation process."""

    metadata_rows: int
    filtered_rows: int
    duplicate_rows: int
    invalid_label_rows: int
    missing_files: int
    unreadable_files: int
    valid_images: int
    class_counts: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load and validate the columns of the Kaggle metadata CSV."""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV tidak ditemukan: {metadata_path}")

    frame = pd.read_csv(metadata_path)
    missing = [column for column in METADATA_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Kolom metadata tidak lengkap: {missing}. "
            f"Kolom tersedia: {frame.columns.tolist()}"
        )
    return frame


def _resolve_image_path(numbers_root: Path, relative_file: str) -> Path:
    """Resolve a metadata ``file`` entry relative to the ``numbers/`` folder."""
    path = Path(str(relative_file).strip().replace("\\", "/"))
    if path.is_absolute():
        return path.resolve()
    return (numbers_root / path).resolve()


def build_valid_dataset(
    metadata_path: Path,
    dataset_dir: Path = config.DATASET_DIR,
) -> tuple[pd.DataFrame, DataAudit, pd.DataFrame]:
    """Filter Chars74k-Hnd rows and validate labels, paths, and readability.

    Returns
    -------
    valid_df:
        Columns ``file``, ``label``, ``path``, ``width``, ``height``.
    audit:
        Counts for every validation stage.
    failures:
        Rows that were rejected (missing/unreadable files or invalid labels).
    """
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {dataset_dir}. "
            "Pastikan archive/numbers/chars74k_png/Hnd tersedia."
        )

    metadata = load_metadata(metadata_path)
    filtered = metadata.loc[
        metadata["origin"].astype(str).str.strip().str.lower()
        == config.TARGET_ORIGIN.lower()
    ].copy()
    filtered = filtered.loc[
        filtered["group"].astype(str).str.strip() == config.TARGET_GROUP
    ].copy()
    filtered_rows = len(filtered)

    failure_rows: list[dict[str, str]] = []

    numeric_labels = pd.to_numeric(filtered["label"], errors="coerce")
    invalid_label_mask = ~numeric_labels.isin(config.CLASSES)
    for _, row in filtered.loc[invalid_label_mask].iterrows():
        failure_rows.append(
            {
                "file": str(row["file"]),
                "path": "",
                "reason": f"label tidak valid: {row['label']!r}",
            }
        )
    filtered = filtered.loc[~invalid_label_mask].copy()
    filtered["label"] = numeric_labels.loc[~invalid_label_mask].astype(int)

    numbers_root = dataset_dir.parent.parent
    filtered["path"] = [
        _resolve_image_path(numbers_root, value) for value in filtered["file"]
    ]

    duplicate_mask = filtered.duplicated(subset=["path"], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    for _, row in filtered.loc[duplicate_mask].iterrows():
        failure_rows.append(
            {"file": str(row["file"]), "path": str(row["path"]), "reason": "duplikat path"}
        )
    filtered = filtered.loc[~duplicate_mask].copy()

    rows: list[dict[str, Any]] = []
    missing_count = 0
    unreadable_count = 0

    for _, row in tqdm(
        filtered.iterrows(), total=len(filtered), desc="Validasi citra", unit="img"
    ):
        path = Path(row["path"])
        if not path.is_file():
            missing_count += 1
            failure_rows.append(
                {"file": str(row["file"]), "path": str(path), "reason": "file tidak ditemukan"}
            )
            continue

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            unreadable_count += 1
            failure_rows.append(
                {"file": str(row["file"]), "path": str(path), "reason": "gagal dibaca"}
            )
            continue

        height, width = image.shape[:2]
        rows.append(
            {
                "file": str(row["file"]),
                "label": int(row["label"]),
                "path": str(path),
                "width": int(width),
                "height": int(height),
            }
        )

    valid = pd.DataFrame(rows, columns=["file", "label", "path", "width", "height"])
    if valid.empty:
        raise RuntimeError("Tidak ada citra valid yang dapat digunakan.")

    counts = valid["label"].value_counts().sort_index()
    class_counts = {digit: int(counts.get(digit, 0)) for digit in config.CLASSES}
    audit = DataAudit(
        metadata_rows=len(metadata),
        filtered_rows=filtered_rows,
        duplicate_rows=duplicate_count,
        invalid_label_rows=int(invalid_label_mask.sum()),
        missing_files=missing_count,
        unreadable_files=unreadable_count,
        valid_images=len(valid),
        class_counts=class_counts,
    )
    failures = pd.DataFrame(
        failure_rows, columns=["file", "path", "reason"], dtype=str
    )
    return valid, audit, failures


def save_data_audit(
    valid_df: pd.DataFrame,
    audit: DataAudit,
    failures: pd.DataFrame,
    reports_dir: Path = config.REPORTS_DIR,
    plots_dir: Path = config.PLOTS_DIR,
) -> pd.DataFrame:
    """Persist the audit JSON/CSV, class distribution table, and sample grid."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    counts = (
        valid_df["label"]
        .value_counts()
        .reindex(config.CLASSES, fill_value=0)
        .sort_index()
        .astype(int)
    )
    distribution = pd.DataFrame(
        {
            "digit": counts.index.astype(int),
            "jumlah": counts.values.astype(int),
        }
    )
    distribution["persentase"] = (
        distribution["jumlah"] / distribution["jumlah"].sum() * 100
    ).round(3)
    distribution.to_csv(reports_dir / "class_distribution.csv", index=False)

    audit_payload = audit.to_dict()
    audit_payload["class_distribution"] = {
        str(digit): count for digit, count in audit.class_counts.items()
    }
    (reports_dir / "data_audit.json").write_text(
        json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    failures.to_csv(reports_dir / "data_failures.csv", index=False)

    _plot_class_distribution(distribution, plots_dir / "class_distribution.png")
    _plot_sample_images(valid_df, plots_dir / "sample_images_per_class.png")
    return distribution


def _plot_class_distribution(distribution: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of sample counts for digits 0-9."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        distribution["digit"].astype(str),
        distribution["jumlah"],
        color="#2b6cb0",
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, value in zip(bars, distribution["jumlah"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Distribusi Kelas Chars74k-Hnd (Sebelum Split)")
    ax.set_xlabel("Digit")
    ax.set_ylabel("Jumlah Citra")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_sample_images(
    valid_df: pd.DataFrame,
    output_path: Path,
    samples_per_class: int = 5,
) -> None:
    """Grid with several example images for every digit class."""
    fig, axes = plt.subplots(
        len(config.CLASSES),
        samples_per_class,
        figsize=(samples_per_class * 2.0, len(config.CLASSES) * 1.8),
        squeeze=False,
    )
    for row, digit in enumerate(config.CLASSES):
        subset = valid_df.loc[valid_df["label"] == digit].head(samples_per_class)
        for col in range(samples_per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(subset):
                continue
            image = cv2.imread(str(subset.iloc[col]["path"]), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                ax.imshow(image, cmap="gray")
            if col == 0:
                ax.set_title(f"Digit {digit}", loc="left", fontsize=9)
    fig.suptitle("Contoh Citra per Kelas (Chars74k-Hnd)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def stratified_split(
    dataframe: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified 80/20 split before any fitting step.

    Raises
    ------
    ValueError
        If a class has fewer than two samples (stratification impossible) or
        if stratification would leave a class out of either split.
    """
    counts = dataframe["label"].value_counts()
    minimum = int(counts.min())
    if minimum < 2:
        raise ValueError(
            "Stratified split tidak mungkin: kelas dengan sampel paling sedikit "
            f"hanya {minimum}. Tambah data atau sesuaikan strategi split secara eksplisit."
        )

    # For a class of size n, the test side must receive at least one sample.
    minimum_required = max(2, math.ceil(1.0 / test_size))
    if minimum < minimum_required:
        logger.warning(
            "Kelas dengan sampel paling sedikit hanya %d; split tetap dilakukan "
            "karena stratifikasi masih memungkinkan, tetapi evaluasi per kelas "
            "akan sangat tidak stabil.",
            minimum,
        )

    indices = np.arange(len(dataframe))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=dataframe["label"].to_numpy(),
        random_state=seed,
    )

    train_df = dataframe.iloc[train_idx].copy()
    test_df = dataframe.iloc[test_idx].copy()
    train_df["split"] = "train"
    test_df["split"] = "test"

    train_labels = set(train_df["label"].unique())
    test_labels = set(test_df["label"].unique())
    missing = set(config.CLASSES) - (train_labels & test_labels)
    if missing:
        raise ValueError(
            f"Stratifikasi gagal: kelas {sorted(missing)} tidak muncul di train dan test."
        )

    logger.info(
        "Split stratified selesai: train=%d (%.1f%%), test=%d (%.1f%%)",
        len(train_df),
        100 * len(train_df) / len(dataframe),
        len(test_df),
        100 * len(test_df) / len(dataframe),
    )
    return train_df, test_df


def save_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reports_dir: Path = config.REPORTS_DIR,
) -> pd.DataFrame:
    """Save the split manifest and per-class train/test counts."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    columns = ["file", "path", "label", "split"]
    manifest = pd.concat(
        [train_df[columns], test_df[columns]], ignore_index=True
    ).sort_values(["split", "label", "file"])
    manifest.to_csv(reports_dir / "train_test_split.csv", index=False)

    rows = []
    for digit in config.CLASSES:
        train_count = int((train_df["label"] == digit).sum())
        test_count = int((test_df["label"] == digit).sum())
        rows.append(
            {
                "digit": digit,
                "train": train_count,
                "test": test_count,
                "total": train_count + test_count,
                "train_persen": round(100 * train_count / len(train_df), 3),
                "test_persen": round(100 * test_count / len(test_df), 3),
            }
        )
    distribution = pd.DataFrame(rows)
    distribution.to_csv(reports_dir / "split_distribution.csv", index=False)
    return distribution


def choose_cv_folds(
    y_train: Sequence[int],
    requested: int = config.CV_FOLDS,
) -> int:
    """Choose a valid stratified fold count based on the rarest class."""
    counts = pd.Series(y_train).value_counts()
    minimum = int(counts.min())
    if minimum < 2:
        raise ValueError(
            "Cross-validation tidak mungkin: kelas dengan sampel paling sedikit "
            f"pada train hanya {minimum}."
        )
    folds = min(requested, minimum)
    if folds != requested:
        logger.warning(
            "Jumlah fold disesuaikan dari %d menjadi %d karena kelas tersedikit "
            "hanya memiliki %d sampel pada data train.",
            requested,
            folds,
            minimum,
        )
    return folds

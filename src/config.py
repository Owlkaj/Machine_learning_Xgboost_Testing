"""Central configuration for the Chars74k-Hnd digit classification project.

All paths are resolved relative to the project root so the pipeline can be run
from any working directory.  Hyperparameters follow ``PRD.md`` exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
ARCHIVE_DIR: Final[Path] = PROJECT_ROOT / "archive"
NUMBERS_DIR: Final[Path] = ARCHIVE_DIR / "numbers"
DATASET_DIR: Final[Path] = NUMBERS_DIR / "chars74k_png" / "Hnd"
METADATA_CANDIDATES: Final[tuple[Path, ...]] = (
    ARCHIVE_DIR / "numbers.csv",
    PROJECT_ROOT / "metadata" / "numbers.csv",
)

OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"
MODELS_DIR: Final[Path] = OUTPUT_DIR / "models"
REPORTS_DIR: Final[Path] = OUTPUT_DIR / "reports"
PLOTS_DIR: Final[Path] = OUTPUT_DIR / "plots"
CACHE_DIR: Final[Path] = PROJECT_ROOT / "cache"

# --------------------------------------------------------------------------- #
# Dataset / experiment constants
# --------------------------------------------------------------------------- #
SEED: Final[int] = 42
TEST_SIZE: Final[float] = 0.2
CV_FOLDS: Final[int] = 5
TARGET_ORIGIN: Final[str] = "chars74k"
TARGET_GROUP: Final[str] = "Hnd"
CLASSES: Final[tuple[int, ...]] = tuple(range(10))

# --------------------------------------------------------------------------- #
# Preprocessing (Tahap 3)
# --------------------------------------------------------------------------- #
IMAGE_SIZE: Final[tuple[int, int]] = (28, 28)  # (width, height) for OpenCV
CLAHE_CLIP_LIMIT: Final[float] = 2.0
CLAHE_TILE_GRID_SIZE: Final[tuple[int, int]] = (4, 4)

HOG_PARAMS: Final[dict[str, object]] = {
    "orientations": 9,
    "pixels_per_cell": (4, 4),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "visualize": False,
}
EXPECTED_HOG_FEATURES: Final[int] = 1_296

# --------------------------------------------------------------------------- #
# PCA (Tahap 4)
# --------------------------------------------------------------------------- #
PCA_VARIANCE: Final[float] = 0.95

# --------------------------------------------------------------------------- #
# XGBoost (Tahap 5)
# --------------------------------------------------------------------------- #
XGB_PARAMS: Final[dict[str, object]] = {
    "objective": "multi:softprob",
    "num_class": len(CLASSES),
    "tree_method": "hist",
    "n_estimators": 400,
    "learning_rate": 0.08,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": SEED,
}

# Names used consistently in reports, file names and plots.
SCENARIO_NO_PCA: Final[str] = "tanpa_pca"
SCENARIO_PCA: Final[str] = "dengan_pca"
SCENARIO_LABELS: Final[dict[str, str]] = {
    SCENARIO_NO_PCA: "XGBoost tanpa PCA",
    SCENARIO_PCA: "XGBoost dengan PCA",
}


def ensure_output_dirs() -> None:
    """Create every output/cache directory required by the pipeline."""
    for directory in (MODELS_DIR, REPORTS_DIR, PLOTS_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a single consistent logging format for all modules."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def resolve_metadata_path(explicit: Path | None = None) -> Path:
    """Return the metadata CSV path or raise a descriptive error."""
    candidates = (explicit,) if explicit is not None else METADATA_CANDIDATES
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        "CSV metadata tidak ditemukan. Dicari di: "
        f"{searched}. Unduh dataset Kaggle 'pintowar/numerical-images' dan "
        "letakkan numbers.csv di folder archive/ atau metadata/."
    )

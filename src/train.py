"""Cross-validation and final training for both feature scenarios (Tahap 5)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from . import config

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    """Stratified k-fold results for one scenario."""

    folds: int
    fold_accuracies: list[float] = field(default_factory=list)
    fold_f1_macro: list[float] = field(default_factory=list)
    fold_fit_times: list[float] = field(default_factory=list)

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean(self.fold_accuracies))

    @property
    def std_accuracy(self) -> float:
        return float(np.std(self.fold_accuracies, ddof=1))

    @property
    def mean_f1_macro(self) -> float:
        return float(np.mean(self.fold_f1_macro))

    @property
    def std_f1_macro(self) -> float:
        return float(np.std(self.fold_f1_macro, ddof=1))

    @property
    def mean_fit_time(self) -> float:
        return float(np.mean(self.fold_fit_times))

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable summary for reports."""
        return {
            "folds": self.folds,
            "fold_accuracies": self.fold_accuracies,
            "fold_f1_macro": self.fold_f1_macro,
            "fold_fit_times": self.fold_fit_times,
            "cv_accuracy_mean": self.mean_accuracy,
            "cv_accuracy_std": self.std_accuracy,
            "cv_f1_macro_mean": self.mean_f1_macro,
            "cv_f1_macro_std": self.std_f1_macro,
            "cv_fit_time_mean": self.mean_fit_time,
        }

    def to_frame(self, scenario: str) -> pd.DataFrame:
        """Per-fold table enriched with the scenario name."""
        frame = pd.DataFrame(
            {
                "fold": np.arange(1, self.folds + 1),
                "accuracy": self.fold_accuracies,
                "f1_macro": self.fold_f1_macro,
                "fit_time_detik": self.fold_fit_times,
            }
        )
        frame.insert(0, "skenario", scenario)
        return frame


def cross_validate_scenario(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    folds: int,
    seed: int = config.SEED,
) -> CVResult:
    """Run leakage-safe stratified k-fold cross-validation on the train set.

    The pipeline is cloned for every fold by scikit-learn, so StandardScaler
    and PCA are fitted only on the training part of each fold.
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_validate(
        clone(pipeline),
        X_train,
        y_train,
        cv=splitter,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        n_jobs=1,
        error_score="raise",
        return_estimator=False,
    )
    result = CVResult(
        folds=folds,
        fold_accuracies=[float(value) for value in scores["test_accuracy"]],
        fold_f1_macro=[float(value) for value in scores["test_f1_macro"]],
        fold_fit_times=[float(value) for value in scores["fit_time"]],
    )
    logger.info(
        "CV %d-fold: accuracy=%.4f +/- %.4f | macro-F1=%.4f +/- %.4f",
        folds,
        result.mean_accuracy,
        result.std_accuracy,
        result.mean_f1_macro,
        result.std_f1_macro,
    )
    return result


def fit_final_model(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[Pipeline, float]:
    """Fit a fresh pipeline on the full training set and time the fit."""
    model = clone(pipeline)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    logger.info("Training final selesai dalam %.3f detik", elapsed)
    return model, elapsed


def collect_learning_curve(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    scenario: str,
    reports_dir: Path = config.REPORTS_DIR,
    plots_dir: Path = config.PLOTS_DIR,
    validation_fraction: float = 0.15,
) -> None:
    """Optional mlogloss curve using an internal validation split of train.

    The official test set is never touched.  The curve is useful to inspect
    overfitting; it does not influence model selection.
    """
    scenario_label = config.SCENARIO_LABELS[scenario]
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_fraction,
        stratify=y_train,
        random_state=config.SEED,
    )

    # XGBoost's eval_set is passed straight to the booster, so the internal
    # validation features must be transformed with preprocessing that was
    # fitted on X_sub only (no leakage from X_val).
    steps = pipeline.named_steps
    if "pca" in steps:
        scaler = clone(steps["scaler"])
        pca = clone(steps["pca"])
        X_sub_ready = pca.fit_transform(scaler.fit_transform(X_sub))
        X_val_ready = pca.transform(scaler.transform(X_val))
    else:
        X_sub_ready = X_sub
        X_val_ready = X_val

    booster = clone(steps["xgb"])
    booster.fit(
        X_sub_ready,
        y_sub,
        eval_set=[(X_sub_ready, y_sub), (X_val_ready, y_val)],
        verbose=False,
    )
    history = booster.evals_result()
    train_loss = history["validation_0"]["mlogloss"]
    valid_loss = history["validation_1"]["mlogloss"]
    epochs = np.arange(1, len(train_loss) + 1)

    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "epoch": epochs,
            "mlogloss_train": train_loss,
            "mlogloss_validasi": valid_loss,
        }
    ).to_csv(reports_dir / f"learning_curve_{scenario}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train", color="#2b6cb0")
    ax.plot(epochs, valid_loss, label="Validasi (internal train)", color="#c53030")
    ax.set_title(f"Kurva mlogloss - {scenario_label}")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("mlogloss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        plots_dir / f"learning_curve_{scenario}.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

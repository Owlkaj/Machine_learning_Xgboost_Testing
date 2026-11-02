"""Model pipelines for the two feature scenarios (Tahap 4 & 5)."""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config


def build_xgb_classifier() -> XGBClassifier:
    """Create a fresh XGBoost classifier using the PRD configuration."""
    return XGBClassifier(**config.XGB_PARAMS)


def build_pipeline(use_pca: bool) -> Pipeline:
    """Build a complete, leakage-safe pipeline.

    Parameters
    ----------
    use_pca:
        ``False`` -> HOG -> XGBoost.
        ``True``  -> HOG -> StandardScaler -> PCA(95% variance) -> XGBoost.

    In both cases the returned pipeline can be passed directly to
    ``cross_validate``; scikit-learn clones and refits every step on each
    training fold, so the scaler and PCA never see validation/test data.
    """
    if not use_pca:
        return Pipeline(steps=[("xgb", build_xgb_classifier())])

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=config.PCA_VARIANCE,
                    svd_solver="full",
                    random_state=config.SEED,
                ),
            ),
            ("xgb", build_xgb_classifier()),
        ]
    )


def scenario_name(use_pca: bool) -> str:
    """Return the canonical scenario key used for files and reports."""
    return config.SCENARIO_PCA if use_pca else config.SCENARIO_NO_PCA

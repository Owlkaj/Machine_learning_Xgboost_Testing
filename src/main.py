"""End-to-end orchestration for Chars74k-Hnd + CLAHE + HOG + PCA + XGBoost.

Run from the project root:

    .venv\\Scripts\\python.exe -m src.main

The script executes every stage in PRD order and stores models, reports, and
plots under ``outputs/``.  The test set is touched exactly once per scenario,
after all cross-validation and final training are complete.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import config, data, evaluate, pipeline, preprocess, train

logger = logging.getLogger(__name__)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Log the start, end, and duration of a pipeline stage."""
    logger.info("=" * 72)
    logger.info("MULAI: %s", name)
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("SELESAI: %s (%.2f detik)", name, time.perf_counter() - start)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Klasifikasi digit Chars74k-Hnd dengan CLAHE + HOG + PCA + XGBoost"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path ke numbers.csv (default: archive/numbers.csv atau metadata/numbers.csv)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.DATASET_DIR,
        help=f"Folder subset Hnd (default: {config.DATASET_DIR})",
    )
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Abaikan cache dan ekstraksi ulang fitur HOG.",
    )
    parser.add_argument(
        "--learning-curve",
        action="store_true",
        help="Buat kurva mlogloss train/validasi internal (opsional).",
    )
    return parser.parse_args()


def _samples_by_class(dataframe: pd.DataFrame) -> dict[int, Path]:
    """Return the first image path of every class."""
    samples: dict[int, Path] = {}
    for digit in config.CLASSES:
        subset = dataframe.loc[dataframe["label"] == digit]
        if subset.empty:
            raise RuntimeError(f"Tidak ada sampel untuk digit {digit}.")
        samples[digit] = Path(subset.iloc[0]["path"])
    return samples


def _save_features_info(
    X_train: np.ndarray,
    X_test: np.ndarray,
    cache_train: Path,
    cache_test: Path,
) -> dict[str, object]:
    """Persist a small JSON description of the extracted HOG matrices."""
    payload: dict[str, object] = {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "expected_features": config.EXPECTED_HOG_FEATURES,
        "dtype": str(X_train.dtype),
        "cache_train": str(cache_train),
        "cache_test": str(cache_test),
    }
    (config.REPORTS_DIR / "features_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def _print_table(frame: pd.DataFrame, title: str) -> None:
    """Log a compact text table without relying on tabulate."""
    logger.info("%s\n%s", title, frame.to_string(index=False))


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the complete experiment and return the summary dictionary."""
    config.ensure_output_dirs()
    metadata_path = config.resolve_metadata_path(args.metadata)
    data_dir = args.data_dir.resolve()

    logger.info("Metadata : %s", metadata_path)
    logger.info("Dataset  : %s", data_dir)
    logger.info("Output   : %s", config.OUTPUT_DIR)

    # ------------------------------------------------------------------ #
    # Tahap 1 - loading, validation, audit
    # ------------------------------------------------------------------ #
    with stage("Tahap 1 - Data loading, validasi, dan audit"):
        valid_df, audit, failures = data.build_valid_dataset(
            metadata_path=metadata_path,
            dataset_dir=data_dir,
        )
        distribution = data.save_data_audit(valid_df, audit, failures)
        logger.info(
            "Metadata=%d | filtered=%d | duplikat=%d | label invalid=%d | "
            "file hilang=%d | gagal dibaca=%d | valid=%d",
            audit.metadata_rows,
            audit.filtered_rows,
            audit.duplicate_rows,
            audit.invalid_label_rows,
            audit.missing_files,
            audit.unreadable_files,
            audit.valid_images,
        )
        _print_table(distribution, "Distribusi kelas sebelum split:")
        if (distribution["jumlah"] == 0).any():
            missing_digits = distribution.loc[distribution["jumlah"] == 0, "digit"].tolist()
            raise RuntimeError(f"Digit tanpa sampel valid: {missing_digits}")

    # ------------------------------------------------------------------ #
    # Tahap 2 - stratified 80/20 split
    # ------------------------------------------------------------------ #
    with stage("Tahap 2 - Stratified train/test split"):
        train_df, test_df = data.stratified_split(
            valid_df,
            test_size=config.TEST_SIZE,
            seed=config.SEED,
        )
        split_distribution = data.save_split(train_df, test_df)
        _print_table(split_distribution, "Distribusi per kelas setelah split:")
        cv_folds = data.choose_cv_folds(
            train_df["label"].to_numpy(),
            requested=config.CV_FOLDS,
        )
        logger.info("Jumlah fold CV yang digunakan: %d", cv_folds)

    # ------------------------------------------------------------------ #
    # Tahap 3 - CLAHE + HOG (cache)
    # ------------------------------------------------------------------ #
    with stage("Tahap 3 - Preprocessing CLAHE dan ekstraksi HOG"):
        cache_train = config.CACHE_DIR / "hog_train.npz"
        cache_test = config.CACHE_DIR / "hog_test.npz"
        X_train = preprocess.load_or_extract_features(
            train_df["path"].tolist(),
            cache_path=cache_train,
            force_rebuild=args.force_features,
        )
        X_test = preprocess.load_or_extract_features(
            test_df["path"].tolist(),
            cache_path=cache_test,
            force_rebuild=args.force_features,
        )
        y_train = train_df["label"].to_numpy(dtype=np.int64)
        y_test = test_df["label"].to_numpy(dtype=np.int64)
        feature_info = _save_features_info(X_train, X_test, cache_train, cache_test)
        logger.info(
            "Fitur train=%s test=%s dtype=%s",
            X_train.shape,
            X_test.shape,
            X_train.dtype,
        )

        samples = _samples_by_class(train_df)
        evaluate.plot_clahe_comparison(samples)
        evaluate.plot_hog_visualization(samples)

    # ------------------------------------------------------------------ #
    # Tahap 4 & 5 - CV + final training for both scenarios
    # ------------------------------------------------------------------ #
    scenarios: dict[str, dict[str, object]] = {}
    evaluation_results: list[evaluate.EvaluationResult] = []
    cv_results: dict[str, train.CVResult] = {}
    fitted_models: dict[str, Pipeline] = {}

    scenario_settings = (
        (config.SCENARIO_NO_PCA, False),
        (config.SCENARIO_PCA, True),
    )

    for scenario, use_pca in scenario_settings:
        label = config.SCENARIO_LABELS[scenario]
        with stage(f"Tahap 4/5 - {label}: CV dan training final"):
            model_pipeline = pipeline.build_pipeline(use_pca=use_pca)
            cv_result = train.cross_validate_scenario(
                model_pipeline,
                X_train,
                y_train,
                folds=cv_folds,
                seed=config.SEED,
            )
            cv_results[scenario] = cv_result
            cv_result.to_frame(scenario).to_csv(
                config.REPORTS_DIR / f"cv_folds_{scenario}.csv", index=False
            )

            final_model, training_time = train.fit_final_model(
                model_pipeline, X_train, y_train
            )
            fitted_models[scenario] = final_model
            joblib.dump(final_model, config.MODELS_DIR / f"xgb_{scenario}.joblib")

            if args.learning_curve:
                train.collect_learning_curve(
                    model_pipeline, X_train, y_train, scenario=scenario
                )

            y_pred = final_model.predict(X_test)
            result = evaluate.evaluate_model(scenario, y_test, y_pred)
            evaluate.save_evaluation_artifacts(result, config.REPORTS_DIR)
            evaluation_results.append(result)

            pca_stats = evaluate.pca_statistics(final_model)
            scenarios[scenario] = {
                "label": label,
                "accuracy": result.accuracy,
                "macro_precision": result.macro_precision,
                "macro_recall": result.macro_recall,
                "macro_f1": result.macro_f1,
                "training_time_sec": training_time,
                "n_features": config.EXPECTED_HOG_FEATURES if not use_pca else pca_stats["n_komponen_pca"],
                "cv_accuracy_mean": cv_result.mean_accuracy,
                "cv_accuracy_std": cv_result.std_accuracy,
                "cv_f1_macro_mean": cv_result.mean_f1_macro,
                "cv_f1_macro_std": cv_result.std_f1_macro,
                "cv_fit_time_mean_sec": cv_result.mean_fit_time,
                **pca_stats,
            }

    # ------------------------------------------------------------------ #
    # Tahap 6 - comparison metrics and remaining visualisations
    # ------------------------------------------------------------------ #
    with stage("Tahap 6 - Perbandingan metrik dan visualisasi"):
        comparison_rows = []
        for scenario, payload in scenarios.items():
            comparison_rows.append(
                {
                    "Skenario": payload["label"],
                    "Accuracy": round(float(payload["accuracy"]), 6),
                    "Macro Precision": round(float(payload["macro_precision"]), 6),
                    "Macro Recall": round(float(payload["macro_recall"]), 6),
                    "Macro F1-Score": round(float(payload["macro_f1"]), 6),
                    "Training Time (detik)": round(float(payload["training_time_sec"]), 4),
                    "Jumlah Fitur": int(payload["n_features"]),
                    "CV Accuracy Mean": round(float(payload["cv_accuracy_mean"]), 6),
                    "CV Accuracy Std": round(float(payload["cv_accuracy_std"]), 6),
                    "CV Macro F1 Mean": round(float(payload["cv_f1_macro_mean"]), 6),
                    "CV Macro F1 Std": round(float(payload["cv_f1_macro_std"]), 6),
                    "CV Mean Fit Time (detik)": round(float(payload["cv_fit_time_mean_sec"]), 4),
                    "Komponen PCA": int(payload["n_komponen_pca"]),
                    "Reduksi Dimensi (%)": float(payload["reduksi_dimensi_persen"]),
                    "Cumulative Explained Variance": float(
                        payload["cumulative_explained_variance"]
                    ),
                }
            )
        comparison = pd.DataFrame(comparison_rows)
        comparison.to_csv(config.REPORTS_DIR / "metrics_comparison.csv", index=False)
        _print_table(comparison, "Tabel perbandingan skenario:")

        pd.concat(
            [
                result.to_frame(scenario)
                for scenario, result in cv_results.items()
            ],
            ignore_index=True,
        ).to_csv(config.REPORTS_DIR / "cv_results.csv", index=False)

        evaluate.plot_cv_accuracy(cv_results)
        evaluate.plot_confusion_matrices(evaluation_results)
        pca_stats = evaluate.plot_pca_variance(fitted_models[config.SCENARIO_PCA])
        evaluate.plot_pca_scatter(X_train, y_train)

    summary: dict[str, object] = {
        "dataset": audit.to_dict(),
        "split": {
            "train": int(len(train_df)),
            "test": int(len(test_df)),
            "test_size": config.TEST_SIZE,
            "seed": config.SEED,
            "cv_folds": int(cv_folds),
        },
        "features": feature_info,
        "pca": pca_stats,
        "scenarios": scenarios,
        "config": {
            "image_size": list(config.IMAGE_SIZE),
            "clahe": {
                "clip_limit": config.CLAHE_CLIP_LIMIT,
                "tile_grid_size": list(config.CLAHE_TILE_GRID_SIZE),
            },
            "hog": {key: str(value) for key, value in config.HOG_PARAMS.items()},
            "pca_variance": config.PCA_VARIANCE,
            "xgb_params": config.XGB_PARAMS,
        },
    }
    (config.REPORTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("=" * 72)
    logger.info("SELESAI. Artefak tersimpan di: %s", config.OUTPUT_DIR)
    return summary


def main() -> None:
    """CLI entry point."""
    config.configure_logging()
    args = parse_args()
    summary = run(args)
    test_accuracies = {
        payload["label"]: round(float(payload["accuracy"]), 4)
        for payload in summary["scenarios"].values()  # type: ignore[union-attr]
    }
    logger.info("Test accuracy: %s", test_accuracies)


if __name__ == "__main__":
    main()

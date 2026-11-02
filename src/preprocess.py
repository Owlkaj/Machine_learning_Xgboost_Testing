"""Image preprocessing (grayscale, resize, CLAHE) and HOG extraction.

Implements Tahap 3 of the PRD.  The exact same deterministic transformation is
used for every image and for both train and test splits.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm

from . import config

logger = logging.getLogger(__name__)


def load_grayscale_image(path: str | Path) -> np.ndarray:
    """Read an image as grayscale and fail loudly when it cannot be decoded."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise ValueError(f"Citra tidak dapat dibaca: {path}")
    return image


def resize_image(
    image: np.ndarray,
    size: tuple[int, int] = config.IMAGE_SIZE,
) -> np.ndarray:
    """Resize to 28x28 with area interpolation (antialiased downscaling)."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = config.CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = config.CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def preprocess_image(path: str | Path) -> np.ndarray:
    """Full deterministic preprocessing: grayscale -> 28x28 -> CLAHE."""
    image = load_grayscale_image(path)
    resized = resize_image(image)
    return apply_clahe(resized)


def extract_hog_from_image(image: np.ndarray) -> np.ndarray:
    """Extract a 1-D float32 HOG descriptor from a preprocessed image."""
    features = hog(image, feature_vector=True, **config.HOG_PARAMS)
    vector = np.asarray(features, dtype=np.float32).ravel()
    if vector.size != config.EXPECTED_HOG_FEATURES:
        raise RuntimeError(
            "Jumlah fitur HOG tidak sesuai: "
            f"{vector.size} (diharapkan {config.EXPECTED_HOG_FEATURES}). "
            f"Periksa HOG_PARAMS={config.HOG_PARAMS}."
        )
    return vector


def extract_hog_from_path(path: str | Path) -> np.ndarray:
    """Preprocess one image and return its HOG descriptor."""
    return extract_hog_from_image(preprocess_image(path))


def extract_hog_features(
    paths: Sequence[str | Path],
    show_progress: bool = True,
) -> np.ndarray:
    """Extract HOG descriptors for a sequence of image paths.

    The result is a ``(n_samples, 1296)`` ``np.float32`` matrix.
    """
    iterator = tqdm(paths, desc="Ekstraksi HOG", unit="img", disable=not show_progress)
    rows = [extract_hog_from_path(path) for path in iterator]
    matrix = np.vstack(rows).astype(np.float32, copy=False)
    if matrix.shape[1] != config.EXPECTED_HOG_FEATURES:
        raise RuntimeError(f"Shape fitur tidak valid: {matrix.shape}")
    logger.info("Fitur HOG selesai: shape=%s dtype=%s", matrix.shape, matrix.dtype)
    return matrix


def _cache_fingerprint(paths: Sequence[str | Path]) -> str:
    """Fingerprint the *set* of paths, file stats, and HOG parameters.

    The fingerprint is intentionally order-independent.  The cache also stores
    the original path order, so ``load_or_extract_features`` can reorder the
    matrix when the caller supplies the same images in a different sequence.
    """
    entries = []
    for path in paths:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        entries.append(f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}")
    entries.sort()

    hasher = hashlib.sha1()
    hasher.update(
        json.dumps(
            {
                "cache_schema": 2,
                "image_size": config.IMAGE_SIZE,
                "clahe": [config.CLAHE_CLIP_LIMIT, config.CLAHE_TILE_GRID_SIZE],
                "hog": config.HOG_PARAMS,
                "expected": config.EXPECTED_HOG_FEATURES,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )
    hasher.update("\n".join(entries).encode("utf-8"))
    return hasher.hexdigest()


def load_or_extract_features(
    paths: Sequence[str | Path],
    cache_path: Path,
    force_rebuild: bool = False,
) -> np.ndarray:
    """Load cached HOG features when valid, otherwise extract and cache them.

    If the cache contains the same images in a different order, rows are
    reordered to match ``paths`` so that features always align with labels.
    """
    fingerprint = _cache_fingerprint(paths)
    requested_paths = [str(Path(path).resolve()) for path in paths]

    if cache_path.is_file() and not force_rebuild:
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_fingerprint = str(cached["fingerprint"].item())
                matrix = cached["X"]
                cached_paths = [str(value) for value in cached["paths"].tolist()]
            if (
                cached_fingerprint == fingerprint
                and matrix.shape == (len(requested_paths), config.EXPECTED_HOG_FEATURES)
                and matrix.dtype == np.float32
                and len(cached_paths) == len(requested_paths)
                and set(cached_paths) == set(requested_paths)
            ):
                if cached_paths == requested_paths:
                    logger.info(
                        "Memuat cache fitur: %s %s", cache_path.name, matrix.shape
                    )
                    return matrix

                position = {path: index for index, path in enumerate(cached_paths)}
                order = np.array(
                    [position[path] for path in requested_paths], dtype=np.int64
                )
                logger.info(
                    "Memuat cache fitur dan menyusun ulang urutan: %s %s",
                    cache_path.name,
                    matrix.shape,
                )
                return np.ascontiguousarray(matrix[order])

            logger.warning("Cache tidak cocok/korup, ekstraksi ulang: %s", cache_path)
        except (KeyError, OSError, ValueError) as error:
            logger.warning("Gagal membaca cache (%s), ekstraksi ulang.", error)

    matrix = extract_hog_features(paths)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X=matrix,
        fingerprint=np.array(fingerprint),
        paths=np.array(requested_paths),
    )
    logger.info("Cache fitur disimpan: %s", cache_path)
    return matrix


def hog_visualization(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return features and a visualisation-ready HOG gradient image."""
    params: dict[str, Any] = dict(config.HOG_PARAMS)
    params["visualize"] = True
    features, hog_image = hog(image, feature_vector=True, **params)
    return np.asarray(features, dtype=np.float32), np.asarray(hog_image)

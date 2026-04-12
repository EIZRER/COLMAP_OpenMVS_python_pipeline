"""
ML Feature Pipeline: SuperPoint + LightGlue → COLMAP Database

Replaces COLMAP's SIFT-based feature_extractor + exhaustive_matcher with:
  1. SuperPoint keypoint + descriptor extraction per image
  2. LightGlue matching for all exhaustive image pairs
  3. OpenCV RANSAC geometric verification (fundamental matrix)
  4. Direct write of keypoints and verified two-view geometries to the
     COLMAP SQLite database (bypasses COLMAP matcher binaries entirely)

The database must already contain the cameras and images tables, which are
populated by running `colmap feature_extractor` first (we still run it so
that COLMAP sets up camera intrinsics; we then replace kps/descs/matches).
"""
import threading
import cv2
import numpy as np
from pathlib import Path
from typing import Callable, List, Optional

from ml.colmap_db import COLMAPDatabase, TWO_VIEW_UNCALIBRATED
from ml.lightglue import LightGlueMatcher

# Image extensions considered by the pipeline
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Minimum inlier count to store a two-view geometry entry
_MIN_INLIERS = 15


def _find_images(image_dir: str) -> List[Path]:
    """Return sorted list of image paths in image_dir."""
    root = Path(image_dir)
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    return paths


def _geometric_verify(
    kp1: np.ndarray, kp2: np.ndarray, matches: np.ndarray
) -> tuple:
    """RANSAC fundamental-matrix estimation.

    Returns (inlier_matches, F_matrix) or (None, None) if not enough points.
    """
    if len(matches) < 8:
        return None, None

    pts1 = kp1[matches[:, 0]]  # (M, 2)
    pts2 = kp2[matches[:, 1]]  # (M, 2)

    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=4.0,
        confidence=0.999,
        maxIters=2000,
    )
    if F is None or mask is None:
        return None, None

    inlier_mask = mask.ravel().astype(bool)
    inliers = matches[inlier_mask]

    if len(inliers) < _MIN_INLIERS:
        return None, None

    return inliers, F


def run_ml_feature_pipeline(
    image_dir: str,
    database_path: str,
    use_gpu: bool = True,
    max_keypoints: int = 2048,
    log_callback: Optional[Callable[[str], None]] = None,
    abort_event: Optional[threading.Event] = None,
):
    """Main entry point called by the photogrammetry pipeline.

    Writes SuperPoint keypoints/descriptors and LightGlue+RANSAC two-view
    geometries directly into the COLMAP SQLite database.

    Parameters
    ----------
    image_dir       : folder containing input images
    database_path   : path to the COLMAP database file (already created)
    use_gpu         : use CUDA if available
    max_keypoints   : maximum keypoints per image (-1 = unlimited)
    log_callback    : optional logging function
    abort_event     : optional threading.Event to support early abort
    """

    def log(msg: str):
        if log_callback:
            log_callback(msg)

    def check_abort():
        if abort_event and abort_event.is_set():
            raise RuntimeError("ABORTED")

    # ── Load models ───────────────────────────────────────────────────────
    log("  [ML] Loading SuperPoint + LightGlue models…")
    try:
        matcher = LightGlueMatcher(use_gpu=use_gpu, max_keypoints=max_keypoints)
    except ImportError as e:
        raise RuntimeError(str(e))

    device_name = str(matcher.device)
    log(f"  [ML] Running on device: {device_name}")

    # ── Discover images ───────────────────────────────────────────────────
    image_paths = _find_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {image_dir}")
    log(f"  [ML] Found {len(image_paths)} image(s)")

    # ── Open DB and map image names → IDs ─────────────────────────────────
    with COLMAPDatabase(database_path) as db:
        name_to_id = db.get_image_name_to_id()

        # Verify all images are registered in the DB
        registered = {p for p in image_paths if p.name in name_to_id}
        if not registered:
            raise RuntimeError(
                "None of the discovered images are registered in the COLMAP database. "
                "Make sure colmap feature_extractor has been run first."
            )
        log(f"  [ML] {len(registered)}/{len(image_paths)} images found in DB")

        # Filter to only registered images
        image_paths = [p for p in image_paths if p.name in name_to_id]

        # Clear existing keypoints/descriptors/geometries so we start fresh
        db.clear_keypoints_and_descriptors()
        db.clear_two_view_geometries()

        # ── Step 1: Feature Extraction ────────────────────────────────────
        log("  [ML] Extracting SuperPoint features…")
        all_features = {}  # image_id → {keypoints, descriptors, feats_dict}

        for i, img_path in enumerate(image_paths):
            check_abort()
            image_id = name_to_id[img_path.name]

            import cv2 as _cv2
            img = _cv2.imread(str(img_path))
            if img is None:
                log(f"  [ML] WARNING: could not read {img_path.name}, skipping")
                continue

            kp, desc, feats = matcher.extract(img)
            db.write_keypoints(image_id, kp)
            db.write_descriptors(image_id, desc)

            all_features[image_id] = {
                "keypoints": kp,
                "descriptors": desc,
                "feats": feats,
                "name": img_path.name,
            }
            log(f"  [ML]   [{i+1}/{len(image_paths)}] {img_path.name}: {len(kp)} keypoints")

        # Commit feature writes before matching
        db._conn.commit()

        # ── Step 2: LightGlue Matching + Geometric Verification ───────────
        ids = list(all_features.keys())
        n_pairs = len(ids) * (len(ids) - 1) // 2
        log(f"  [ML] Matching {n_pairs} image pair(s) with LightGlue…")

        matched_pairs = 0
        from itertools import combinations

        for pair_idx, (id1, id2) in enumerate(combinations(ids, 2)):
            check_abort()

            f1 = all_features[id1]
            f2 = all_features[id2]

            raw_matches = matcher.match(f1["feats"], f2["feats"])

            if len(raw_matches) < 8:
                continue

            inliers, F = _geometric_verify(f1["keypoints"], f2["keypoints"], raw_matches)
            if inliers is None:
                continue

            db.write_two_view_geometry(
                id1, id2, inliers,
                F=F,
                config=TWO_VIEW_UNCALIBRATED,
            )
            matched_pairs += 1

            if (pair_idx + 1) % 50 == 0 or (pair_idx + 1) == n_pairs:
                log(
                    f"  [ML]   Processed {pair_idx+1}/{n_pairs} pairs "
                    f"({matched_pairs} with sufficient matches)"
                )

    log(f"  [ML] Done. {matched_pairs}/{n_pairs} pairs written to DB.")

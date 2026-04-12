"""
Utilities for reading and writing the COLMAP SQLite database directly.

COLMAP uses a well-defined schema.  We write SuperPoint keypoints,
descriptors, and LightGlue+RANSAC verified matches without going through
COLMAP's own matcher binary.
"""
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import combinations


# ── Pair-ID encoding (matches COLMAP's internal convention) ───────────────

def _pair_id(id1: int, id2: int) -> int:
    """Encode two image IDs as a single int64 pair key (smaller id first)."""
    if id1 > id2:
        id1, id2 = id2, id1
    return id1 * 2_147_483_647 + id2


# ── Two-view geometry config codes ────────────────────────────────────────
#   COLMAP defines: UNDEFINED=0, DEGENERATE=1, CALIBRATED=2,
#                   UNCALIBRATED=3, PLANAR=4, PANORAMIC=5, PLANAR_OR_PANORAMIC=6
TWO_VIEW_UNCALIBRATED = 3


class COLMAPDatabase:
    """Thin wrapper around the COLMAP SQLite database for reading and writing
    keypoints, descriptors, and two-view geometries.

    Usage (context manager):
    >>> with COLMAPDatabase("database.db") as db:
    ...     names = db.get_image_names()
    """

    def __init__(self, db_path: str):
        self._path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> "COLMAPDatabase":
        self._conn = sqlite3.connect(self._path)
        return self

    def __exit__(self, *_):
        if self._conn:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    # ── Read helpers ──────────────────────────────────────────────────────

    def get_image_name_to_id(self) -> Dict[str, int]:
        """Return {image_name: image_id} for all images in the DB."""
        cur = self._conn.execute("SELECT image_id, name FROM images")
        return {name: image_id for image_id, name in cur.fetchall()}

    def get_all_image_ids(self) -> List[int]:
        cur = self._conn.execute("SELECT image_id FROM images ORDER BY image_id")
        return [row[0] for row in cur.fetchall()]

    def get_all_pairs(self) -> List[Tuple[int, int]]:
        """Return all image ID pairs for exhaustive matching."""
        ids = self.get_all_image_ids()
        return list(combinations(ids, 2))

    # ── Write helpers ─────────────────────────────────────────────────────

    def write_keypoints(self, image_id: int, keypoints: np.ndarray):
        """Write or overwrite keypoints for an image.

        keypoints : (N, 2) float32 – (x, y) pixel coordinates.
        COLMAP accepts 2, 4, or 6 columns; we write 2.
        """
        kp = np.ascontiguousarray(keypoints, dtype=np.float32)
        n, cols = kp.shape
        self._conn.execute(
            "INSERT OR REPLACE INTO keypoints (image_id, rows, cols, data) VALUES (?,?,?,?)",
            (image_id, n, cols, kp.tobytes()),
        )

    def write_descriptors(self, image_id: int, descriptors: np.ndarray):
        """Write or overwrite descriptors for an image.

        descriptors : (N, D) float32
        """
        desc = np.ascontiguousarray(descriptors, dtype=np.float32)
        n, d = desc.shape
        self._conn.execute(
            "INSERT OR REPLACE INTO descriptors (image_id, rows, cols, data) VALUES (?,?,?,?)",
            (image_id, n, d, desc.tobytes()),
        )

    def write_two_view_geometry(
        self,
        image_id1: int,
        image_id2: int,
        inlier_matches: np.ndarray,
        F: Optional[np.ndarray] = None,
        E: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        config: int = TWO_VIEW_UNCALIBRATED,
    ):
        """Write verified matches + geometry for a pair into two_view_geometries.

        inlier_matches : (M, 2) int32 – (idx_in_img1, idx_in_img2)
        F, E, H        : optional 3x3 float64 matrices
        config         : COLMAP geometry type (default: UNCALIBRATED=3)
        """
        # Ensure ordering: smaller id first (COLMAP convention)
        if image_id1 > image_id2:
            image_id1, image_id2 = image_id2, image_id1
            inlier_matches = inlier_matches[:, ::-1].copy()

        pair_id = _pair_id(image_id1, image_id2)
        m = len(inlier_matches)
        match_data = np.ascontiguousarray(inlier_matches, dtype=np.uint32).tobytes()

        def _mat(mat):
            if mat is None:
                return b""
            return np.ascontiguousarray(mat, dtype=np.float64).flatten().tobytes()

        self._conn.execute(
            "INSERT OR REPLACE INTO two_view_geometries "
            "(pair_id, rows, cols, data, config, F, E, H, qvec, tvec) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (pair_id, m, 2, match_data, config, _mat(F), _mat(E), _mat(H), b"", b""),
        )

    def clear_keypoints_and_descriptors(self):
        """Remove all rows from keypoints and descriptors tables."""
        self._conn.execute("DELETE FROM keypoints")
        self._conn.execute("DELETE FROM descriptors")

    def clear_two_view_geometries(self):
        """Remove all rows from two_view_geometries table."""
        self._conn.execute("DELETE FROM two_view_geometries")
        self._conn.execute("DELETE FROM matches")

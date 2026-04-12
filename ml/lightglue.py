"""
LightGlue matcher wrapper.

Uses the official ETH CVG lightglue package:
  pip install git+https://github.com/cvg/LightGlue.git

Provides a unified API for SuperPoint feature extraction + LightGlue matching
that is used by the ML feature pipeline.
"""
import torch
from typing import Dict, Tuple
import numpy as np


class LightGlueMatcher:
    """Wraps lightglue.SuperPoint + lightglue.LightGlue into a single helper.

    Usage
    -----
    matcher = LightGlueMatcher(use_gpu=True, max_keypoints=2048)
    kp, desc, feats = matcher.extract(image_np)
    matches = matcher.match(feats0, feats1)  # (M, 2) int32
    """

    def __init__(self, use_gpu: bool = True, max_keypoints: int = 2048):
        try:
            from lightglue import LightGlue, SuperPoint
            from lightglue.utils import rbd
            self._rbd = rbd
        except ImportError as e:
            raise ImportError(
                "The 'lightglue' package is required for ML feature matching.\n"
                "Install it with:\n"
                "  pip install git+https://github.com/cvg/LightGlue.git"
            ) from e

        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    # ── Feature Extraction ────────────────────────────────────────────────

    @torch.no_grad()
    def extract(self, image_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Extract SuperPoint features from a numpy image.

        image_np : (H, W) grayscale float32 in [0, 1]
                   OR (H, W, 3) RGB uint8

        Returns
        -------
        keypoints   : (N, 2) float32  – (x, y) pixel coords
        descriptors : (N, 256) float32
        feats_dict  : batched feature dict (kept in CPU memory for later matching)
        """
        import cv2

        # Convert to (H, W) float32 in [0, 1]
        if image_np.ndim == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        elif image_np.dtype == np.uint8:
            gray = image_np.astype(np.float32) / 255.0
        else:
            gray = image_np.astype(np.float32)

        # lightglue expects (1, 1, H, W) or can load from path via load_image
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)

        feats = self.extractor.extract(tensor)
        feats_cpu = {k: v.cpu() for k, v in feats.items()}

        kp = self._rbd(feats)["keypoints"].cpu().numpy()        # (N, 2)
        desc = self._rbd(feats)["descriptors"].cpu().numpy()    # (N, D)

        return kp, desc, feats_cpu

    # ── Matching ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def match(self, feats0: Dict, feats1: Dict) -> np.ndarray:
        """Run LightGlue on two feature dicts.

        Parameters
        ----------
        feats0, feats1 : feature dicts returned by :meth:`extract` (CPU tensors)

        Returns
        -------
        matches : (M, 2) int32 – (idx_in_image0, idx_in_image1)
        """
        f0 = {k: v.to(self.device) for k, v in feats0.items()}
        f1 = {k: v.to(self.device) for k, v in feats1.items()}

        result = self.matcher({"image0": f0, "image1": f1})
        matches = self._rbd(result)["matches"].cpu().numpy()  # (M, 2)
        return matches.astype(np.int32)

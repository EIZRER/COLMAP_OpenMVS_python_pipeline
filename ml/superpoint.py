"""
SuperPoint: Self-Supervised Interest Point Detection and Description
Ref: DeTone et al. 2018 (https://arxiv.org/abs/1712.07629)

Architecture is compatible with the official Magic Leap pretrained weights:
  https://github.com/magicleap/SuperPointPretrainedNetwork
  (superpoint_v1.pth)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple


def _nms_keypoints(heatmap: np.ndarray, threshold: float, nms_radius: int) -> np.ndarray:
    """Non-Maximum Suppression on a 2-D score heatmap.

    Returns (N, 2) float32 array of (x, y) pixel coordinates.
    """
    # Apply threshold
    heatmap = heatmap * (heatmap >= threshold)

    # Max-pool NMS
    pad = nms_radius
    h_padded = np.pad(heatmap, pad, mode="constant", constant_values=0.0)
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(h_padded, size=nms_radius * 2 + 1)[pad:-pad, pad:-pad]
    keep = (heatmap == local_max) & (heatmap > 0)
    ys, xs = np.where(keep)
    scores = heatmap[ys, xs]

    # Sort by score descending
    order = np.argsort(scores)[::-1]
    xs = xs[order]
    ys = ys[order]
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _sample_descriptors(
    descriptors: torch.Tensor, keypoints: np.ndarray, image_hw: Tuple[int, int]
) -> np.ndarray:
    """Bilinear-sample descriptor map at keypoint locations.

    descriptors : (1, D, Hc, Wc)  – the coarse feature map (H/8, W/8)
    keypoints   : (N, 2) float32  – (x, y) pixel coordinates in the full image
    image_hw    : (H, W) of the full-resolution input image

    Returns (N, D) float32.
    """
    H, W = image_hw
    N = len(keypoints)
    if N == 0:
        D = descriptors.shape[1]
        return np.zeros((0, D), dtype=np.float32)

    kp_t = torch.from_numpy(keypoints).float()  # (N, 2)
    # Normalise to [-1, 1] as expected by grid_sample
    norm_x = 2.0 * kp_t[:, 0] / (W - 1) - 1.0
    norm_y = 2.0 * kp_t[:, 1] / (H - 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=1)  # (N, 2)
    grid = grid.view(1, 1, N, 2)  # (1, 1, N, 2)

    sampled = F.grid_sample(descriptors, grid, mode="bilinear", align_corners=True)
    sampled = sampled.squeeze(0).squeeze(1)  # (D, N)
    sampled = F.normalize(sampled, p=2, dim=0)  # L2-normalise each descriptor
    return sampled.T.cpu().numpy()  # (N, D)


class SuperPoint(nn.Module):
    """SuperPoint network (encoder + detector head + descriptor head).

    Pretrained weights can be loaded with :meth:`load_weights`.
    Compatible with the official ``superpoint_v1.pth`` from Magic Leap.
    """

    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,  # -1 → keep all above threshold
        "remove_borders": 4,
    }

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        cfg = {**self.default_config, **(config or {})}
        self.nms_radius = cfg["nms_radius"]
        self.keypoint_threshold = cfg["keypoint_threshold"]
        self.max_keypoints = cfg["max_keypoints"]
        self.remove_borders = cfg["remove_borders"]

        # ── Shared Encoder (VGG-like, no BN) ──────────────────────────────
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1a = nn.Conv2d(1, 64, 3, 1, 1)
        self.conv1b = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2a = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2b = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3a = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3b = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4a = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4b = nn.Conv2d(128, 128, 3, 1, 1)

        # ── Detector head ─────────────────────────────────────────────────
        self.convPa = nn.Conv2d(128, 256, 3, 1, 1)
        self.convPb = nn.Conv2d(256, 65, 1, 1, 0)

        # ── Descriptor head ───────────────────────────────────────────────
        self.convDa = nn.Conv2d(128, 256, 3, 1, 1)
        self.convDb = nn.Conv2d(256, 256, 1, 1, 0)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (heatmap, dense_descriptor_map).

        x        : (B, 1, H, W) float32, values in [0, 1]
        heatmap  : (B, H, W)    – per-pixel keypoint probability
        desc_map : (B, 256, H/8, W/8) – L2-normalised descriptor map
        """
        # Encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Detector head
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)  # (B, 65, H/8, W/8)
        B, _, Hc, Wc = semi.shape
        semi = semi.permute(0, 2, 3, 1)  # (B, Hc, Wc, 65)
        semi = semi.softmax(dim=-1)
        semi = semi[:, :, :, :-1]  # drop dustbin → (B, Hc, Wc, 64)
        semi = semi.reshape(B, Hc, Wc, 8, 8)
        semi = semi.permute(0, 1, 3, 2, 4).contiguous()
        heatmap = semi.reshape(B, Hc * 8, Wc * 8)  # (B, H, W)

        # Descriptor head
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)  # (B, 256, H/8, W/8)
        desc = F.normalize(desc, p=2, dim=1)

        return heatmap, desc

    # ── Extraction API ────────────────────────────────────────────────────

    @torch.no_grad()
    def extract(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract keypoints and descriptors from a single image.

        image : (H, W) uint8 or float32 grayscale array
                OR (H, W, 3) BGR/RGB uint8 — converted to grayscale internally.

        Returns
        -------
        keypoints   : (N, 2) float32  – (x, y) pixel coordinates
        descriptors : (N, 256) float32 – L2-normalised descriptors
        """
        gray = self._to_gray(image)
        H, W = gray.shape

        tensor = (
            torch.from_numpy(gray.astype(np.float32) / 255.0)
            .unsqueeze(0)  # (1, H, W)
            .unsqueeze(0)  # (1, 1, H, W)
        )
        device = next(self.parameters()).device
        tensor = tensor.to(device)

        heatmap, desc_map = self.forward(tensor)
        heatmap_np = heatmap[0].cpu().numpy()  # (H, W)

        # Remove border responses
        b = self.remove_borders
        heatmap_np[:b, :] = 0
        heatmap_np[-b:, :] = 0
        heatmap_np[:, :b] = 0
        heatmap_np[:, -b:] = 0

        # NMS
        keypoints = _nms_keypoints(heatmap_np, self.keypoint_threshold, self.nms_radius)

        if self.max_keypoints > 0 and len(keypoints) > self.max_keypoints:
            scores = heatmap_np[keypoints[:, 1].astype(int), keypoints[:, 0].astype(int)]
            idx = np.argsort(scores)[::-1][: self.max_keypoints]
            keypoints = keypoints[idx]

        descriptors = _sample_descriptors(desc_map, keypoints, (H, W))
        return keypoints, descriptors

    # ── Weight Loading ────────────────────────────────────────────────────

    def load_weights(self, weights_path: str):
        """Load pretrained weights from a .pth file.

        Compatible with the official Magic Leap weights
        (https://github.com/magicleap/SuperPointPretrainedNetwork).
        """
        state = torch.load(weights_path, map_location="cpu")
        # Official weights are stored directly as a state dict
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.load_state_dict(state)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

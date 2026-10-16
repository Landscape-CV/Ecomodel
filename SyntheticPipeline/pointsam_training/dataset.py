"""PyTorch dataset over voxelized Point-SAM crops."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_unit_sphere(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Center + scale to unit sphere (matches SegmenterPointSAM / upstream)."""
    centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale < 1e-8:
        scale = 1.0
    return (centered / scale).astype(np.float32), centroid.astype(np.float32), scale


def instance_ids_to_masks(
    instance_ids: np.ndarray,
    num_masks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample up to num_masks tree instances → bool [M, N]."""
    trees = np.unique(instance_ids)
    trees = trees[trees >= 0]
    if len(trees) == 0:
        # degenerate: all-false then force one random point
        m = np.zeros((num_masks, len(instance_ids)), dtype=bool)
        if len(instance_ids):
            m[0, rng.integers(0, len(instance_ids))] = True
        return m
    if len(trees) >= num_masks:
        chosen = rng.choice(trees, size=num_masks, replace=False)
    else:
        extra = rng.choice(trees, size=num_masks - len(trees), replace=True)
        chosen = np.concatenate([trees, extra])
    masks = np.stack([(instance_ids == tid) for tid in chosen], axis=0)
    return masks


def random_subsample(
    xyz: np.ndarray,
    rgb: np.ndarray,
    instance_ids: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(xyz)
    if n <= num_samples:
        # Pad by repeating points so every sample has the patch count the encoder expects.
        idx = np.concatenate([np.arange(n), rng.choice(n, size=num_samples - n, replace=True)])
        return xyz[idx], rgb[idx], instance_ids[idx]
    # Prefer keeping some foreground from each sampled tree later; here bias FG
    trees = np.unique(instance_ids)
    trees = trees[trees >= 0]
    if len(trees) == 0:
        idx = rng.choice(n, size=num_samples, replace=False)
        return xyz[idx], rgb[idx], instance_ids[idx]

    fg = np.nonzero(instance_ids >= 0)[0]
    n_fg = min(len(fg), max(num_samples // 2, 1))
    fg_sel = rng.choice(fg, size=n_fg, replace=False)
    remaining = num_samples - n_fg
    pool = np.setdiff1d(np.arange(n), fg_sel, assume_unique=False)
    if len(pool) < remaining:
        bg_sel = rng.choice(n, size=remaining, replace=True)
    else:
        bg_sel = rng.choice(pool, size=remaining, replace=False)
    idx = rng.permutation(np.concatenate([fg_sel, bg_sel]))
    return xyz[idx], rgb[idx], instance_ids[idx]


def stem_click_indices(xyz: np.ndarray, member: np.ndarray, click_h: float = 1.3,
                       upper_h: float = 4.0) -> Tuple[int, int]:
    """(stem click, upper click) point indices for one tree mask, as SegmenterPointSAM's stem prompts.

    Stem click: the tree point nearest the stem axis about ``click_h`` m above the tree's lowest point;
    upper click: about ``upper_h`` m up near the same axis (falls back to the stem click).
    """
    idx = np.flatnonzero(member)
    p = xyz[idx]
    z0 = p[:, 2].min()
    band = idx[(p[:, 2] >= z0 + click_h - 0.5) & (p[:, 2] <= z0 + click_h + 0.7)]
    if len(band) == 0:
        band = idx[p[:, 2] <= np.quantile(p[:, 2], 0.1)]
    axis = np.median(xyz[band, :2], axis=0)

    def nearest(target_h: float, cand: np.ndarray) -> int:
        cost = np.hypot(*(xyz[cand, :2] - axis).T) + np.abs(xyz[cand, 2] - (z0 + target_h))
        return int(cand[np.argmin(cost)])

    stem = nearest(click_h, band)
    up = idx[(p[:, 2] >= z0 + upper_h - 1.0) & (p[:, 2] <= z0 + upper_h + 1.0)
             & (np.hypot(*(p[:, :2] - axis).T) < 0.8)]
    return stem, (nearest(upper_h, up) if len(up) else stem)


def augment_scanner(xyz: np.ndarray, rgb: np.ndarray, ids: np.ndarray, rng: np.random.Generator,
                    cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scanner-type robustness: density loss, occlusion holes, height-dependent sparsity, intensity loss."""
    keep = np.ones(len(xyz), dtype=bool)
    if rng.random() < cfg.get("p_drop_ground", 0.0):
        keep &= ids >= 0
    if rng.random() < cfg.get("p_density", 0.0):
        keep &= rng.random(len(xyz)) < rng.uniform(0.35, 1.0)
    if rng.random() < cfg.get("p_occlusion", 0.0):
        for _ in range(int(rng.integers(1, 5))):
            c = xyz[rng.integers(len(xyz))]
            half = rng.uniform(0.3, 1.5, 3)
            keep &= ~np.all(np.abs(xyz - c) < half, axis=1)
    if rng.random() < cfg.get("p_height", 0.0):
        h = (xyz[:, 2] - xyz[:, 2].min()) / max(1e-6, np.ptp(xyz[:, 2]))
        k = rng.uniform(0.3, 0.8)
        # Ground scanners lose the canopy top; airborne ones lose the stems.
        p_keep = 1 - k * h if rng.random() < 0.6 else 1 - k * (1 - h)
        keep &= rng.random(len(xyz)) < p_keep
    if keep.sum() >= 256:
        xyz, rgb, ids = xyz[keep], rgb[keep], ids[keep]
    if rng.random() < cfg.get("p_no_intensity", 0.0):
        rgb = np.full_like(rgb, 0.5)
    else:
        rgb = np.clip(rgb * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1), 0, 1).astype(np.float32)
    sigma = cfg.get("jitter", 0.0)
    if sigma > 0:
        xyz = xyz + rng.normal(0, rng.uniform(0, sigma), xyz.shape).astype(np.float32)
    tilt = np.deg2rad(cfg.get("tilt_deg", 0.0))
    if tilt > 0:
        ax, ay = rng.uniform(-tilt, tilt, 2)
        rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]], dtype=np.float32)
        ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]], dtype=np.float32)
        xyz = xyz @ (rx @ ry).T
    return xyz.astype(np.float32), rgb.astype(np.float32), ids


class ForestCropDataset(Dataset):
    """
    Loads ``*.npz`` crops with keys xyz, rgb, instance_ids.

    Returns dict compatible with PointCloudSAM.forward:
      coords [N,3], features [N,3], gt_masks [M,N], stem_idx [M,2] (stem / upper click point indices)
    """

    def __init__(
        self,
        crop_dir: Path | str,
        num_points: int = 10000,
        num_masks: int = 2,
        augment: bool = True,
        seed: int = 0,
        file_list: Optional[Sequence[str]] = None,
        aug: Optional[dict] = None,
        min_mask_points: int = 1,
    ):
        """``crop_dir`` may be one folder or a list of folders (mixed datasets)."""
        dirs = [Path(d) for d in crop_dir] if isinstance(crop_dir, (list, tuple)) else [Path(crop_dir)]
        self.crop_dir = dirs[0]
        if file_list is not None:
            self.files = [self.crop_dir / f for f in file_list]
        else:
            self.files = sorted(f for d in dirs for f in d.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz crops in {dirs}")
        self.num_points = num_points
        self.num_masks = num_masks
        self.augment = augment
        self.seed = seed
        self.aug = aug or {}
        self.min_mask_points = min_mask_points
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]
        data = np.load(path)
        xyz = np.asarray(data["xyz"], dtype=np.float32)
        rgb = np.asarray(data["rgb"], dtype=np.float32)
        ids = np.asarray(data["instance_ids"], dtype=np.int32)

        # Training draws fresh randomness each epoch; validation stays fixed.
        rng = np.random.default_rng(self.seed + index * 10007 + (self.epoch * 7919 if self.augment else 0))
        if self.augment and self.aug:
            xyz, rgb, ids = augment_scanner(xyz, rgb, ids, rng, self.aug)
        elif self.aug.get("val_drop_ground"):
            if (ids >= 0).sum() >= 256:
                xyz, rgb, ids = xyz[ids >= 0], rgb[ids >= 0], ids[ids >= 0]
        xyz, rgb, ids = random_subsample(xyz, rgb, ids, self.num_points, rng)

        if self.augment:
            # random yaw
            angle = float(rng.uniform(-np.pi, np.pi))
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            xyz = xyz @ R.T
            # light scale jitter
            scale_j = float(rng.uniform(0.85, 1.05))
            xyz = xyz * scale_j

        xyz_n, _, _ = normalize_unit_sphere(xyz)
        # Match upstream NormalizeColor(mean=0.5, std=0.5) on [0,1] features
        feats = (rgb - 0.5) / 0.5
        if self.min_mask_points > 1:
            tids, cnt = np.unique(ids[ids >= 0], return_counts=True)
            small = tids[cnt < self.min_mask_points]
            if len(small) and len(small) < len(tids):
                ids = np.where(np.isin(ids, small), -1, ids)
        masks = instance_ids_to_masks(ids, self.num_masks, rng)
        # Guarantee each mask has FG and BG for Point-SAM border sampling
        for mi in range(masks.shape[0]):
            if masks[mi].all() or not masks[mi].any():
                # flip a few points to restore a border
                n = masks.shape[1]
                flip = rng.choice(n, size=min(32, n), replace=False)
                if masks[mi].all():
                    masks[mi, flip] = False
                else:
                    masks[mi, flip] = True
        stem_idx = np.array([stem_click_indices(xyz, masks[mi]) for mi in range(masks.shape[0])], dtype=np.int64)

        return {
            "coords": torch.from_numpy(xyz_n),
            "features": torch.from_numpy(feats.astype(np.float32)),
            "gt_masks": torch.from_numpy(masks),
            "stem_idx": torch.from_numpy(stem_idx),
            "path": str(path),
        }


def collate_forest(batch: List[dict]) -> dict:
    return {
        "coords": torch.stack([b["coords"] for b in batch], dim=0),
        "features": torch.stack([b["features"] for b in batch], dim=0),
        "gt_masks": torch.stack([b["gt_masks"] for b in batch], dim=0),
        "stem_idx": torch.stack([b["stem_idx"] for b in batch], dim=0),
    }

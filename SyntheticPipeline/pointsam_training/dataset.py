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
        return xyz, rgb, instance_ids
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


class ForestCropDataset(Dataset):
    """
    Loads ``*.npz`` crops with keys xyz, rgb, instance_ids.

    Returns dict compatible with PointCloudSAM.forward:
      coords [N,3], features [N,3], gt_masks [M,N]
    """

    def __init__(
        self,
        crop_dir: Path | str,
        num_points: int = 10000,
        num_masks: int = 2,
        augment: bool = True,
        seed: int = 0,
        file_list: Optional[Sequence[str]] = None,
    ):
        self.crop_dir = Path(crop_dir)
        if file_list is not None:
            self.files = [self.crop_dir / f for f in file_list]
        else:
            self.files = sorted(self.crop_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz crops in {self.crop_dir}")
        self.num_points = num_points
        self.num_masks = num_masks
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]
        data = np.load(path)
        xyz = np.asarray(data["xyz"], dtype=np.float32)
        rgb = np.asarray(data["rgb"], dtype=np.float32)
        ids = np.asarray(data["instance_ids"], dtype=np.int32)

        rng = np.random.default_rng(self.seed + index * 10007)
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

        return {
            "coords": torch.from_numpy(xyz_n),
            "features": torch.from_numpy(feats.astype(np.float32)),
            "gt_masks": torch.from_numpy(masks),
            "path": str(path),
        }


def collate_forest(batch: List[dict]) -> dict:
    return {
        "coords": torch.stack([b["coords"] for b in batch], dim=0),
        "features": torch.stack([b["features"] for b in batch], dim=0),
        "gt_masks": torch.stack([b["gt_masks"] for b in batch], dim=0),
    }

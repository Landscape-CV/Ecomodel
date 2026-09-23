"""Wood / leaf reference classifiers for the instance annotator."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

METHODS = (
    "stem_grow",
    "eigen",
    "percentile",
    "intensity",
    "otsu",
    "rgi",
    "gbseparation",
)

# RegionGrowing is a pure-Python per-point kNN loop — never run on full plot tiles.
RGI_MAX_POINTS = 60_000
# GBSeparation builds a dense kNN index (N x knn x 8 bytes).
GB_MAX_POINTS = 120_000
# Eigen / stem-grow feature computation budget.
GEO_MAX_POINTS = 200_000


def _as_intensity(intensity: np.ndarray, n: int) -> np.ndarray:
    inten = np.asarray(intensity, dtype=np.float64).reshape(-1)
    if len(inten) != n:
        raise ValueError(f"intensity length {len(inten)} != points {n}")
    return inten


def _nn_paint_masks(
    xyz_full: np.ndarray,
    xyz_sub: np.ndarray,
    wood_sub: np.ndarray,
    leaf_sub: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz_sub)
    _, nn = tree.query(xyz_full, k=1, workers=-1)
    nn = np.asarray(nn, dtype=np.int64).reshape(-1)
    return wood_sub[nn], leaf_sub[nn]


def _voxel_subsample_indices(
    xyz: np.ndarray,
    max_points: int,
    *,
    intensity: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """One point per voxel (prefer highest intensity). Better than pure random for stems."""
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    max_pts = max(50, int(max_points))
    if n <= max_pts:
        return None

    mins = xyz.min(axis=0)
    extent = np.maximum(xyz.max(axis=0) - mins, 1e-6)
    vol = float(np.prod(extent))
    vox = max((vol / float(max_pts)) ** (1.0 / 3.0), 1e-4)

    keys = np.floor((xyz - mins) / vox).astype(np.int64)
    span = keys.max(axis=0) - keys.min(axis=0) + 1
    packed = (
        (keys[:, 0] - keys[:, 0].min())
        + (keys[:, 1] - keys[:, 1].min()) * int(span[0])
        + (keys[:, 2] - keys[:, 2].min()) * int(span[0] * span[1])
    )

    if intensity is not None:
        inten = np.asarray(intensity, dtype=np.float64).reshape(-1)
        # Highest intensity first within each voxel
        order = np.lexsort((-inten, packed))
        packed_s = packed[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = packed_s[1:] != packed_s[:-1]
        chosen = order[first]
    else:
        _, first_idx = np.unique(packed, return_index=True)
        chosen = first_idx

    if len(chosen) > max_pts:
        rng = np.random.default_rng(seed)
        if intensity is not None:
            scores = inten[chosen].astype(np.float64)
            scores = scores - scores.min() + 1e-6
            probs = scores / scores.sum()
            pick = rng.choice(len(chosen), size=max_pts, replace=False, p=probs)
            chosen = chosen[pick]
        else:
            chosen = rng.choice(chosen, size=max_pts, replace=False)

    return np.sort(chosen.astype(np.int64))


def classify_intensity_threshold(
    xyz: np.ndarray,
    intensity: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Wood = intensity >= threshold; leaf = below."""
    xyz = np.asarray(xyz, dtype=np.float64)
    inten = _as_intensity(intensity, len(xyz))
    wood = inten >= float(threshold)
    leaf = ~wood
    return wood, leaf


def classify_intensity_percentile(
    xyz: np.ndarray,
    intensity: np.ndarray,
    percentile: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Wood = intensity >= P-th percentile; leaf = below."""
    xyz = np.asarray(xyz, dtype=np.float64)
    inten = _as_intensity(intensity, len(xyz))
    p = float(np.clip(percentile, 0.0, 100.0))
    thr = float(np.percentile(inten, p))
    return classify_intensity_threshold(xyz, inten, thr)


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    lo, hi = float(np.min(v)), float(np.max(v))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float(np.median(v))
    hist, edges = np.histogram(v, bins=int(bins), range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5 * (lo + hi)
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (edges[:-1] + edges[1:]) * 0.5)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    sigma_b[~np.isfinite(sigma_b)] = 0.0
    k = int(np.argmax(sigma_b))
    return float(0.5 * (edges[k] + edges[k + 1]))


def classify_otsu(
    xyz: np.ndarray,
    intensity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Otsu threshold on intensity (wood = above)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    inten = _as_intensity(intensity, len(xyz))
    thr = _otsu_threshold(inten)
    return classify_intensity_threshold(xyz, inten, thr)


def _eigenfeatures(
    xyz: np.ndarray,
    *,
    k: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """linearity, verticality (principal-axis |z|), curvature."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    k = int(max(5, min(k, n - 1)))
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k + 1, workers=-1)
    idx = np.asarray(idx)[:, 1:]

    nbrs = xyz[idx]
    mean = nbrs.mean(axis=1, keepdims=True)
    c = nbrs - mean
    cov = np.matmul(np.transpose(c, (0, 2, 1)), c) / float(k)
    evals, evecs = np.linalg.eigh(cov)
    l0 = np.maximum(evals[:, 0], 0.0)
    l1 = np.maximum(evals[:, 1], 0.0)
    l2 = np.maximum(evals[:, 2], 0.0)
    s = l0 + l1 + l2 + 1e-12
    curvature = l0 / s
    linearity = (l2 - l1) / (l2 + 1e-12)
    axis = evecs[:, :, 2]
    verticality = np.abs(axis @ np.array([0.0, 0.0, 1.0]))
    return linearity, verticality, curvature


def classify_eigen(
    xyz: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    *,
    linearity_min: float = 0.3,
    verticality_min: float = 0.7,
    curvature_max: float = 0.12,
    k: int = 20,
    max_points: int = GEO_MAX_POINTS,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Wood ~= linear + vertical + low-curvature; leaf = rest.

    Defaults from LeWoS holdout sweep (macro-F1 ~0.80).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    if n < 30:
        raise ValueError(f"eigen needs >=30 points (got {n})")
    inten = _as_intensity(intensity, n) if intensity is not None else None

    sub_idx = _voxel_subsample_indices(xyz, max_points, intensity=inten, seed=seed)
    if sub_idx is None:
        sub_xyz = xyz
    else:
        sub_xyz = xyz[sub_idx]
        print(f"[wood_leaf] eigen: voxel-subsample {n:,} -> {len(sub_idx):,}")

    lin, vert, curv = _eigenfeatures(sub_xyz, k=k)
    wood_sub = (
        (lin >= float(linearity_min))
        & (vert >= float(verticality_min))
        & (curv <= float(curvature_max))
    )
    leaf_sub = ~wood_sub
    if sub_idx is None:
        return wood_sub, leaf_sub
    return _nn_paint_masks(xyz, sub_xyz, wood_sub, leaf_sub)


def classify_stem_grow(
    xyz: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    *,
    verticality_min: float = 0.8,
    height_percentile: float = 15.0,
    grow_radius: float = 0.2,
    max_points: int = GEO_MAX_POINTS,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Seed low-Z high-verticality points; grow by radius; leftover = leaf.

    Defaults from LeWoS holdout sweep (best macro-F1 ~0.81).
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    if n < 30:
        raise ValueError(f"stem_grow needs >=30 points (got {n})")
    inten = _as_intensity(intensity, n) if intensity is not None else None

    sub_idx = _voxel_subsample_indices(xyz, max_points, intensity=inten, seed=seed)
    if sub_idx is None:
        sub_xyz = xyz
    else:
        sub_xyz = xyz[sub_idx]
        print(f"[wood_leaf] stem_grow: voxel-subsample {n:,} -> {len(sub_idx):,}")

    _, vert, _ = _eigenfeatures(sub_xyz, k=20)
    z = sub_xyz[:, 2]
    z_cut = float(np.percentile(z, float(np.clip(height_percentile, 1.0, 99.0))))
    seeds = (vert >= float(verticality_min)) & (z <= z_cut)
    if not np.any(seeds):
        low = z <= z_cut
        if np.any(low):
            thr = float(np.percentile(vert[low], 75))
            seeds = low & (vert >= thr)
        if not np.any(seeds):
            seeds = vert >= float(np.percentile(vert, 90))

    wood_sub = np.zeros(len(sub_xyz), dtype=bool)
    tree = cKDTree(sub_xyz)
    can_grow = vert >= (float(verticality_min) * 0.85)
    queue = list(np.flatnonzero(seeds))
    wood_sub[seeds] = True
    r = float(max(grow_radius, 1e-3))
    head = 0
    while head < len(queue):
        i = queue[head]
        head += 1
        nbrs = tree.query_ball_point(sub_xyz[i], r=r)
        for j in nbrs:
            if wood_sub[j] or not can_grow[j]:
                continue
            wood_sub[j] = True
            queue.append(j)

    leaf_sub = ~wood_sub
    if sub_idx is None:
        return wood_sub, leaf_sub
    return _nn_paint_masks(xyz, sub_xyz, wood_sub, leaf_sub)


def classify_rgi(
    xyz: np.ndarray,
    intensity: np.ndarray,
    *,
    max_points: int = RGI_MAX_POINTS,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """RGI on voxel+intensity subsample, then NN-paint back."""
    xyz = np.asarray(xyz, dtype=np.float64)
    inten = _as_intensity(intensity, len(xyz))
    n = len(xyz)
    if n < 100:
        raise ValueError(f"RGI needs >=100 points (got {n})")

    from SegmentRGI.SegmentRGI import classify_wood_leaf_point_cloud

    sub_idx = _voxel_subsample_indices(xyz, max_points, intensity=inten, seed=seed)
    if sub_idx is None:
        sub_xyz, sub_inten = xyz, inten
    else:
        sub_xyz = xyz[sub_idx]
        sub_inten = inten[sub_idx]
        print(
            f"[wood_leaf] RGI: voxel/intensity subsample {n:,} -> {len(sub_idx):,}"
        )

    cloud = np.column_stack([sub_xyz, sub_inten])
    wood_sub, leaf_sub = classify_wood_leaf_point_cloud(
        cloud,
        noise_percentile=5,
        angle_deg=15.0,
        curv_thresh=0.07,
        resid_thresh=0.05,
        k=30,
        minClusterSize=10,
        maxClusterSize=100000,
        smoothMode=True,
        useResidualTest=True,
        useCurvatureTest=True,
    )
    if wood_sub is None or leaf_sub is None:
        raise RuntimeError(
            "RGI found no clusters. Try Eigen / Stem-grow / Intensity percentile."
        )
    wood_sub = np.asarray(wood_sub, dtype=bool).reshape(-1)
    leaf_sub = np.asarray(leaf_sub, dtype=bool).reshape(-1)
    if len(wood_sub) != len(sub_xyz):
        raise RuntimeError(
            f"RGI mask length mismatch: wood={len(wood_sub)} n={len(sub_xyz)}"
        )
    if np.any(wood_sub & leaf_sub):
        leaf_sub = leaf_sub & ~wood_sub

    if sub_idx is None:
        return wood_sub, leaf_sub
    wood_mask, leaf_mask = _nn_paint_masks(xyz, sub_xyz, wood_sub, leaf_sub)
    print(f"[wood_leaf] RGI: NN-painted back to {n:,} pts")
    return wood_mask, leaf_mask


def _gb_knn_for_n(n: int) -> int:
    if n <= 20_000:
        return 120
    if n <= 60_000:
        return 60
    return 40


def classify_gbseparation(
    xyz: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    *,
    max_points: int = GB_MAX_POINTS,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """GBSeparation; voxel-subsample + NN-paint on large clouds."""
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    if n < 50:
        raise ValueError(f"GBSeparation needs >=50 points (got {n})")
    inten = _as_intensity(intensity, n) if intensity is not None else None

    from GBSeparation.remove_leaves import LeafRemover

    sub_idx = _voxel_subsample_indices(xyz, max_points, intensity=inten, seed=seed)
    if sub_idx is None:
        sub_xyz = xyz
        note_n = None
    else:
        sub_xyz = xyz[sub_idx]
        note_n = n

    remover = LeafRemover()
    remover.knn = _gb_knn_for_n(len(sub_xyz))
    try:
        wood_sub, leaf_sub = remover.process(sub_xyz[:, :3].copy(), return_mask=True)
    except MemoryError as exc:
        raise MemoryError(
            f"GBSeparation OOM on {len(sub_xyz):,} pts (knn={remover.knn}). "
            "Try Eigen, Stem-grow, or Intensity methods."
        ) from exc

    wood_sub = np.asarray(wood_sub, dtype=bool).reshape(-1)
    leaf_sub = np.asarray(leaf_sub, dtype=bool).reshape(-1)
    if len(wood_sub) != len(sub_xyz):
        raise RuntimeError(
            f"GBSeparation mask length mismatch: wood={len(wood_sub)} n={len(sub_xyz)}"
        )
    leaf_sub = leaf_sub & ~wood_sub

    if sub_idx is None:
        return wood_sub, leaf_sub
    wood_mask, leaf_mask = _nn_paint_masks(xyz, sub_xyz, wood_sub, leaf_sub)
    print(
        f"[wood_leaf] GBSeparation: {note_n:,} -> {len(sub_xyz):,} pts "
        f"(knn={remover.knn}), NN-painted"
    )
    return wood_mask, leaf_mask


def classify_wood_leaf(
    method: str,
    xyz: np.ndarray,
    intensity: np.ndarray,
    *,
    threshold: Optional[float] = None,
    percentile: float = 40.0,
    linearity_min: float = 0.3,
    verticality_min: float = 0.8,
    curvature_max: float = 0.12,
    height_percentile: float = 15.0,
    grow_radius: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unified wood/leaf classification.

    Default method params tuned on LeWoS LabelledPC holdout (see
    output/wood_leaf_benchmark/ and wood_leaf_defaults.json).
    """
    key = str(method).strip().lower()
    if key not in METHODS:
        raise ValueError(f"Unknown wood/leaf method {method!r}; choose from {METHODS}")

    xyz = np.asarray(xyz, dtype=np.float64)
    inten = _as_intensity(intensity, len(xyz))

    if key == "percentile":
        return classify_intensity_percentile(xyz, inten, percentile=percentile)
    if key == "intensity":
        thr = float(np.median(inten)) if threshold is None else float(threshold)
        return classify_intensity_threshold(xyz, inten, thr)
    if key == "otsu":
        return classify_otsu(xyz, inten)
    if key == "eigen":
        return classify_eigen(
            xyz,
            inten,
            linearity_min=linearity_min,
            verticality_min=verticality_min,
            curvature_max=curvature_max,
        )
    if key == "stem_grow":
        return classify_stem_grow(
            xyz,
            inten,
            verticality_min=verticality_min,
            height_percentile=height_percentile,
            grow_radius=grow_radius,
        )
    if key == "rgi":
        return classify_rgi(xyz, inten)
    return classify_gbseparation(xyz, inten)


def mask_counts(
    wood_mask: np.ndarray,
    leaf_mask: np.ndarray,
) -> dict:
    wood = np.asarray(wood_mask, dtype=bool)
    leaf = np.asarray(leaf_mask, dtype=bool)
    n = len(wood)
    return {
        "n": n,
        "wood": int(wood.sum()),
        "leaf": int(leaf.sum()),
        "unknown": int((~(wood | leaf)).sum()),
        "overlap": int((wood & leaf).sum()),
    }

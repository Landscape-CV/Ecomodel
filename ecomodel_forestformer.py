"""ForestFormer3D and a disabled SegmentAnyTreeV2 slot.

ForestFormer uses the authors' FF3D_inference wrapper
(thirdparty/ForestFormer3D/ff3d_forestsens) and the released
``epoch_3000_fix.pth`` checkpoint. The official stack is PyTorch 1.13 /
CUDA 11.6 / spconv-cu116. ``segment`` raises if that stack is not
importable, rather than returning an empty labelling.

SegmentAnyTreeV2 has no published code or weights. It stays disabled
until ``SEGMENTANYTREE_V2_CKPT`` points at a local file.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent
FF3D_ROOT = _ROOT / "thirdparty" / "ForestFormer3D" / "ff3d_forestsens"
_CKPT_DIR = _ROOT / "thirdparty" / "checkpoints" / "forestformer3d"
_REQUIRED = ("mmengine", "mmdet", "mmdet3d", "spconv", "segmentator")


def forestformer_checkpoint() -> Optional[Path]:
    env = os.environ.get("FORESTFORMER3D_CKPT", "").strip()
    if env and os.path.isfile(env):
        return Path(env)
    if not _CKPT_DIR.is_dir():
        return None
    hits = sorted(_CKPT_DIR.rglob("epoch_3000_fix.pth"))
    return hits[0] if hits else None


def forestformer_block_reason() -> Optional[str]:
    """Why ForestFormer cannot run here, or None when the stack imports."""
    missing = []
    for name in _REQUIRED:
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    if not missing:
        if forestformer_checkpoint() is None:
            return (
                "ForestFormer3D checkpoint epoch_3000_fix.pth is not in "
                f"{_CKPT_DIR}. Download clean_forestformer.zip from Zenodo record 16742708."
            )
        return None
    try:
        import torch
        torch_desc = f"torch {torch.__version__}"
        if torch.cuda.is_available():
            torch_desc += f" on {torch.cuda.get_device_name(0)}"
    except Exception:
        torch_desc = "this Python"
    return (
        "ForestFormer3D cannot run here: missing "
        + ", ".join(missing)
        + ". The released wrapper needs PyTorch 1.13, CUDA 11.6, and spconv-cu116 "
        f"(see thirdparty/ForestFormer3D/INSTALL_ECOMODEL.md). {torch_desc} is not that stack."
    )


def segmentanytree_v2_ckpt() -> Optional[str]:
    path = os.environ.get("SEGMENTANYTREE_V2_CKPT", "").strip()
    if path and os.path.isfile(path):
        return path
    return None


def segmentanytree_v2_reason() -> Optional[str]:
    if segmentanytree_v2_ckpt():
        return None
    return (
        "No published weights. Set SEGMENTANYTREE_V2_CKPT to a local checkpoint "
        "when one exists. The 2024 SegmentAnyTree model is a different network."
    )


def _write_xyz_ply(path: Path, xyz: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    xyz = np.asarray(xyz, dtype=np.float32)
    vertex = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def _run_ff3d(xyz: np.ndarray, ckpt: Path) -> Tuple[np.ndarray, np.ndarray]:
    """One inference pass. Returns (points, instance ids), -1 for non-tree."""
    root = str(FF3D_ROOT)
    scan = "ecomodel_tile"
    test_dir = FF3D_ROOT / "data" / "ForAINetV2" / "test_data"
    meta = FF3D_ROOT / "data" / "ForAINetV2" / "meta_data"
    out_dir = FF3D_ROOT / "work_dirs" / "output"
    for d in (test_dir, meta, out_dir):
        d.mkdir(parents=True, exist_ok=True)
    for old in test_dir.glob("*.ply"):
        old.unlink()
    for old in out_dir.glob(f"{scan}*"):
        if old.is_file():
            old.unlink()
    _write_xyz_ply(test_dir / f"{scan}.ply", xyz[:, :3])
    (meta / "test_list.txt").write_text(scan + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["FF3D_ROOT"] = root
    py = sys.executable
    data_dir = FF3D_ROOT / "data" / "ForAINetV2"
    steps = [
        [py, str(data_dir / "create_testlist.py")],
        [py, str(data_dir / "batch_load_ForAINetV2_data.py"),
         "--test_scan_names_file", "meta_data/test_list.txt"],
        [py, str(FF3D_ROOT / "tools" / "create_data_forainetv2.py"), "forainetv2",
         "--root-path", str(data_dir), "--out-dir", str(data_dir)],
        [py, str(FF3D_ROOT / "tools" / "test.py"),
         str(FF3D_ROOT / "configs" / "oneformer3d_qs_radius16_qp300_2many.py"),
         str(ckpt)],
    ]
    for i, cmd in enumerate(steps):
        cwd = str(data_dir) if i == 1 else root
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
            raise RuntimeError("ForestFormer3D inference failed:\n" + "\n".join(tail))

    pred_path = out_dir / f"{scan}_1.ply"
    if not pred_path.is_file():
        hits = sorted(out_dir.glob(f"{scan}*.ply"))
        if not hits:
            raise RuntimeError(f"ForestFormer3D wrote no ply under {out_dir}")
        pred_path = hits[-1]
    from plyfile import PlyData
    from scipy.spatial import cKDTree

    vert = PlyData.read(str(pred_path))["vertex"]
    pred_xyz = np.column_stack([vert["x"], vert["y"], vert["z"]]).astype(np.float64)
    inst = np.asarray(vert["instance_pred"], dtype=np.int32)
    offset_path = data_dir / "forainetv2_instance_data" / f"{scan}_offsets.npy"
    if offset_path.is_file():
        pred_xyz = pred_xyz + np.load(offset_path).astype(np.float64).reshape(1, 3)
    labels = np.full(len(xyz), -1, dtype=np.int32)
    if len(pred_xyz):
        _, nn = cKDTree(xyz[:, :3]).query(pred_xyz, k=1)
        labels[nn] = inst
        labels[labels < 0] = -1
    return xyz, labels


class SegmenterForestFormer:
    """Ground-removed XYZI in, per-point instance ids out (trees >= 0, non-tree -1)."""

    def __init__(self, checkpoint_path: str = ""):
        self.checkpoint_path = checkpoint_path

    def segment(self, point_cloud, output_dir=None):
        reason = forestformer_block_reason()
        if reason:
            raise RuntimeError(reason)
        ckpt = Path(self.checkpoint_path) if self.checkpoint_path else forestformer_checkpoint()
        if ckpt is None or not ckpt.is_file():
            raise RuntimeError(forestformer_block_reason() or "ForestFormer3D checkpoint missing")
        pts = np.asarray(point_cloud, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) < 50:
            return None, None
        return _run_ff3d(pts[:, :4] if pts.shape[1] >= 4 else pts, ckpt)


class SegmenterSegmentAnyTreeV2:
    """Placeholder. Enabled only when a local checkpoint path is configured."""

    def __init__(self, checkpoint_path: str = ""):
        self.checkpoint_path = checkpoint_path or (segmentanytree_v2_ckpt() or "")

    def segment(self, point_cloud, output_dir=None):
        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            raise RuntimeError(segmentanytree_v2_reason())
        raise RuntimeError(
            "SegmentAnyTreeV2 weights were given, but no public inference code exists "
            f"to load {self.checkpoint_path}."
        )

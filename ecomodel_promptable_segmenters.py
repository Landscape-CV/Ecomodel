"""Promptable instance segmenters: Point-SAM and SNAP."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


def _repo_thirdparty(*parts: str) -> Path:
    return Path(__file__).resolve().parent.joinpath("thirdparty", *parts)


class SegmenterPointSAM:
    """
    Point-SAM wrapper as a drop-in instance segmenter.

    Automatic mode: stem/grid seed clicks (see Utils.promptable_instance).
    Oracle / interactive: pass ``prompts`` as a list of (3,) XYZ clicks in the
    same coordinate frame as ``point_cloud``.

    Label convention matches Scanline/TreeLearn: -1 = unassigned, 0+ = trees.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_name: str = "large",
        config_dir: str = None,
        use_gpu: bool = True,
        max_points: int = 150000,
        voxel_size: float = 0.05,
        group_number: int = 2048,
        group_size: int = 256,
    ):
        self.checkpoint_path = checkpoint_path
        self.config_name = config_name
        self.config_dir = config_dir
        self.use_gpu = use_gpu
        self.max_points = max_points
        self.voxel_size = voxel_size
        self.group_number = group_number
        self.group_size = group_size
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import sys
        import hydra
        from omegaconf import OmegaConf
        from safetensors.torch import load_model

        root = _repo_thirdparty("Point-SAM")
        if not root.is_dir() or not (root / "pc_sam").is_dir():
            raise FileNotFoundError(
                f"Point-SAM not installed at {root}. See thirdparty/Point-SAM/README.md"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from pc_sam.utils.torch_utils import replace_with_fused_layernorm

        cfg_dir = self.config_dir or str(root / "configs")
        with hydra.initialize_config_dir(config_dir=os.path.abspath(cfg_dir), version_base=None):
            cfg = hydra.compose(config_name=self.config_name)
            OmegaConf.resolve(cfg)

        model = hydra.utils.instantiate(cfg.model)
        try:
            model.apply(replace_with_fused_layernorm)
        except Exception as exc:
            # apex CUDA build is optional; standard LayerNorm is fine for inference.
            print(f"[SegmenterPointSAM] fused LayerNorm unavailable ({exc}); using nn.LayerNorm")
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Point-SAM checkpoint not found: {self.checkpoint_path}")
        load_model(model, self.checkpoint_path)
        model.eval()
        if self.use_gpu:
            model.cuda()
        else:
            model.cpu()
        self._model = model

    def segment(self, point_cloud: np.ndarray, output_dir: str = None, prompts=None):
        from Utils.promptable_instance import (
            auto_stem_grid_prompts,
            intensity_to_rgb,
            masks_to_instance_labels,
            propagate_labels_nn,
            voxel_downsample_with_index,
        )

        try:
            self._ensure_model()
        except Exception as exc:
            print(f"[SegmenterPointSAM] model load failed: {exc}")
            return None, None

        xyz_full = point_cloud[:, :3].astype(np.float64)
        rgb_full = intensity_to_rgb(point_cloud)

        if len(xyz_full) > self.max_points:
            xyz_ds, keep = voxel_downsample_with_index(xyz_full, self.voxel_size)
            if len(xyz_ds) > self.max_points:
                rng = np.random.default_rng(0)
                sel = rng.choice(len(xyz_ds), size=self.max_points, replace=False)
                keep = keep[sel]
                xyz_ds = xyz_full[keep]
            rgb_ds = rgb_full[keep]
        else:
            xyz_ds, rgb_ds = xyz_full, rgb_full

        centroid = xyz_ds.mean(axis=0)
        centered = xyz_ds - centroid
        scale = np.linalg.norm(centered, axis=1).max()
        if scale < 1e-8:
            scale = 1.0
        xyz_n = (centered / scale).astype(np.float32)
        rgb_n = rgb_ds.astype(np.float32)

        if prompts is None:
            world_prompts = auto_stem_grid_prompts(xyz_ds)
        else:
            world_prompts = [np.asarray(p, dtype=np.float64).reshape(3) for p in prompts]

        norm_prompts = [((p - centroid) / scale).astype(np.float32) for p in world_prompts]

        import torch

        pc_xyz = torch.from_numpy(xyz_n).float().unsqueeze(0)
        pc_rgb = torch.from_numpy(rgb_n).float().unsqueeze(0)
        if self.use_gpu:
            pc_xyz = pc_xyz.cuda()
            pc_rgb = pc_rgb.cuda()

        try:
            grouper = self._model.pc_encoder.patch_embed.grouper
            npts = xyz_n.shape[0]
            grouper.num_groups = min(self.group_number, max(1, npts))
            grouper.group_size = min(self.group_size, max(2, npts))
        except Exception:
            pass

        masks = []
        scores = []
        try:
            with torch.no_grad():
                # PointCloudSAM.predict_masks(coords, features, prompt_coords, prompt_labels, ...)
                for click in norm_prompts:
                    pp = torch.from_numpy(click.reshape(1, 1, 3)).float()
                    pl = torch.ones((1, 1), dtype=torch.long)
                    if self.use_gpu:
                        pp = pp.cuda()
                        pl = pl.cuda()
                    mask, sc = self._model.predict_masks(
                        pc_xyz, pc_rgb, pp, pl, prompt_masks=None, multimask_output=True
                    )
                    # mask: [B*M, num_outputs, N], sc: [B*M, num_outputs]
                    best = int(torch.argmax(sc[0]).item())
                    m = (mask[0][best] > 0).detach().cpu().numpy().astype(bool)
                    score = float(sc[0][best].detach().cpu().numpy())
                    masks.append(m)
                    scores.append(score)
        except Exception as exc:
            import traceback
            print(f"[SegmenterPointSAM] inference failed: {exc}")
            traceback.print_exc()
            return None, None

        labels_ds = masks_to_instance_labels(masks, scores)
        if len(xyz_ds) != len(xyz_full):
            return point_cloud, propagate_labels_nn(xyz_ds, labels_ds, xyz_full)
        return point_cloud, labels_ds


class SegmenterSNAP:
    """
    SNAP wrapper as a drop-in instance segmenter (Outdoor domain default).

    Automatic mode: SNAP ``segment_everything`` (HDBSCAN seeds).
    Oracle / manual: pass ``prompts`` as list of (3,) XYZ clicks.
    """

    def __init__(
        self,
        checkpoint_path: str,
        domain: str = "Outdoor",
        grid_size: float = 0.05,
        use_gpu: bool = True,
        max_points: int = 400000,
        voxel_size: float = 0.05,
    ):
        self.checkpoint_path = checkpoint_path
        self.domain = domain
        self.grid_size = grid_size
        self.use_gpu = use_gpu
        self.max_points = max_points
        self.voxel_size = voxel_size
        self._seg_model = None

    def _ensure_model(self):
        if self._seg_model is not None:
            return
        import sys

        root = _repo_thirdparty("SNAP")
        if not (root / "src").is_dir():
            raise FileNotFoundError(
                f"SNAP not installed at {root}. See thirdparty/SNAP/README.md"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from python_demo import SegmentationModel

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"SNAP checkpoint not found: {self.checkpoint_path}")
        self._seg_model = SegmentationModel(
            checkpoint_path=self.checkpoint_path,
            domain=self.domain,
            grid_size=self.grid_size,
        )

    def segment(self, point_cloud: np.ndarray, output_dir: str = None, prompts=None):
        from Utils.promptable_instance import (
            auto_stem_grid_prompts,
            intensity_to_rgb,
            masks_to_instance_labels,
            propagate_labels_nn,
            voxel_downsample_with_index,
        )
        from scipy.spatial import cKDTree

        try:
            self._ensure_model()
        except Exception as exc:
            print(f"[SegmenterSNAP] model load failed: {exc}")
            return None, None

        xyz_full = point_cloud[:, :3].astype(np.float64)
        rgb_full = intensity_to_rgb(point_cloud)

        if len(xyz_full) > self.max_points:
            xyz_ds, keep = voxel_downsample_with_index(xyz_full, self.voxel_size)
            if len(xyz_ds) > self.max_points:
                rng = np.random.default_rng(0)
                sel = rng.choice(len(xyz_ds), size=self.max_points, replace=False)
                keep = keep[sel]
                xyz_ds = xyz_full[keep]
            rgb_ds = rgb_full[keep]
            inten_src = point_cloud[keep, 3] if point_cloud.shape[1] >= 4 else None
        else:
            xyz_ds, rgb_ds = xyz_full, rgb_full
            inten_src = point_cloud[:, 3] if point_cloud.shape[1] >= 4 else None

        data_in = {"coord": xyz_ds.astype(np.float32), "color": rgb_ds.astype(np.float32)}
        if inten_src is not None:
            inten = inten_src.astype(np.float64)
            if inten.max() > 1.5:
                inten = inten / 65535.0
            data_in["intensity"] = inten.astype(np.float32).reshape(-1, 1)

        try:
            data = self._seg_model.intialize_pointcloud(data_in)
            self._seg_model.extract_backbone_features(data)

            if prompts is None:
                try:
                    masks, _text, iou_out, _seeds = self._seg_model.segment_everything(data)
                    scores = [
                        float(np.asarray(s).reshape(-1)[0]) if s is not None else 1.0
                        for s in iou_out
                    ]
                except Exception as exc:
                    print(
                        f"[SegmenterSNAP] segment_everything failed ({exc}); "
                        "using stem-grid prompts"
                    )
                    world_prompts = auto_stem_grid_prompts(xyz_ds)
                    prompt_points = [
                        [[float(p[0]), float(p[1]), float(p[2])]] for p in world_prompts
                    ]
                    masks, _text, iou_out = self._seg_model.segment(
                        data, prompt_points, text_prompt=None
                    )
                    scores = [
                        float(np.asarray(s).reshape(-1)[0]) if s is not None else 1.0
                        for s in iou_out
                    ]
            else:
                prompt_points = [
                    [[float(p[0]), float(p[1]), float(p[2])]] for p in prompts
                ]
                masks, _text, iou_out = self._seg_model.segment(
                    data, prompt_points, text_prompt=None
                )
                scores = [
                    float(np.asarray(s).reshape(-1)[0]) if s is not None else 1.0
                    for s in iou_out
                ]

            coord_out = data["coord"]
            if hasattr(coord_out, "cpu"):
                coord_out = coord_out.cpu().numpy()
            coord_out = np.asarray(coord_out)

            aligned_masks = []
            for m in masks:
                m = np.asarray(m).astype(bool).reshape(-1)
                if len(m) == len(coord_out):
                    tree = cKDTree(xyz_ds)
                    _, nn = tree.query(coord_out, k=1)
                    m_ds = np.zeros(len(xyz_ds), dtype=bool)
                    m_ds[nn[m]] = True
                    aligned_masks.append(m_ds)
                elif len(m) == len(xyz_ds):
                    aligned_masks.append(m)
                else:
                    print(f"[SegmenterSNAP] skipping mask length {len(m)}")

            labels_ds = masks_to_instance_labels(aligned_masks, scores)
            if len(xyz_ds) != len(xyz_full):
                return point_cloud, propagate_labels_nn(xyz_ds, labels_ds, xyz_full)
            return point_cloud, labels_ds

        except Exception as exc:
            import traceback
            print(f"[SegmenterSNAP] inference failed: {exc}")
            traceback.print_exc()
            return None, None

"""
Fine-tune Point-SAM on voxelized synthetic forest crops.

Modes:
  decoder_ft — freeze pc_encoder; train prompt_encoder / mask_encoder / mask_decoder
  full_ft    — train all; encoder uses lower LR

Usage (from SyntheticPipeline/pointsam_training/):
  python train.py --config configs/forest_decoder_ft.yaml
  python train.py --config configs/forest_full_ft.yaml --max_steps 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_SP = _HERE.parent
_ROOT = _SP.parent


def _resolve(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model(cfg: dict, device: torch.device):
    import hydra
    from omegaconf import OmegaConf
    from safetensors.torch import load_model

    point_sam_root = _ROOT / "thirdparty" / "Point-SAM"
    if str(point_sam_root) not in sys.path:
        sys.path.insert(0, str(point_sam_root))

    from pc_sam.utils.torch_utils import replace_with_fused_layernorm

    cfg_dir = str(_resolve(cfg["config_dir"], _HERE))
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        hcfg = hydra.compose(config_name=cfg["config_name"])
        OmegaConf.resolve(hcfg)

    model = hydra.utils.instantiate(hcfg.model)
    try:
        model.apply(replace_with_fused_layernorm)
    except Exception as exc:
        print(f"[train] fused LayerNorm unavailable ({exc}); using nn.LayerNorm")

    ckpt = _resolve(cfg["pretrained_ckpt"], _HERE)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt}")
    load_model(model, str(ckpt))
    print(f"[train] loaded pretrained weights from {ckpt}")

    # Tune patch grouper for training point counts
    try:
        grouper = model.pc_encoder.patch_embed.grouper
        grouper.num_groups = int(cfg.get("group_number", 512))
        grouper.group_size = int(cfg.get("group_size", 64))
    except Exception:
        pass

    model.to(device)
    return model


def _detach_tree(obj):
    """Detach tensors in nested structures so frozen encoder graph can free."""
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, dict):
        return {k: _detach_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [_detach_tree(v) for v in obj]
        return type(obj)(out) if not isinstance(obj, list) else out
    return obj


def wrap_encoder_no_grad(model) -> None:
    """
    For decoder_ft: run pc_encoder under no_grad and detach outputs.
    Freezing requires_grad alone still spikes VRAM; this drops encoder autograd state.
    """
    enc = model.pc_encoder
    orig_forward = enc.forward

    def forward_no_grad(*args, **kwargs):
        with torch.no_grad():
            out = orig_forward(*args, **kwargs)
        return _detach_tree(out)

    enc.forward = forward_no_grad  # type: ignore[method-assign]
    print("[train] pc_encoder wrapped with torch.no_grad()+detach (VRAM guard)")


def set_trainable(model, mode: str):
    if mode == "decoder_ft":
        for p in model.pc_encoder.parameters():
            p.requires_grad = False
        for name in ("point_encoder", "mask_encoder", "mask_decoder"):
            for p in getattr(model, name).parameters():
                p.requires_grad = True
        wrap_encoder_no_grad(model)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"[train] decoder_ft: trainable {n_train:,} / {n_all:,} params")
    elif mode == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
        print("[train] full_ft: all params trainable")
    else:
        raise ValueError(f"Unknown mode: {mode}")


def gpu_mem_mb() -> dict:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "alloc_mb": torch.cuda.memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
        "peak_alloc_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
        "free_mb": free / (1024**2),
        "total_mb": total / (1024**2),
        "used_frac": 1.0 - (free / total),
    }


def enforce_vram_guard(cfg: dict, step: int, *, hard: bool = False) -> dict:
    """
    Soft: empty_cache when used_frac > soft threshold.
    Hard: raise RuntimeError when used_frac > hard threshold (abort training).
    """
    mem = gpu_mem_mb()
    if not mem:
        return mem
    soft = float(cfg.get("vram_soft_frac", 0.88))
    hard_frac = float(cfg.get("vram_hard_frac", 0.96))
    if mem["used_frac"] >= soft:
        torch.cuda.empty_cache()
        mem = gpu_mem_mb()
        print(
            f"[vram] step={step} soft-trim used={mem['used_frac']:.1%} "
            f"alloc={mem['alloc_mb']:.0f}MB reserved={mem['reserved_mb']:.0f}MB "
            f"free={mem['free_mb']:.0f}MB",
            flush=True,
        )
    if hard and mem["used_frac"] >= hard_frac:
        raise RuntimeError(
            f"VRAM guard abort at step={step}: used={mem['used_frac']:.1%} "
            f"(hard_frac={hard_frac:.0%}). Lower num_points/group_number or "
            f"batch_size, or raise vram_hard_frac in the config."
        )
    return mem


def build_optimizer(model, cfg: dict):
    mode = cfg["mode"]
    lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 0.01))
    if mode == "full_ft":
        enc_lr = float(cfg.get("encoder_lr", lr * 0.1))
        params = [
            {"params": model.pc_encoder.parameters(), "lr": enc_lr},
            {
                "params": list(model.point_encoder.parameters())
                + list(model.mask_encoder.parameters())
                + list(model.mask_decoder.parameters()),
                "lr": lr,
            },
        ]
        return torch.optim.AdamW(params, weight_decay=wd)
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp: bool) -> dict:
    model.eval()
    ious = []
    losses = []
    for batch in loader:
        coords = batch["coords"].to(device)
        feats = batch["features"].to(device)
        gt = batch["gt_masks"].to(device)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            outputs = model(coords=coords, features=feats, gt_masks=gt, is_eval=True)
            gt_flat = gt.flatten(0, 1)
            loss, aux = criterion(outputs, gt_flat)
        losses.append(float(loss.item()))
        if aux:
            ious.append(float(aux[-1]["iou"].mean().item()))
    model.train()
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_iou": float(np.mean(ious)) if ious else float("nan"),
    }


def save_safetensors(model, path: Path):
    from safetensors.torch import save_model

    path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--max_steps", type=int, default=None, help="Override config max_steps")
    ap.add_argument("--out_dir", type=str, default=None, help="Override config out_dir")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg_path = _resolve(args.config, Path.cwd())
    if not cfg_path.is_file():
        cfg_path = _resolve(args.config, _HERE)
    cfg = load_config(cfg_path)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        frac = float(cfg.get("cuda_mem_fraction", 0.92))
        try:
            torch.cuda.set_per_process_memory_fraction(frac)
            print(f"[train] cuda mem fraction capped at {frac:.0%}", flush=True)
        except Exception as exc:
            print(f"[train] could not set mem fraction ({exc})")
        torch.cuda.reset_peak_memory_stats()
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = _resolve(cfg["data_dir"], _HERE)
    out_dir = _resolve(cfg["out_dir"], _HERE)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure Point-SAM + local dataset import
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    point_sam_root = _ROOT / "thirdparty" / "Point-SAM"
    if str(point_sam_root) not in sys.path:
        sys.path.insert(0, str(point_sam_root))

    from dataset import ForestCropDataset, collate_forest
    from pc_sam.model.loss import Criterion

    train_ds = ForestCropDataset(
        data_dir / "train",
        num_points=int(cfg["num_points"]),
        num_masks=int(cfg["num_masks"]),
        augment=True,
        seed=seed,
    )
    val_ds = ForestCropDataset(
        data_dir / "val",
        num_points=int(cfg["num_points"]),
        num_masks=int(cfg["num_masks"]),
        augment=False,
        seed=seed + 1,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_forest,
        drop_last=True,
    )
    val_cap = int(cfg.get("val_max_crops", 32))
    if len(val_ds) > val_cap:
        from torch.utils.data import Subset

        rng = np.random.default_rng(seed + 2)
        idxs = sorted(rng.choice(len(val_ds), size=val_cap, replace=False).tolist())
        val_ds_eval = Subset(val_ds, idxs)
        print(f"[train] val capped to {val_cap}/{len(val_ds)} crops for speed")
    else:
        val_ds_eval = val_ds
    val_loader = DataLoader(
        val_ds_eval,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_forest,
    )
    print(f"[train] crops train={len(train_ds)} val={len(val_ds)} device={device}")

    model = build_model(cfg, device)
    set_trainable(model, cfg["mode"])
    optimizer = build_optimizer(model, cfg)
    criterion = Criterion(use_soft_iou=False)
    amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    max_steps = int(cfg["max_steps"])
    val_every = int(cfg.get("val_every", 100))
    log_every = int(cfg.get("log_every", 20))
    patience = int(cfg.get("early_stop_patience", 8))
    max_grad = float(cfg.get("max_grad_norm", 1.0))

    log = {
        "config": cfg,
        "config_path": str(cfg_path),
        "steps": [],
        "best_val_iou": -1.0,
    }
    best_iou = -1.0
    bad_rounds = 0
    step = 0
    model.train()
    t0 = time.time()
    epoch = 0

    while step < max_steps:
        epoch += 1
        for batch in train_loader:
            if step >= max_steps:
                break
            coords = batch["coords"].to(device)
            feats = batch["features"].to(device)
            gt = batch["gt_masks"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(coords=coords, features=feats, gt_masks=gt)
                gt_flat = gt.flatten(0, 1)
                loss, aux = criterion(outputs, gt_flat)

            scaler.scale(loss).backward()
            if max_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_grad
                )
            scaler.step(optimizer)
            scaler.update()

            step += 1
            try:
                mem = enforce_vram_guard(cfg, step, hard=True)
            except RuntimeError as exc:
                print(f"[train] {exc}", flush=True)
                log["abort_reason"] = str(exc)
                log["vram_at_abort"] = gpu_mem_mb()
                save_safetensors(model, out_dir / "last.safetensors")
                step = max_steps
                break

            if step % log_every == 0:
                iou0 = float(aux[0]["iou"].mean().item()) if aux else float("nan")
                mem = mem or gpu_mem_mb()
                mem_s = ""
                if mem:
                    mem_s = (
                        f" vram={mem['used_frac']:.0%}"
                        f" alloc={mem['alloc_mb']:.0f}MB"
                        f" peak={mem['peak_reserved_mb']:.0f}MB"
                    )
                print(
                    f"step {step}/{max_steps} loss={loss.item():.4f} iou0={iou0:.3f} "
                    f"epoch={epoch}{mem_s}",
                    flush=True,
                )
                entry = {"step": step, "loss": float(loss.item()), "iou0": iou0}
                if mem:
                    entry["vram"] = {k: float(v) for k, v in mem.items()}
                log["steps"].append(entry)

            if step % val_every == 0 or step == max_steps:
                metrics = evaluate(model, val_loader, criterion, device, amp)
                print(
                    f"[val] step={step} loss={metrics['val_loss']:.4f} "
                    f"iou={metrics['val_iou']:.3f}"
                )
                log["steps"].append({"step": step, **metrics})
                save_safetensors(model, out_dir / "last.safetensors")
                if metrics["val_iou"] > best_iou:
                    best_iou = metrics["val_iou"]
                    log["best_val_iou"] = best_iou
                    save_safetensors(model, out_dir / "best.safetensors")
                    print(f"[val] new best iou={best_iou:.3f} -> best.safetensors", flush=True)
                    bad_rounds = 0
                else:
                    bad_rounds += 1
                    if bad_rounds >= patience:
                        print(
                            f"[train] early stop at step={step} "
                            f"(patience={patience}, best_iou={best_iou:.3f})"
                        )
                        step = max_steps
                        break

    log["elapsed_sec"] = time.time() - t0
    log["best_val_iou"] = best_iou
    log["vram_peak"] = gpu_mem_mb()
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"[train] done in {log['elapsed_sec']:.1f}s best_iou={best_iou:.3f}")
    if log.get("vram_peak"):
        vp = log["vram_peak"]
        print(
            f"[train] peak_reserved={vp.get('peak_reserved_mb', 0):.0f}MB "
            f"peak_alloc={vp.get('peak_alloc_mb', 0):.0f}MB",
            flush=True,
        )
    print(f"[train] weights -> {out_dir}")


if __name__ == "__main__":
    # Avoid OpenMP / BLAS oversubscription on Windows
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()

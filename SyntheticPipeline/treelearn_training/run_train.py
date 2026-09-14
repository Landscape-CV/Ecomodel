"""
Run TreeLearn heads-only FT and export best.pth + freeze verification.

Usage (from SyntheticPipeline/treelearn_training/ OR TreeLearn/):
  python ../SyntheticPipeline/treelearn_training/run_train.py
  python run_train.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SP = _HERE.parent
_ROOT = _SP.parent
_TL = _ROOT / "TreeLearn"
_CKPT_DIR = _SP / "treelearn_checkpoints"


def _run_train(smoke: bool) -> Path:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_TL) + os.pathsep + env.get("PYTHONPATH", "")
    cfg = "configs/training/ecomodel_ft_heads.yaml"
    work = "ecomodel_ft_heads_smoke" if smoke else "ecomodel_ft_heads"
    cmd = [sys.executable, "tools/training/train.py", "--config", cfg, "--work_dir", work]
    if smoke:
        # Patch via env not available; write temp config with 2 epochs
        src = _TL / cfg
        tmp = _TL / "configs" / "training" / "_ecomodel_ft_heads_smoke.yaml"
        text = src.read_text(encoding="utf-8")
        text = re.sub(r"^epochs:\s*\d+", "epochs: 2", text, flags=re.M)
        text = re.sub(r"^examples_per_epoch:\s*\d+", "examples_per_epoch: 20", text, flags=re.M)
        text = re.sub(r"^validation_frequency:\s*\d+", "validation_frequency: 1", text, flags=re.M)
        text = re.sub(r"^save_frequency:\s*\d+", "save_frequency: 1", text, flags=re.M)
        tmp.write_text(text, encoding="utf-8")
        cmd = [
            sys.executable,
            "tools/training/train.py",
            "--config",
            str(tmp.relative_to(_TL)),
            "--work_dir",
            work,
        ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(_TL), env=env)
    return _TL / "work_dirs" / work


def _parse_best_epoch(work_dir: Path) -> int:
    """Pick epoch with highest logged val/semantic_acc; fallback to latest epoch_*.pth."""
    best_epoch = None
    best_acc = -1.0
    log_candidates = list(work_dir.glob("*.log")) + list(work_dir.glob("**/*.log"))
    # also tensorboard-less: scan any text logs
    for log in work_dir.rglob("*"):
        if not log.is_file():
            continue
        if log.suffix not in {".log", ".txt"} and "log" not in log.name.lower():
            continue
        try:
            text = log.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in re.finditer(
            r"\[VALIDATION\]\s*\[(\d+)/.*?\]\s*val/semantic_acc\s*([0-9.]+)",
            text,
        ):
            ep, acc = int(m.group(1)), float(m.group(2))
            if acc >= best_acc:
                best_acc = acc
                best_epoch = ep
    if best_epoch is not None:
        cand = work_dir / f"epoch_{best_epoch}.pth"
        if cand.is_file():
            print(f"Best by val acc={best_acc:.2f} -> epoch_{best_epoch}.pth")
            return best_epoch
    epochs = []
    for p in work_dir.glob("epoch_*.pth"):
        m = re.match(r"epoch_(\d+)\.pth", p.name)
        if m:
            epochs.append(int(m.group(1)))
    if not epochs:
        raise FileNotFoundError(f"No epoch_*.pth in {work_dir}")
    latest = max(epochs)
    print(f"No val acc parsed; using latest epoch_{latest}.pth")
    return latest


def verify_encoder_frozen(ckpt_path: Path, vanilla_path: Path) -> dict:
    """Confirm backbone tensors match vanilla (heads-only FT)."""
    ft = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    van = torch.load(str(vanilla_path), map_location="cpu", weights_only=False)
    # TreeLearn checkpoints may wrap state dict
    def state(obj):
        if isinstance(obj, dict):
            for k in ("state_dict", "model", "net"):
                if k in obj and isinstance(obj[k], dict):
                    return obj[k]
            # maybe raw
            if any(isinstance(v, torch.Tensor) for v in obj.values()):
                return obj
        raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")

    sd_ft = state(ft)
    sd_van = state(van)
    # strip module. prefix
    def norm(sd):
        out = {}
        for k, v in sd.items():
            nk = k[7:] if k.startswith("module.") else k
            out[nk] = v
        return out

    sd_ft, sd_van = norm(sd_ft), norm(sd_van)
    backbone_prefixes = ("input_conv.", "unet.", "output_layer.")
    head_prefixes = ("semantic_linear.", "offset_linear.")
    enc_changed = 0
    head_changed = 0
    for k in sd_van:
        if k not in sd_ft:
            continue
        if not torch.is_tensor(sd_van[k]):
            continue
        diff = (sd_van[k].float() - sd_ft[k].float()).abs().max().item()
        if diff <= 1e-6:
            continue
        if k.startswith(backbone_prefixes):
            enc_changed += 1
        elif k.startswith(head_prefixes):
            head_changed += 1
    ok = enc_changed == 0 and head_changed > 0
    report = {
        "ok": ok,
        "backbone_tensors_changed": enc_changed,
        "head_tensors_changed": head_changed,
        "ckpt": str(ckpt_path),
        "vanilla": str(vanilla_path),
    }
    print(
        f"[verify] backbone_changed={enc_changed} head_changed={head_changed} ok={ok}"
    )
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip_train", action="store_true", help="Only export/verify existing work_dir")
    ap.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Relative work_dirs name (default ecomodel_ft_heads or _smoke)",
    )
    args = ap.parse_args()

    work_name = args.work_dir or ("ecomodel_ft_heads_smoke" if args.smoke else "ecomodel_ft_heads")
    work_dir = _TL / "work_dirs" / work_name
    if not args.skip_train:
        work_dir = _run_train(smoke=args.smoke)

    best_ep = _parse_best_epoch(work_dir)
    src = work_dir / f"epoch_{best_ep}.pth"
    _CKPT_DIR.mkdir(parents=True, exist_ok=True)
    dst = _CKPT_DIR / ("best_smoke.pth" if args.smoke else "best.pth")
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")

    vanilla = _TL / "data" / "model_weights" / "model_weights_with_small_20241213.pth"
    report = verify_encoder_frozen(dst, vanilla)
    ( _CKPT_DIR / ("verify_smoke.json" if args.smoke else "verify_report.json")).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if not report["ok"] and not args.smoke:
        # smoke may change BN running stats if somehow trained — heads-only should be ok
        print("WARNING: freeze verification failed", flush=True)
        if report["backbone_tensors_changed"] > 0:
            raise SystemExit("Backbone changed despite fixed_modules — abort")


if __name__ == "__main__":
    main()

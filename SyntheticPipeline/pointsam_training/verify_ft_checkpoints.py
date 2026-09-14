"""Verify fine-tuned Point-SAM checkpoints differ from vanilla as expected.

Usage (from SyntheticPipeline/):
  python pointsam_training/verify_ft_checkpoints.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from safetensors.torch import load_file

_HERE = Path(__file__).resolve().parent
_SP = _HERE.parent
_ROOT = _SP.parent


def _l2_and_changed(a: dict, b: dict):
    l2_sq = 0.0
    n_changed = 0
    enc_changed = 0
    other_changed = 0
    enc_l2_sq = 0.0
    other_l2_sq = 0.0
    max_rel = 0.0
    max_key = None
    for k, ta in a.items():
        tb = b[k]
        d = (ta.float() - tb.float()).norm().item()
        l2_sq += d * d
        if d > 1e-8:
            n_changed += 1
            rel = d / (ta.float().norm().item() + 1e-12)
            if rel > max_rel:
                max_rel = rel
                max_key = k
            if k.startswith("pc_encoder"):
                enc_changed += 1
                enc_l2_sq += d * d
            else:
                other_changed += 1
                other_l2_sq += d * d
    return {
        "l2": l2_sq**0.5,
        "tensors_changed": n_changed,
        "n_tensors": len(a),
        "encoder_tensors_changed": enc_changed,
        "non_encoder_tensors_changed": other_changed,
        "encoder_l2": enc_l2_sq**0.5,
        "non_encoder_l2": other_l2_sq**0.5,
        "max_rel_change": max_rel,
        "max_rel_key": max_key,
    }


def verify_one(name: str, path: Path, vanilla: dict, expect_encoder_frozen: bool) -> dict:
    if not path.is_file():
        return {"name": name, "path": str(path), "ok": False, "error": "missing file"}
    sd = load_file(str(path))
    if set(sd.keys()) != set(vanilla.keys()):
        return {
            "name": name,
            "path": str(path),
            "ok": False,
            "error": "key mismatch vs vanilla",
        }
    stats = _l2_and_changed(vanilla, sd)
    ok = True
    reasons = []
    if stats["l2"] < 1e-3:
        ok = False
        reasons.append("weights identical to vanilla (training did not stick)")
    if expect_encoder_frozen and stats["encoder_tensors_changed"] != 0:
        ok = False
        reasons.append(
            f"encoder should be frozen but {stats['encoder_tensors_changed']} tensors changed"
        )
    if expect_encoder_frozen and stats["non_encoder_tensors_changed"] < 10:
        ok = False
        reasons.append("too few non-encoder tensors changed for a real decoder FT")
    if (not expect_encoder_frozen) and stats["encoder_tensors_changed"] < 10:
        ok = False
        reasons.append("full_ft expected encoder updates; almost none found")
    return {
        "name": name,
        "path": str(path),
        "ok": ok,
        "expect_encoder_frozen": expect_encoder_frozen,
        "reasons": reasons,
        **stats,
        "n_params": int(sum(t.numel() for t in sd.values())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--vanilla",
        type=Path,
        default=_ROOT / "thirdparty" / "checkpoints" / "point_sam" / "model.safetensors",
    )
    ap.add_argument(
        "--decoder",
        type=Path,
        default=_SP / "pointsam_checkpoints" / "decoder_ft" / "best.safetensors",
    )
    ap.add_argument(
        "--full",
        type=Path,
        default=_SP / "pointsam_checkpoints" / "full_ft" / "best.safetensors",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_SP / "pointsam_checkpoints" / "verify_report.json",
    )
    args = ap.parse_args()

    if not args.vanilla.is_file():
        raise SystemExit(f"Vanilla checkpoint missing: {args.vanilla}")

    print(f"Loading vanilla: {args.vanilla}")
    vanilla = load_file(str(args.vanilla))
    results = [
        verify_one("decoder_ft/best", args.decoder, vanilla, expect_encoder_frozen=True),
        verify_one("full_ft/best", args.full, vanilla, expect_encoder_frozen=False),
    ]
    # also last if present
    for label, base, frozen in (
        ("decoder_ft/last", args.decoder.parent / "last.safetensors", True),
        ("full_ft/last", args.full.parent / "last.safetensors", False),
    ):
        if base.is_file():
            results.append(verify_one(label, base, vanilla, expect_encoder_frozen=frozen))

    report = {
        "vanilla": str(args.vanilla),
        "results": results,
        "all_ok": all(r.get("ok") for r in results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for r in results:
        status = "OK" if r.get("ok") else "FAIL"
        print(
            f"[{status}] {r['name']}: L2={r.get('l2', float('nan')):.4f} "
            f"changed={r.get('tensors_changed')}/{r.get('n_tensors')} "
            f"enc_changed={r.get('encoder_tensors_changed')} "
            f"other_changed={r.get('non_encoder_tensors_changed')}"
        )
        for reason in r.get("reasons") or []:
            print(f"  - {reason}")
    print(f"Wrote {args.out}")
    if not report["all_ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

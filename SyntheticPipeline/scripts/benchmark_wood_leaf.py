"""
Benchmark wood/leaf classifiers on LeWoS (+ optional Heidelberg) GT.

  python scripts/benchmark_wood_leaf.py
  python scripts/benchmark_wood_leaf.py --lewos_root ... --heidelberg_root ...

Writes SyntheticPipeline/output/wood_leaf_benchmark/:
  trials.csv, summary.json, compare_*.png, per_tree_heatmap.png
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

_SP = Path(__file__).resolve().parents[1]
_ROOT = _SP.parent
for p in (str(_ROOT), str(_SP)):
    if p not in sys.path:
        sys.path.insert(0, p)

from instance_annotator.wood_leaf import (  # noqa: E402
    classify_eigen,
    classify_gbseparation,
    classify_intensity_percentile,
    classify_intensity_threshold,
    classify_otsu,
    classify_rgi,
    classify_stem_grow,
)

OUT_DEFAULT = _SP / "output" / "wood_leaf_benchmark"


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp + fp + fn == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def metrics(gt_wood: np.ndarray, pred_wood: np.ndarray, pred_leaf: np.ndarray) -> Dict[str, float]:
    """
    Unknown preds (neither) count as incorrect for both classes.
    Pred wood wins on overlap.
    """
    gt_wood = np.asarray(gt_wood, dtype=bool)
    pred_wood = np.asarray(pred_wood, dtype=bool).copy()
    pred_leaf = np.asarray(pred_leaf, dtype=bool).copy()
    pred_leaf &= ~pred_wood
    unknown = ~(pred_wood | pred_leaf)
    # Force unknown to wrong side: treat as leaf when GT wood and wood when GT leaf
    # Equivalent: unknown never contributes TP
    pred_wood_eff = pred_wood & ~unknown
    pred_leaf_eff = pred_leaf & ~unknown

    gt_leaf = ~gt_wood
    # wood
    tp_w = int((pred_wood_eff & gt_wood).sum())
    fp_w = int((pred_wood_eff & gt_leaf).sum())
    fn_w = int((gt_wood & ~pred_wood_eff).sum())  # includes unknown on wood GT
    # leaf
    tp_l = int((pred_leaf_eff & gt_leaf).sum())
    fp_l = int((pred_leaf_eff & gt_wood).sum())
    fn_l = int((gt_leaf & ~pred_leaf_eff).sum())

    wood_f1 = _f1(tp_w, fp_w, fn_w)
    leaf_f1 = _f1(tp_l, fp_l, fn_l)
    correct = int(((pred_wood_eff & gt_wood) | (pred_leaf_eff & gt_leaf)).sum())
    acc = correct / max(len(gt_wood), 1)
    return {
        "wood_precision": tp_w / (tp_w + fp_w) if (tp_w + fp_w) else 0.0,
        "wood_recall": tp_w / (tp_w + fn_w) if (tp_w + fn_w) else 0.0,
        "wood_f1": wood_f1,
        "leaf_precision": tp_l / (tp_l + fp_l) if (tp_l + fp_l) else 0.0,
        "leaf_recall": tp_l / (tp_l + fn_l) if (tp_l + fn_l) else 0.0,
        "leaf_f1": leaf_f1,
        "macro_f1": 0.5 * (wood_f1 + leaf_f1),
        "accuracy": float(acc),
        "unknown_frac": float(unknown.mean()),
    }


def load_tiles(root: Path) -> Dict[str, Dict[str, Any]]:
    tiles_dir = root / "tiles"
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for tid in split["tune"] + split["holdout"]:
        xyz = np.load(tiles_dir / f"{tid}_xyz.npy")
        wood = np.load(tiles_dir / f"{tid}_wood.npy").astype(bool)
        inten_path = tiles_dir / f"{tid}_intensity.npy"
        inten = np.load(inten_path) if inten_path.exists() else np.ones(len(xyz))
        meta = json.loads((tiles_dir / f"{tid}_meta.json").read_text(encoding="utf-8"))
        out[tid] = {"xyz": xyz, "wood": wood, "intensity": inten, "meta": meta}
    return out, split


def _param_tag(d: Dict[str, Any]) -> str:
    if not d:
        return "default"
    return "|".join(f"{k}={v}" for k, v in sorted(d.items()))


def eigen_grid() -> List[Dict[str, Any]]:
    grid = []
    for lin, vert, curv in itertools.product(
        (0.3, 0.45, 0.6),
        (0.4, 0.55, 0.7),
        (0.08, 0.12, 0.2),
    ):
        grid.append(
            {
                "linearity_min": lin,
                "verticality_min": vert,
                "curvature_max": curv,
                "max_points": 50_000,
            }
        )
    return grid


def stem_grid() -> List[Dict[str, Any]]:
    grid = []
    for vert, hp, rad in itertools.product(
        (0.5, 0.65, 0.8),
        (15.0, 25.0, 40.0),
        (0.2, 0.35, 0.5),
    ):
        grid.append(
            {
                "verticality_min": vert,
                "height_percentile": hp,
                "grow_radius": rad,
                "max_points": 50_000,
            }
        )
    return grid


def run_method(
    method: str,
    xyz: np.ndarray,
    inten: np.ndarray,
    params: Dict[str, Any],
    *,
    has_real_intensity: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if method == "eigen":
        return classify_eigen(xyz, inten, **params)
    if method == "stem_grow":
        return classify_stem_grow(xyz, inten, **params)
    if method == "rgi":
        return classify_rgi(xyz, inten, max_points=int(params.get("max_points", 40_000)))
    if method == "gbseparation":
        return classify_gbseparation(xyz, inten, max_points=int(params.get("max_points", 80_000)))
    if method == "percentile":
        if not has_real_intensity:
            raise RuntimeError("N/A: no real intensity")
        return classify_intensity_percentile(xyz, inten, percentile=float(params["percentile"]))
    if method == "intensity":
        if not has_real_intensity:
            raise RuntimeError("N/A: no real intensity")
        thr = params.get("threshold")
        if thr is None:
            thr = float(np.median(inten))
        return classify_intensity_threshold(xyz, inten, float(thr))
    if method == "otsu":
        if not has_real_intensity:
            raise RuntimeError("N/A: no real intensity")
        return classify_otsu(xyz, inten)
    raise ValueError(method)


def evaluate_dataset(
    dataset_id: str,
    root: Path,
    out_rows: List[Dict[str, Any]],
    *,
    skip_gb: bool = False,
    time_budget_s: float = 180.0,
) -> None:
    tiles, split = load_tiles(root)
    has_real_intensity = dataset_id.startswith("heidelberg")

    configs: List[Tuple[str, Dict[str, Any]]] = []
    configs += [("eigen", p) for p in eigen_grid()]
    configs += [("stem_grow", p) for p in stem_grid()]
    configs += [
        ("rgi", {"max_points": 40_000, "variant": "default"}),
        ("rgi", {"max_points": 60_000, "variant": "more_points"}),
    ]
    if not skip_gb:
        configs += [("gbseparation", {"max_points": 80_000})]
    if has_real_intensity:
        for p in (20.0, 40.0, 60.0):
            configs.append(("percentile", {"percentile": p}))
        configs.append(("otsu", {}))
        configs.append(("intensity", {"threshold": None}))  # median

    tune_ids = list(split["tune"])
    hold_ids = list(split["holdout"])
    print(f"\n=== {dataset_id}: {len(tiles)} trees, tune={len(tune_ids)} holdout={len(hold_ids)} ===")
    print(f"configs={len(configs)} has_intensity={has_real_intensity}")

    for method, params in configs:
        tag = _param_tag({k: v for k, v in params.items() if k != "max_points" or method in ("rgi", "gbseparation")})
        print(f"\n[{dataset_id}] {method} {tag}")
        for split_name, ids in (("tune", tune_ids), ("holdout", hold_ids)):
            for tid in ids:
                t = tiles[tid]
                xyz, wood, inten = t["xyz"], t["wood"], t["intensity"]
                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "dataset": dataset_id,
                    "tree_id": tid,
                    "split": split_name,
                    "method": method,
                    "params": tag,
                    "params_json": json.dumps(params),
                    "n_points": int(len(xyz)),
                    "status": "ok",
                    "seconds": None,
                    "error": "",
                }
                t0 = time.time()
                try:
                    if method == "gbseparation" and len(xyz) > 150_000:
                        raise RuntimeError("skip: too many points for GBSeparation")
                    pred_w, pred_l = run_method(
                        method, xyz, inten, params, has_real_intensity=has_real_intensity
                    )
                    elapsed = time.time() - t0
                    if elapsed > time_budget_s and method == "gbseparation":
                        raise RuntimeError(f"time budget exceeded ({elapsed:.1f}s)")
                    m = metrics(wood, pred_w, pred_l)
                    row.update(m)
                    row["seconds"] = round(elapsed, 3)
                except Exception as exc:
                    row["status"] = "error" if "N/A" not in str(exc) else "na"
                    row["error"] = str(exc)[:300]
                    row["seconds"] = round(time.time() - t0, 3)
                    for k in (
                        "wood_precision",
                        "wood_recall",
                        "wood_f1",
                        "leaf_precision",
                        "leaf_recall",
                        "leaf_f1",
                        "macro_f1",
                        "accuracy",
                        "unknown_frac",
                    ):
                        row[k] = None
                    if row["status"] == "error":
                        print(f"  ! {tid}: {exc}")
                out_rows.append(row)


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(np.mean(a)), float(np.std(a))


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick best params per (dataset, method) on tune macro_f1; report holdout."""
    from collections import defaultdict

    # group tune scores
    tune_scores: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in rows:
        if r["status"] != "ok" or r["split"] != "tune" or r["macro_f1"] is None:
            continue
        key = (r["dataset"], r["method"], r["params"])
        tune_scores[key].append(float(r["macro_f1"]))

    best_by_method: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (ds, method, params), scores in tune_scores.items():
        mean_f1, std_f1 = _mean_std(scores)
        cur = best_by_method.get((ds, method))
        if cur is None or mean_f1 > cur["tune_macro_f1_mean"]:
            best_by_method[(ds, method)] = {
                "dataset": ds,
                "method": method,
                "params": params,
                "tune_macro_f1_mean": mean_f1,
                "tune_macro_f1_std": std_f1,
                "tune_n": len(scores),
            }

    # holdout for best params
    for key, info in best_by_method.items():
        ds, method = key
        hold = [
            float(r["macro_f1"])
            for r in rows
            if r["dataset"] == ds
            and r["method"] == method
            and r["params"] == info["params"]
            and r["split"] == "holdout"
            and r["status"] == "ok"
            and r["macro_f1"] is not None
        ]
        wood = [
            float(r["wood_f1"])
            for r in rows
            if r["dataset"] == ds
            and r["method"] == method
            and r["params"] == info["params"]
            and r["split"] == "holdout"
            and r["status"] == "ok"
            and r["wood_f1"] is not None
        ]
        leaf = [
            float(r["leaf_f1"])
            for r in rows
            if r["dataset"] == ds
            and r["method"] == method
            and r["params"] == info["params"]
            and r["split"] == "holdout"
            and r["status"] == "ok"
            and r["leaf_f1"] is not None
        ]
        hm, hs = _mean_std(hold)
        wm, ws = _mean_std(wood)
        lm, ls = _mean_std(leaf)
        info["holdout_macro_f1_mean"] = hm
        info["holdout_macro_f1_std"] = hs
        info["holdout_wood_f1_mean"] = wm
        info["holdout_wood_f1_std"] = ws
        info["holdout_leaf_f1_mean"] = lm
        info["holdout_leaf_f1_std"] = ls
        info["holdout_n"] = len(hold)

    # global best across datasets prioritizing lewos then heidelberg
    ranking = sorted(
        best_by_method.values(),
        key=lambda x: (
            0 if x["dataset"].startswith("lewos") else 1,
            -(x.get("holdout_macro_f1_mean") or -1),
        ),
    )
    overall = ranking[0] if ranking else None
    return {"best_by_method": list(best_by_method.values()), "overall_best": overall}


def write_trials_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def make_plots(summary: Dict[str, Any], rows: List[Dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    best = summary["best_by_method"]
    if not best:
        print("[plots] nothing to plot")
        return

    # Prefer LeWoS for main chart; if empty use all
    lewos = [b for b in best if b["dataset"].startswith("lewos")]
    plot_set = lewos if lewos else best
    # one bar per method
    methods = [b["method"] for b in plot_set]
    means = [b.get("holdout_macro_f1_mean") or 0 for b in plot_set]
    stds = [b.get("holdout_macro_f1_std") or 0 for b in plot_set]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Holdout macro-F1")
    ax.set_ylim(0, 1.05)
    ds = plot_set[0]["dataset"]
    ax.set_title(f"Wood/leaf methods — {ds} (best params on tune)")
    ax.axhline(0.70, color="#F58518", ls="--", lw=1, label="target 0.70")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_macro_f1.png", dpi=140)
    plt.close(fig)

    # wood vs leaf F1
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.35
    wood_m = [b.get("holdout_wood_f1_mean") or 0 for b in plot_set]
    leaf_m = [b.get("holdout_leaf_f1_mean") or 0 for b in plot_set]
    ax.bar(x - w / 2, wood_m, w, label="wood F1", color="#8B5A2B")
    ax.bar(x + w / 2, leaf_m, w, label="leaf F1", color="#2E8B57")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Holdout F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Wood vs leaf F1 — {ds}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "compare_wood_leaf_f1.png", dpi=140)
    plt.close(fig)

    # heatmap: trees x methods (best params) on holdout+tune combined for that dataset
    best_params = {b["method"]: b["params"] for b in plot_set}
    tree_ids = sorted(
        {
            r["tree_id"]
            for r in rows
            if r["dataset"] == ds and r["status"] == "ok" and r["method"] in best_params
        }
    )
    meths = list(best_params.keys())
    if tree_ids and meths:
        mat = np.full((len(tree_ids), len(meths)), np.nan)
        for i, tid in enumerate(tree_ids):
            for j, m in enumerate(meths):
                vals = [
                    float(r["macro_f1"])
                    for r in rows
                    if r["dataset"] == ds
                    and r["tree_id"] == tid
                    and r["method"] == m
                    and r["params"] == best_params[m]
                    and r["status"] == "ok"
                    and r["macro_f1"] is not None
                ]
                if vals:
                    mat[i, j] = vals[0]
        fig, ax = plt.subplots(figsize=(max(6, len(meths)), max(4, 0.28 * len(tree_ids))))
        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(meths)))
        ax.set_xticklabels(meths, rotation=30, ha="right")
        ax.set_yticks(range(len(tree_ids)))
        ax.set_yticklabels(tree_ids, fontsize=7)
        ax.set_title(f"Per-tree macro-F1 — {ds}")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        fig.tight_layout()
        fig.savefig(out_dir / "per_tree_heatmap.png", dpi=140)
        plt.close(fig)

    # Heidelberg intensity chart if present
    hd = [b for b in best if b["dataset"].startswith("heidelberg")]
    if hd:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        methods = [b["method"] for b in hd]
        means = [b.get("holdout_macro_f1_mean") or 0 for b in hd]
        stds = [b.get("holdout_macro_f1_std") or 0 for b in hd]
        x = np.arange(len(methods))
        ax.bar(x, means, yerr=stds, capsize=4, color="#54A24B")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel("Holdout macro-F1")
        ax.set_ylim(0, 1.05)
        ax.set_title("Wood/leaf methods — Heidelberg (with intensity)")
        ax.axhline(0.70, color="#F58518", ls="--", lw=1)
        fig.tight_layout()
        fig.savefig(out_dir / "compare_macro_f1_heidelberg.png", dpi=140)
        plt.close(fig)


def promote_defaults(summary: Dict[str, Any], wood_leaf_path: Path, app_path: Path) -> Optional[Dict]:
    """Update wood_leaf / app defaults from overall best (prefer lewos geometry winner)."""
    best = summary.get("overall_best")
    if not best:
        return None
    # Prefer best lewos geometry method
    lewos_best = None
    for b in summary["best_by_method"]:
        if b["dataset"].startswith("lewos") and b["method"] in ("eigen", "stem_grow", "rgi", "gbseparation"):
            if lewos_best is None or (b.get("holdout_macro_f1_mean") or -1) > (
                lewos_best.get("holdout_macro_f1_mean") or -1
            ):
                lewos_best = b
    chosen = lewos_best or best
    method = chosen["method"]
    # parse params from params string like linearity_min=0.45|verticality_min=0.55|...
    params: Dict[str, Any] = {}
    if chosen.get("params") and chosen["params"] != "default":
        for part in chosen["params"].split("|"):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            try:
                params[k] = float(v) if "." in v else int(v)
            except ValueError:
                params[k] = v

    # Write a small defaults sidecar consumed conceptually; also patch app radio defaults
    defaults_path = wood_leaf_path.parent / "wood_leaf_defaults.json"
    payload = {
        "method": method,
        "params": params,
        "source": chosen,
        "note": "Selected by benchmark_wood_leaf.py on holdout macro-F1",
    }
    defaults_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[promote] wrote {defaults_path}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lewos_root", type=Path, default=_SP / "testdataset" / "lewos_labelled")
    ap.add_argument(
        "--heidelberg_root",
        type=Path,
        default=_SP / "testdataset" / "heidelberg_woodleaf",
    )
    ap.add_argument("--out_dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--skip_gb", action="store_true")
    ap.add_argument("--lewos_only", action="store_true")
    ap.add_argument("--heidelberg_only", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    if not args.heidelberg_only and (args.lewos_root / "split.json").exists():
        evaluate_dataset("lewos_labelledpc", args.lewos_root, rows, skip_gb=args.skip_gb)
    elif not args.heidelberg_only:
        print(f"[warn] LeWoS not prepared at {args.lewos_root}")

    if not args.lewos_only and (args.heidelberg_root / "split.json").exists():
        evaluate_dataset(
            "heidelberg_uumedi", args.heidelberg_root, rows, skip_gb=args.skip_gb
        )
    elif not args.lewos_only:
        print(f"[warn] Heidelberg not prepared at {args.heidelberg_root}")

    if not rows:
        print("No results — prepare datasets first.")
        sys.exit(1)

    write_trials_csv(out_dir / "trials.csv", rows)
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_plots(summary, rows, out_dir)
    promote_defaults(
        summary,
        _SP / "instance_annotator" / "wood_leaf.py",
        _SP / "instance_annotator" / "app.py",
    )

    best = summary.get("overall_best")
    print("\n========== SUMMARY ==========")
    for b in summary["best_by_method"]:
        print(
            f"  {b['dataset']:20s} {b['method']:12s} params={b['params'][:60]} "
            f"tune={b['tune_macro_f1_mean']:.3f} holdout={b.get('holdout_macro_f1_mean', float('nan')):.3f}"
        )
    if best:
        print(
            f"\nOVERALL BEST: {best['dataset']} / {best['method']} "
            f"holdout_macro_f1={best.get('holdout_macro_f1_mean'):.3f}"
        )
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()

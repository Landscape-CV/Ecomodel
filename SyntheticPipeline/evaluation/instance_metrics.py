"""Hungarian-matching metrics for tree instance segmentation."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def pairwise_overlap(
    gt_labels: np.ndarray, pred_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """IoU and point counts between GT ids (>=0) and pred ids (>=0) on one aligned cloud.

    Returns ``(iou, gt_ids, pred_ids, inter, gt_counts, pred_counts)``. ``inter[i, j]`` is the
    number of points in both instance ``gt_ids[i]`` and ``pred_ids[j]``.
    """
    gt_ids = np.unique(gt_labels[gt_labels >= 0])
    pred_ids = np.unique(pred_labels[pred_labels >= 0])
    n_g, n_p = len(gt_ids), len(pred_ids)
    iou = np.zeros((n_g, n_p), dtype=float)
    inter = np.zeros((n_g, n_p), dtype=np.int64)
    gt_counts = np.zeros(n_g, dtype=np.int64)
    pred_counts = np.zeros(n_p, dtype=np.int64)
    if n_g == 0 or n_p == 0:
        return iou, gt_ids.astype(int), pred_ids.astype(int), inter, gt_counts, pred_counts

    for j, p in enumerate(pred_ids):
        pred_counts[j] = int(np.sum(pred_labels == p))
    for i, g in enumerate(gt_ids):
        gmask = gt_labels == g
        gt_counts[i] = int(gmask.sum())
        if gt_counts[i] == 0:
            continue
        for j, p in enumerate(pred_ids):
            hit = int(np.sum(gmask & (pred_labels == p)))
            if hit == 0:
                continue
            inter[i, j] = hit
            union = int(gt_counts[i]) + int(pred_counts[j]) - hit
            iou[i, j] = hit / union if union > 0 else 0.0
    return iou, gt_ids.astype(int), pred_ids.astype(int), inter, gt_counts, pred_counts


def pairwise_iou_matrix(
    gt_labels: np.ndarray, pred_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build IoU matrix between GT instance IDs (>=0) and pred instance IDs (>=0)
    over the same (aligned) point array.
    """
    iou, gt_ids, pred_ids, _, _, _ = pairwise_overlap(gt_labels, pred_labels)
    return iou, gt_ids, pred_ids


def _tree_level(matched_ious, tree_precs, tree_recs, tree_f1s) -> Dict[str, float]:
    """Means over matched trees. Undefined (NaN) when nothing matched at the IoU threshold."""
    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    iou = mean(matched_ious)
    return {
        "mean_matched_iou": iou,
        "tree_iou": iou,
        "tree_precision": mean(tree_precs),
        "tree_recall": mean(tree_recs),
        "tree_f1": mean(tree_f1s),
    }


def match_instances(
    iou: np.ndarray,
    iou_thresh: float = 0.5,
    *,
    inter: np.ndarray | None = None,
    gt_counts: np.ndarray | None = None,
    pred_counts: np.ndarray | None = None,
) -> Dict[str, float]:
    """Hungarian matching maximizing IoU; score detections at ``iou_thresh``.

    Plot-level precision, recall and F1 count matched *trees*. When ``inter`` and the
    per-instance counts are passed, tree-level precision, recall, F1 and IoU are the
    means of the point scores of those matched trees only.
    """
    n_gt, n_pred = iou.shape
    nan_trees = _tree_level([], [], [], [])
    if n_gt == 0 and n_pred == 0:
        return {
            "tp": 0, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "pq": 1.0, "sq": 1.0, "rq": 1.0,
            "num_gt": 0, "num_pred": 0,
            **nan_trees,
        }
    if n_gt == 0:
        return {
            "tp": 0, "fp": n_pred, "fn": 0,
            "precision": 0.0, "recall": 1.0, "f1": 0.0,
            "pq": 0.0, "sq": 0.0, "rq": 0.0,
            "num_gt": 0, "num_pred": n_pred,
            **nan_trees,
        }
    if n_pred == 0:
        return {
            "tp": 0, "fp": 0, "fn": n_gt,
            "precision": 1.0 if n_gt == 0 else 0.0, "recall": 0.0, "f1": 0.0,
            "pq": 0.0, "sq": 0.0, "rq": 0.0,
            "num_gt": n_gt, "num_pred": 0,
            **nan_trees,
        }

    cost = 1.0 - iou
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_ious = []
    tree_precs, tree_recs, tree_f1s = [], [], []
    tp = 0
    have_counts = inter is not None and gt_counts is not None and pred_counts is not None
    for r, c in zip(row_ind, col_ind):
        if iou[r, c] < iou_thresh:
            continue
        tp += 1
        matched_ious.append(float(iou[r, c]))
        if not have_counts:
            continue
        hit = float(inter[r, c])
        p = float(pred_counts[c])
        g = float(gt_counts[r])
        prec = hit / p if p else 0.0
        rec = hit / g if g else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        tree_precs.append(prec)
        tree_recs.append(rec)
        tree_f1s.append(f1)

    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    trees = _tree_level(matched_ious, tree_precs, tree_recs, tree_f1s)
    sq = trees["tree_iou"] if matched_ious else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    pq = sq * rq

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "num_gt": n_gt,
        "num_pred": n_pred,
        **trees,
    }

"""Hungarian-matching metrics for tree instance segmentation."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def pairwise_iou_matrix(
    gt_labels: np.ndarray, pred_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build IoU matrix between GT instance IDs (>=0) and pred instance IDs (>=0)
    over the same (aligned) point array.
    """
    gt_ids = np.unique(gt_labels[gt_labels >= 0])
    pred_ids = np.unique(pred_labels[pred_labels >= 0])
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return (
            np.zeros((len(gt_ids), len(pred_ids)), dtype=float),
            gt_ids.astype(int),
            pred_ids.astype(int),
        )

    iou = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
    for i, g in enumerate(gt_ids):
        gmask = gt_labels == g
        gcount = int(gmask.sum())
        if gcount == 0:
            continue
        for j, p in enumerate(pred_ids):
            pmask = pred_labels == p
            inter = int(np.sum(gmask & pmask))
            if inter == 0:
                continue
            union = gcount + int(pmask.sum()) - inter
            iou[i, j] = inter / union if union > 0 else 0.0
    return iou, gt_ids.astype(int), pred_ids.astype(int)


def match_instances(iou: np.ndarray, iou_thresh: float = 0.5) -> Dict[str, float]:
    """Hungarian matching maximizing IoU; score detections at ``iou_thresh``."""
    n_gt, n_pred = iou.shape
    if n_gt == 0 and n_pred == 0:
        return {
            "tp": 0, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "pq": 1.0, "sq": 1.0, "rq": 1.0,
            "mean_matched_iou": float("nan"),
            "num_gt": 0, "num_pred": 0,
        }
    if n_gt == 0:
        return {
            "tp": 0, "fp": n_pred, "fn": 0,
            "precision": 0.0, "recall": 1.0, "f1": 0.0,
            "pq": 0.0, "sq": 0.0, "rq": 0.0,
            "mean_matched_iou": float("nan"),
            "num_gt": 0, "num_pred": n_pred,
        }
    if n_pred == 0:
        return {
            "tp": 0, "fp": 0, "fn": n_gt,
            "precision": 1.0 if n_gt == 0 else 0.0, "recall": 0.0, "f1": 0.0,
            "pq": 0.0, "sq": 0.0, "rq": 0.0,
            "mean_matched_iou": float("nan"),
            "num_gt": n_gt, "num_pred": 0,
        }

    cost = 1.0 - iou
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_ious = []
    tp = 0
    for r, c in zip(row_ind, col_ind):
        if iou[r, c] >= iou_thresh:
            tp += 1
            matched_ious.append(float(iou[r, c]))

    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    sq = float(np.mean(matched_ious)) if matched_ious else 0.0
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
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else float("nan"),
        "num_gt": n_gt,
        "num_pred": n_pred,
    }

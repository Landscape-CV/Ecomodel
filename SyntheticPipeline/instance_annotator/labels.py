"""Pure-numpy instance label editor with undo/redo."""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def compact_ids(labels: np.ndarray) -> np.ndarray:
    """Remap tree IDs to contiguous 0..T-1; keep -1."""
    out = np.asarray(labels, dtype=np.int32).copy()
    pos = out >= 0
    if not np.any(pos):
        return out
    uniq = np.unique(out[pos])
    # Fast LUT: map each unique id → rank (handles sparse / large ids)
    remap = {int(u): i for i, u in enumerate(uniq)}
    flat = out[pos]
    out[pos] = np.fromiter((remap[int(v)] for v in flat), dtype=np.int32, count=len(flat))
    return out


def remap_treelearn_labels(labels: np.ndarray) -> np.ndarray:
    """
    TreeLearn segmenter uses -1 non-tree and trees often starting at 1.
    Normalize to Ecomodel: non-tree -1, trees contiguous >=0.
    """
    lab = np.asarray(labels, dtype=np.int32).copy()
    # Treat 0 as non-tree if any positive trees exist (TreeLearn convention)
    if np.any(lab > 0):
        lab[lab == 0] = -1
    lab[lab < 0] = -1
    return compact_ids(lab)


class LabelEditor:
    """In-memory int32 labels with undo/redo stacks."""

    def __init__(self, labels: Optional[np.ndarray] = None, n_points: Optional[int] = None):
        if labels is not None:
            self.labels = np.asarray(labels, dtype=np.int32).reshape(-1).copy()
        elif n_points is not None:
            self.labels = np.full(int(n_points), -1, dtype=np.int32)
        else:
            raise ValueError("Provide labels or n_points")
        self._undo: List[np.ndarray] = []
        self._redo: List[np.ndarray] = []
        self.dirty = False

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def num_trees(self) -> int:
        return int(len(np.unique(self.labels[self.labels >= 0])))

    def tree_ids(self) -> np.ndarray:
        return np.unique(self.labels[self.labels >= 0])

    def counts(self) -> dict:
        ids, cnt = np.unique(self.labels, return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, cnt)}

    def _push(self) -> None:
        self._undo.append(self.labels.copy())
        self._redo.clear()
        self.dirty = True

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self.labels.copy())
        self.labels = self._undo.pop()
        self.dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self.labels.copy())
        self.labels = self._redo.pop()
        self.dirty = True
        return True

    def set_all(self, labels: np.ndarray, *, record_undo: bool = True) -> None:
        lab = np.asarray(labels, dtype=np.int32).reshape(-1)
        if len(lab) != len(self.labels):
            raise ValueError(f"label length {len(lab)} != {len(self.labels)}")
        if record_undo:
            self._push()
        self.labels = lab.copy()
        self.dirty = True

    def reassign(self, mask: np.ndarray, target_id: int) -> int:
        """Assign selection to target_id (may be -1). Returns #points changed."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.labels.shape:
            raise ValueError("mask shape mismatch")
        tid = int(target_id)
        if tid < -1:
            raise ValueError("target_id must be >= -1")
        self._push()
        n = int(mask.sum())
        self.labels[mask] = tid
        return n

    def mark_nontree(self, mask: np.ndarray) -> int:
        return self.reassign(mask, -1)

    def paint_new(self, mask: np.ndarray) -> int:
        """Paint selection as a new tree ID (max+1). Returns new id."""
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            raise ValueError("empty selection")
        self._push()
        new_id = int(self.labels.max()) + 1 if np.any(self.labels >= 0) else 0
        if new_id < 0:
            new_id = 0
        self.labels[mask] = new_id
        return new_id

    def merge(self, source_id: int, target_id: int) -> int:
        """Merge all points of source_id into target_id. Returns #points moved."""
        sid, tid = int(source_id), int(target_id)
        if sid == tid:
            return 0
        if sid < 0 or tid < -1:
            raise ValueError("source must be a tree id (>=0); target >= -1")
        mask = self.labels == sid
        n = int(mask.sum())
        if n == 0:
            return 0
        self._push()
        self.labels[mask] = tid
        return n

    def split_to_new(self, mask: np.ndarray) -> int:
        """Alias for paint_new (split subset of a tree into a new instance)."""
        return self.paint_new(mask)

    def compact(self) -> None:
        self._push()
        self.labels = compact_ids(self.labels)

    def next_tree_id(self) -> int:
        if not np.any(self.labels >= 0):
            return 0
        return int(self.labels.max()) + 1

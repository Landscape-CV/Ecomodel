"""Pure-numpy instance label editor with diff-based undo/redo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Undo history budget; oldest steps are dropped beyond this.
DEFAULT_UNDO_BYTES = 512 * 1024 * 1024


def compact_ids(labels: np.ndarray) -> np.ndarray:
    """Remap tree IDs to contiguous 0..T-1; keep -1."""
    out = np.asarray(labels, dtype=np.int32).copy()
    pos = out >= 0
    if not np.any(pos):
        return out
    uniq, inv = np.unique(out[pos], return_inverse=True)
    out[pos] = inv.astype(np.int32)
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


@dataclass
class _Diff:
    idx: np.ndarray  # int64 changed point indices
    old: np.ndarray  # int32 labels before
    new: np.ndarray  # int32 labels after
    extra: Any = None  # side effects to undo/redo alongside (e.g. project-wide merges)

    @property
    def nbytes(self) -> int:
        return int(self.idx.nbytes + self.old.nbytes + self.new.nbytes)


class LabelEditor:
    """In-memory int32 labels with capped, diff-based undo/redo stacks."""

    def __init__(
        self,
        labels: Optional[np.ndarray] = None,
        n_points: Optional[int] = None,
        *,
        max_undo_bytes: int = DEFAULT_UNDO_BYTES,
    ):
        if labels is not None:
            self.labels = np.asarray(labels, dtype=np.int32).reshape(-1).copy()
        elif n_points is not None:
            self.labels = np.full(int(n_points), -1, dtype=np.int32)
        else:
            raise ValueError("Provide labels or n_points")
        self._undo: List[_Diff] = []
        self._redo: List[_Diff] = []
        self.max_undo_bytes = int(max_undo_bytes)
        self.dirty = False
        self.version = 0
        self.last_changed: np.ndarray = np.zeros(0, dtype=np.int64)
        # When set, new tree ids come from here (island-wide ids in project regions).
        self.id_allocator: Optional[Callable[[], int]] = None

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def num_trees(self) -> int:
        return int(len(self.tree_ids()))

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def tree_ids(self) -> np.ndarray:
        return np.unique(self.labels[self.labels >= 0])

    def counts(self) -> Dict[int, int]:
        ids, cnt = np.unique(self.labels, return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, cnt)}

    def undo_bytes(self) -> int:
        return sum(d.nbytes for d in self._undo)

    # ── internals ──────────────────────────────────────────────────────────
    def _commit(self, new_labels: np.ndarray) -> int:
        """Record diff between current and new labels; returns #changed."""
        changed = np.flatnonzero(self.labels != new_labels)
        if len(changed) == 0:
            self.last_changed = changed
            return 0
        diff = _Diff(
            idx=changed.astype(np.int64),
            old=self.labels[changed].copy(),
            new=new_labels[changed].astype(np.int32, copy=True),
        )
        self.labels[changed] = diff.new
        self._undo.append(diff)
        self._redo.clear()
        self._trim()
        self._touch(changed)
        return int(len(changed))

    @property
    def last_diff(self) -> Optional[_Diff]:
        return self._undo[-1] if self._undo else None

    @property
    def next_redo(self) -> Optional[_Diff]:
        return self._redo[-1] if self._redo else None

    def _commit_mask(self, mask: np.ndarray, target: int) -> int:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.labels.shape:
            raise ValueError("mask shape mismatch")
        idx = np.flatnonzero(mask & (self.labels != target))
        if len(idx) == 0:
            self.last_changed = idx
            return 0
        diff = _Diff(
            idx=idx.astype(np.int64),
            old=self.labels[idx].copy(),
            new=np.full(len(idx), target, dtype=np.int32),
        )
        self.labels[idx] = target
        self._undo.append(diff)
        self._redo.clear()
        self._trim()
        self._touch(idx)
        return int(len(idx))

    def _trim(self) -> None:
        total = self.undo_bytes()
        while len(self._undo) > 1 and total > self.max_undo_bytes:
            total -= self._undo.pop(0).nbytes

    def _touch(self, changed: np.ndarray) -> None:
        self.dirty = True
        self.version += 1
        self.last_changed = changed

    # ── public edits ───────────────────────────────────────────────────────
    def undo(self) -> bool:
        if not self._undo:
            return False
        d = self._undo.pop()
        self.labels[d.idx] = d.old
        self._redo.append(d)
        self._touch(d.idx)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        d = self._redo.pop()
        self.labels[d.idx] = d.new
        self._undo.append(d)
        self._touch(d.idx)
        return True

    def set_all(self, labels: np.ndarray, *, record_undo: bool = True) -> int:
        lab = np.asarray(labels, dtype=np.int32).reshape(-1)
        if len(lab) != len(self.labels):
            raise ValueError(f"label length {len(lab)} != {len(self.labels)}")
        if record_undo:
            return self._commit(lab)
        changed = np.flatnonzero(self.labels != lab)
        self.labels = lab.copy()
        self._touch(changed)
        return int(len(changed))

    def reassign(self, mask: np.ndarray, target_id: int) -> int:
        """Assign selection to target_id (may be -1). Returns #points changed."""
        tid = int(target_id)
        if tid < -1:
            raise ValueError("target_id must be >= -1")
        return self._commit_mask(mask, tid)

    def mark_nontree(self, mask: np.ndarray) -> int:
        return self.reassign(mask, -1)

    def paint_new(self, mask: np.ndarray) -> int:
        """Paint selection as a new tree ID (max+1). Returns new id."""
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            raise ValueError("empty selection")
        new_id = self.id_allocator() if self.id_allocator else self.next_tree_id()
        self._commit_mask(mask, new_id)
        return new_id

    def merge(self, source_id: int, target_id: int) -> int:
        """Merge all points of source_id into target_id. Returns #points moved."""
        sid, tid = int(source_id), int(target_id)
        if sid < 0 or tid < -1:
            raise ValueError("source must be a tree id (>=0); target >= -1")
        if sid == tid:
            return 0
        return self._commit_mask(self.labels == sid, tid)

    def split_to_new(self, mask: np.ndarray) -> int:
        """Alias for paint_new (split subset of a tree into a new instance)."""
        return self.paint_new(mask)

    def compact(self) -> int:
        """Renumber trees to 0..T-1. Returns #points relabeled (0 = no-op)."""
        return self._commit(compact_ids(self.labels))

    def next_tree_id(self) -> int:
        if not np.any(self.labels >= 0):
            return 0
        return int(self.labels.max()) + 1

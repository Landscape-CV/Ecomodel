"""
Streamlit TLS instance annotator (bugfix rewrite).

  cd SyntheticPipeline
  python -m streamlit run instance_annotator/app.py
"""
from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np

_SP_DIR = Path(__file__).resolve().parents[1]
_ROOT = _SP_DIR.parent
for p in (str(_ROOT), str(_SP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

# Streamlit re-executes this script on edit but keeps sibling modules cached.
# Reload so viz / wood_leaf signature changes apply without a full server restart.
import instance_annotator.io as _io
import instance_annotator.labels as _labels
import instance_annotator.segment as _segment
import instance_annotator.viz as _viz
import instance_annotator.wood_leaf as _wood_leaf

for _mod in (_io, _labels, _segment, _viz, _wood_leaf):
    importlib.reload(_mod)

from instance_annotator.io import load_cloud, load_tile_prefix, save_tile
from instance_annotator.labels import LabelEditor
from instance_annotator.segment import METHODS, run_method
from instance_annotator.wood_leaf import (
    GB_MAX_POINTS,
    RGI_MAX_POINTS,
    METHODS as WOOD_LEAF_METHODS,
    classify_wood_leaf,
    mask_counts,
)
from instance_annotator.viz import (
    build_plotly_figure,
    downsample_indices,
    parse_plotly_point_indices,
    selection_from_click,
)

st.set_page_config(page_title="TLS Instance Annotator", layout="wide")
st.title("TLS Instance Annotator")
st.caption(
    "Load TLS → optional wood/leaf reference → optional segmentation → "
    "inspect/edit in 3D → export `*_scan.laz` + `*_instances.npy`."
)


def _init_state() -> None:
    ss = st.session_state
    defaults = {
        "xyz": None,
        "intensity": None,
        "editor": None,
        "display_idx": None,
        "selection": None,
        "name": "tile",
        "meta": {},
        "source_method": "manual",
        "highlight_id": None,
        "loaded": False,
        "max_display": 150_000,
        "last_pick_sig": None,
        "fig_rev": 0,
        "status_msg": "",
        "wood_mask": None,
        "leaf_mask": None,
        "wood_leaf_method": None,
    }
    for k, v in defaults.items():
        ss.setdefault(k, v)


def _ensure_selection(n: int) -> np.ndarray:
    sel = st.session_state.selection
    if sel is None or not isinstance(sel, np.ndarray) or len(sel) != n:
        st.session_state.selection = np.zeros(n, dtype=bool)
    return st.session_state.selection


def _set_cloud(xyz, intensity, labels, name: str, meta: dict, method: str = "manual") -> None:
    xyz = np.asarray(xyz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64).reshape(-1)
    n = len(xyz)
    if len(intensity) != n:
        raise ValueError(f"intensity length {len(intensity)} != points {n}")
    if labels is None:
        labels = np.full(n, -1, dtype=np.int32)
    else:
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        if len(labels) != n:
            raise ValueError(f"labels length {len(labels)} != points {n}")

    st.session_state.xyz = xyz
    st.session_state.intensity = intensity
    st.session_state.editor = LabelEditor(labels)
    want = int(st.session_state.get("max_display", 150_000))
    st.session_state.display_idx = downsample_indices(n, max_points=want)
    st.session_state.selection = np.zeros(n, dtype=bool)
    st.session_state.name = name
    st.session_state.meta = dict(meta or {})
    st.session_state.source_method = method
    st.session_state.highlight_id = None
    st.session_state.last_pick_sig = None
    st.session_state.fig_rev = int(st.session_state.get("fig_rev", 0)) + 1
    st.session_state.loaded = True
    st.session_state.wood_mask = None
    st.session_state.leaf_mask = None
    st.session_state.wood_leaf_method = None


def _bump_fig() -> None:
    st.session_state.fig_rev = int(st.session_state.get("fig_rev", 0)) + 1


def _apply_brush(full_indices, radius_m: float, *, add: bool = True) -> int:
    xyz = st.session_state.xyz
    sel = _ensure_selection(len(xyz))
    mask = np.zeros(len(xyz), dtype=bool)
    for fi in full_indices:
        mask |= selection_from_click(xyz, int(fi), radius_m)
    if add:
        st.session_state.selection = sel | mask
    else:
        st.session_state.selection = mask
    _bump_fig()
    return int(mask.sum())


_init_state()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Load")
    mode = st.radio(
        "Start mode",
        ["Run segmentation", "Manual tagging only", "Load existing GT"],
        index=2,
        key="start_mode",
    )
    path_in = st.text_input(
        "Path to LAZ/PLY or tile prefix",
        value=str(_SP_DIR / "testdataset" / "real_instance" / "l1w_t00_03"),
        key="path_in",
        help="Relative paths resolve against SyntheticPipeline/.",
    )
    st.caption("Prefer a **disk path** for large TLS. Upload is optional.")
    uploaded = st.file_uploader("Or upload LAZ/LAS/PLY", type=["laz", "las", "ply"])

    max_display = st.number_input(
        "Max display points",
        min_value=20_000,
        max_value=2_000_000,
        value=int(st.session_state.max_display),
        step=25_000,
        key="max_display_input",
        help="3D view only; edits stay full-res. Lower if the browser lags.",
    )
    st.session_state.max_display = int(max_display)

    if st.button("Load cloud", type="primary", key="btn_load"):
        try:
            labels = None
            meta: dict = {}
            if uploaded is not None:
                suffix = Path(uploaded.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getbuffer())
                    tmp_path = tmp.name
                xyz, inten = load_cloud(tmp_path)
                name = Path(uploaded.name).stem
                meta = {"source_file": uploaded.name}
                if mode == "Load existing GT":
                    st.warning("Upload has no sidecar GT — starting blank labels.")
            else:
                # Prefer tile prefix whenever *_scan.laz exists beside the path
                from instance_annotator.io import _as_path

                resolved = _as_path(path_in)
                prefix_ok = (
                    Path(str(resolved) + "_scan.laz").exists()
                    or Path(str(resolved) + "_scan.las").exists()
                    or (
                        resolved.is_file()
                        and any(
                            resolved.name.endswith(s)
                            for s in ("_scan.laz", "_scan.las", "_instances.npy", "_meta.json")
                        )
                    )
                )
                if prefix_ok:
                    tile = load_tile_prefix(path_in)
                    xyz, inten = tile["xyz"], tile["intensity"]
                    name = tile["name"]
                    meta = tile["meta"]
                    labels = tile["labels"]
                    if mode == "Manual tagging only":
                        labels = None
                    elif mode == "Load existing GT" and labels is None:
                        raise FileNotFoundError("No *_instances.npy next to this prefix.")
                    elif mode == "Run segmentation":
                        labels = None
                else:
                    if mode == "Load existing GT":
                        raise FileNotFoundError(
                            "Load existing GT needs a tile prefix with *_instances.npy"
                        )
                    xyz, inten = load_cloud(path_in)
                    name = Path(path_in).stem.replace("_scan", "")
                    meta = {"source_file": str(path_in)}
                    labels = None

            _set_cloud(
                xyz,
                inten,
                labels,
                name,
                meta,
                method="existing_gt" if labels is not None else "manual",
            )
            st.session_state.status_msg = (
                f"Loaded {name}: {len(xyz):,} pts, trees={st.session_state.editor.num_trees}"
            )
            st.rerun()
        except Exception as exc:
            st.exception(exc)

    st.header("2. Wood / Leaf (reference)")
    st.caption(
        "Optional reference overlay only — does not change instance labels. "
        "Re-run replaces masks."
    )
    wl_labels = {
        "stem_grow": "Stem-grow (verticality) ★",
        "eigen": "Eigenfeatures (geom)",
        "percentile": "Intensity percentile",
        "intensity": "Intensity threshold",
        "otsu": "Otsu (intensity)",
        "rgi": "RGI",
        "gbseparation": "GBSeparation",
    }
    wl_method = st.selectbox(
        "Method",
        list(WOOD_LEAF_METHODS),
        index=0,
        format_func=lambda k: wl_labels.get(k, k),
        key="wl_method",
        disabled=not st.session_state.loaded,
        help="★ = best on LeWoS wood/leaf GT (holdout macro-F1≈0.81). See output/wood_leaf_benchmark/.",
    )
    wl_percentile = 40.0
    wl_threshold = None
    wl_lin = 0.3
    wl_vert = 0.7
    wl_curv = 0.12
    wl_hperc = 15.0
    wl_grow_r = 0.2
    if wl_method == "percentile":
        wl_percentile = st.slider(
            "Wood ≥ percentile",
            min_value=5.0,
            max_value=95.0,
            value=40.0,
            step=1.0,
            key="wl_percentile",
            disabled=not st.session_state.loaded,
        )
    elif wl_method == "intensity":
        med = 0.0
        if st.session_state.loaded and st.session_state.intensity is not None:
            med = float(np.median(st.session_state.intensity))
        wl_threshold = st.number_input(
            "Intensity threshold (wood ≥ T)",
            value=med,
            key="wl_threshold",
            disabled=not st.session_state.loaded,
            help="Default = cloud median intensity.",
        )
    elif wl_method == "otsu":
        st.caption("Auto intensity split (wood = brighter class). Fast on full cloud.")
    elif wl_method == "eigen":
        st.caption(
            "Wood = high linearity + verticality + low curvature. "
            "Defaults from LeWoS sweep."
        )
        wl_lin = st.slider("Min linearity", 0.1, 0.9, 0.3, 0.05, key="wl_lin")
        wl_vert = st.slider("Min verticality", 0.1, 0.95, 0.7, 0.05, key="wl_vert")
        wl_curv = st.slider("Max curvature", 0.02, 0.4, 0.12, 0.02, key="wl_curv")
    elif wl_method == "stem_grow":
        st.caption(
            "Seed low-Z vertical points, grow by radius. "
            "Best LeWoS holdout macro-F1≈0.81 (defaults baked in)."
        )
        wl_vert = st.slider("Min verticality", 0.3, 0.95, 0.8, 0.05, key="wl_vert_sg")
        wl_hperc = st.slider("Seed height ≤ percentile", 5.0, 50.0, 15.0, 1.0, key="wl_hperc")
        wl_grow_r = st.slider("Grow radius (m)", 0.1, 1.0, 0.2, 0.05, key="wl_grow_r")
    elif wl_method == "rgi":
        n_pts = int(len(st.session_state.xyz)) if st.session_state.loaded else 0
        if n_pts > RGI_MAX_POINTS:
            st.caption(
                f"Cloud has {n_pts:,} pts — voxel+intensity subsample to {RGI_MAX_POINTS:,}, "
                "then NN-paint. Weak on LeWoS/Heidelberg GT; prefer Stem-grow / Eigen."
            )
        else:
            st.caption("Region-growing wood/leaf (in-memory).")
    elif wl_method == "gbseparation":
        n_pts = int(len(st.session_state.xyz)) if st.session_state.loaded else 0
        if n_pts > GB_MAX_POINTS:
            st.caption(
                f"Cloud has {n_pts:,} pts — subsample to {GB_MAX_POINTS:,}, then NN-paint. "
                "Prefer Stem-grow / Eigen on large plots."
            )
        else:
            st.caption("Best on small / single-tree clouds.")
    if st.button(
        "Run wood/leaf",
        disabled=not st.session_state.loaded,
        key="btn_wood_leaf",
    ):
        with st.spinner(f"Running wood/leaf ({wl_labels.get(wl_method, wl_method)})…"):
            try:
                wood, leaf = classify_wood_leaf(
                    wl_method,
                    st.session_state.xyz,
                    st.session_state.intensity,
                    threshold=wl_threshold,
                    percentile=wl_percentile,
                    linearity_min=wl_lin,
                    verticality_min=wl_vert,
                    curvature_max=wl_curv,
                    height_percentile=wl_hperc,
                    grow_radius=wl_grow_r,
                )
                st.session_state.wood_mask = wood
                st.session_state.leaf_mask = leaf
                st.session_state.wood_leaf_method = wl_method
                counts = mask_counts(wood, leaf)
                n_cloud = len(st.session_state.xyz) if st.session_state.xyz is not None else 0
                extra = ""
                if wl_method == "gbseparation" and n_cloud > GB_MAX_POINTS:
                    extra = f" (sub->{GB_MAX_POINTS:,})"
                elif wl_method == "rgi" and n_cloud > RGI_MAX_POINTS:
                    extra = f" (sub->{RGI_MAX_POINTS:,})"
                elif wl_method in ("eigen", "stem_grow") and n_cloud > 200_000:
                    extra = " (voxel sub + NN)"
                st.session_state.status_msg = (
                    f"Wood/leaf ({wl_labels.get(wl_method, wl_method)}{extra}): "
                    f"wood={counts['wood']:,} leaf={counts['leaf']:,} "
                    f"unknown={counts['unknown']:,}"
                )
                _bump_fig()
                st.rerun()
            except Exception as exc:
                st.exception(exc)

    has_wl = (
        st.session_state.wood_mask is not None
        and st.session_state.leaf_mask is not None
        and st.session_state.loaded
    )
    show_wood = st.checkbox(
        "Show wood",
        value=True,
        key="show_wood",
        disabled=not has_wl,
    )
    show_leaf = st.checkbox(
        "Show leaves",
        value=True,
        key="show_leaf",
        disabled=not has_wl,
    )
    color_by = st.radio(
        "Color by",
        ["Instance IDs", "Wood / Leaf"],
        index=0,
        key="color_by",
        disabled=not has_wl,
        horizontal=True,
    )
    if has_wl:
        c = mask_counts(st.session_state.wood_mask, st.session_state.leaf_mask)
        st.caption(
            f"Last run: {st.session_state.wood_leaf_method} · "
            f"wood={c['wood']:,} leaf={c['leaf']:,} unknown={c['unknown']:,}"
        )

    st.header("3. Segment")
    method = st.selectbox("Method", list(METHODS), index=list(METHODS).index("treelearn"))
    leaf_removal = st.checkbox("Leaf removal (RGI)", value=False)
    treex_stock = st.checkbox("TreeX stock TLS (real data)", value=True)
    if st.button("Run method", disabled=not st.session_state.loaded, key="btn_run"):
        with st.spinner(f"Running {method}…"):
            try:
                lab, info = run_method(
                    st.session_state.xyz,
                    st.session_state.intensity,
                    method,
                    leaf_removal=leaf_removal,
                    treex_stock_tls=treex_stock,
                )
                st.session_state.editor.set_all(lab, record_undo=True)
                st.session_state.source_method = method
                st.session_state.selection = np.zeros(len(lab), dtype=bool)
                st.session_state.last_pick_sig = None
                _bump_fig()
                st.session_state.status_msg = info.get("message") or (
                    "ok" if info.get("ok") else "No instances"
                )
                st.rerun()
            except Exception as exc:
                st.exception(exc)

    st.header("4. Export")
    out_dir = st.text_input(
        "Output directory",
        value=str(_SP_DIR / "output" / "annotator_exports"),
        key="out_dir",
    )
    # Don't bind value= to session name every run (resets typing); sync via key default once
    if "out_name" not in st.session_state:
        st.session_state.out_name = st.session_state.name or "tile"
    out_name = st.text_input("Tile name", key="out_name")
    if st.button("Sync name from loaded tile", key="btn_sync_name"):
        st.session_state.out_name = st.session_state.name or "tile"
        st.rerun()
    if st.button("Save corrected GT", disabled=not st.session_state.loaded, key="btn_save"):
        try:
            meta = dict(st.session_state.meta)
            meta["source_method"] = st.session_state.source_method
            meta["edited"] = True
            paths = save_tile(
                out_dir,
                out_name,
                st.session_state.xyz,
                st.session_state.intensity,
                st.session_state.editor.labels,
                meta=meta,
            )
            st.session_state.editor.dirty = False
            st.session_state.status_msg = "Saved:\n" + "\n".join(paths.values())
            st.success(st.session_state.status_msg)
        except Exception as exc:
            st.exception(exc)

if st.session_state.status_msg and not st.session_state.loaded:
    st.info(st.session_state.status_msg)

if not st.session_state.loaded:
    st.info("Load a cloud from the sidebar to begin.")
else:
    # ── Main pane ────────────────────────────────────────────────────────────
    ed: LabelEditor = st.session_state.editor
    xyz = st.session_state.xyz
    n = len(xyz)
    sel = _ensure_selection(n)

    # Refresh display subsample if cap changed
    want = int(st.session_state.max_display)
    cur = st.session_state.display_idx
    need = (
        cur is None
        or (n <= want and len(cur) != n)
        or (n > want and len(cur) != want)
    )
    if need:
        st.session_state.display_idx = downsample_indices(n, max_points=want)
        _bump_fig()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Points", f"{n:,}")
    c2.metric("Trees", ed.num_trees)
    c3.metric("Selected", f"{int(sel.sum()):,}")
    c4.metric("Dirty", "yes" if ed.dirty else "no")
    if st.session_state.status_msg:
        st.caption(st.session_state.status_msg)

    st.subheader("Edit tools")
    tool = st.radio(
        "Tool",
        ["Brush select", "Reassign", "Merge", "Paint new / split", "Mark non-tree", "Undo/Redo"],
        horizontal=True,
        key="tool",
    )
    brush_r = st.slider("Brush radius (m)", 0.05, 3.0, 0.25, 0.05, key="brush_r")
    col_a, col_b = st.columns(2)
    with col_a:
        hide_nt = st.checkbox("Hide non-tree (−1)", value=False, key="hide_nt")
    with col_b:
        add_mode = st.checkbox("Add to selection (vs replace)", value=True, key="add_mode")

    tree_ids = [int(t) for t in ed.tree_ids().tolist()]
    id_options = [-1] + tree_ids
    next_id = ed.next_tree_id()
    reassign_opts = list(dict.fromkeys(id_options + [next_id]))

    left, right = st.columns([2.2, 1])

    with right:
        st.markdown("**Instances**")
        if tree_ids:
            pick_tid = st.selectbox(
                "Tree ID",
                tree_ids,
                format_func=lambda t: f"tree {t} ({int((ed.labels == t).sum()):,} pts)",
                key="pick_tid",
            )
            b_sel, b_hi = st.columns(2)
            if b_sel.button("Select tree", key="btn_sel_tree"):
                st.session_state.selection = ed.labels == int(pick_tid)
                st.session_state.highlight_id = int(pick_tid)
                st.session_state.last_pick_sig = None
                _bump_fig()
                st.rerun()
            if b_hi.button("Highlight only", key="btn_hi_tree"):
                st.session_state.highlight_id = int(pick_tid)
                _bump_fig()
                st.rerun()
        else:
            st.caption("No tree instances yet.")

        nt_count = int((ed.labels < 0).sum())
        st.caption(f"Non-tree (−1): {nt_count:,} pts")

        seed_idx = st.number_input(
            "Seed point index (full cloud)",
            min_value=0,
            max_value=max(n - 1, 0),
            value=0,
            step=1,
            key="seed_idx",
            help="From hover tooltip `idx=` on the plot.",
        )
        if st.button("Brush from seed", key="btn_seed"):
            added = _apply_brush([int(seed_idx)], brush_r, add=add_mode)
            st.session_state.status_msg = (
                f"Brush @ {seed_idx}: touched {added:,}; "
                f"selection={int(st.session_state.selection.sum()):,}"
            )
            st.rerun()
        if st.button("Clear selection", key="btn_clear"):
            st.session_state.selection = np.zeros(n, dtype=bool)
            st.session_state.highlight_id = None
            st.session_state.last_pick_sig = ("__cleared__",)
            _bump_fig()
            st.rerun()

        sel = st.session_state.selection

        if tool == "Reassign":
            target = st.selectbox(
                "Target ID",
                reassign_opts,
                format_func=lambda x: "non-tree (−1)" if x < 0 else str(x),
                key="reassign_tgt",
            )
            if st.button("Apply reassign", type="primary", key="btn_reassign"):
                if not np.any(sel):
                    st.error("Selection is empty.")
                else:
                    nchg = ed.reassign(sel, int(target))
                    st.session_state.selection = np.zeros(n, dtype=bool)
                    st.session_state.last_pick_sig = None
                    _bump_fig()
                    st.session_state.status_msg = f"Reassigned {nchg:,} → {target}"
                    st.rerun()
        elif tool == "Merge":
            if not tree_ids:
                st.warning("Need at least one tree to merge.")
            else:
                src = st.selectbox("Source tree", tree_ids, key="merge_src")
                tgt = st.selectbox(
                    "Target",
                    id_options,
                    format_func=lambda x: "non-tree (−1)" if x < 0 else str(x),
                    key="merge_tgt",
                )
                if st.button("Apply merge", type="primary", key="btn_merge"):
                    nchg = ed.merge(int(src), int(tgt))
                    _bump_fig()
                    st.session_state.status_msg = f"Merged {nchg:,} from {src} → {tgt}"
                    st.rerun()
        elif tool == "Paint new / split":
            if st.button("Paint selection as new tree", type="primary", key="btn_paint"):
                if not np.any(sel):
                    st.error("Selection is empty.")
                else:
                    new_id = ed.paint_new(sel)
                    st.session_state.selection = np.zeros(n, dtype=bool)
                    st.session_state.last_pick_sig = None
                    _bump_fig()
                    st.session_state.status_msg = f"Created tree {new_id}"
                    st.rerun()
        elif tool == "Mark non-tree":
            if st.button("Mark selection as −1", type="primary", key="btn_nontree"):
                if not np.any(sel):
                    st.error("Selection is empty.")
                else:
                    nchg = ed.mark_nontree(sel)
                    st.session_state.selection = np.zeros(n, dtype=bool)
                    st.session_state.last_pick_sig = None
                    _bump_fig()
                    st.session_state.status_msg = f"Marked {nchg:,} non-tree"
                    st.rerun()
        elif tool == "Undo/Redo":
            b1, b2, b3 = st.columns(3)
            if b1.button("Undo", key="btn_undo"):
                if ed.undo():
                    _bump_fig()
                    st.rerun()
                else:
                    st.warning("Nothing to undo.")
            if b2.button("Redo", key="btn_redo"):
                if ed.redo():
                    _bump_fig()
                    st.rerun()
                else:
                    st.warning("Nothing to redo.")
            if b3.button("Compact IDs", key="btn_compact"):
                ed.compact()
                _bump_fig()
                st.rerun()

    with left:
        color_mode = (
            "material"
            if (
                st.session_state.wood_mask is not None
                and color_by == "Wood / Leaf"
            )
            else "instance"
        )
        fig, _origin = build_plotly_figure(
            xyz,
            ed.labels,
            st.session_state.display_idx,
            highlight_id=st.session_state.highlight_id,
            selection_mask_full=st.session_state.selection,
            hide_nontree=hide_nt,
            wood_mask=st.session_state.wood_mask,
            leaf_mask=st.session_state.leaf_mask,
            show_wood=bool(show_wood) if st.session_state.wood_mask is not None else True,
            show_leaf=bool(show_leaf) if st.session_state.leaf_mask is not None else True,
            color_mode=color_mode,
            title=f"{st.session_state.name} · {st.session_state.source_method}",
        )
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="main_cloud",
        )

        if tool == "Brush select":
            full_indices = parse_plotly_point_indices(event, st.session_state.display_idx)
            if full_indices:
                sig = tuple(sorted(full_indices))
                prev = st.session_state.last_pick_sig
                if prev == ("__cleared__",):
                    st.session_state.last_pick_sig = sig
                elif sig != prev:
                    st.session_state.last_pick_sig = sig
                    added = _apply_brush(full_indices, brush_r, add=add_mode)
                    st.session_state.status_msg = (
                        f"Plot pick {len(full_indices)} pt(s), brush touched {added:,}; "
                        f"selection={int(st.session_state.selection.sum()):,}"
                    )
                    st.rerun()

        st.caption(
            "Plot: click points (Brush select tool) or use seed index. "
            "Magenta = selection, yellow = highlight. "
            "Wood/Leaf colors: brown = wood, green = leaf, gray = unknown. "
            "Camera stays put across edits."
        )

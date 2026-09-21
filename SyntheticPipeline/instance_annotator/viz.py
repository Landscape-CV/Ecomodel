"""Plotly visualization helpers for instance-colored point clouds."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

PathLike = Union[str, Path]


def instance_colors(labels: np.ndarray) -> np.ndarray:
    """Deterministic RGB in [0,1] for instance IDs (-1 → gray)."""
    rgb = np.full((len(labels), 3), 0.55, dtype=np.float64)
    ids = labels.astype(np.int64)
    pos = ids >= 0
    if not np.any(pos):
        return rgb
    u = ids[pos]
    h = (u * 2654435761) % (2**32)
    rgb[pos, 0] = ((h % 256) / 255.0) * 0.85 + 0.1
    rgb[pos, 1] = (((h // 256) % 256) / 255.0) * 0.85 + 0.1
    rgb[pos, 2] = (((h // 65536) % 256) / 255.0) * 0.85 + 0.1
    return rgb


def downsample_indices(
    n: int,
    max_points: int = 150_000,
    seed: int = 0,
) -> np.ndarray:
    """Return sorted indices into the full cloud for display."""
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=int(max_points), replace=False)
    return np.sort(idx.astype(np.int64))


def write_colored_ply(path: PathLike, xyz: np.ndarray, rgb01: np.ndarray) -> None:
    """Binary little-endian colored PLY."""
    path = Path(path)
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.clip(np.asarray(rgb01) * 255.0, 0, 255).astype(np.uint8)
    m = len(xyz)
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {m}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        f.write(header)
        verts = np.empty(
            m,
            dtype=[
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("r", "u1"),
                ("g", "u1"),
                ("b", "u1"),
            ],
        )
        verts["x"] = xyz[:, 0]
        verts["y"] = xyz[:, 1]
        verts["z"] = xyz[:, 2]
        verts["r"] = rgb[:, 0]
        verts["g"] = rgb[:, 1]
        verts["b"] = rgb[:, 2]
        f.write(verts.tobytes())


def _rgb_css(colors: np.ndarray) -> List[str]:
    """Vectorized float RGB [0,1] → css strings (still needed for Scatter3d)."""
    c = np.clip(np.asarray(colors) * 255.0, 0, 255).astype(np.uint8)
    # Faster than Python loop over f-strings for moderate N
    return [f"rgb({r},{g},{b})" for r, g, b in c]


# Material reference colors (wood / leaf / unknown)
_WOOD_RGB = np.array([139 / 255.0, 90 / 255.0, 43 / 255.0])   # #8B5A2B
_LEAF_RGB = np.array([46 / 255.0, 139 / 255.0, 87 / 255.0])   # #2E8B57
_UNKNOWN_RGB = np.array([0.55, 0.55, 0.55])


def material_colors(
    wood_mask: np.ndarray,
    leaf_mask: np.ndarray,
) -> np.ndarray:
    """RGB in [0,1]: wood brown, leaf green, neither gray. Wood wins on overlap."""
    wood = np.asarray(wood_mask, dtype=bool)
    leaf = np.asarray(leaf_mask, dtype=bool)
    rgb = np.tile(_UNKNOWN_RGB, (len(wood), 1))
    rgb[leaf & ~wood] = _LEAF_RGB
    rgb[wood] = _WOOD_RGB
    return rgb


def build_plotly_figure(
    xyz: np.ndarray,
    labels: np.ndarray,
    display_idx: np.ndarray,
    *,
    highlight_id: Optional[int] = None,
    selection_mask_full: Optional[np.ndarray] = None,
    hide_nontree: bool = False,
    wood_mask: Optional[np.ndarray] = None,
    leaf_mask: Optional[np.ndarray] = None,
    show_wood: bool = True,
    show_leaf: bool = True,
    color_mode: str = "instance",
    point_size: float = 1.8,
    title: str = "Instance labels",
    max_rgb_points: int = 200_000,
):
    """
    Build a Plotly Scatter3d figure from a display subset.

    ``display_idx`` maps figure point i → full-cloud index (before hide filter).
    Customdata columns: [full_idx, label] as plain Python ints for Streamlit.

    When ``wood_mask`` / ``leaf_mask`` are set, ``show_wood`` / ``show_leaf``
    filter visibility (unknown points always kept). ``color_mode`` is
    ``"instance"`` or ``"material"``.
    """
    import plotly.graph_objects as go
    import plotly.express as px

    xyz = np.asarray(xyz, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    idx = np.asarray(display_idx, dtype=np.int64)
    if len(idx) == 0 or len(xyz) == 0:
        fig = go.Figure()
        fig.update_layout(title=title or "Empty cloud")
        return fig, np.zeros(3)

    sub_xyz = xyz[idx]
    sub_lab = labels[idx]

    keep = np.ones(len(idx), dtype=bool)
    if hide_nontree:
        keep &= sub_lab >= 0

    has_material = (
        wood_mask is not None
        and leaf_mask is not None
        and len(wood_mask) == len(xyz)
        and len(leaf_mask) == len(xyz)
    )
    if has_material:
        w_full = np.asarray(wood_mask, dtype=bool)
        l_full = np.asarray(leaf_mask, dtype=bool)
        sub_w = w_full[idx]
        sub_l = l_full[idx]
        # Hide toggled-off classes; unknown (neither) always shown
        if not show_wood:
            keep &= ~sub_w
        if not show_leaf:
            keep &= ~sub_l

    if not np.any(keep):
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no points to show)")
        return fig, np.zeros(3)

    sub_xyz = sub_xyz[keep]
    sub_lab = sub_lab[keep]
    idx = idx[keep]

    mode = str(color_mode or "instance").lower()
    if has_material and mode == "material":
        colors = material_colors(w_full[idx], l_full[idx])
    else:
        colors = instance_colors(sub_lab)

    if highlight_id is not None:
        hi = sub_lab == int(highlight_id)
        colors[hi] = np.array([1.0, 0.92, 0.2])
    if selection_mask_full is not None:
        sel_full = np.asarray(selection_mask_full, dtype=bool)
        if len(sel_full) == len(xyz):
            sel = sel_full[idx]
            colors[sel] = np.array([1.0, 0.25, 0.85])

    origin = sub_xyz.mean(axis=0)
    plot_xyz = sub_xyz - origin

    # Always honor selection/highlight colors. Cap CSS rgb list size for speed;
    # above that, quantize to a discrete colorscale keyed by visual class.
    n = len(sub_xyz)
    use_css = n <= max_rgb_points or (has_material and mode == "material")
    if use_css:
        marker = dict(
            size=point_size,
            opacity=0.92,
            color=_rgb_css(colors),
        )
    else:
        # Encode: nontree=-2, selection=-1, highlight=max+1, else tree id
        code = sub_lab.astype(np.float64)
        code[sub_lab < 0] = -2.0
        if highlight_id is not None:
            code[sub_lab == int(highlight_id)] = float(np.nanmax(code) + 1)
        if selection_mask_full is not None and len(selection_mask_full) == len(xyz):
            code[selection_mask_full[idx]] = -1.0
        marker = dict(
            size=point_size,
            opacity=0.92,
            color=code,
            colorscale=px.colors.sample_colorscale("Turbo", np.linspace(0, 1, 12)),
            cmin=float(np.nanmin(code)),
            cmax=float(np.nanmax(code)),
            showscale=False,
        )

    # Plain Python ints — Streamlit selection serialization is picky about numpy types
    custom = np.column_stack([idx.astype(np.int64), sub_lab.astype(np.int64)])

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=plot_xyz[:, 0],
                y=plot_xyz[:, 1],
                z=plot_xyz[:, 2],
                mode="markers",
                marker=marker,
                customdata=custom,
                hovertemplate=(
                    "idx=%{customdata[0]} id=%{customdata[1]}<br>"
                    "x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra></extra>"
                ),
                name="points",
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m, centered)",
            yaxis_title="Y (m, centered)",
            zaxis_title="Z (m, centered)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        # Stable camera across edits; bump revision when labels change via caller key
        uirevision="instance-annotator-cam",
        height=700,
    )
    return fig, origin


def selection_from_click(
    xyz: np.ndarray,
    click_full_idx: int,
    radius_m: float,
) -> np.ndarray:
    """Boolean mask: points within radius_m of the clicked full-cloud point."""
    xyz = np.asarray(xyz, dtype=np.float64)
    i = int(click_full_idx)
    if i < 0 or i >= len(xyz):
        return np.zeros(len(xyz), dtype=bool)
    c = xyz[i]
    d2 = np.sum((xyz - c) ** 2, axis=1)
    return d2 <= (float(radius_m) ** 2)


def parse_plotly_point_indices(event, display_idx: np.ndarray) -> List[int]:
    """
    Extract full-cloud indices from a Streamlit plotly selection event.
    Prefers customdata[0]; falls back to point_index → display_idx.
    """
    if event is None:
        return []
    sel = getattr(event, "selection", None)
    if sel is None:
        return []
    pts = None
    if isinstance(sel, dict):
        pts = sel.get("points")
    else:
        pts = getattr(sel, "points", None)
    if not pts:
        return []

    out: List[int] = []
    disp = np.asarray(display_idx, dtype=np.int64)
    for p in pts:
        if not isinstance(p, dict):
            continue
        cd = p.get("customdata")
        full = None
        if cd is not None:
            try:
                # customdata may be [idx, label] or nested
                if isinstance(cd, (list, tuple, np.ndarray)):
                    full = int(cd[0])
                else:
                    full = int(cd)
            except (TypeError, ValueError, IndexError):
                full = None
        if full is None:
            for key in ("point_index", "pointNumber", "point_number"):
                if key in p:
                    try:
                        di = int(p[key])
                        if 0 <= di < len(disp):
                            full = int(disp[di])
                    except (TypeError, ValueError):
                        pass
                    break
        if full is not None:
            out.append(full)
    # unique, stable
    return list(dict.fromkeys(out))


def save_plotly_html(fig, path: PathLike) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return str(path)

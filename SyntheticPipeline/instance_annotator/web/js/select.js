// Selection tools: lasso, box, sphere brush, click-to-select-tree, hover readout.
import { state, emit, recountSelection } from "./state.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const CLICK_SLOP = 4;

export class SelectionTools {
  constructor(viewer, viewportEl, overlayEl, hoverEl) {
    this.viewer = viewer;
    this.el = viewportEl;
    this.overlay = overlayEl;
    this.hoverEl = hoverEl;
    this.drag = null;
    this.brushCursor = null;
    this._hoverTimer = 0;
    this.brushRadius = 0.3;

    const c = viewer.canvas;
    c.addEventListener("pointerdown", (e) => this._down(e));
    window.addEventListener("pointermove", (e) => this._move(e));
    window.addEventListener("pointerup", (e) => this._up(e));
    c.addEventListener("dblclick", (e) => this._dbl(e));
    c.addEventListener("pointerleave", () => { this.hoverEl.textContent = ""; this._hideBrush(); });
    c.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  setTool(tool) {
    state.tool = tool;
    this.el.classList.remove("tool-lasso", "tool-box", "tool-brush");
    if (tool !== "navigate") this.el.classList.add(`tool-${tool}`);
    this.viewer.setToolMode(tool !== "navigate");
    this._hideBrush();
    emit("tool", tool);
  }

  _local(e) {
    const r = this.viewer.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  _op(e) {
    if (e.ctrlKey || e.metaKey) return "sub";
    if (e.shiftKey) return "add";
    return "replace";
  }

  // Alt restricts to the focused tree, or to the tree under the drag start.
  _restrictLabel(e, x, y) {
    if (!e.altKey) return null;
    if (state.focusedTree !== null) return state.focusedTree;
    const i = this.viewer.pick(x, y);
    return i >= 0 ? state.labels[i] : null;
  }

  _down(e) {
    if (!state.n || e.button !== 0) return;
    const [x, y] = this._local(e);
    const tool = state.tool;
    this.drag = { tool, x0: x, y0: y, pts: [[x, y]], op: this._op(e), restrict: this._restrictLabel(e, x, y), moved: false };
    if (tool === "navigate") return;
    e.preventDefault();
    this.viewer.canvas.setPointerCapture(e.pointerId);
    if (tool === "brush") {
      if (this.drag.op === "replace") this._clearSel();
      this.drag.op = this.drag.op === "sub" ? "sub" : "add";
      this.viewer.project();
      this._brushAt(x, y);
    }
  }

  _move(e) {
    if (!state.n) return;
    const [x, y] = this._local(e);
    const d = this.drag;
    if (!d) {
      if (state.tool === "brush") this._showBrush(x, y);
      this._scheduleHover(x, y);
      return;
    }
    if (Math.hypot(x - d.x0, y - d.y0) > CLICK_SLOP) d.moved = true;
    if (d.tool === "lasso") {
      d.pts.push([x, y]);
      this._drawPoly(d.pts);
    } else if (d.tool === "box") {
      this._drawBox(d.x0, d.y0, x, y);
    } else if (d.tool === "brush") {
      this._showBrush(x, y);
      this._brushAt(x, y);
    }
  }

  _up(e) {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    const [x, y] = this._local(e);
    this._clearOverlay();
    if (d.tool === "navigate") {
      if (!d.moved && e.button === 0) this._clickTree(x, y, d.op);
      return;
    }
    if (d.tool === "lasso" && d.pts.length > 2) {
      this._applyScreen(d, (px, py) => pointInPoly(px, py, d.pts), bbox(d.pts));
    } else if (d.tool === "box" && d.moved) {
      const x0 = Math.min(d.x0, x), x1 = Math.max(d.x0, x), y0 = Math.min(d.y0, y), y1 = Math.max(d.y0, y);
      this._applyScreen(d, (px, py) => px >= x0 && px <= x1 && py >= y0 && py <= y1, [x0, y0, x1, y1]);
    } else if (d.tool !== "brush" && !d.moved) {
      this._clickTree(x, y, d.op);
    }
    if (d.tool === "brush") this._finish();
  }

  _dbl(e) {
    if (!state.n) return;
    const [x, y] = this._local(e);
    const i = this.viewer.pick(x, y);
    if (i < 0) return;
    const t = state.labels[i];
    if (t >= 0) emit("isolate-tree", t);
  }

  _applyScreen(d, inside, [bx0, by0, bx1, by1]) {
    const { sx, sy, depth } = this.viewer.project();
    const sel = state.selection, lab = state.labels;
    if (d.op === "replace") sel.fill(0);
    const val = d.op === "sub" ? 0 : 1;
    for (let i = 0; i < state.n; i++) {
      const dd = depth[i];
      if (!(dd === dd)) continue;
      const px = sx[i], py = sy[i];
      if (px < bx0 || px > bx1 || py < by0 || py > by1) continue;
      if (d.restrict !== null && lab[i] !== d.restrict) continue;
      if (inside(px, py)) sel[i] = val;
    }
    this._finish();
  }

  _brushAt(x, y) {
    const d = this.drag;
    const c = this.viewer.pick(x, y, 10);
    if (c < 0) return;
    const p = state.positions, sel = state.selection, lab = state.labels;
    const cx = p[3 * c], cy = p[3 * c + 1], cz = p[3 * c + 2];
    const r2 = this.brushRadius * this.brushRadius;
    const { depth } = this.viewer.project();
    const val = d.op === "sub" ? 0 : 1;
    for (let i = 0; i < state.n; i++) {
      const dd = depth[i];
      if (!(dd === dd)) continue;
      const dx = p[3 * i] - cx, dy = p[3 * i + 1] - cy, dz = p[3 * i + 2] - cz;
      if (dx * dx + dy * dy + dz * dz > r2) continue;
      if (d.restrict !== null && lab[i] !== d.restrict) continue;
      sel[i] = val;
    }
    this.viewer.selectionChanged();
    recountSelection();
    emit("selection");
  }

  _clickTree(x, y, op) {
    const i = this.viewer.pick(x, y);
    if (i < 0) return;
    const t = state.labels[i];
    if (t < 0) {
      emit("toast", { msg: "Non-tree point. Use Lasso / Box / Brush to select non-tree points.", kind: "info" });
      return;
    }
    selectTree(t, op);
    emit("focus-tree", { id: t, frame: false });
    this.viewer.selectionChanged();
  }

  _clearSel() {
    state.selection.fill(0);
    this._finish();
  }

  _finish() {
    recountSelection();
    this.viewer.selectionChanged();
    emit("selection");
  }

  // ── hover readout ──────────────────────────────────────────────────────
  _scheduleHover(x, y) {
    clearTimeout(this._hoverTimer);
    this._hoverTimer = setTimeout(() => this._hover(x, y), 70);
  }

  _hover(x, y) {
    if (this.drag || !state.n) return;
    const i = this.viewer.pick(x, y);
    if (i < 0) { this.hoverEl.textContent = ""; return; }
    const t = state.labels[i];
    const o = state.info.origin || [0, 0, 0];
    const p = state.positions;
    const xyz = [0, 1, 2].map((k) => (p[3 * i + k] + o[k]).toFixed(2)).join(", ");
    let txt;
    if (t < 0) txt = "non-tree";
    else {
      const tr = state.treeById.get(t);
      txt = `tree ${t}` + (tr ? ` · ${tr.count.toLocaleString()} pts · h ${tr.height.toFixed(1)} m` : "");
    }
    this.hoverEl.textContent = `${txt}\nxyz ${xyz}`;
  }

  // ── overlay drawing ────────────────────────────────────────────────────
  _clearOverlay() {
    for (const n of [...this.overlay.querySelectorAll(".lasso")]) n.remove();
  }

  _drawPoly(pts) {
    let el = this.overlay.querySelector(".lasso");
    if (!el) {
      el = document.createElementNS(SVG_NS, "polygon");
      el.setAttribute("class", "lasso");
      this.overlay.appendChild(el);
    }
    el.setAttribute("points", pts.map((p) => p.join(",")).join(" "));
  }

  _drawBox(x0, y0, x1, y1) {
    this._drawPoly([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
  }

  _showBrush(x, y) {
    if (!this.brushCursor) {
      this.brushCursor = document.createElementNS(SVG_NS, "circle");
      this.brushCursor.setAttribute("class", "brush");
      this.overlay.appendChild(this.brushCursor);
    }
    // Approximate the sphere's on-screen radius at the target distance.
    const cam = this.viewer.camera;
    const dist = cam.position.distanceTo(this.viewer.controls.target);
    const pxPerM = this.viewer.canvas.clientHeight / (2 * Math.tan((cam.fov * Math.PI) / 360) * dist);
    this.brushCursor.setAttribute("cx", x);
    this.brushCursor.setAttribute("cy", y);
    this.brushCursor.setAttribute("r", Math.max(4, this.brushRadius * pxPerM));
    this.brushCursor.style.display = "";
  }

  _hideBrush() {
    if (this.brushCursor) this.brushCursor.style.display = "none";
  }
}

export function selectTree(t, op = "replace") {
  const sel = state.selection, lab = state.labels;
  if (op === "replace") sel.fill(0);
  const val = op === "sub" ? 0 : 1;
  for (let i = 0; i < state.n; i++) if (lab[i] === t) sel[i] = val;
  recountSelection();
  emit("selection");
}

function bbox(pts) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) {
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  return [x0, y0, x1, y1];
}

function pointInPoly(x, y, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

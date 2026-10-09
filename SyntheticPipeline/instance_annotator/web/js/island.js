// Island mode: streaming octree view, island-wide tree table, mini-map, edit regions,
// island segmentation / border-merge review / export.
import * as THREE from "three";
import { state, emit, on, registerTreeIds, resetColorSlots, cssColor } from "./state.js";
import { getJSON, postJSON } from "./api.js";
import { OctreeLayer } from "./octree.js";
import { renderTrees, showTab, initPointSamSelect } from "./panels.js";
import { el, toast, confirmDialog, modal } from "./ui.js";

const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";

export class Island {
  /**
   * hooks: { updateChrome, loadCloudData(opts), resetSingle(), errorToast, guarded(fn,label),
   *          runJob(url, body, label, pane, onDone), setFocus(id, frame) }
   */
  constructor(viewer, hooks) {
    this.v = viewer;
    this.hooks = hooks;
    this.layer = new OctreeLayer(viewer);
    this.project = null;
    this.pendingBox = null;   // {min:[x,y], max:[x,y]} proposed region (project-local)
    this.regionBox = null;    // open region
    this.helper = null;
    this.drawing = false;
    this.drag = null;
    this.candidates = [];
    this._treesTimer = 0;
    this._ctxTimer = 0;
    this._mmPending = false;
    this._hoverTimer = 0;

    this.layer.onStats = (s) => this._streamStats(s);
    on("camera", () => { if (this.active) { this._adaptNear(); this._drawMinimapSoon(); } });
    this._initCanvasEvents();
    this._initMinimap();
    this._initPanel();
  }

  get active() { return state.mode !== "single"; }
  get inRegion() { return state.mode === "region"; }

  // ── lifecycle ──────────────────────────────────────────────────────────
  async enter(info) {
    const hier = await getJSON("/api/project/hierarchy");
    this.project = info;
    state.mode = info.region ? "region" : "island";
    resetColorSlots();
    this.layer.open(hier);
    const b = this.layer.bounds();
    this.v.frameBox(b, true);
    this.v.setHeightRange(b.min.z, b.max.z);
    for (const id of ["iseg-result", "istitch-result", "iexp-result", "region-info"]) $(id).textContent = "";
    if (!state.info.loaded) state.info = { loaded: false, origin: info.origin };
    this.hooks.viewPanel.resetSlab(b.min.z, b.max.z);
    $("tab-island").classList.remove("hidden");
    showTab("island");
    this._fillSummary();
    this._drawMinimapSoon();
    if (info.region) {
      this.regionBox = { min: info.region.bmin.slice(0, 2), max: info.region.bmax.slice(0, 2) };
      this._showHelper(this.regionBox);
      this.layer.setContext(this.regionBox);
      await this.hooks.loadCloudData({ frame: false });
    } else {
      state.trees = [];
      state.treeById = new Map();
      renderTrees();
      this.refreshTrees();
    }
    this.loadCandidates();
    this.hooks.updateChrome();
  }

  async leave() {
    if (this.inRegion) await postJSON("/api/region/close");
    await postJSON("/api/project/close");
    this.layer.close();
    this._removeHelper();
    this.project = null;
    this.regionBox = null;
    this.pendingBox = null;
    this._syncBoxEditor();
    state.mode = "single";
    $("tab-island").classList.add("hidden");
    showTab("trees");
    this._drawMinimapSoon();
    this.hooks.updateChrome();
  }

  // ── trees ──────────────────────────────────────────────────────────────
  async refreshTrees() {
    if (state.mode !== "island") return;
    clearTimeout(this._treesTimer);
    const r = await getJSON("/api/island/trees");
    if (state.mode !== "island") return;
    if (!r.ready) {
      $("nontree-label").textContent = "Computing island tree stats...";
      this._treesTimer = setTimeout(() => this.refreshTrees().catch(this.hooks.errorToast), 1500);
      return;
    }
    state.trees = r.trees;
    state.treeById = new Map(r.trees.map((t) => [t.id, t]));
    if (registerTreeIds(r.trees.map((t) => t.id))) {
      this.v.ensurePalette(r.trees.length ? r.trees[r.trees.length - 1].id : 0, true);
    }
    if (state.focusedTree !== null && !state.treeById.has(state.focusedTree)) this.hooks.setFocus(null);
    renderTrees();
    $("nontree-label").textContent = `${r.trees.length.toLocaleString()} trees on the island`;
    this._drawMinimapSoon();
  }

  flyToTree(id) {
    const t = state.treeById.get(id);
    if (!t || !t.bmin) { toast(`No location known for tree ${id} yet.`, "warn"); return; }
    this.v.frameBox(new THREE.Box3(new THREE.Vector3(...t.bmin), new THREE.Vector3(...t.bmax)));
  }

  frame() {
    if (state.mode === "island" && state.focusedTree !== null) this.flyToTree(state.focusedTree);
    else this.v.frameBox(this.layer.bounds());
  }

  async applyResult(res) {
    this.project = res.info;
    await this.layer.refreshLabels(res.affected_nodes ?? null);
    await this.refreshTrees();
    this.hooks.updateChrome();
  }

  async undoRedo(which) {
    await this.hooks.guarded(async () => {
      const res = await postJSON(`/api/island/${which}`);
      await this.applyResult(res);
      toast(`${which === "undo" ? "Undid" : "Redid"} ${res.changed_points.toLocaleString()} pts`);
    });
  }

  async setReviewed(id, value) {
    try {
      this.project = await postJSON("/api/island/review", { tree_id: id, reviewed: value });
      const t = state.treeById.get(id);
      if (t) t.reviewed = value;
      renderTrees();
      this.hooks.updateChrome();
    } catch (err) { this.hooks.errorToast(err); }
  }

  async mergeFocused() {
    const src = state.focusedTree;
    if (src === null) { toast("Focus a tree first (click it in the view or the table).", "warn"); return; }
    const inp = el("input", { type: "number", min: -1, step: 1, value: "" });
    const tgt = await modal({
      title: `Merge tree ${src}`,
      body: el("div", { style: "display:flex;flex-direction:column;gap:8px" },
        el("p", { text: `All points of tree ${src} (every tile) get the target ID. Tip: click the other tree in the view, note its ID.` }),
        el("label", {}, "Target tree ID (-1 = non-tree)", inp)),
      buttons: [{ label: "Cancel", value: null }, { label: "Merge", value: "ok", primary: true }],
      collect: () => {
        const n = Number(inp.value);
        if (inp.value === "" || !Number.isInteger(n) || n < -1) { inp.focus(); return undefined; }
        return n;
      },
    });
    if (tgt === null || tgt === undefined) return;
    await this.hooks.guarded(async () => {
      const res = await postJSON("/api/island/merge", { sources: [src], target: tgt });
      this.hooks.setFocus(tgt >= 0 ? tgt : null);
      await this.applyResult(res);
      toast(`Merged tree ${src} into ${tgt < 0 ? "non-tree" : tgt} (${res.changed_points.toLocaleString()} pts)`, "success");
    }, "Merging...");
  }

  async nontreeFocused() {
    const id = state.focusedTree;
    if (id === null) { toast("Focus a tree first.", "warn"); return; }
    if (!(await confirmDialog("Mark as non-tree?", `Every point of tree ${id} becomes non-tree (undoable).`, "Mark non-tree", true))) return;
    await this.hooks.guarded(async () => {
      const res = await postJSON("/api/island/nontree", { ids: [id] });
      this.hooks.setFocus(null);
      await this.applyResult(res);
      toast(`Tree ${id} marked non-tree (${res.changed_points.toLocaleString()} pts)`, "success");
    }, "Updating...");
  }

  // ── regions ────────────────────────────────────────────────────────────
  startDraw() {
    if (state.mode !== "island") return;
    this.drawing = true;
    this.v.setToolMode(true);
    $("viewport").classList.add("tool-box");
    toast("Drag a box on the ground plane. Right-drag orbits, Esc cancels.", "info");
  }

  _stopDraw() {
    this.drawing = false;
    this.drag = null;
    this.v.setToolMode(false);
    $("viewport").classList.remove("tool-box");
    for (const n of [...$("overlay").querySelectorAll(".lasso")]) n.remove();
  }

  regionAroundTree() {
    const t = state.treeById.get(state.focusedTree);
    if (!t || !t.bmin) { toast("Focus a tree first.", "warn"); return; }
    const m = Math.max(0, Number($("region-margin").value) || 0);
    this.setPendingBox({ min: [t.bmin[0] - m, t.bmin[1] - m], max: [t.bmax[0] + m, t.bmax[1] + m] });
  }

  // Point-level tools need full-resolution points: open a region around the focused tree, or the
  // area the camera is looking at. Resolves true once a region is open.
  async ensureRegion() {
    if (state.mode === "region") return true;
    if (state.mode !== "island" || state.busy) return false;
    const t = state.treeById.get(state.focusedTree);
    let box;
    if (t && t.bmin) {
      const m = Math.max(0, Number($("region-margin").value) || 0);
      box = { min: [t.bmin[0] - m, t.bmin[1] - m], max: [t.bmax[0] + m, t.bmax[1] + m] };
    } else {
      const c = this.v.controls.target;
      const half = Math.min(25, Math.max(8, 0.25 * this.v.camera.position.distanceTo(c)));
      box = { min: [c.x - half, c.y - half], max: [c.x + half, c.y + half] };
    }
    toast(t ? `Opening an edit region around tree ${t.id} for point editing...`
      : "Opening an edit region around the view centre for point editing (focus a tree first to centre on it)...", "info");
    await this.setPendingBox(box);
    await this.openRegion();
    return state.mode === "region";
  }

  async setPendingBox(box) {
    this.pendingBox = box;
    this._showHelper(box);
    this._syncBoxEditor();
    await this._estimate(box);
    this.hooks.updateChrome();
  }

  async _estimate(box) {
    const seq = (this._estSeq = (this._estSeq || 0) + 1);
    const w = box.max[0] - box.min[0], h = box.max[1] - box.min[1];
    $("region-info").textContent = `${w.toFixed(1)} x ${h.toFixed(1)} m · estimating...`;
    try {
      const r = await getJSON(`/api/project/estimate?x0=${box.min[0]}&y0=${box.min[1]}&x1=${box.max[0]}&y1=${box.max[1]}`);
      if (seq !== this._estSeq) return;
      const warn = r.upper > r.max_points ? ` (limit ${(r.max_points / 1e6).toFixed(0)}M, may be too large)` : "";
      $("region-info").textContent = `${w.toFixed(1)} x ${h.toFixed(1)} m · up to ${r.upper.toLocaleString()} pts${warn}`;
    } catch (err) { this.hooks.errorToast(err); }
  }

  // ── box editor (centre / size sliders, absolute easting / northing) ────
  _initBoxEditor() {
    const root = $("region-edit");
    root.addEventListener("input", (e) => {
      const k = e.target.dataset.k;
      if (!k || !this.pendingBox) return;
      const v = Number(e.target.value);
      if (!Number.isFinite(v)) return;
      for (const inp of root.querySelectorAll(`input[data-k="${k}"]`)) if (inp !== e.target) inp.value = e.target.value;
      const o = this.project.origin, b = this.pendingBox;
      let cx = (b.min[0] + b.max[0]) / 2, cy = (b.min[1] + b.max[1]) / 2;
      let w = b.max[0] - b.min[0], h = b.max[1] - b.min[1];
      if (k === "cx") cx = v - o[0];
      else if (k === "cy") cy = v - o[1];
      else if (k === "w") w = Math.max(1, v);
      else if (k === "h") h = Math.max(1, v);
      this._updateBox({ min: [cx - w / 2, cy - h / 2], max: [cx + w / 2, cy + h / 2] });
    });
    $("btn-region-here").addEventListener("click", () => {
      const b = this.pendingBox;
      if (!b) return;
      const t = this.v.controls.target, w = b.max[0] - b.min[0], h = b.max[1] - b.min[1];
      this._updateBox({ min: [t.x - w / 2, t.y - h / 2], max: [t.x + w / 2, t.y + h / 2] });
      this._syncBoxEditor();
    });
    $("btn-region-look").addEventListener("click", () => {
      const b = this.pendingBox;
      if (!b) return;
      const lb = this.layer.bounds();
      this.v.frameBox(new THREE.Box3(new THREE.Vector3(b.min[0], b.min[1], lb.min.z), new THREE.Vector3(b.max[0], b.max[1], lb.max.z)));
    });
    $("btn-region-clear").addEventListener("click", () => this.clearPendingBox());
  }

  // Live update while dragging a slider; the point estimate waits until the slider settles.
  _updateBox(box) {
    this.pendingBox = box;
    this._showHelper(box);
    clearTimeout(this._estTimer);
    this._estTimer = setTimeout(() => this._estimate(box), 300);
  }

  _syncBoxEditor() {
    const root = $("region-edit"), b = this.pendingBox;
    root.classList.toggle("hidden", !b || state.mode !== "island");
    if (!b) return;
    const o = this.project.origin, lb = this.layer.bounds();
    const vals = {
      cx: (b.min[0] + b.max[0]) / 2 + o[0], cy: (b.min[1] + b.max[1]) / 2 + o[1],
      w: b.max[0] - b.min[0], h: b.max[1] - b.min[1],
    };
    const ranges = { cx: [lb.min.x + o[0], lb.max.x + o[0]], cy: [lb.min.y + o[1], lb.max.y + o[1]] };
    for (const inp of root.querySelectorAll("input[data-k]")) {
      const k = inp.dataset.k;
      if (ranges[k] && inp.type === "range") {
        inp.min = Math.floor(ranges[k][0]);
        inp.max = Math.ceil(ranges[k][1]);
        inp.step = 0.5;
      }
      if (k === "w" || k === "h") {
        const cap = Math.max(150, Math.ceil(vals[k]));
        if (inp.type === "range") inp.max = cap;
      }
      inp.value = vals[k].toFixed(1);
    }
  }

  clearPendingBox() {
    this.pendingBox = null;
    this._estSeq = (this._estSeq || 0) + 1;
    clearTimeout(this._estTimer);
    if (!this.regionBox) this._removeHelper();
    $("region-info").textContent = "";
    this._syncBoxEditor();
    this._drawMinimapSoon();
    this.hooks.updateChrome();
  }

  async openRegion() {
    const box = this.pendingBox;
    if (!box || state.mode !== "island") return;
    await this.hooks.guarded(async () => {
      const t = toast("Loading region at full resolution...", "busy", 0);
      try {
        const maxDisplay = Number(localStorage.getItem("maxDisplay")) || 2000000;
        const r = await postJSON("/api/region/open", { bmin: box.min, bmax: box.max, max_display: maxDisplay });
        this.project = r.project;
        state.mode = "region";
        this.regionBox = box;
        this.pendingBox = null;
        clearTimeout(this._estTimer);
        this._estSeq = (this._estSeq || 0) + 1;
        this._syncBoxEditor();
        this.layer.setContext(box);
        await this.hooks.loadCloudData({ frame: false });
        if (this.v.bounds && !this.v.bounds.isEmpty()) this.v.frameBox(this.v.bounds);
        $("region-info").textContent = `Region open: ${r.info.num_points.toLocaleString()} pts. Edits go straight to the project.`;
        showTab("trees");
      } finally { t.close(); }
      toast("Region loaded. Use Lasso / Box / Brush as usual; Close region when done.", "success");
    }, "Loading region...");
  }

  async closeRegion() {
    if (!this.inRegion) return;
    await this.hooks.guarded(async () => {
      this.project = await postJSON("/api/region/close");
      state.mode = "island";
      this.regionBox = null;
      this._removeHelper();
      this.hooks.resetSingle();
      this.layer.setContext(null);
      $("region-info").textContent = "";
      await this.layer.refreshLabels(null);
      await this.refreshTrees();
      showTab("island");
    }, "Closing region...");
  }

  // Region edits can relabel whole trees outside the box; refresh the context lazily.
  contextChanged() {
    if (!this.inRegion) return;
    clearTimeout(this._ctxTimer);
    this._ctxTimer = setTimeout(() => this.layer.refreshLabels(null), 800);
  }

  _showHelper(box) {
    this._removeHelper();
    const b = this.layer.bounds();
    const b3 = new THREE.Box3(new THREE.Vector3(box.min[0], box.min[1], b.min.z), new THREE.Vector3(box.max[0], box.max[1], b.max.z));
    this.helper = new THREE.Box3Helper(b3, 0xffa640);
    this.v.scene.add(this.helper);
    this.v.requestRender();
    this._drawMinimapSoon();
  }

  _removeHelper() {
    if (this.helper) {
      this.v.scene.remove(this.helper);
      this.helper.geometry.dispose();
      this.helper = null;
      this.v.requestRender();
    }
  }

  _groundPoint(x, y, z) {
    const c = this.v.canvas;
    const ndc = new THREE.Vector2((x / c.clientWidth) * 2 - 1, 1 - (y / c.clientHeight) * 2);
    const rc = new THREE.Raycaster();
    rc.setFromCamera(ndc, this.v.camera);
    const out = new THREE.Vector3();
    return rc.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -z), out) ? out : null;
  }

  // ── canvas interaction (island mode only; region points use SelectionTools) ──
  _initCanvasEvents() {
    const c = this.v.canvas;
    const local = (e) => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    c.addEventListener("pointerdown", (e) => {
      if (state.mode !== "island" || e.button !== 0) return;
      const [x, y] = local(e);
      this.drag = { x0: x, y0: y, moved: false, draw: this.drawing };
      if (this.drawing) { e.preventDefault(); c.setPointerCapture(e.pointerId); }
    });
    window.addEventListener("pointermove", (e) => {
      if (state.mode !== "island") return;
      const [x, y] = local(e);
      const d = this.drag;
      if (!d) { this._scheduleHover(x, y); return; }
      if (Math.hypot(x - d.x0, y - d.y0) > 4) d.moved = true;
      if (d.draw) this._drawRect(d.x0, d.y0, x, y);
    });
    window.addEventListener("pointerup", (e) => {
      const d = this.drag;
      if (!d || state.mode !== "island") return;
      this.drag = null;
      const [x, y] = local(e);
      if (d.draw) {
        this._stopDraw();
        if (!d.moved) return;
        const z = this.v.controls.target.z;
        const pts = [[d.x0, d.y0], [x, d.y0], [x, y], [d.x0, y]].map(([a, b]) => this._groundPoint(a, b, z));
        if (pts.some((p) => !p)) { toast("Box corners miss the ground plane. Tilt the view down and retry.", "warn"); return; }
        const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
        this.setPendingBox({ min: [Math.min(...xs), Math.min(...ys)], max: [Math.max(...xs), Math.max(...ys)] });
        return;
      }
      if (!d.moved && state.tool === "navigate") {
        const hit = this.layer.pick(x, y);
        if (!hit) return;
        if (hit.label < 0) { toast("Non-tree point.", "info"); return; }
        emit("focus-tree", { id: hit.label, frame: false });
      }
    });
    c.addEventListener("dblclick", (e) => {
      if (state.mode !== "island") return;
      const [x, y] = local(e);
      const hit = this.layer.pick(x, y);
      if (hit && hit.label >= 0) emit("isolate-tree", hit.label);
    });
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.drawing) { this._stopDraw(); e.stopPropagation(); }
    }, true);
  }

  _drawRect(x0, y0, x1, y1) {
    let node = $("overlay").querySelector(".lasso");
    if (!node) {
      node = document.createElementNS(SVG_NS, "polygon");
      node.setAttribute("class", "lasso");
      $("overlay").appendChild(node);
    }
    node.setAttribute("points", [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map((p) => p.join(",")).join(" "));
  }

  _scheduleHover(x, y) {
    clearTimeout(this._hoverTimer);
    this._hoverTimer = setTimeout(() => {
      if (state.mode !== "island" || this.drag) return;
      const hit = this.layer.pick(x, y);
      const hov = $("hud-hover");
      if (!hit) { hov.textContent = ""; return; }
      const o = this.project.origin;
      const xyz = hit.pos.map((v, k) => (v + o[k]).toFixed(2)).join(", ");
      const t = state.treeById.get(hit.label);
      const txt = hit.label < 0 ? "non-tree"
        : `tree ${hit.label}` + (t ? ` · ${t.count.toLocaleString()} pts · h ${t.height.toFixed(1)} m` + (t.tiles.length > 1 ? ` · ${t.tiles.length} tiles` : "") : "");
      hov.textContent = `${txt}\nxyz ${xyz}`;
    }, 90);
  }

  // Keep depth precision usable when zooming from island scale into a single tree.
  _adaptNear() {
    const cam = this.v.camera;
    const d = cam.position.distanceTo(this.v.controls.target);
    const near = Math.max(0.02, d / 1000);
    if (Math.abs(near - cam.near) / cam.near > 0.25) {
      cam.near = near;
      cam.far = Math.max(cam.far, d * 20);
      cam.updateProjectionMatrix();
    }
  }

  // ── stream stats / summary ─────────────────────────────────────────────
  _streamStats(s) {
    $("isl-stream").textContent =
      `${(s.visiblePoints / 1e6).toFixed(2)}M pts shown in ${s.visibleNodes} nodes` +
      (s.loading ? ` · loading ${s.loading}` : "") + ` · cache ${(s.loadedPts / 1e6).toFixed(1)}M`;
  }

  _fillSummary() {
    const p = this.project;
    if (!p) return;
    const crs = (p.crs || "").match(/^[A-Z]*CRS\["([^"]+)"/)?.[1] || p.crs || "";
    $("island-summary").textContent = `${p.name}: ${p.tiles.length} tiles, ${p.num_points.toLocaleString()} pts` +
      (crs ? `\n${crs}` : "");
    $("island-summary").title = `${p.root}\n${p.crs || ""}`;
  }

  // ── panel ──────────────────────────────────────────────────────────────
  _initPanel() {
    const budget = $("isl-budget");
    const saved = Number(localStorage.getItem("pointBudget")) || 8;
    budget.value = saved;
    const setB = () => {
      const m = Number(budget.value);
      $("isl-budget-l").textContent = `${m}M`;
      localStorage.setItem("pointBudget", String(m));
      this.layer.setBudget(m * 1e6);
    };
    budget.addEventListener("input", setB);
    setB();
    $("isl-adaptive").addEventListener("change", (e) => this.layer.setAdaptive(e.target.checked));
    $("btn-region-draw").addEventListener("click", () => this.startDraw());
    this._initBoxEditor();
    $("btn-region-tree").addEventListener("click", () => this.regionAroundTree());
    $("btn-region-open").addEventListener("click", () => this.openRegion());
    $("btn-region-close").addEventListener("click", () => this.closeRegion());
    $("btn-isl-merge").addEventListener("click", () => this.mergeFocused());
    $("btn-isl-nontree").addEventListener("click", () => this.nontreeFocused());
    $("btn-isl-close").addEventListener("click", () => this.hooks.guarded(() => this.leave(), "Closing project..."));
    $("btn-iseg-run").addEventListener("click", () => this.runSegment());
    $("btn-istitch-run").addEventListener("click", () => this.runStitch());
    $("btn-iexp-run").addEventListener("click", () => this.runExport());
  }

  initMethods(methods, weights) {
    const sel = $("iseg-method");
    sel.replaceChildren(...methods.map((m) => el("option", { value: m, text: m })));
    if (methods.includes("treelearn")) sel.value = "treelearn";
    const syncVoxel = () => { $("iseg-voxel").value = sel.value === "treex" ? "0.03" : "0.05"; };
    sel.addEventListener("change", syncVoxel);
    syncVoxel();
    this._pointsamCkpt = initPointSamSelect($("iseg-ps"), sel, weights);
  }

  updateChrome() {
    const island = state.mode === "island", region = state.mode === "region";
    const busy = state.busy;
    const p = this.project;
    $("btn-region-draw").disabled = !island || busy;
    $("btn-region-tree").disabled = !island || busy || state.focusedTree === null;
    $("btn-region-open").disabled = !island || busy || !this.pendingBox;
    $("btn-region-close").classList.toggle("hidden", !region);
    $("btn-region-open").classList.toggle("hidden", region);
    $("btn-isl-merge").disabled = !island || busy || state.focusedTree === null;
    $("btn-isl-nontree").disabled = !island || busy || state.focusedTree === null;
    for (const id of ["btn-iseg-run", "btn-istitch-run", "btn-iexp-run"]) $(id).disabled = !island || busy;
    $("island-export").classList.toggle("hidden", !this.active);
    $("single-export").classList.toggle("hidden", island);
    $("iexp-focused").disabled = state.focusedTree === null;
    $("cand-count").textContent = p && this.candidates.length ? `${this.candidates.length} open` : "";
    this._renderCandidates();
    this._drawMinimapSoon();
  }

  // ── island jobs ────────────────────────────────────────────────────────
  async runSegment() {
    const method = $("iseg-method").value;
    const ok = await confirmDialog("Segment the whole island?",
      `Runs ${method} tile by tile with a ${$("iseg-buffer").value} m buffer, then stitches trees across borders. ` +
      "This replaces all island labels (a backup of the old labels is kept in the project) and can take a long time.",
      "Run", true);
    if (!ok) return;
    const body = {
      method, buffer: Number($("iseg-buffer").value) || 10, voxel: Number($("iseg-voxel").value) || null,
      leaf_removal: $("iseg-leaf").checked, pointsam_ckpt: this._pointsamCkpt?.() ?? null,
    };
    await this.hooks.runJob("/api/island/segment", body, `Island segmentation (${method})`, "iseg", async (done) => {
      this.project = await getJSON("/api/project/status");
      this.hooks.setFocus(null);
      await this.layer.refreshLabels(null);
      await this.refreshTrees();
      await this.loadCandidates();
      toast(`Island segmentation: ${done.message}`, "success");
    });
  }

  async runStitch() {
    await this.hooks.runJob("/api/island/stitch", {}, "Finding border merges", "istitch", async (done) => {
      await this.loadCandidates();
      toast(done.message, "success");
    });
  }

  async runExport() {
    const onlyFocused = $("iexp-focused").checked && state.focusedTree !== null;
    const body = {
      out_dir: $("iexp-out").value.trim(),
      write_tiles: $("iexp-tiles").checked,
      tree_ids: onlyFocused ? [state.focusedTree] : null,
    };
    await this.hooks.runJob("/api/island/export", body, "Exporting island", "iexp", async (done) => {
      $("iexp-out").value = done.result?.out_dir || body.out_dir;
      toast(done.message, "success");
    });
  }

  async loadCandidates() {
    try {
      const r = await getJSON("/api/island/candidates");
      this.candidates = r.candidates || [];
    } catch { this.candidates = []; }
    this.hooks.updateChrome();
  }

  _renderCandidates() {
    const list = $("cand-list");
    const island = state.mode === "island";
    const rows = this.candidates.slice(0, 300).map((c) => el("div", { class: "cand" },
      el("span", { class: "swatch", style: `background:${cssColor(c.a)}` }),
      el("span", { text: `${c.a} + ${c.b}` }),
      el("span", { class: "muted", text: `score ${Number(c.score).toFixed(2)}` }),
      el("span", { class: "spacer" }),
      el("button", { class: "btn small", text: "Fly", onclick: () => this.flyToCandidate(c) }),
      el("button", { class: "btn small primary", text: "Merge", disabled: !island || state.busy, onclick: () => this.resolveCandidate(c, true) }),
      el("button", { class: "btn small", text: "Reject", disabled: !island || state.busy, onclick: () => this.resolveCandidate(c, false) }),
    ));
    if (this.candidates.length > 300) rows.push(el("div", { class: "muted", text: `... ${this.candidates.length - 300} more` }));
    list.replaceChildren(...rows);
  }

  flyToCandidate(c) {
    const ta = state.treeById.get(c.a), tb = state.treeById.get(c.b);
    const box = new THREE.Box3();
    for (const t of [ta, tb]) if (t?.bmin) { box.expandByPoint(new THREE.Vector3(...t.bmin)); box.expandByPoint(new THREE.Vector3(...t.bmax)); }
    if (box.isEmpty() && c.location) {
      const p = new THREE.Vector3(...c.location);
      box.setFromCenterAndSize(p, new THREE.Vector3(15, 15, 15));
    }
    if (!box.isEmpty()) this.v.frameBox(box);
    this.hooks.setFocus(c.a);
  }

  async resolveCandidate(c, accept) {
    await this.hooks.guarded(async () => {
      const res = await postJSON(`/api/island/candidates/${encodeURIComponent(c.key)}/${accept ? "accept" : "reject"}`);
      if (accept) await this.applyResult(res);
      else this.project = res.info;
      await this.loadCandidates();
      if (accept) toast(`Merged ${c.b} into ${c.a}`, "success");
    }, accept ? "Merging..." : "Updating...");
  }

  // ── mini-map ───────────────────────────────────────────────────────────
  _initMinimap() {
    const cv = $("minimap");
    cv.addEventListener("click", (e) => {
      const m = this._mm;
      if (!m) return;
      const r = cv.getBoundingClientRect();
      const x = m.x0 + (e.clientX - r.left - m.pad) / m.s;
      const y = m.y1 - (e.clientY - r.top - m.pad) / m.s;
      const t = this.v.controls.target, cam = this.v.camera;
      const dx = x - t.x, dy = y - t.y;
      t.x += dx; t.y += dy;
      cam.position.x += dx; cam.position.y += dy;
      this.v.controls.update();
      this.v._camChanged();
    });
  }

  _drawMinimapSoon() {
    if (this._mmPending) return;
    this._mmPending = true;
    requestAnimationFrame(() => { this._mmPending = false; this._drawMinimap(); });
  }

  _drawMinimap() {
    const cv = $("minimap");
    cv.classList.toggle("hidden", !this.active || !this.project);
    if (!this.active || !this.project) return;
    const W = cv.clientWidth, H = cv.clientHeight, dpr = window.devicePixelRatio || 1;
    if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const p = this.project, pad = 6;
    const x0 = p.bmin[0], y0 = p.bmin[1], x1 = p.bmax[0], y1 = p.bmax[1];
    const s = Math.min((W - 2 * pad) / Math.max(x1 - x0, 1), (H - 2 * pad) / Math.max(y1 - y0, 1));
    this._mm = { x0, y1, s, pad };
    const X = (x) => pad + (x - x0) * s, Y = (y) => pad + (y1 - y) * s;
    g.strokeStyle = "#3a4150";
    g.lineWidth = 1;
    for (const t of p.tiles) g.strokeRect(X(t.bmin[0]), Y(t.bmax[1]), (t.bmax[0] - t.bmin[0]) * s, (t.bmax[1] - t.bmin[1]) * s);
    for (const t of state.mode === "island" ? state.trees : []) {
      if (!t.base) continue;
      g.fillStyle = t.id === state.focusedTree ? "#ffffff" : cssColor(t.id);
      const r = t.id === state.focusedTree ? 3 : 1.2;
      g.fillRect(X(t.base[0]) - r, Y(t.base[1]) - r, 2 * r, 2 * r);
    }
    const box = this.regionBox || this.pendingBox;
    if (box) {
      g.strokeStyle = "#ffa640";
      g.strokeRect(X(box.min[0]), Y(box.max[1]), (box.max[0] - box.min[0]) * s, (box.max[1] - box.min[1]) * s);
    }
    // Camera footprint on the target's ground plane.
    const c = this.v.canvas, z = this.v.controls.target.z;
    const corners = [[0, 0], [c.clientWidth, 0], [c.clientWidth, c.clientHeight], [0, c.clientHeight]]
      .map(([a, b]) => this._groundPoint(a, b, z));
    g.strokeStyle = "#7fb2ff";
    if (corners.every(Boolean)) {
      g.beginPath();
      corners.forEach((q, k) => (k ? g.lineTo(X(q.x), Y(q.y)) : g.moveTo(X(q.x), Y(q.y))));
      g.closePath();
      g.stroke();
    }
    const t = this.v.controls.target;
    g.fillStyle = "#7fb2ff";
    g.beginPath();
    g.arc(X(t.x), Y(t.y), 3, 0, 2 * Math.PI);
    g.fill();
  }
}

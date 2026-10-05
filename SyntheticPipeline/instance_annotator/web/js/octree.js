// Streaming octree layer (Potree-style) for island projects.
// Nodes are fetched from /api/node/{gid} (xyz float32 + labels int32, project-local frame).
// Node 0 is a coarse island-wide preview whose children are the per-tile roots.
import * as THREE from "three";
import { VERT, FRAG, extraUniforms } from "./pointshader.js";
import { state, on } from "./state.js";

const MAX_CONCURRENT = 6;
const MIN_PX = 60;          // stop refining when a node's projected radius is below this
const CACHE_FACTOR = 1.6;   // loaded points kept (visible + cached) relative to the budget

export class OctreeLayer {
  constructor(viewer) {
    this.v = viewer;
    this.group = new THREE.Group();
    viewer.scene.add(this.group);
    this.uniforms = { ...viewer.uniforms, ...extraUniforms() };
    this.material = new THREE.ShaderMaterial({
      uniforms: this.uniforms, vertexShader: VERT, fragmentShader: FRAG,
      transparent: false, depthWrite: true,
    });
    this.budget = 8_000_000;
    this.adaptive = true;
    this.h = null;
    this.entries = new Map();   // gid -> {state, points, geom, labelAttr, n, lastUsed, leafish}
    this.visible = new Set();
    this.queue = [];
    this.loading = 0;
    this.frame = 0;
    this.visiblePoints = 0;
    this.onStats = null;
    this._pending = false;
    this._generation = 0;
    this._frustum = new THREE.Frustum();
    this._box = new THREE.Box3();
    this._m = new THREE.Matrix4();
    on("camera", () => this.schedule());
  }

  get active() { return !!this.h; }

  open(hier) {
    this.close();
    const N = hier.n.length;
    this.h = {
      N,
      n: Int32Array.from(hier.n),
      tile: Int32Array.from(hier.tile),
      level: Int32Array.from(hier.level),
      bmin: Float32Array.from(hier.bmin),
      bmax: Float32Array.from(hier.bmax),
      spacing: Float32Array.from(hier.spacing),
      childStart: Int32Array.from(hier.child_start),
      children: Int32Array.from(hier.children),
    };
    this.group.visible = true;
    this._load(0);
    this.schedule();
  }

  close() {
    this._generation++;
    for (const e of this.entries.values()) this._dispose(e);
    this.entries.clear();
    this.visible.clear();
    this.queue = [];
    this.h = null;
    this.visiblePoints = 0;
    this.setContext(null);
    this.v.requestRender();
  }

  bounds() {
    const h = this.h;
    if (!h) return null;
    return new THREE.Box3(new THREE.Vector3(h.bmin[0], h.bmin[1], h.bmin[2]), new THREE.Vector3(h.bmax[0], h.bmax[1], h.bmax[2]));
  }

  setBudget(n) { this.budget = n; this.schedule(); }
  setAdaptive(on_) { this.adaptive = on_; this.v.requestRender(); }
  setVisible(vis) { this.group.visible = vis; this.v.requestRender(); }

  // Dim everything and hide points inside the xy box (the edit region is drawn by the viewer).
  setContext(box) {
    const u = this.uniforms;
    u.uDim.value = box ? 1 : 0;
    u.uClipOn.value = box ? 1 : 0;
    if (box) {
      u.uClipMin.value.set(box.min[0], box.min[1]);
      u.uClipMax.value.set(box.max[0], box.max[1]);
    }
    this.v.requestRender();
  }

  applyView() {
    const transparent = state.view.opacity < 0.999;
    if (this.material.transparent !== transparent) {
      this.material.transparent = transparent;
      this.material.depthWrite = !transparent;
      this.material.needsUpdate = true;
    }
    this.v.requestRender();
  }

  schedule() {
    if (this._pending || !this.h) return;
    this._pending = true;
    requestAnimationFrame(() => this._update());
  }

  _nodeBox(id) {
    const h = this.h, b = this._box;
    b.min.set(h.bmin[3 * id], h.bmin[3 * id + 1], h.bmin[3 * id + 2]);
    b.max.set(h.bmax[3 * id], h.bmax[3 * id + 1], h.bmax[3 * id + 2]);
    return b;
  }

  _priority(id, camPos, scale) {
    const h = this.h;
    const cx = (h.bmin[3 * id] + h.bmax[3 * id]) / 2, cy = (h.bmin[3 * id + 1] + h.bmax[3 * id + 1]) / 2, cz = (h.bmin[3 * id + 2] + h.bmax[3 * id + 2]) / 2;
    const ex = h.bmax[3 * id] - h.bmin[3 * id], ey = h.bmax[3 * id + 1] - h.bmin[3 * id + 1], ez = h.bmax[3 * id + 2] - h.bmin[3 * id + 2];
    const r = Math.max(0.5 * Math.hypot(ex, ey, ez), 0.01);
    const d = Math.hypot(camPos.x - cx, camPos.y - cy, camPos.z - cz);
    if (d <= r) return 1e9;
    return (r / d) * scale;
  }

  _update() {
    this._pending = false;
    const h = this.h;
    if (!h) return;
    this.frame++;
    const cam = this.v.camera;
    cam.updateMatrixWorld();
    this._m.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    this._frustum.setFromProjectionMatrix(this._m);
    const scale = this.v.uniforms.uScale.value;
    const camPos = cam.position;

    // Best-first traversal by projected size under the point budget.
    const heap = new MaxHeap();
    for (let k = h.childStart[0]; k < h.childStart[1]; k++) {
      const c = h.children[k];
      if (this._frustum.intersectsBox(this._nodeBox(c))) heap.push(c, this._priority(c, camPos, scale));
    }
    const vis = [];
    let pts = 0;
    while (heap.size) {
      const [id] = heap.pop();
      if (pts + h.n[id] > this.budget) continue;
      vis.push(id);
      pts += h.n[id];
      for (let k = h.childStart[id]; k < h.childStart[id + 1]; k++) {
        const c = h.children[k];
        if (!this._frustum.intersectsBox(this._nodeBox(c))) continue;
        const pr = this._priority(c, camPos, scale);
        if (pr >= MIN_PX) heap.push(c, pr);
      }
    }
    const visSet = new Set(vis);
    this.visible = visSet;
    this.queue = [];
    let shown = 0, rootsReady = true;
    for (const id of vis) {
      const e = this.entries.get(id);
      if (e && e.state === "loaded") {
        e.lastUsed = this.frame;
        shown += e.n;
      } else {
        if (!e) this.queue.push(id);
        if (h.level[id] === 0) rootsReady = false;
      }
    }
    for (const [id, e] of this.entries) {
      if (!e.points) continue;
      if (id === 0) { e.points.visible = !rootsReady || vis.length === 0; continue; }
      e.points.visible = visSet.has(id);
      if (e.points.visible) {
        let childShown = false;
        for (let k = h.childStart[id]; k < h.childStart[id + 1]; k++) {
          const c = h.children[k];
          if (visSet.has(c) && this.entries.get(c)?.state === "loaded") { childShown = true; break; }
        }
        e.leafish = !childShown;
      }
    }
    this.visiblePoints = shown;
    this._pump();
    this._evict();
    this.v.requestRender();
    this.onStats?.(this.stats());
  }

  stats() {
    let loadedPts = 0, loadedNodes = 0;
    for (const e of this.entries.values()) if (e.state === "loaded") { loadedPts += e.n; loadedNodes++; }
    return { visiblePoints: this.visiblePoints, visibleNodes: this.visible.size, loading: this.loading + this.queue.length, loadedPts, loadedNodes };
  }

  _pump() {
    while (this.loading < MAX_CONCURRENT && this.queue.length) {
      const id = this.queue.shift();
      if (!this.entries.has(id)) this._load(id);
    }
  }

  async _load(id) {
    const gen = this._generation;
    const entry = { state: "loading", n: 0, points: null, lastUsed: this.frame, leafish: true };
    this.entries.set(id, entry);
    this.loading++;
    try {
      const r = await fetch(`/api/node/${id}`);
      if (!r.ok) throw new Error(`node ${id}: HTTP ${r.status}`);
      const buf = await r.arrayBuffer();
      if (gen !== this._generation) return;
      const n = buf.byteLength / 16;
      const pos = new Float32Array(buf, 0, 3 * n);
      const lab = new Int32Array(buf, 12 * n, n);
      this._build(id, entry, pos, lab);
    } catch (err) {
      console.warn(err);
      if (gen === this._generation) this.entries.delete(id);
    } finally {
      if (gen === this._generation) {
        this.loading--;
        this.schedule();
      }
    }
  }

  _build(id, entry, pos, lab) {
    const n = lab.length;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pos), 3));
    const labelAttr = new THREE.BufferAttribute(Float32Array.from(lab), 1);
    labelAttr.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute("label", labelAttr);
    // Constant attributes: no selection / material info in streamed nodes.
    g.setAttribute("selected", new THREE.BufferAttribute(new Float32Array(n), 1));
    g.setAttribute("material", new THREE.BufferAttribute(new Float32Array(n), 1));
    const pts = new THREE.Points(g, this.material);
    pts.frustumCulled = false;
    pts.visible = false;
    // True leaves hold every remaining point, so the grid spacing overstates their gaps.
    const isLeaf = this.h.childStart[id + 1] === this.h.childStart[id];
    const spacing = isLeaf ? 0 : this.h.spacing[id];
    pts.onBeforeRender = (renderer, scene, camera, geometry, material) => {
      this.uniforms.uSpacing.value = this.adaptive && entry.leafish ? spacing : 0;
      material.uniformsNeedUpdate = true;
    };
    let mx = 0;
    for (let i = 0; i < n; i++) if (lab[i] > mx) mx = lab[i];
    this.v.ensurePalette(mx);
    Object.assign(entry, { state: "loaded", n, points: pts, geom: g, labelAttr, pos: g.attributes.position.array });
    this.group.add(pts);
  }

  _dispose(e) {
    if (e.points) {
      this.group.remove(e.points);
      e.geom.dispose();
    }
    e.points = null;
    e.state = "disposed";
  }

  _evict() {
    let total = 0;
    const cand = [];
    for (const [id, e] of this.entries) {
      if (e.state !== "loaded") continue;
      total += e.n;
      if (id !== 0 && !this.visible.has(id)) cand.push([e.lastUsed, id, e]);
    }
    const cap = this.budget * CACHE_FACTOR;
    if (total <= cap) return;
    cand.sort((a, b) => a[0] - b[0]);
    for (const [, id, e] of cand) {
      if (total <= cap) break;
      total -= e.n;
      this._dispose(e);
      this.entries.delete(id);
    }
  }

  // Re-fetch labels for the given nodes (null = all). Hidden cached nodes are dropped instead.
  async refreshLabels(ids = null) {
    const gen = this._generation;
    const want = ids === null ? [...this.entries.keys()] : ids.filter((id) => this.entries.has(id));
    const jobs = [];
    for (const id of want) {
      const e = this.entries.get(id);
      if (!e || e.state !== "loaded") {
        if (e) this.entries.delete(id);
        continue;
      }
      if (id !== 0 && !this.visible.has(id)) {
        this._dispose(e);
        this.entries.delete(id);
        continue;
      }
      jobs.push([id, e]);
    }
    let k = 0;
    const worker = async () => {
      while (k < jobs.length) {
        const [id, e] = jobs[k++];
        const r = await fetch(`/api/node/${id}/labels`);
        if (!r.ok || gen !== this._generation || e.state !== "loaded") continue;
        const lab = new Int32Array(await r.arrayBuffer());
        if (lab.length !== e.n) continue;
        let mx = 0;
        for (let i = 0; i < lab.length; i++) if (lab[i] > mx) mx = lab[i];
        this.v.ensurePalette(mx);
        e.labelAttr.array.set(lab);
        e.labelAttr.needsUpdate = true;
        this.v.requestRender();
      }
    };
    await Promise.all(Array.from({ length: MAX_CONCURRENT }, worker));
    this.schedule();
  }

  // Front-most visible streamed point near the cursor: {label, pos:[x,y,z]} or null.
  pick(x, y, radiusPx = 7) {
    if (!this.h) return null;
    const cam = this.v.camera;
    cam.updateMatrixWorld();
    const m = new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse).elements;
    const w = this.v.canvas.clientWidth, hgt = this.v.canvas.clientHeight;
    const nx = (x / w) * 2 - 1, ny = 1 - (y / hgt) * 2;
    const rx = (2 * radiusPx) / w, ry = (2 * radiusPx) / hgt;
    const r2 = radiusPx * radiusPx;
    const v = state.view, u = this.uniforms;
    const clip = u.uClipOn.value > 0.5, cmin = u.uClipMin.value, cmax = u.uClipMax.value;
    const zmin = Number.isFinite(v.zmin) ? v.zmin : -1e9, zmax = Number.isFinite(v.zmax) ? v.zmax : 1e9;
    const iso = v.isolate && state.focusedTree !== null;
    let best = null, bestD = Infinity;
    for (const [id, e] of this.entries) {
      if (!e.points || !e.points.visible) continue;
      if (!this._screenOverlap(id, m, nx, ny, rx, ry)) continue;
      const p = e.pos, lab = e.labelAttr.array;
      for (let i = 0; i < e.n; i++) {
        const px = p[3 * i], py = p[3 * i + 1], pz = p[3 * i + 2];
        const cw = m[3] * px + m[7] * py + m[11] * pz + m[15];
        if (cw <= 1e-6 || cw >= bestD) continue;
        const cx = (m[0] * px + m[4] * py + m[8] * pz + m[12]) / cw;
        const cy = (m[1] * px + m[5] * py + m[9] * pz + m[13]) / cw;
        const dx = (cx - nx) * w / 2, dy = (cy - ny) * hgt / 2;
        if (dx * dx + dy * dy > r2) continue;
        const l = lab[i];
        if (pz < zmin || pz > zmax) continue;
        if (v.hideNontree && l < 0) continue;
        if (iso && l !== state.focusedTree) continue;
        if (clip && px >= cmin.x && px <= cmax.x && py >= cmin.y && py <= cmax.y) continue;
        bestD = cw;
        best = { label: l, pos: [px, py, pz] };
      }
    }
    return best;
  }

  _screenOverlap(id, m, nx, ny, rx, ry) {
    const h = this.h;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let c = 0; c < 8; c++) {
      const px = c & 1 ? h.bmax[3 * id] : h.bmin[3 * id];
      const py = c & 2 ? h.bmax[3 * id + 1] : h.bmin[3 * id + 1];
      const pz = c & 4 ? h.bmax[3 * id + 2] : h.bmin[3 * id + 2];
      const cw = m[3] * px + m[7] * py + m[11] * pz + m[15];
      if (cw <= 1e-6) return true;  // box crosses the camera plane
      const cx = (m[0] * px + m[4] * py + m[8] * pz + m[12]) / cw;
      const cy = (m[1] * px + m[5] * py + m[9] * pz + m[13]) / cw;
      if (cx < x0) x0 = cx; if (cx > x1) x1 = cx;
      if (cy < y0) y0 = cy; if (cy > y1) y1 = cy;
    }
    return nx >= x0 - rx && nx <= x1 + rx && ny >= y0 - ry && ny <= y1 + ry;
  }
}

class MaxHeap {
  constructor() { this.k = []; this.p = []; }
  get size() { return this.k.length; }
  push(key, pr) {
    const k = this.k, p = this.p;
    let i = k.length;
    k.push(key); p.push(pr);
    while (i > 0) {
      const j = (i - 1) >> 1;
      if (p[j] >= p[i]) break;
      [k[i], k[j]] = [k[j], k[i]]; [p[i], p[j]] = [p[j], p[i]];
      i = j;
    }
  }
  pop() {
    const k = this.k, p = this.p;
    const top = [k[0], p[0]];
    const lk = k.pop(), lp = p.pop();
    if (k.length) {
      k[0] = lk; p[0] = lp;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < k.length && p[l] > p[m]) m = l;
        if (r < k.length && p[r] > p[m]) m = r;
        if (m === i) break;
        [k[i], k[m]] = [k[m], k[i]]; [p[i], p[m]] = [p[m], p[i]];
        i = m;
      }
    }
    return top;
  }
}

// three.js point-cloud viewer with per-point label / selection / material attributes.
import * as THREE from "three";
import { OrbitControls } from "/vendor/OrbitControls.js";
import { state, emit, treeColor } from "./state.js";

const PAL_W = 1024;
// IDs beyond this wrap around the palette (colors repeat).
const PAL_MAX = PAL_W * 1024;

const VERT = /* glsl */ `
uniform float uSize;
uniform float uPixelRatio;
uniform float uAtten;
uniform float uScale;
uniform float uMode;        // 0 instance, 1 material
uniform float uFocus;
uniform float uFocusOn;
uniform float uIsolate;
uniform float uHideNT;
uniform float uZmin;
uniform float uZmax;
uniform float uShowWood;
uniform float uShowLeaf;
uniform sampler2D uPalette;
uniform vec2 uPalSize;
attribute float label;
attribute float selected;
attribute float material;
varying vec3 vColor;

vec3 treeColor(float id) {
  if (id < 0.0) return vec3(0.42, 0.42, 0.45);
  float id2 = mod(id, uPalSize.x * uPalSize.y);
  vec2 uv = vec2((mod(id2, uPalSize.x) + 0.5) / uPalSize.x, (floor(id2 / uPalSize.x) + 0.5) / uPalSize.y);
  return texture2D(uPalette, uv).rgb;
}

void main() {
  bool isFocus = uFocusOn > 0.5 && abs(label - uFocus) < 0.5;
  bool hide = (uHideNT > 0.5 && label < 0.0)
    || (uIsolate > 0.5 && uFocusOn > 0.5 && !isFocus)
    || position.z < uZmin || position.z > uZmax
    || (material > 0.5 && material < 1.5 && uShowWood < 0.5)
    || (material > 1.5 && uShowLeaf < 0.5);
  if (hide) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    return;
  }
  vec3 col;
  if (uMode > 0.5) {
    col = material > 1.5 ? vec3(0.18, 0.55, 0.34) : (material > 0.5 ? vec3(0.55, 0.35, 0.17) : vec3(0.45));
  } else {
    col = treeColor(label);
  }
  if (uFocusOn > 0.5 && uIsolate < 0.5 && uMode < 0.5 && !isFocus) col = mix(col, vec3(0.12), 0.65);
  if (selected > 0.5) col = vec3(1.0, 0.25, 0.85);
  vColor = col;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float sz = uSize * uPixelRatio;
  if (uAtten > 0.5) sz = uSize * uScale / max(-mv.z, 0.01);
  if (selected > 0.5) sz = max(sz, 2.0 * uPixelRatio) * 1.15;
  gl_PointSize = clamp(sz, 1.0, 64.0);
}
`;

const FRAG = /* glsl */ `
uniform float uOpacity;
varying vec3 vColor;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  if (dot(d, d) > 0.25) discard;
  gl_FragColor = vec4(vColor, uOpacity);
}
`;

export class Viewer {
  constructor(host) {
    this.host = host;
    this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x15171c);
    host.appendChild(this.renderer.domElement);
    this.canvas = this.renderer.domElement;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 5000);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(30, -30, 20);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = false;
    this.controls.zoomToCursor = true;
    this.controls.screenSpacePanning = true;
    this.controls.addEventListener("change", () => this._camChanged());

    this.uniforms = {
      uSize: { value: 2 }, uPixelRatio: { value: this.renderer.getPixelRatio() },
      uAtten: { value: 0 }, uScale: { value: 400 }, uMode: { value: 0 },
      uFocus: { value: -1 }, uFocusOn: { value: 0 }, uIsolate: { value: 0 },
      uHideNT: { value: 0 }, uZmin: { value: -1e9 }, uZmax: { value: 1e9 },
      uShowWood: { value: 1 }, uShowLeaf: { value: 1 }, uOpacity: { value: 1 },
      uPalette: { value: null }, uPalSize: { value: new THREE.Vector2(PAL_W, 1) },
    };
    this._palRows = 0;
    this.ensurePalette(0);
    this.material = new THREE.ShaderMaterial({
      uniforms: this.uniforms, vertexShader: VERT, fragmentShader: FRAG,
      transparent: false, depthWrite: true,
    });
    this.points = null;
    this.geom = null;
    this._proj = null;
    this._projKey = -1;
    this._camVersion = 0;
    this._visVersion = 0;
    this._needsRender = true;
    this.bounds = null;

    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    const loop = () => {
      if (this._needsRender) {
        this._needsRender = false;
        this.renderer.render(this.scene, this.camera);
      }
      requestAnimationFrame(loop);
    };
    loop();
  }

  _camChanged() {
    this._camVersion++;
    this.requestRender();
    emit("camera");
  }

  requestRender() { this._needsRender = true; }

  // Grow the label -> color texture so it covers maxLabel.
  ensurePalette(maxLabel, force = false) {
    const need = Math.max(1, Math.ceil((Math.min(maxLabel, PAL_MAX - 1) + 1) / PAL_W));
    if (need <= this._palRows && !force) return;
    const rows = need <= this._palRows ? this._palRows : Math.max(need, this._palRows * 2, 1);
    const data = new Uint8Array(PAL_W * rows * 4);
    for (let id = 0; id < PAL_W * rows; id++) {
      const [r, g, b] = treeColor(id);
      data[4 * id] = r * 255; data[4 * id + 1] = g * 255; data[4 * id + 2] = b * 255; data[4 * id + 3] = 255;
    }
    const tex = new THREE.DataTexture(data, PAL_W, rows, THREE.RGBAFormat);
    tex.magFilter = tex.minFilter = THREE.NearestFilter;
    tex.needsUpdate = true;
    this.uniforms.uPalette.value?.dispose();
    this.uniforms.uPalette.value = tex;
    this.uniforms.uPalSize.value.set(PAL_W, rows);
    this._palRows = rows;
    this.requestRender();
  }

  _maxLabel(arr) {
    let m = 0;
    for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
    return m;
  }

  resize() {
    const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.uniforms.uScale.value = h / (2 * Math.tan((this.camera.fov * Math.PI) / 360));
    this._camVersion++;
    this.requestRender();
  }

  setCloud(positions, labels) {
    if (this.points) {
      this.scene.remove(this.points);
      this.geom.dispose();
    }
    const n = labels.length;
    this.ensurePalette(this._maxLabel(labels));
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.labelAttr = new THREE.BufferAttribute(Float32Array.from(labels), 1);
    this.labelAttr.setUsage(THREE.DynamicDrawUsage);
    this.selAttr = new THREE.BufferAttribute(state.selection, 1);
    this.selAttr.setUsage(THREE.DynamicDrawUsage);
    this.matAttr = new THREE.BufferAttribute(state.material, 1);
    g.setAttribute("label", this.labelAttr);
    g.setAttribute("selected", this.selAttr);
    g.setAttribute("material", this.matAttr);
    g.computeBoundingSphere();
    g.computeBoundingBox();
    this.geom = g;
    this.points = new THREE.Points(g, this.material);
    this.points.frustumCulled = false;
    this.scene.add(this.points);
    this.bounds = g.boundingBox.clone();
    this.n = n;
    this._visVersion++;
    this.frameBox(this.bounds, true);
  }

  // Patch a subset of labels (display indices -> new label).
  patchLabels(idx, labs) {
    const a = this.labelAttr.array;
    let mx = 0;
    for (let k = 0; k < idx.length; k++) { a[idx[k]] = labs[k]; if (labs[k] > mx) mx = labs[k]; }
    this.ensurePalette(mx);
    this.labelAttr.needsUpdate = true;
    this._visVersion++;
    this.requestRender();
  }

  setAllLabels(labels) {
    this.ensurePalette(this._maxLabel(labels));
    this.labelAttr.array.set(labels);
    this.labelAttr.needsUpdate = true;
    this._visVersion++;
    this.requestRender();
  }

  selectionChanged() {
    if (!this.selAttr) return;
    this.selAttr.needsUpdate = true;
    this.requestRender();
  }

  materialChanged() {
    if (!this.matAttr) return;
    this.matAttr.needsUpdate = true;
    this._visVersion++;
    this.requestRender();
  }

  applyView() {
    const v = state.view, u = this.uniforms;
    u.uSize.value = v.size;
    u.uOpacity.value = v.opacity;
    u.uAtten.value = v.atten ? 1 : 0;
    u.uMode.value = v.colorMode;
    u.uHideNT.value = v.hideNontree ? 1 : 0;
    u.uIsolate.value = v.isolate ? 1 : 0;
    u.uZmin.value = Number.isFinite(v.zmin) ? v.zmin : -1e9;
    u.uZmax.value = Number.isFinite(v.zmax) ? v.zmax : 1e9;
    u.uShowWood.value = v.showWood ? 1 : 0;
    u.uShowLeaf.value = v.showLeaf ? 1 : 0;
    u.uFocusOn.value = state.focusedTree === null ? 0 : 1;
    u.uFocus.value = state.focusedTree === null ? -1 : state.focusedTree;
    const transparent = v.opacity < 0.999;
    if (this.material.transparent !== transparent) {
      this.material.transparent = transparent;
      this.material.depthWrite = !transparent;
      this.material.needsUpdate = true;
    }
    this._visVersion++;
    this.requestRender();
  }

  // JS mirror of the vertex-shader visibility test.
  isVisible(i) {
    const v = state.view;
    const lab = state.labels[i];
    if (v.hideNontree && lab < 0) return false;
    if (v.isolate && state.focusedTree !== null && lab !== state.focusedTree) return false;
    const z = state.positions[3 * i + 2];
    if (z < v.zmin || z > v.zmax) return false;
    const m = state.material[i];
    if (m === 1 && !v.showWood) return false;
    if (m === 2 && !v.showLeaf) return false;
    return true;
  }

  // Screen-space projection of all visible points (cached per camera/visibility).
  project() {
    const key = this._camVersion * 1e6 + this._visVersion;
    if (this._proj && this._projKey === key) return this._proj;
    const n = state.n;
    if (!this._proj || this._proj.sx.length !== n) {
      this._proj = { sx: new Float32Array(n), sy: new Float32Array(n), depth: new Float32Array(n) };
    }
    const { sx, sy, depth } = this._proj;
    this.camera.updateMatrixWorld();
    const m = new THREE.Matrix4().multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse).elements;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    const p = state.positions;
    for (let i = 0; i < n; i++) {
      if (!this.isVisible(i)) { depth[i] = NaN; continue; }
      const x = p[3 * i], y = p[3 * i + 1], z = p[3 * i + 2];
      const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (cw <= 1e-6) { depth[i] = NaN; continue; }
      const cx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / cw;
      const cy = (m[1] * x + m[5] * y + m[9] * z + m[13]) / cw;
      if (cx < -1.05 || cx > 1.05 || cy < -1.05 || cy > 1.05) { depth[i] = NaN; continue; }
      sx[i] = (cx * 0.5 + 0.5) * w;
      sy[i] = (1 - (cy * 0.5 + 0.5)) * h;
      depth[i] = cw;
    }
    this._projKey = key;
    return this._proj;
  }

  // Nearest visible point to the cursor (front-most within radiusPx).
  pick(x, y, radiusPx = 7) {
    const { sx, sy, depth } = this.project();
    const r2 = radiusPx * radiusPx;
    let best = -1, bestD = Infinity;
    for (let i = 0; i < state.n; i++) {
      const d = depth[i];
      if (!(d === d)) continue;
      const dx = sx[i] - x, dy = sy[i] - y;
      if (dx * dx + dy * dy > r2) continue;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  frameBox(box, resetDirection = false) {
    if (!box || box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(size.length() / 2, 0.5);
    const dist = radius / Math.sin((this.camera.fov * Math.PI) / 360) * 1.05;
    let dir;
    if (resetDirection) dir = new THREE.Vector3(1, -1, 0.7).normalize();
    else dir = this.camera.position.clone().sub(this.controls.target).normalize();
    this.controls.target.copy(center);
    this.camera.position.copy(center.clone().add(dir.multiplyScalar(dist)));
    this.camera.near = Math.max(dist / 2000, 0.01);
    this.camera.far = dist * 20 + radius * 4;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this._camChanged();
  }

  frameIndices(pred) {
    const p = state.positions;
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    let any = false;
    for (let i = 0; i < state.n; i++) {
      if (!pred(i)) continue;
      v.set(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
      box.expandByPoint(v);
      any = true;
    }
    if (any) this.frameBox(box);
    return any;
  }

  frameAll() { this.frameBox(this.bounds); }

  // Left button behaviour: orbit in navigate mode; tools take left, right orbits, middle pans.
  setToolMode(isTool) {
    const M = THREE.MOUSE;
    this.controls.mouseButtons = isTool
      ? { LEFT: -1, MIDDLE: M.PAN, RIGHT: M.ROTATE }
      : { LEFT: M.ROTATE, MIDDLE: M.DOLLY, RIGHT: M.PAN };
  }
}

// Local terrain model from the loaded (display) points, used to drop ground from a selection.
import { state, recountSelection, emit } from "./state.js";

const CELL = 1.0;        // m
const WIN = 2;           // neighbourhood radius in cells for canopy-only / empty cells
const MAX_SLOPE = 0.5;   // m rise per m; cells whose lowest point is higher above their
                         // neighbourhood than this allows hold no ground returns

const cache = new WeakMap();

function buildDtm(p, n) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = p[3 * i], y = p[3 * i + 1];
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  const nx = Math.max(1, Math.ceil((x1 - x0) / CELL) + 1), ny = Math.max(1, Math.ceil((y1 - y0) / CELL) + 1);
  const cmin = new Float32Array(nx * ny).fill(Infinity);
  for (let i = 0; i < n; i++) {
    const c = Math.floor((p[3 * i + 1] - y0) / CELL) * nx + Math.floor((p[3 * i] - x0) / CELL);
    if (p[3 * i + 2] < cmin[c]) cmin[c] = p[3 * i + 2];
  }
  // Lowest value in a (2*WIN+1)^2 window, as a separable min filter.
  const winMin = (src) => {
    const tmp = new Float32Array(nx * ny), out = new Float32Array(nx * ny);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      let m = Infinity;
      for (let k = Math.max(0, i - WIN); k <= Math.min(nx - 1, i + WIN); k++) m = Math.min(m, src[j * nx + k]);
      tmp[j * nx + i] = m;
    }
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      let m = Infinity;
      for (let k = Math.max(0, j - WIN); k <= Math.min(ny - 1, j + WIN); k++) m = Math.min(m, tmp[k * nx + i]);
      out[j * nx + i] = m;
    }
    return out;
  };
  const rise = MAX_SLOPE * WIN * CELL;
  const nb = winMin(cmin);
  const dtm = new Float32Array(nx * ny);
  for (let c = 0; c < nx * ny; c++) dtm[c] = Math.min(cmin[c], nb[c] + rise);
  // Fill cells with no points at all by growing from their neighbours.
  for (let pass = 0; pass < 64; pass++) {
    let empty = 0;
    const g = winMin(dtm);
    for (let c = 0; c < nx * ny; c++) if (dtm[c] === Infinity) { dtm[c] = g[c]; if (g[c] === Infinity) empty++; }
    if (!empty) break;
  }
  return { x0, y0, nx, ny, dtm };
}

function groundZ(g, x, y) {
  // Bilinear between cell centres.
  const fx = Math.min(Math.max((x - g.x0) / CELL - 0.5, 0), g.nx - 1);
  const fy = Math.min(Math.max((y - g.y0) / CELL - 0.5, 0), g.ny - 1);
  const i = Math.floor(fx), j = Math.floor(fy);
  const i1 = Math.min(i + 1, g.nx - 1), j1 = Math.min(j + 1, g.ny - 1);
  const tx = fx - i, ty = fy - j, d = g.dtm, nx = g.nx;
  const a = d[j * nx + i] * (1 - tx) + d[j * nx + i1] * tx;
  const b = d[j1 * nx + i] * (1 - tx) + d[j1 * nx + i1] * tx;
  return a * (1 - ty) + b * ty;
}

// Deselect selected points less than `height` m above the local ground. Returns how many.
export function dropGroundFromSelection(height) {
  const p = state.positions, sel = state.selection;
  if (!p || !sel) return 0;
  let g = cache.get(p);
  if (!g) { g = buildDtm(p, state.n); cache.set(p, g); }
  let dropped = 0;
  for (let i = 0; i < state.n; i++) {
    if (!sel[i]) continue;
    if (p[3 * i + 2] - groundZ(g, p[3 * i], p[3 * i + 1]) < height) { sel[i] = 0; dropped++; }
  }
  if (dropped) { recountSelection(); emit("selection"); }
  return dropped;
}

// Shared client state + tiny event bus.

const listeners = new Map();

export function on(evt, fn) {
  if (!listeners.has(evt)) listeners.set(evt, new Set());
  listeners.get(evt).add(fn);
}

export function emit(evt, payload) {
  for (const fn of listeners.get(evt) || []) fn(payload);
}

export const state = {
  info: { loaded: false },
  n: 0,                  // display point count
  positions: null,       // Float32Array (n*3), origin-centered
  labels: null,          // Int32Array (n)
  material: null,        // Uint8Array (n): 0 unknown, 1 wood, 2 leaf
  selection: null,       // Uint8Array (n)
  selCount: 0,
  trees: [],             // [{id,count,height,zmin,center,reviewed}]
  treeById: new Map(),
  focusedTree: null,     // tree id or null
  tool: "navigate",
  busy: false,
  view: {
    colorMode: 0, size: 2, opacity: 1, atten: false,
    hideNontree: false, isolate: false, showWood: true, showLeaf: true,
    zmin: -Infinity, zmax: Infinity,
  },
};

// Each tree ID gets a color slot in first-seen order; slots are spread with
// golden-ratio hues so neighbouring slots differ. Slots never change within a
// session, so merges/deletes do not recolor other trees. The viewer uploads
// these colors into a palette texture indexed by label.
const colorSlot = new Map();
let nextSlot = 0;

export function resetColorSlots() {
  colorSlot.clear();
  nextSlot = 0;
}

// Returns true when new IDs were added (palette must be refreshed).
export function registerTreeIds(ids) {
  let added = false;
  for (const id of [...ids].sort((a, b) => a - b)) {
    if (id >= 0 && !colorSlot.has(id)) { colorSlot.set(id, nextSlot++); added = true; }
  }
  return added;
}

function hash32(x) {
  x = (x ^ 61) ^ (x >>> 16);
  x = Math.imul(x, 9);
  x ^= x >>> 4;
  x = Math.imul(x, 0x27d4eb2d);
  x ^= x >>> 15;
  return x >>> 0;
}

export function treeColor(id) {
  if (id < 0) return [0.42, 0.42, 0.45];
  const k = colorSlot.has(id) ? colorSlot.get(id) : 5000 + (hash32(id + 1) % 5000);
  const h = (k * 0.61803398875 + 0.05) % 1;
  const s = [0.85, 0.6, 0.95][k % 3];
  const v = [0.98, 0.82, 0.7][Math.floor(k / 3) % 3];
  const c = v * s;
  const hp = h * 6;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r = 0, g = 0, b = 0;
  if (hp < 1) [r, g, b] = [c, x, 0];
  else if (hp < 2) [r, g, b] = [x, c, 0];
  else if (hp < 3) [r, g, b] = [0, c, x];
  else if (hp < 4) [r, g, b] = [0, x, c];
  else if (hp < 5) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  const m = v - c;
  return [r + m, g + m, b + m];
}

export function cssColor(id) {
  const [r, g, b] = treeColor(id);
  return `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
}

export function selectedIndices() {
  const out = new Uint32Array(state.selCount);
  const sel = state.selection;
  let k = 0;
  for (let i = 0; i < state.n; i++) if (sel[i]) out[k++] = i;
  return k === out.length ? out : out.subarray(0, k);
}

export function recountSelection() {
  let c = 0;
  const sel = state.selection;
  for (let i = 0; i < state.n; i++) c += sel[i];
  state.selCount = c;
  return c;
}

// Labels present in the current selection, with display-point counts.
export function selectionLabelCounts() {
  const m = new Map();
  const sel = state.selection, lab = state.labels;
  for (let i = 0; i < state.n; i++) {
    if (sel[i]) m.set(lab[i], (m.get(lab[i]) || 0) + 1);
  }
  return m;
}

// Side panels: tabs, tree table, view controls, wood/leaf, segmentation, export.
import { state, emit, on, cssColor, selectionLabelCounts } from "./state.js";
import { el } from "./ui.js";

const $ = (id) => document.getElementById(id);

// ── tabs ─────────────────────────────────────────────────────────────────
export function initTabs() {
  for (const t of document.querySelectorAll(".tab")) {
    t.addEventListener("click", () => showTab(t.dataset.tab));
  }
}

export function showTab(name) {
  for (const t of document.querySelectorAll(".tab")) t.classList.toggle("active", t.dataset.tab === name);
  for (const p of document.querySelectorAll(".pane")) p.classList.toggle("active", p.id === `pane-${name}`);
}

// ── tree table ───────────────────────────────────────────────────────────
export function orderedTrees() {
  const q = $("tree-search").value.trim();
  const sort = $("tree-sort").value;
  let list = state.trees;
  if (q) list = list.filter((t) => String(t.id).includes(q));
  list = [...list];
  if (sort === "count") list.sort((a, b) => b.count - a.count);
  else if (sort === "height") list.sort((a, b) => b.height - a.height);
  else if (sort === "unreviewed") list.sort((a, b) => (a.reviewed - b.reviewed) || (a.id - b.id));
  else list.sort((a, b) => a.id - b.id);
  return list;
}

export function initTreeTable() {
  $("tree-search").addEventListener("input", renderTrees);
  $("tree-sort").addEventListener("change", renderTrees);
  const tbody = $("tree-table").querySelector("tbody");
  tbody.addEventListener("click", (e) => {
    const tr = e.target.closest("tr");
    if (!tr || !tr.dataset.id || e.target.matches("input")) return;
    const id = Number(tr.dataset.id);
    if (e.shiftKey || e.ctrlKey) emit("select-tree", { id, op: e.ctrlKey ? "sub" : "add" });
    else emit("focus-tree", { id, frame: false });
  });
  tbody.addEventListener("dblclick", (e) => {
    const tr = e.target.closest("tr");
    if (tr?.dataset.id) emit("isolate-tree", Number(tr.dataset.id));
  });
  tbody.addEventListener("change", (e) => {
    if (!e.target.matches("input[type=checkbox]")) return;
    emit("review", { id: Number(e.target.closest("tr").dataset.id), value: e.target.checked });
  });
  on("selection", markSelectedRows);
}

const MAX_ROWS = 2000;

export function renderTrees() {
  const tbody = $("tree-table").querySelector("tbody");
  const frag = document.createDocumentFragment();
  const list = orderedTrees();
  let shown = list.length > MAX_ROWS ? list.slice(0, MAX_ROWS) : list;
  const f = state.focusedTree;
  if (f !== null && shown !== list && !shown.some((t) => t.id === f)) {
    const ft = state.treeById.get(f);
    if (ft) shown = [ft, ...shown];
  }
  for (const t of shown) {
    const row = el("tr", { "data-id": t.id, class: t.id === state.focusedTree ? "focused" : null },
      el("td", {}, el("span", { class: "swatch", style: `background:${cssColor(t.id)}` })),
      el("td", { text: String(t.id) }),
      el("td", { class: "num", text: t.count.toLocaleString() }),
      el("td", { class: "num", text: `${t.height.toFixed(1)} m` }),
      el("td", {}, el("input", { type: "checkbox", checked: t.reviewed, title: "Reviewed (K)" })),
    );
    frag.appendChild(row);
  }
  if (list.length > MAX_ROWS) {
    frag.appendChild(el("tr", {}, el("td", { colspan: 5, class: "muted",
      text: `${(list.length - MAX_ROWS).toLocaleString()} more trees: use the filter or sort` })));
  }
  tbody.replaceChildren(frag);
  const nRev = state.trees.filter((t) => t.reviewed).length;
  const total = state.trees.length;
  $("review-bar").style.width = total ? `${(100 * nRev) / total}%` : "0";
  $("review-label").textContent = `${nRev} / ${total} reviewed`;
  let nt = 0;
  if (state.labels) for (let i = 0; i < state.n; i++) nt += state.labels[i] < 0;
  $("nontree-label").textContent = state.n ? `Non-tree: ${nt.toLocaleString()} display pts` : "";
  markSelectedRows();
}

function markSelectedRows() {
  const present = state.selCount ? selectionLabelCounts() : new Map();
  for (const tr of $("tree-table").querySelectorAll("tbody tr")) {
    tr.classList.toggle("in-sel", present.has(Number(tr.dataset.id)));
  }
}

export function markFocusedRow(scroll = true) {
  for (const tr of $("tree-table").querySelectorAll("tbody tr")) {
    const f = Number(tr.dataset.id) === state.focusedTree;
    tr.classList.toggle("focused", f);
    if (f && scroll) tr.scrollIntoView({ block: "nearest" });
  }
}

// ── view panel ───────────────────────────────────────────────────────────
export function initViewPanel(viewer) {
  const v = state.view;
  const bind = (id, key, parse, evt = "input") => {
    $(id).addEventListener(evt, () => {
      v[key] = parse($(id));
      viewer.applyView();
      emit("view");
    });
  };
  bind("color-mode", "colorMode", (e) => Number(e.value), "change");
  bind("pt-size", "size", (e) => Number(e.value));
  bind("pt-opacity", "opacity", (e) => Number(e.value));
  bind("pt-atten", "atten", (e) => e.checked, "change");
  bind("hide-nt", "hideNontree", (e) => e.checked, "change");
  bind("isolate", "isolate", (e) => e.checked, "change");
  bind("show-wood", "showWood", (e) => e.checked, "change");
  bind("show-leaf", "showLeaf", (e) => e.checked, "change");

  const slab = () => {
    let a = Number($("z-min").value), b = Number($("z-max").value);
    if (a > b) [a, b] = [b, a];
    const full = Number($("z-min").min) === a && Number($("z-max").max) === b;
    v.zmin = full ? -Infinity : a;
    v.zmax = full ? Infinity : b;
    $("z-min-l").textContent = (a + (state.info.origin?.[2] || 0)).toFixed(1);
    $("z-max-l").textContent = (b + (state.info.origin?.[2] || 0)).toFixed(1);
    viewer.applyView();
    emit("view");
  };
  $("z-min").addEventListener("input", slab);
  $("z-max").addEventListener("input", slab);
  $("slab-reset").addEventListener("click", () => {
    $("z-min").value = $("z-min").min;
    $("z-max").value = $("z-max").max;
    slab();
  });
  return { resetSlab: (zmin, zmax) => {
    for (const id of ["z-min", "z-max"]) {
      $(id).min = zmin.toFixed(2);
      $(id).max = zmax.toFixed(2);
      $(id).step = Math.max((zmax - zmin) / 500, 0.01).toFixed(3);
    }
    $("z-min").value = zmin;
    $("z-max").value = zmax;
    slab();
  } };
}

export function syncViewControls() {
  const v = state.view;
  $("color-mode").value = String(v.colorMode);
  $("hide-nt").checked = v.hideNontree;
  $("isolate").checked = v.isolate;
  $("show-wood").checked = v.showWood;
  $("show-leaf").checked = v.showLeaf;
  $("wl-toggles").classList.toggle("hidden", !state.info.wood_leaf_method);
}

// ── wood / leaf ──────────────────────────────────────────────────────────
const WL_LABELS = {
  stem_grow: "Stem-grow (best on LeWoS)",
  eigen: "Eigenfeatures",
  percentile: "Intensity percentile",
  intensity: "Intensity threshold",
  otsu: "Otsu (intensity)",
  rgi: "RGI",
  gbseparation: "GBSeparation",
};

const WL_PARAMS = {
  stem_grow: [
    ["verticality_min", "Min verticality", 0.8, 0.05],
    ["height_percentile", "Seed height percentile", 15, 1],
    ["grow_radius", "Grow radius (m)", 0.2, 0.05],
  ],
  eigen: [
    ["linearity_min", "Min linearity", 0.3, 0.05],
    ["verticality_min", "Min verticality", 0.7, 0.05],
    ["curvature_max", "Max curvature", 0.12, 0.01],
  ],
  percentile: [["percentile", "Wood at or above percentile", 40, 1]],
  intensity: [["threshold", "Intensity threshold (blank = median)", "", 0.01]],
};

let wlNeedsIntensity = [];

export function initWoodLeaf(methods, run, needsIntensity = []) {
  wlNeedsIntensity = needsIntensity;
  const sel = $("wl-method");
  sel.replaceChildren(...methods.map((m) => el("option", { value: m, text: WL_LABELS[m] || m })));
  const renderParams = () => {
    const spec = WL_PARAMS[sel.value] || [];
    $("wl-params").replaceChildren(...spec.map(([key, label, def, step]) =>
      el("label", {}, label, el("input", { type: "number", "data-key": key, value: def, step }))));
    if (!spec.length) $("wl-params").appendChild(el("p", { class: "muted", text: "No parameters." }));
  };
  sel.addEventListener("change", renderParams);
  renderParams();
  $("btn-wl-run").addEventListener("click", () => {
    const params = {};
    for (const inp of $("wl-params").querySelectorAll("input")) {
      if (inp.value !== "") params[inp.dataset.key] = Number(inp.value);
    }
    run(sel.value, params);
  });
}

export function setWoodLeafAvailability(hasIntensity) {
  const sel = $("wl-method");
  for (const opt of sel.options) {
    const off = !hasIntensity && wlNeedsIntensity.includes(opt.value);
    opt.disabled = off;
    opt.textContent = (WL_LABELS[opt.value] || opt.value) + (off ? " (needs intensity)" : "");
  }
  if (sel.selectedOptions[0]?.disabled) {
    sel.value = [...sel.options].find((o) => !o.disabled)?.value;
    sel.dispatchEvent(new Event("change"));
  }
}

// ── segmentation ─────────────────────────────────────────────────────────
/** Fill a Point-SAM weights <select>; it is only shown while `methodSel` is "pointsam". */
export function initPointSamSelect(psSel, methodSel, weights) {
  psSel.replaceChildren(...(weights || []).map((w) => el("option", { value: w.path, text: w.label, title: w.path })));
  const sync = () => psSel.closest("label").classList.toggle("hidden", methodSel.value !== "pointsam");
  methodSel.addEventListener("change", sync);
  sync();
  return () => (methodSel.value === "pointsam" && psSel.value) || null;
}

export function initSegment(methods, weights, run) {
  const sel = $("seg-method");
  sel.replaceChildren(...methods.map((m) => el("option", { value: m, text: m })));
  if (methods.includes("treelearn")) sel.value = "treelearn";
  const syncOptions = () => $("seg-treex").closest("label").classList.toggle("hidden", sel.value !== "treex");
  sel.addEventListener("change", syncOptions);
  syncOptions();
  const ckpt = initPointSamSelect($("seg-ps"), sel, weights);
  $("btn-seg-run").addEventListener("click", () =>
    run(sel.value, $("seg-leaf").checked, $("seg-treex").checked, ckpt()));
}

// ── export ───────────────────────────────────────────────────────────────
export function initExport(save) {
  $("btn-export").addEventListener("click", () => save());
}

export function setExportDefaults(info) {
  if (!$("out-dir").value) $("out-dir").value = info.default_export_dir || "";
  $("out-name").value = info.name || "tile";
}

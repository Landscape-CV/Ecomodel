// Annotator bootstrap: wires viewer, tools, panels, shortcuts and the API.
import {
  state, on, emit, selectedIndices, selectionLabelCounts, recountSelection, registerTreeIds, resetColorSlots,
} from "./state.js";
import { getJSON, postJSON, postBinary, getBinary, upload, pollJob, ApiError } from "./api.js";
import { Viewer } from "./viewer.js";
import { SelectionTools, selectTree } from "./select.js";
import {
  initTabs, initTreeTable, renderTrees, orderedTrees, markFocusedRow, initViewPanel,
  syncViewControls, initWoodLeaf, initSegment, initExport, setExportDefaults, showTab, setWoodLeafAvailability,
} from "./panels.js";
import { initShortcuts, showHelp } from "./shortcuts.js";
import { modal, confirmDialog, toast, el, busyStart, setProgress, fmtElapsed } from "./ui.js";

const $ = (id) => document.getElementById(id);

const viewer = new Viewer($("canvas-host"));
const tools = new SelectionTools(viewer, $("viewport"), $("overlay"), $("hud-hover"));
const viewPanel = initViewPanel(viewer);
window.__annotator = { state, viewer, tools };

// ── helpers ──────────────────────────────────────────────────────────────
function errorToast(err) {
  console.error(err);
  toast(err instanceof ApiError ? err.message : String(err), "error", 6000);
}

async function guarded(fn, label = "Working...") {
  if (state.busy) return;
  state.busy = true;
  document.body.style.cursor = "progress";
  // Only show the bar for operations that take noticeable time.
  let bar = null;
  const timer = setTimeout(() => { bar = busyStart(label); }, 150);
  try { return await fn(); }
  catch (err) { errorToast(err); }
  finally {
    clearTimeout(timer);
    bar?.end();
    state.busy = false;
    document.body.style.cursor = "";
    updateChrome();
  }
}

function updateChrome() {
  const info = state.info;
  const loaded = !!info.loaded;
  $("empty-state").classList.toggle("hidden", loaded);
  $("hud").classList.toggle("hidden", !loaded);
  $("view-panel").classList.toggle("hidden", !loaded);
  $("file-name").textContent = loaded
    ? `${info.name} · ${info.num_points.toLocaleString()} pts` + (info.num_display < info.num_points ? ` (showing ${info.num_display.toLocaleString()})` : "")
    : "No cloud loaded";
  $("file-name").title = info.source || "";
  $("dirty-dot").classList.toggle("hidden", !(loaded && info.dirty));
  $("btn-undo").disabled = !loaded || !info.can_undo;
  $("btn-redo").disabled = !loaded || !info.can_redo;
  $("btn-save").disabled = !loaded;
  $("btn-wl-run").disabled = !loaded || state.busy;
  $("btn-seg-run").disabled = !loaded || state.busy;
  const noSel = state.selCount === 0;
  for (const b of document.querySelectorAll(".act")) b.disabled = !loaded || noSel;
  $("hud-sel").textContent = `Selected: ${state.selCount.toLocaleString()} display pts`;
  const as = info.autosave;
  $("autosave-label").textContent = loaded
    ? (info.last_saved ? `Last saved ${info.last_saved}. ` : "Not saved this session. ") +
      (as ? `Autosave: ${new Date(as.mtime * 1000).toLocaleTimeString()}` : "Autosave every 10 edits.")
    : "";
  document.title = loaded ? `${info.dirty ? "* " : ""}${info.name} - TLS Annotator` : "TLS Instance Annotator";
}

async function refreshTrees() {
  const { trees } = await getJSON("/api/trees");
  state.trees = trees;
  state.treeById = new Map(trees.map((t) => [t.id, t]));
  if (registerTreeIds(trees.map((t) => t.id))) {
    viewer.ensurePalette(trees.length ? trees[trees.length - 1].id : 0, true);
  }
  if (state.focusedTree !== null && !state.treeById.has(state.focusedTree)) setFocus(null);
  renderTrees();
}

let treeTimer = 0;
function refreshTreesSoon() {
  clearTimeout(treeTimer);
  treeTimer = setTimeout(() => refreshTrees().catch(errorToast), 150);
}

// ── loading ──────────────────────────────────────────────────────────────
async function loadCloudData() {
  state.info = await getJSON("/api/status");
  if (!state.info.loaded) { updateChrome(); return; }
  const [pts, labs] = await Promise.all([getBinary("/api/points"), getBinary("/api/labels")]);
  state.positions = new Float32Array(pts);
  state.labels = new Int32Array(labs);
  resetColorSlots();
  state.n = state.labels.length;
  state.selection = new Uint8Array(state.n);
  state.material = new Uint8Array(state.n);
  state.selCount = 0;
  state.focusedTree = null;
  state.view.isolate = false;
  if (state.info.wood_leaf_method) state.material.set(new Uint8Array(await getBinary("/api/material")));
  else state.view.colorMode = 0;
  viewer.setCloud(state.positions, state.labels);
  const b = viewer.bounds;
  viewPanel.resetSlab(b.min.z, b.max.z);
  syncViewControls();
  viewer.applyView();
  setExportDefaults(state.info);
  setWoodLeafAvailability(state.info.has_intensity);
  await refreshTrees();
  updateChrome();
}

async function openDialog() {
  if (state.info.loaded && state.info.dirty &&
      !(await confirmDialog("Unsaved changes", "Opening another cloud discards unsaved edits (autosave is kept). Continue?", "Discard and open", true))) return;

  const recent = JSON.parse(localStorage.getItem("recentPaths") || "[]");
  const pathIn = el("input", { type: "text", value: recent[0] || "testdataset/real_instance/l1w_t00_03", style: "width:100%" });
  const listEl = el("div", { class: "browser" });
  const dirLabel = el("div", { class: "muted" });
  const modeGt = el("input", { type: "radio", name: "mode", value: "gt", checked: true });
  const modeBlank = el("input", { type: "radio", name: "mode", value: "blank" });
  const maxDisp = el("input", { type: "number", value: localStorage.getItem("maxDisplay") || 2000000, step: 250000, min: 50000 });
  const fileIn = el("input", { type: "file", accept: ".laz,.las,.ply" });

  async function browse(dir) {
    try {
      const r = await getJSON(`/api/browse?dir=${encodeURIComponent(dir)}`);
      dirLabel.textContent = r.dir;
      const items = [];
      if (r.parent) items.push(el("div", { class: "item dir", onclick: () => browse(r.parent) }, ".."));
      for (const d of r.dirs) items.push(el("div", { class: "item dir", onclick: () => browse(`${r.dir}/${d}`) }, d));
      for (const t of r.tiles) {
        items.push(el("div", {
          class: "item", title: t.path,
          onclick: (e) => { pathIn.value = t.path; mark(e.currentTarget); },
          ondblclick: () => document.querySelector(".modal .btn.primary")?.click(),
        }, el("span", { text: t.name }), el("span", {},
          el("span", { class: t.has_gt ? "badge" : "badge gray", text: t.has_gt ? "labels" : "no labels" }),
          ` ${t.size_mb} MB`)));
      }
      for (const f of r.files) {
        items.push(el("div", {
          class: "item", title: f.path,
          onclick: (e) => { pathIn.value = f.path; mark(e.currentTarget); },
          ondblclick: () => document.querySelector(".modal .btn.primary")?.click(),
        }, el("span", { text: f.name }), el("span", { class: "muted", text: `${f.size_mb} MB` })));
      }
      if (!items.length) items.push(el("div", { class: "item muted", text: "No clouds here." }));
      listEl.replaceChildren(...items);
    } catch (err) { errorToast(err); }
  }
  function mark(node) {
    for (const n of listEl.querySelectorAll(".sel")) n.classList.remove("sel");
    node.classList.add("sel");
  }

  const body = el("div", { style: "display:flex;flex-direction:column;gap:8px" },
    el("label", {}, "Tile prefix or LAZ/LAS/PLY path", pathIn),
    recent.length ? el("div", { class: "chips" }, recent.slice(0, 6).map((p) =>
      el("span", { class: "chip", title: p, text: p.split(/[\\/]/).pop(), onclick: () => { pathIn.value = p; } }))) : null,
    dirLabel, listEl,
    el("div", { class: "row" },
      el("label", { class: "check" }, modeGt, "Use existing labels (*_instances.npy)"),
      el("label", { class: "check" }, modeBlank, "Start blank")),
    el("label", {}, "Max display points (full resolution is kept for edits)", maxDisp),
    el("label", {}, "Or upload a file", fileIn),
  );
  browse(pathIn.value.includes("/") || pathIn.value.includes("\\") ? pathIn.value.replace(/[\\/][^\\/]*$/, "") : "testdataset");

  const choice = await modal({
    title: "Open cloud", body, wide: true,
    buttons: [{ label: "Cancel", value: null }, { label: "Open", value: "open", primary: true }],
    collect: () => ({ path: pathIn.value.trim(), mode: modeBlank.checked ? "blank" : "gt", maxDisplay: Number(maxDisp.value) || null, file: fileIn.files[0] }),
  });
  if (!choice) return;
  localStorage.setItem("maxDisplay", String(choice.maxDisplay || 2000000));

  await guarded(async () => {
    const t = toast(choice.file ? `Uploading ${choice.file.name}...` : `Loading ${choice.path}...`, "busy", 0);
    try {
      if (choice.file) await upload(choice.file, choice.maxDisplay);
      else {
        await postJSON("/api/load", { path: choice.path, mode: choice.mode, max_display: choice.maxDisplay });
        const rec = [choice.path, ...recent.filter((p) => p !== choice.path)].slice(0, 8);
        localStorage.setItem("recentPaths", JSON.stringify(rec));
      }
      t.update("Preparing view...");
      await loadCloudData();
    } finally { t.close(); }
    toast(`Loaded ${state.info.name}: ${state.info.num_points.toLocaleString()} pts, ${state.info.num_trees} trees`, "success");
    await offerAutosave();
  }, choice.file ? "Uploading..." : "Loading cloud...");
}

async function offerAutosave() {
  const as = state.info.autosave;
  if (!as) return;
  const when = new Date(as.mtime * 1000).toLocaleString();
  const v = await modal({
    title: "Restore autosave?",
    body: `An unsaved autosave of this tile from ${when} exists.`,
    buttons: [
      { label: "Discard it", value: "discard", danger: true },
      { label: "Keep for later", value: null },
      { label: "Restore", value: "restore", primary: true },
    ],
  });
  if (v === "restore") {
    const res = await postJSON("/api/autosave/restore");
    await applyEditResult(res);
    toast("Autosave restored (Ctrl+Z to revert).", "success");
  } else if (v === "discard") {
    state.info = await postJSON("/api/autosave/discard");
    updateChrome();
  }
}

// ── edits ────────────────────────────────────────────────────────────────
async function applyEditResult(res) {
  if (res.reload_labels) {
    const labs = new Int32Array(await getBinary("/api/labels"));
    state.labels.set(labs);
    viewer.setAllLabels(labs);
  } else if (res.display_idx?.length) {
    const idx = res.display_idx, labs = res.display_labels;
    for (let k = 0; k < idx.length; k++) state.labels[idx[k]] = labs[k];
    viewer.patchLabels(idx, labs);
  }
  state.info = res.info;
  updateChrome();
  refreshTreesSoon();
}

function clearSelection() {
  if (!state.selection) return;
  state.selection.fill(0);
  recountSelection();
  viewer.selectionChanged();
  emit("selection");
}

async function editOp(op, params = {}) {
  const q = new URLSearchParams(params).toString();
  const res = await postBinary(`/api/edit/${op}${q ? `?${q}` : ""}`, selectedIndices());
  await applyEditResult(res);
  return res;
}

function largestTreeInSelection() {
  let best = null, bestC = -1;
  for (const [t, c] of selectionLabelCounts()) if (t >= 0 && c > bestC) { best = t; bestC = c; }
  return best;
}

async function promptTreeId(title, message, def, extraChips = []) {
  const inp = el("input", { type: "number", value: def ?? "", min: -1, step: 1 });
  const chips = el("div", { class: "chips" }, extraChips.map(([label, val]) =>
    el("span", { class: "chip", text: label, onclick: () => { inp.value = val; } })));
  const v = await modal({
    title,
    body: el("div", { style: "display:flex;flex-direction:column;gap:8px" }, el("p", { text: message }), el("label", {}, "Target tree ID (-1 = non-tree)", inp), chips),
    buttons: [{ label: "Cancel", value: null }, { label: "Apply", value: "ok", primary: true }],
    collect: () => {
      const n = Number(inp.value);
      if (inp.value === "" || !Number.isInteger(n) || n < -1) { inp.focus(); return undefined; }
      return n;
    },
  });
  return v === null ? null : v;
}

const ACTIONS = {
  async new() {
    const res = await editOp("new");
    clearSelection();
    setFocus(res.new_id);
    toast(`Created tree ${res.new_id} (${res.changed_points.toLocaleString()} pts)`, "success");
  },
  async assign() {
    const def = state.focusedTree ?? largestTreeInSelection() ?? state.info.next_tree_id;
    const chips = [[`new tree (${state.info.next_tree_id})`, state.info.next_tree_id], ["non-tree", -1]];
    if (state.focusedTree !== null) chips.unshift([`focused (${state.focusedTree})`, state.focusedTree]);
    const tgt = await promptTreeId("Assign selection", `Assign ${state.selCount.toLocaleString()} selected display points to:`, def, chips);
    if (tgt === null) return;
    const res = await editOp("assign", { target: tgt });
    clearSelection();
    if (tgt >= 0) setFocus(tgt);
    toast(`Assigned ${res.changed_points.toLocaleString()} pts to ${tgt < 0 ? "non-tree" : `tree ${tgt}`}`, "success");
  },
  async merge() {
    const trees = [...selectionLabelCounts().keys()].filter((t) => t >= 0);
    if (trees.length < 1) { toast("Select points from the trees you want to merge.", "warn"); return; }
    const def = state.focusedTree !== null && trees.includes(state.focusedTree) ? state.focusedTree : largestTreeInSelection();
    const chips = trees.slice(0, 12).map((t) => [`tree ${t}`, t]);
    const tgt = await promptTreeId("Merge trees", `Merge whole trees ${trees.join(", ")} into:`, def, chips);
    if (tgt === null) return;
    if (tgt < 0 && !(await confirmDialog("Merge into non-tree?", `This marks every point of trees ${trees.join(", ")} as non-tree.`, "Mark as non-tree", true))) return;
    const res = await editOp("merge", { target: tgt });
    clearSelection();
    setFocus(tgt >= 0 ? tgt : null);
    toast(`Merged ${res.merged.join(", ")} into ${tgt < 0 ? "non-tree" : tgt} (${res.changed_points.toLocaleString()} pts)`, "success");
  },
  async nontree() {
    const res = await editOp("nontree");
    clearSelection();
    toast(`Marked ${res.changed_points.toLocaleString()} pts as non-tree`, "success");
  },
  async grow() {
    const r = tools.brushRadius;
    const buf = await postBinary(`/api/grow?radius=${r}&same_label=true`, selectedIndices());
    const idx = new Uint32Array(buf);
    for (let k = 0; k < idx.length; k++) state.selection[idx[k]] = 1;
    recountSelection();
    viewer.selectionChanged();
    emit("selection");
    toast(`Selection grown to ${state.selCount.toLocaleString()} pts (radius ${r} m)`);
  },
  async clear() { clearSelection(); },
};

function runAction(name) {
  if (!state.info.loaded) return;
  if (name === "clear") { clearSelection(); return; }
  if (state.selCount === 0) { toast("Nothing selected. Click a tree, or use Lasso (L) / Box (B) / Brush (R).", "warn"); return; }
  guarded(() => ACTIONS[name]());
}

async function undoRedo(which) {
  if (!state.info.loaded) return;
  await guarded(async () => {
    const res = await postJSON(`/api/${which}`);
    await applyEditResult(res);
    toast(`${which === "undo" ? "Undid" : "Redid"} ${res.changed_points.toLocaleString()} pts`);
  });
}

// ── focus / navigation ───────────────────────────────────────────────────
function setFocus(id, frame = false) {
  state.focusedTree = id;
  viewer.applyView();
  markFocusedRow();
  if (frame && id !== null) viewer.frameIndices((i) => state.labels[i] === id);
}

function stepTree(dir) {
  const list = orderedTrees();
  if (!list.length) return;
  let k = list.findIndex((t) => t.id === state.focusedTree);
  k = k < 0 ? (dir > 0 ? 0 : list.length - 1) : (k + dir + list.length) % list.length;
  setFocus(list[k].id, true);
}

async function setReviewed(id, value) {
  try {
    state.info = await postJSON("/api/review", { tree_id: id, reviewed: value });
    const t = state.treeById.get(id);
    if (t) t.reviewed = value;
    renderTrees();
    updateChrome();
  } catch (err) { errorToast(err); }
}

function frame() {
  if (state.selCount) viewer.frameIndices((i) => state.selection[i] === 1);
  else if (state.focusedTree !== null) viewer.frameIndices((i) => state.labels[i] === state.focusedTree);
  else viewer.frameAll();
}

// ── save ─────────────────────────────────────────────────────────────────
async function save(overwrite = false) {
  if (!state.info.loaded) return;
  await guarded(async () => {
    const body = { out_dir: $("out-dir").value.trim(), name: $("out-name").value.trim(), overwrite };
    try {
      const r = await postJSON("/api/save", body);
      state.info = r.info;
      $("export-result").textContent = `Saved:\n${Object.values(r.paths).join("\n")}`;
      toast(`Saved ${body.name}`, "success");
    } catch (err) {
      if (err instanceof ApiError && err.status === 409 && err.data.existing) {
        state.busy = false;
        const ok = await confirmDialog("Overwrite existing files?",
          `These files already exist:\n${err.data.existing.join("\n")}`, "Overwrite", true);
        if (ok) { await save(true); }
        return;
      }
      throw err;
    }
  }, "Saving...");
}

// ── background jobs ──────────────────────────────────────────────────────
function showJobError(box, message, traceback) {
  box.className = "result-box";
  box.replaceChildren(el("div", { class: "error-text", text: message }));
  if (traceback) box.appendChild(el("details", {}, el("summary", { class: "muted", text: "Traceback" }), el("pre", { text: traceback })));
}

/** pane: "wl" | "seg" (ids btn-<pane>-run, <pane>-status, <pane>-result). */
async function runJob(url, body, label, pane, onDone) {
  if (!state.info.loaded || state.busy) return;
  const btn = $(`btn-${pane}-run`), status = $(`${pane}-status`), result = $(`${pane}-result`);
  const track = status.querySelector(".progress"), msgEl = status.querySelector(".job-msg"), detEl = status.querySelector(".job-detail");
  const btnText = btn.textContent;
  const top = busyStart(`${label}...`);
  state.busy = true;
  updateChrome();
  btn.textContent = "Running...";
  result.textContent = "";
  status.classList.remove("hidden");
  setProgress(track, null);
  msgEl.textContent = "Starting...";
  detEl.textContent = "";
  const tick = (j) => {
    const pct = typeof j.progress === "number" && j.progress > 0 && j.progress < 1 ? ` ${Math.round(j.progress * 100)}%` : "";
    msgEl.textContent = `${j.message || "Running"}${pct} · ${fmtElapsed(j.elapsed)}`;
    detEl.textContent = j.detail || "";
    detEl.title = j.detail || "";
    setProgress(track, j.progress);
    top.set(j.progress, `${label}${pct} · ${fmtElapsed(j.elapsed)}`);
  };
  try {
    const job = await postJSON(url, body);
    tick(job);
    const done = await pollJob(job.id, tick);
    if (done.state === "error") {
      showJobError(result, done.message, done.result?.traceback);
      toast(`${label} failed (details in the panel)`, "error", 6000);
      return;
    }
    msgEl.textContent = "Loading results...";
    setProgress(track, null);
    top.set(null, `${label}: loading results...`);
    await onDone(done);
    result.className = "muted result-box";
    result.textContent = `${done.message}\nFinished in ${fmtElapsed(done.elapsed)}.`;
  } catch (err) {
    console.error(err);
    showJobError(result, err instanceof ApiError ? err.message : String(err));
    toast(`${label} failed (details in the panel)`, "error", 6000);
  } finally {
    state.busy = false;
    btn.textContent = btnText;
    status.classList.add("hidden");
    top.end();
    updateChrome();
  }
}

async function runSegmentation(method, leafRemoval, treexStock) {
  if (state.info.num_trees > 0 && !(await confirmDialog("Replace all labels?",
    `Running ${method} replaces the current ${state.info.num_trees} trees. You can undo it afterwards.`, "Run", true))) return;
  await runJob("/api/segment", { method, leaf_removal: leafRemoval, treex_stock: treexStock }, `Segmenting (${method})`, "seg", async (done) => {
    const labs = new Int32Array(await getBinary("/api/labels"));
    state.labels.set(labs);
    viewer.setAllLabels(labs);
    clearSelection();
    setFocus(null);
    state.info = await getJSON("/api/status");
    await refreshTrees();
    toast(`Segmentation done: ${done.message}`, "success");
  });
}

async function runWoodLeaf(method, params) {
  await runJob("/api/woodleaf", { method, params }, `Wood/leaf (${method})`, "wl", async (done) => {
    state.material.set(new Uint8Array(await getBinary("/api/material")));
    viewer.materialChanged();
    state.info = await getJSON("/api/status");
    state.view.colorMode = 1;
    syncViewControls();
    viewer.applyView();
    toast(`Wood/leaf: ${done.message}`, "success");
  });
}

// ── wiring ───────────────────────────────────────────────────────────────
initTabs();
initTreeTable();
initExport(() => save());

for (const b of document.querySelectorAll(".tool")) b.addEventListener("click", () => tools.setTool(b.dataset.tool));
for (const b of document.querySelectorAll(".act")) b.addEventListener("click", () => runAction(b.dataset.act));
$("btn-open").addEventListener("click", openDialog);
$("btn-open-empty").addEventListener("click", openDialog);
$("btn-undo").addEventListener("click", () => undoRedo("undo"));
$("btn-redo").addEventListener("click", () => undoRedo("redo"));
$("btn-save").addEventListener("click", () => save());
$("btn-help").addEventListener("click", showHelp);
$("btn-frame").addEventListener("click", frame);
$("btn-prev").addEventListener("click", () => stepTree(-1));
$("btn-next").addEventListener("click", () => stepTree(1));
$("btn-select-tree").addEventListener("click", () => { if (state.focusedTree !== null) selectTree(state.focusedTree); viewer.selectionChanged(); });
$("btn-review").addEventListener("click", () => { if (state.focusedTree !== null) setReviewed(state.focusedTree, !state.treeById.get(state.focusedTree)?.reviewed); });
$("btn-compact").addEventListener("click", async () => {
  if (!(await confirmDialog("Compact IDs?", "Renumber trees to 0..T-1. Reviewed marks are cleared.", "Compact"))) return;
  guarded(async () => {
    const res = await postBinary("/api/edit/compact", new Uint32Array(0));
    setFocus(null);
    await applyEditResult(res);
    toast(res.changed_points ? `Renumbered ${res.changed_points.toLocaleString()} pts` : "IDs already compact");
  });
});
$("brush-r").addEventListener("input", () => { tools.brushRadius = Math.max(0.02, Number($("brush-r").value) || 0.3); });

on("tool", (t) => {
  for (const b of document.querySelectorAll(".tool")) b.classList.toggle("active", b.dataset.tool === t);
});
on("selection", updateChrome);
on("toast", ({ msg, kind }) => toast(msg, kind));
on("focus-tree", ({ id, frame: f }) => setFocus(id, f));
on("select-tree", ({ id, op }) => { selectTree(id, op); viewer.selectionChanged(); });
on("isolate-tree", (id) => {
  state.view.isolate = true;
  syncViewControls();
  setFocus(id, true);
});
on("review", ({ id, value }) => setReviewed(id, value));

initShortcuts({
  undo: () => undoRedo("undo"),
  redo: () => undoRedo("redo"),
  save: () => save(),
  tool: (t) => { if (state.info.loaded) tools.setTool(t); },
  act: runAction,
  selectFocused: () => { if (state.focusedTree !== null) { selectTree(state.focusedTree); viewer.selectionChanged(); } },
  step: stepTree,
  toggleReviewed: () => { if (state.focusedTree !== null) setReviewed(state.focusedTree, !state.treeById.get(state.focusedTree)?.reviewed); },
  toggleIsolate: () => { state.view.isolate = !state.view.isolate; syncViewControls(); viewer.applyView(); },
  toggleHideNontree: () => { state.view.hideNontree = !state.view.hideNontree; syncViewControls(); viewer.applyView(); },
  toggleColor: () => {
    if (!state.info.wood_leaf_method) { toast("Run wood/leaf first (Wood / Leaf tab).", "warn"); showTab("woodleaf"); return; }
    state.view.colorMode = state.view.colorMode ? 0 : 1; syncViewControls(); viewer.applyView();
  },
  frame,
  brush: (d) => {
    const r = Math.max(0.02, Math.round((tools.brushRadius * (d > 0 ? 1.25 : 0.8)) * 100) / 100);
    tools.brushRadius = r;
    $("brush-r").value = r;
  },
  open: openDialog,
});

window.addEventListener("beforeunload", (e) => {
  if (state.info.loaded && state.info.dirty) { e.preventDefault(); e.returnValue = ""; }
});

// Initial state: methods for the side panels, then whatever the server already has loaded.
(async () => {
  try {
    const m = await getJSON("/api/methods");
    initWoodLeaf(m.woodleaf, runWoodLeaf, m.woodleaf_needs_intensity || []);
    initSegment(m.segment, runSegmentation);
    await loadCloudData();
    document.body.classList.remove("booting");
    if (state.info.loaded) await offerAutosave();
  } catch (err) { errorToast(err); }
  document.body.classList.remove("booting");
  updateChrome();
})();

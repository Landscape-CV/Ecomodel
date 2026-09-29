// Global keyboard shortcuts and the cheat sheet.
import { isModalOpen, modal, el } from "./ui.js";

export const SHORTCUTS = [
  ["V", "Navigate (click a point to select its tree)"],
  ["L / B / R", "Lasso / Box / Sphere brush"],
  ["Shift + drag", "Add to selection"],
  ["Ctrl + drag", "Subtract from selection"],
  ["Alt + drag", "Only affect the focused tree (or the tree under the cursor)"],
  ["Right drag / Middle drag", "Orbit / pan while a selection tool is active"],
  ["Double-click", "Isolate a tree and frame it"],
  ["N", "Selection becomes a new tree"],
  ["A", "Assign selection to a tree ID"],
  ["M", "Merge trees in the selection"],
  ["Del / Backspace", "Mark selection as non-tree"],
  ["G", "Grow selection (connected, same label)"],
  ["T", "Select focused tree"],
  ["Esc", "Clear selection (or close dialog)"],
  ["[ / ]", "Previous / next tree"],
  ["K", "Toggle reviewed on focused tree"],
  ["I", "Isolate focused tree"],
  ["H", "Hide non-tree points"],
  ["C", "Toggle instance / wood-leaf colors"],
  ["F", "Frame selection, focused tree, or everything"],
  ["- / =", "Shrink / grow brush radius"],
  ["Ctrl+Z / Ctrl+Y", "Undo / redo"],
  ["Ctrl+S", "Save"],
  ["O", "Open a cloud"],
  ["?", "This help"],
];

export function showHelp() {
  const grid = el("div", { class: "shortcut-grid" },
    SHORTCUTS.flatMap(([k, d]) => [el("kbd", { text: k }), el("span", { text: d })]));
  return modal({ title: "Keyboard shortcuts", body: grid, buttons: [{ label: "Close", value: true, primary: true }] });
}

export function initShortcuts(h) {
  document.addEventListener("keydown", (e) => {
    if (isModalOpen()) return;
    const t = e.target;
    if (t instanceof HTMLInputElement || t instanceof HTMLSelectElement || t instanceof HTMLTextAreaElement) {
      if (e.key === "Escape") t.blur();
      return;
    }
    const ctrl = e.ctrlKey || e.metaKey;
    const k = e.key;
    let handled = true;
    if (ctrl && (k === "z" || k === "Z")) e.shiftKey ? h.redo() : h.undo();
    else if (ctrl && (k === "y" || k === "Y")) h.redo();
    else if (ctrl && (k === "s" || k === "S")) h.save();
    else if (ctrl) handled = false;
    else if (k === "v" || k === "V") h.tool("navigate");
    else if (k === "l" || k === "L") h.tool("lasso");
    else if (k === "b" || k === "B") h.tool("box");
    else if (k === "r" || k === "R") h.tool("brush");
    else if (k === "n" || k === "N") h.act("new");
    else if (k === "a" || k === "A") h.act("assign");
    else if (k === "m" || k === "M") h.act("merge");
    else if (k === "Delete" || k === "Backspace") h.act("nontree");
    else if (k === "g" || k === "G") h.act("grow");
    else if (k === "Escape") h.act("clear");
    else if (k === "t" || k === "T") h.selectFocused();
    else if (k === "[") h.step(-1);
    else if (k === "]") h.step(1);
    else if (k === "k" || k === "K") h.toggleReviewed();
    else if (k === "i" || k === "I") h.toggleIsolate();
    else if (k === "h" || k === "H") h.toggleHideNontree();
    else if (k === "c" || k === "C") h.toggleColor();
    else if (k === "f" || k === "F") h.frame();
    else if (k === "-" || k === "_") h.brush(-1);
    else if (k === "=" || k === "+") h.brush(1);
    else if (k === "o" || k === "O") h.open();
    else if (k === "?") showHelp();
    else handled = false;
    if (handled) e.preventDefault();
  });
}

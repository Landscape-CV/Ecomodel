// Modals and toasts.

const modalRoot = () => document.getElementById("modal-root");

export function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return n;
}

export function isModalOpen() {
  return modalRoot().childElementCount > 0;
}

/**
 * Show a modal. buttons: [{label, value, primary, danger}]. Resolves with the
 * clicked button's value, or null on Escape / backdrop click.
 * `collect` (optional) is called before resolving a non-null value; returning
 * undefined keeps the modal open (validation failed).
 */
export function modal({ title, body, buttons = [{ label: "OK", value: true, primary: true }], collect, onOpen, wide }) {
  return new Promise((resolve) => {
    const bodyEl = el("div", { class: "body" });
    if (typeof body === "string") bodyEl.appendChild(el("p", { text: body }));
    else if (body) bodyEl.appendChild(body);

    const close = (v) => {
      document.removeEventListener("keydown", onKey, true);
      back.remove();
      resolve(v);
    };
    const submit = (v) => {
      if (v === null || v === false || !collect) return close(v);
      const out = collect(v);
      if (out !== undefined) close(out);
    };
    const btnEls = buttons.map((b) =>
      el("button", {
        class: `btn${b.primary ? " primary" : ""}${b.danger ? " danger" : ""}`,
        text: b.label,
        onclick: () => submit(b.value),
      }),
    );
    const box = el("div", { class: "modal", style: wide ? "min-width:640px" : null },
      el("h3", { text: title }), bodyEl, el("div", { class: "buttons" }, btnEls));
    const back = el("div", { class: "modal-back", onmousedown: (e) => { if (e.target === back) close(null); } }, box);

    const onKey = (e) => {
      if (e.key === "Escape") { e.stopPropagation(); e.preventDefault(); close(null); }
      else if (e.key === "Enter" && !(e.target instanceof HTMLTextAreaElement)) {
        const primary = buttons.find((b) => b.primary);
        if (primary) { e.stopPropagation(); e.preventDefault(); submit(primary.value); }
      }
    };
    document.addEventListener("keydown", onKey, true);
    modalRoot().appendChild(back);
    if (onOpen) onOpen(box, close);
    const first = box.querySelector("input, select");
    (first || btnEls.find((b) => b.classList.contains("primary")) || btnEls[0])?.focus();
    if (first && first.select) first.select();
  });
}

export async function confirmDialog(title, message, okLabel = "OK", danger = false) {
  const v = await modal({
    title,
    body: message,
    buttons: [
      { label: "Cancel", value: false },
      { label: okLabel, value: true, primary: true, danger },
    ],
  });
  return v === true;
}

/** progress: 0..1 for a determinate bar, null/undefined for an indeterminate one. */
export function setProgress(track, progress) {
  const known = typeof progress === "number" && progress > 0;
  track.classList.toggle("indeterminate", !known);
  track.querySelector(".bar").style.width = known ? `${Math.round(progress * 100)}%` : "";
}

export function fmtElapsed(s) {
  s = Math.round(s || 0);
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

// Global busy bar under the top bar; nested start() calls share it.
let busyDepth = 0;
export function busyStart(text = "Working...") {
  const root = document.getElementById("busy");
  busyDepth++;
  root.classList.remove("hidden");
  let ended = false;
  const h = {
    set(progress, label) {
      if (ended) return;
      setProgress(root.querySelector(".busy-track"), progress);
      if (label !== undefined) root.querySelector(".busy-text").textContent = label;
    },
    end() {
      if (ended) return;
      ended = true;
      if (--busyDepth <= 0) { busyDepth = 0; root.classList.add("hidden"); }
    },
  };
  h.set(null, text);
  return h;
}

export function toast(msg, kind = "info", ms = 3500) {
  const t = el("div", { class: `toast ${kind}` });
  if (kind === "busy") t.innerHTML = `<span class="spinner"></span>`;
  t.appendChild(document.createTextNode(msg));
  document.getElementById("toasts").appendChild(t);
  if (ms > 0) setTimeout(() => t.remove(), ms);
  return {
    update(text) { t.lastChild.textContent = text; },
    close() { t.remove(); },
  };
}

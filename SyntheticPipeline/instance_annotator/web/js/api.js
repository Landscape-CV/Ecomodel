// Thin fetch wrappers for the FastAPI backend.

export class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.status = status;
    this.data = data || {};
  }
}

async function handle(res) {
  const type = res.headers.get("content-type") || "";
  if (!res.ok) {
    let data = {};
    try { data = type.includes("json") ? await res.json() : { error: await res.text() }; } catch { /* empty */ }
    throw new ApiError(data.error || data.detail || `HTTP ${res.status}`, res.status, data);
  }
  if (type.includes("json")) return res.json();
  return res.arrayBuffer();
}

export async function getJSON(url) {
  return handle(await fetch(url));
}

export async function postJSON(url, body) {
  return handle(await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  }));
}

export async function postBinary(url, typed) {
  return handle(await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: typed,
  }));
}

export async function getBinary(url) {
  return handle(await fetch(url));
}

export async function upload(file, maxDisplay) {
  const fd = new FormData();
  fd.append("file", file);
  if (maxDisplay) fd.append("max_display", String(maxDisplay));
  return handle(await fetch("/api/upload", { method: "POST", body: fd }));
}

export async function pollJob(id, onTick) {
  for (;;) {
    const j = await getJSON(`/api/jobs/${id}`);
    if (onTick) onTick(j);
    if (j.state !== "running") return j;
    await new Promise((r) => setTimeout(r, 500));
  }
}

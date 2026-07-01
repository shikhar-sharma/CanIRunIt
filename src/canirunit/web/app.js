// canirunit — web UI, vanilla JS.
//
// Design rule: NO estimation logic here. Every number displayed comes from
// the Python core via /api/*. The frontend formats + arranges; the backend
// computes. If you feel like doing math here, add an endpoint.

// --------------------------------------------------------------------------
// Shared state (kept in one place so commits 6/7 can slot in cleanly)
// --------------------------------------------------------------------------
const state = {
  system: null,
  models: [],
  selection: {
    model: null,       // logical id OR raw ref (Advanced path)
    runtime: "gguf",
  },
  lastCheck: null,     // whole /api/check response for the current selection
};

// --------------------------------------------------------------------------
// API wrappers — normalize error handling so callers don't have to think
// about response.ok / detail extraction.
// --------------------------------------------------------------------------
async function apiGet(path) {
  const r = await fetch(path);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}

async function apiPost(path, payload) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}

const API = {
  system:  ()      => apiGet("/api/system"),
  models:  ()      => apiGet("/api/models"),
  check:   (body)  => apiPost("/api/check", body),
  compare: (body)  => apiPost("/api/compare", body),
  refresh: ()      => apiPost("/api/refresh", {}),
};

// --------------------------------------------------------------------------
// Formatters
// --------------------------------------------------------------------------
const fmt = {
  gb: (b) => (b == null ? "—" : (b / 1e9).toFixed(1) + " GB"),
  int: (n) => (n == null ? "—" : n.toLocaleString()),
  bandwidth: (gbs) => (gbs == null ? "—" : `${gbs.toFixed(0)} GB/s`),
};

// Tiny DOM helper. Keeps event wiring readable without a framework.
function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v;
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (v === true) el.setAttribute(k, "");
      else if (v != null && v !== false) el.setAttribute(k, v);
    }
  }
  for (const c of children) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// --------------------------------------------------------------------------
// A. Machine panel — from /api/system
// --------------------------------------------------------------------------
async function renderMachine() {
  const el = document.getElementById("machine-content");
  try {
    const s = await API.system();
    state.system = s;
    el.classList.remove("pending", "error");

    const grid = h("div", { class: "machine-grid" },
      metric("Chip", s.chip_id),
      metric("Accelerator", s.accelerator),
      metric("Memory bandwidth", fmt.bandwidth(s.memory_bandwidth_gbs)),
      metric("Total memory", fmt.gb(s.total_memory_bytes)),
      metric(`Usable (${s.usable_basis})`, fmt.gb(s.usable_memory_bytes)),
      s.hard_usable_memory_bytes
        ? metric("Loads-at-all ceiling", fmt.gb(s.hard_usable_memory_bytes))
        : null,
      metric("Storage free", fmt.gb(s.storage_free_bytes)),
    );
    el.replaceChildren(grid);
    if (!s.chip_is_known) {
      el.append(h("p", { class: "note" },
        "! Chip not in the bandwidth table — decode uses a coarse default. Calibrate for real numbers."));
    }
  } catch (err) {
    el.classList.remove("pending");
    el.classList.add("error");
    el.textContent = `Error detecting machine: ${err.message}`;
  }
}

function metric(label, value) {
  return h("div", { class: "metric" },
    h("span", { class: "lbl" }, label),
    h("span", { class: "val" }, value ?? "—"),
  );
}

// --------------------------------------------------------------------------
// B. Model picker — from /api/models
// --------------------------------------------------------------------------
async function loadModels() {
  try {
    const { models } = await API.models();
    state.models = models;
    renderModelList("");
  } catch (err) {
    const sel = document.getElementById("model-select");
    sel.replaceChildren(h("option", { disabled: true }, `Error: ${err.message}`));
  }
}

function renderModelList(query) {
  const sel = document.getElementById("model-select");
  const q = query.trim().toLowerCase();
  const filtered = state.models.filter((m) =>
    !q ||
    m.id.toLowerCase().includes(q) ||
    m.display_name.toLowerCase().includes(q) ||
    m.family.toLowerCase().includes(q),
  );
  sel.replaceChildren(...filtered.map((m) =>
    h("option", { value: m.id },
      `${m.display_name}  —  ${m.family}  (${m.runtimes.join(", ")})`,
    )
  ));
}

function wirePicker() {
  document.getElementById("model-search").addEventListener("input", (e) => {
    renderModelList(e.target.value);
  });
  document.getElementById("model-select").addEventListener("change", (e) => {
    state.selection.model = e.target.value;
    state.selection.runtime = defaultRuntimeFor(e.target.value);
    onSelectionChanged();
  });
  document.getElementById("refresh-btn").addEventListener("click", onRefreshClick);
  document.getElementById("advanced-check").addEventListener("click", () => {
    const ref = document.getElementById("advanced-ref").value.trim();
    if (!ref) return;
    state.selection.model = ref;
    state.selection.runtime = document.getElementById("advanced-runtime").value;
    onSelectionChanged();
  });
}

function defaultRuntimeFor(logical_id) {
  const entry = state.models.find((m) => m.id === logical_id);
  if (!entry) return "gguf";
  // Prefer gguf; if the model only has other runtimes, use the first listed.
  return entry.runtimes.includes("gguf") ? "gguf" : entry.runtimes[0];
}

async function onRefreshClick() {
  const btn = document.getElementById("refresh-btn");
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Refreshing…";
  try {
    const r = await API.refresh();
    await loadModels();
    btn.textContent = `Refreshed — ${r.models} models`;
  } catch (err) {
    btn.textContent = `Refresh failed: ${err.message}`;
  }
  setTimeout(() => {
    btn.disabled = false;
    btn.textContent = original;
  }, 2500);
}

// --------------------------------------------------------------------------
// C. Fit summary — from /api/check
// --------------------------------------------------------------------------
async function onSelectionChanged() {
  const { model, runtime } = state.selection;
  if (!model) return;

  const fitEl = document.getElementById("fit-content");
  fitEl.classList.remove("empty", "error");
  fitEl.classList.add("pending");
  fitEl.textContent = "Fetching model details…";

  try {
    const resp = await API.check({ model, runtime });
    state.lastCheck = resp;
    fitEl.classList.remove("pending");
    renderFit(resp);
  } catch (err) {
    fitEl.classList.remove("pending");
    fitEl.classList.add("error");
    fitEl.textContent = `Error: ${err.message}`;
  }
}

function renderFit(resp) {
  const { spec, fit, memory_curve } = resp;
  const b = fit.breakdown;
  const usable = b.usable_bytes;
  const hard = memory_curve.hard_usable_bytes;
  const total = b.required_bytes_at_native;
  // Bar scale: enough to show weights/KV/overhead AND the ceiling markers.
  const scale = Math.max(usable, total, hard || 0, 1);

  const el = document.getElementById("fit-content");
  el.replaceChildren(
    h("div", { class: "fit-header" },
      h("div", { class: "fit-title" },
        h("div", { class: "model-title" }, spec.repo_id),
        h("div", { class: "model-sub" },
          `${spec.runtime} · ${spec.quant_label || spec.quant} · native ${fmt.int(spec.native_ctx)} tokens`),
      ),
      h("div", { class: `fit-verdict ${fit.fits_at_native_ctx ? "yes" : "no"}` },
        fit.fits_at_native_ctx ? "Fits at native context" : "Doesn't fit at native context"),
    ),
    h("div", { class: "fit-bar-wrap" },
      h("div", { class: "fit-bar" },
        barSeg(b.weight_bytes, scale, "weight", `Weights: ${fmt.gb(b.weight_bytes)}`),
        barSeg(b.kv_bytes_at_native, scale, "kv", `KV cache @ native: ${fmt.gb(b.kv_bytes_at_native)}`),
        barSeg(b.compute_overhead_bytes, scale, "overhead", `Compute overhead: ${fmt.gb(b.compute_overhead_bytes)}`),
        ceilingLine(usable, scale, "comfort", `Comfort ceiling: ${fmt.gb(usable)}`),
        hard ? ceilingLine(hard, scale, "hard", `Hard ceiling: ${fmt.gb(hard)}`) : null,
      ),
      h("div", { class: "fit-bar-legend" },
        legendChip("weight", "Weights"),
        legendChip("kv", "KV cache (native)"),
        legendChip("overhead", "Overhead"),
        legendChip("comfort", "Comfort ceiling", true),
        hard ? legendChip("hard", "Hard ceiling", true) : null,
      ),
    ),
    h("div", { class: "fit-metrics" },
      metric("Max context that fits", `${fmt.int(fit.max_ctx_that_fits)} tokens`),
      (fit.hard_max_ctx_that_fits && fit.hard_max_ctx_that_fits > fit.max_ctx_that_fits)
        ? metric("Loads (with slowdown) to", `${fmt.int(fit.hard_max_ctx_that_fits)} tokens`)
        : null,
      metric("Storage", fit.storage_ok ? "ok" : "INSUFFICIENT"),
    ),
    fit.kv_quant_suggestion
      ? h("p", { class: "tip" }, `Tip: ${fit.kv_quant_suggestion}`)
      : null,
    ...fit.notes.map((n) => h("p", { class: "note" }, n)),
  );
}

function barSeg(bytes, scale, cls, title) {
  const pct = (bytes / scale) * 100;
  return h("div", {
    class: `bar-seg bar-${cls}`,
    style: `width:${pct}%`,
    title,
  });
}

function ceilingLine(bytes, scale, cls, title) {
  const pct = (bytes / scale) * 100;
  return h("div", {
    class: `ceiling ceiling-${cls}`,
    style: `left:${pct}%`,
    title,
  }, h("span", { class: "ceiling-label" }, cls));
}

function legendChip(cls, label, isCeiling = false) {
  return h("span", { class: `legend-chip legend-${cls}${isCeiling ? " legend-ceiling" : ""}` }, label);
}

// --------------------------------------------------------------------------
// Boot
// --------------------------------------------------------------------------
wirePicker();
renderMachine();
loadModels();

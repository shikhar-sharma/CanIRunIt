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
    quant: "Q4_K_M",   // weight quant (GGUF only; other runtimes ignore it)
    kv_quant: "f16",   // KV cache element quant
  },
  lastCheck: null,     // whole /api/check response for the current selection
  charts: { kv: null, decode: null },
};

// Debounce delay for control changes -> /api/check re-fetch. Signed off in
// the spec discussion.
const CONTROL_DEBOUNCE_MS = 250;

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
  const { model, runtime, quant, kv_quant } = state.selection;
  if (!model) return;

  const fitEl = document.getElementById("fit-content");
  fitEl.classList.remove("empty", "error");
  fitEl.classList.add("pending");
  fitEl.textContent = "Fetching model details…";

  try {
    const resp = await API.check({ model, runtime, quant, kv_quant });
    state.lastCheck = resp;
    fitEl.classList.remove("pending");
    renderFit(resp);
    // Reveal the quant controls + curves once we have data.
    document.getElementById("quant-panel").hidden = false;
    document.getElementById("kv-curve-panel").hidden = false;
    document.getElementById("decode-curve-panel").hidden = false;
    reconcileQuantControls(resp);
    renderKVCurve(resp);
    renderDecodeCurve(resp);
  } catch (err) {
    fitEl.classList.remove("pending");
    fitEl.classList.add("error");
    fitEl.textContent = `Error: ${err.message}`;
  }
}

// The debounced re-check invoked when a quant control changes. Coalesces
// rapid changes (dropdown clicks in a row) into one server call.
let _controlDebounceTimer = null;
function scheduleRecheck() {
  clearTimeout(_controlDebounceTimer);
  _controlDebounceTimer = setTimeout(() => {
    _controlDebounceTimer = null;
    onSelectionChanged();
  }, CONTROL_DEBOUNCE_MS);
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
// F. Quantization controls
// --------------------------------------------------------------------------
function wireQuantControls() {
  const weight = document.getElementById("weight-quant");
  const kv = document.getElementById("kv-quant");
  weight.addEventListener("change", (e) => {
    state.selection.quant = e.target.value;
    scheduleRecheck();
  });
  kv.addEventListener("change", (e) => {
    state.selection.kv_quant = e.target.value;
    scheduleRecheck();
  });
}

function reconcileQuantControls(resp) {
  // Weight quant is meaningful only for GGUF; MLX and Ollama bake it into
  // the repo/tag. Disable the control on the other runtimes and label it
  // with the intrinsic quant that came back so the user isn't confused.
  const runtime = resp.spec.runtime;
  const weight = document.getElementById("weight-quant");
  if (runtime === "gguf") {
    weight.disabled = false;
    weight.title = "";
  } else {
    weight.disabled = true;
    weight.title = `intrinsic to ${runtime} — set by the repo/tag (${resp.spec.quant_label || resp.spec.quant})`;
  }
  document.getElementById("kv-quant").value = state.selection.kv_quant;
}

// --------------------------------------------------------------------------
// D. KV-cache curve  (uPlot)
// --------------------------------------------------------------------------
function renderKVCurve(resp) {
  const container = document.getElementById("kv-curve-chart");
  container.replaceChildren();      // wipe any prior plot

  const { memory_curve } = resp;
  const points = memory_curve.points;
  const xs = points.map((p) => p.ctx);
  // uPlot needs y-series aligned with x. Y in GB for readability.
  const toGB = (b) => b / 1e9;
  const weights = points.map(() => toGB(memory_curve.weight_bytes));
  const withKV = points.map((p) => toGB(memory_curve.weight_bytes + p.kv_bytes));
  const withOverhead = points.map((p) => toGB(memory_curve.weight_bytes + p.kv_bytes + memory_curve.overhead_bytes));

  const comfortGB = toGB(memory_curve.usable_bytes);
  const hardGB = memory_curve.hard_usable_bytes != null ? toGB(memory_curve.hard_usable_bytes) : null;

  const opts = {
    width: container.clientWidth || 720,
    height: 260,
    scales: {
      x: { time: false },
      y: { auto: true },
    },
    axes: [
      { label: "Context (tokens)", labelSize: 24 },
      { label: "Memory (GB)", labelSize: 34, size: 60 },
    ],
    series: [
      { label: "Context" },
      { label: "Weights",        stroke: getVar("--weight"),   fill: getVar("--weight"), width: 1, points: { show: false } },
      { label: "Weights + KV",   stroke: getVar("--kv"),       fill: getVar("--kv"),     width: 1, points: { show: false } },
      { label: "Total (+overhead)", stroke: getVar("--overhead"), width: 1, points: { show: false } },
    ],
    hooks: {
      // Draw comfort + hard ceilings and the max-ctx marker on top of the plot.
      draw: [
        (u) => drawCeilingLines(u, comfortGB, hardGB),
        (u) => drawMaxCtxMarker(u, resp.fit.max_ctx_that_fits),
      ],
    },
  };

  state.charts.kv = new uPlot(opts, [xs, weights, withKV, withOverhead], container);
}

function drawCeilingLines(u, comfortGB, hardGB) {
  const ctx = u.ctx;
  const { left, top, width, height } = u.bbox;
  const drawLine = (yVal, color, dash) => {
    if (yVal == null) return;
    const y = u.valToPos(yVal, "y", true);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(left + width, y);
    ctx.stroke();
    ctx.restore();
    // Small right-side label
    ctx.save();
    ctx.fillStyle = color;
    ctx.font = "10px system-ui, -apple-system, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(dash ? "hard" : "comfort", left + width - 4, y - 4);
    ctx.restore();
  };
  drawLine(comfortGB, getVar("--comfort"), null);
  if (hardGB != null) drawLine(hardGB, getVar("--hard"), [4, 3]);
}

function drawMaxCtxMarker(u, maxCtx) {
  if (!maxCtx || maxCtx <= 0) return;
  const ctx = u.ctx;
  const { top, height } = u.bbox;
  const x = u.valToPos(maxCtx, "x", true);
  ctx.save();
  ctx.strokeStyle = getVar("--accent");
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 3]);
  ctx.beginPath();
  ctx.moveTo(x, top);
  ctx.lineTo(x, top + height);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = getVar("--accent");
  ctx.font = "10px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(`max ctx: ${maxCtx.toLocaleString()}`, x + 4, top + 12);
  ctx.restore();
}

// --------------------------------------------------------------------------
// E. Decode-decay curve  (uPlot)
// --------------------------------------------------------------------------
function renderDecodeCurve(resp) {
  const container = document.getElementById("decode-curve-chart");
  container.replaceChildren();

  const points = resp.speed.points;
  const xs = points.map((p) => p.ctx);
  const decode = points.map((p) => p.decode_tok_s);
  const measured = resp.speed.confidence === "measured";

  const opts = {
    width: container.clientWidth || 720,
    height: 220,
    scales: { x: { time: false }, y: { auto: true } },
    axes: [
      { label: "Context (tokens)", labelSize: 24 },
      { label: "Decode (tok/s)", labelSize: 34, size: 60 },
    ],
    series: [
      { label: "Context" },
      {
        label: measured ? "Decode (measured)" : "Decode (estimated)",
        stroke: measured ? getVar("--accent") : getVar("--text-dim"),
        width: measured ? 2 : 1.5,
        dash: measured ? null : [4, 3],
        points: { show: true, size: 4, stroke: measured ? getVar("--accent") : getVar("--text-dim") },
      },
    ],
  };

  state.charts.decode = new uPlot(opts, [xs, decode], container);

  // Update caption + verdict styling to match confidence.
  const cap = document.getElementById("decode-caption");
  cap.classList.toggle("measured", measured);
  cap.classList.toggle("estimated", !measured);
}

function getVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Resize handler: uPlot doesn't reflow on its own. Debounce so we don't
// tear down / rebuild on every resize event.
let _resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(() => {
    if (state.lastCheck) {
      renderKVCurve(state.lastCheck);
      renderDecodeCurve(state.lastCheck);
    }
  }, 120);
});

// --------------------------------------------------------------------------
// Boot
// --------------------------------------------------------------------------
wirePicker();
wireQuantControls();
renderMachine();
loadModels();

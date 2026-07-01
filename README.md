# canirunit

Estimate whether a local **GGUF**, **MLX**, or **Ollama** model will fit and
run well on your machine — before you download gigabytes to find out. CLI or
local web UI.

It answers two separate questions, because they have different physics and
different reliability:

- **Will it fit?** Can you store the weights, and load them plus the working
  memory (KV cache, compute buffers) without the OS choking? This is
  deterministic and answered with confidence.
- **Will it run fast enough?** What decode throughput and time-to-first-token
  will you actually see? This is estimable but uncertain — so it's given as a
  curve, and can be replaced with a *measured* number via on-machine calibration.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/shikhar-sharma/CanIRunIt.git
cd CanIRunIt
pip install -e .

# Optional: the local web UI
pip install -e ".[ui]"
```

## Usage

Check a specific model by its Hugging Face repo id:

```bash
canirunit check bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
canirunit check bartowski/Meta-Llama-3.1-8B-Instruct-GGUF --quant Q5_K_M --ctx 16384

# Other runtimes:
canirunit check mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --runtime mlx
canirunit check llama3.1:8b --runtime ollama

# Compare a logical model across every runtime that has a source:
canirunit compare llama-3.1-8b-instruct
canirunit compare llama-3.1-8b-instruct --calibrate

# Browse / refresh the alias table:
canirunit models
canirunit refresh
```

### Web UI

Once `canirunit[ui]` is installed:

```bash
canirunit serve
```

Opens a page on `http://127.0.0.1:8765/` (bound to localhost only; no auth, no
external exposure). The UI is a teaching layer over the CLI: pick a model,
see whether it fits at native context, watch the KV cache overtake the
usable-memory ceiling as context grows, compare runtimes side by side, and
calibrate to trade the estimated decode curve for a measured one.

No bundled desktop app; the floor for non-CLI users is `canirunit serve`
plus a browser. That's on purpose — a native app bundle is a separate
future effort.

Example output:

```
canirunit — bartowski/Meta-Llama-3.1-8B-Instruct-GGUF  (Q4_K_M)
Machine: Apple M1  [apple_metal]  ·  68 GB/s  ·  17.2 GB total, 11.8 GB usable (Metal working set)

FIT
  Fits at native context (131072):   no
  Max context that fits:           49655 tokens
  Storage for weights:             ok  (4.9 GB needed, 47.9 GB free)
  Memory at native context:        weights 4.9 + KV 17.2 + overhead 0.4 = 22.5 GB  (usable 11.8 GB)
  Suggestion: q4 KV cache reaches native context (131072 tokens)

SPEED  [estimated]
  context     decode        time-to-first-token
      2048      9.2 tok/s    ...
      8192      8.0 tok/s    ...
```

### Calibration (measured speed)

Speed defaults to a *static estimate* from hardware tables. For numbers measured
on your actual machine, add `--calibrate`:

```bash
canirunit check bartowski/Meta-Llama-3.1-8B-Instruct-GGUF --calibrate
```

Calibration runs a small model through an existing runtime and backs out your
machine's real effective throughput. It needs:

- **GGUF / Ollama**: llama.cpp's `llama-bench` on your PATH
  (`brew install llama.cpp`).
- **MLX**: `pip install mlx-lm` on an Apple Silicon Mac.

Without it, the tool falls back to the static estimate and tells you so.
Calibrations are runtime-tagged: an MLX-measured constant won't be applied to
a GGUF target (or vice versa) — the estimator drops cross-runtime
calibrations and notes the fallback.

## How it works

- **Sizes come from the file, not the model card.** The weight footprint is the
  actual GGUF file size(s) from the repo listing. Architecture parameters
  (layers, KV heads, head dims, context) are range-read directly from the GGUF
  header — a small request, no full download.
- **Fit is KV-cache-aware.** The KV cache grows linearly with context and can
  rival the weights at long context, so the tool reports the *maximum context
  that fits* (binary-searched), not a single yes/no, and suggests a cheaper KV
  quantization when that's what's needed to reach native context.
- **Decode is memory-bandwidth-bound**, prefill is compute-bound — estimated
  separately. Decode slows as context fills (KV in the denominator), which is why
  speed is a curve.
- **MoE models** carry two footprints: the full weights for fit, the active
  weights for decode speed.
- **Calibration** measures one small model and inverts the same formulas to
  recover your machine's effective bytes/s and FLOP/s.

## Scope and honest caveats

- **GGUF, MLX, and Ollama.** Other runtimes (raw transformers, vLLM) have
  different footprints and aren't covered.
- **MLX `fetch` works anywhere; `--calibrate` needs Apple Silicon + `mlx_lm`.**
  An MLX-tagged calibration only applies to MLX targets.
- **Apple's wired-memory limit is soft.** The fit report shows a two-ceiling
  view: the *comfort* `Max context that fits` (fully resident under the wired
  limit) and the *hard* `Loads (with slowdown past wired limit)` ceiling that
  uses compression/swap. CUDA shows one ceiling (VRAM is hard).
- **Prefill / TTFT is the soft spot until you calibrate.** The static
  `peak_flops` table is coarse; decode is reliable, prefill is flagged
  low-confidence and is what calibration fixes.
- **MoE active footprint** is approximated by the parameter fraction (experts
  vs the rest), not exact per-tensor bytes.
- **MLA / compressed-KV architectures** (e.g. DeepSeek V2/V3) are *flagged*,
  not modelled — the standard KV formula doesn't apply.
- **Ollama calibration** uses the GGUF path (it IS GGUF physics); running
  `--calibrate --runtime ollama` requires `llama-bench`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

### Maintainer: regenerating the alias table

`canirunit compare` resolves a logical id (e.g. `llama-3.1-8b-instruct`) to
per-runtime sources via `src/canirunit/data/aliases.json`. To regenerate from
Hugging Face listings:

```bash
python scripts/build_aliases.py                                   # dry skeleton
python scripts/build_aliases.py --live                            # crawl HF, print JSON
python scripts/build_aliases.py --live --out data/aliases.json    # write the publication mirror
```

The script never auto-commits. Review the diff, fix any `family: "TODO"`
and any wrong Ollama tags, then commit. `data/aliases.json` is the published
artifact `canirunit refresh` pulls; keep `src/canirunit/data/aliases.json`
(shipped) in sync with it at release time.

## License

MIT — see `LICENSE`.

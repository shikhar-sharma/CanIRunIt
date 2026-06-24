# canirunit

Estimate whether a local **GGUF** model will fit and run well on your machine —
before you download gigabytes to find out.

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
```

## Usage

Check a specific model by its Hugging Face repo id:

```bash
canirunit check bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
canirunit check bartowski/Meta-Llama-3.1-8B-Instruct-GGUF --quant Q5_K_M --ctx 16384
```

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
machine's real effective throughput. It needs **llama.cpp's `llama-bench`** on
your PATH (e.g. `brew install llama.cpp`). Without it, the tool falls back to the
static estimate and tells you so. Calibration is what makes the prefill / TTFT
numbers trustworthy.

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

## Scope and honest caveats (v1)

- **GGUF only.** The llama.cpp / Ollama / LM Studio path. Other runtimes (MLX,
  raw transformers) have different footprints and aren't covered yet.
- **One flow:** check a specific model. The "recommend models that fit" flow is
  planned, and reuses the same core.
- **Prefill / TTFT is the soft spot until you calibrate.** The static `peak_flops`
  table is coarse; decode is reliable, prefill is flagged low-confidence and is
  what calibration fixes.
- **Apple Metal working set** uses a ~0.75-of-unified-memory approximation when
  macOS doesn't expose `iogpu.wired_limit_mb`; verify against real behaviour on
  low-RAM machines.
- **MoE active footprint** is approximated by the parameter fraction (experts vs
  the rest), not exact per-tensor bytes.
- **MLA / compressed-KV architectures** (e.g. DeepSeek) are *flagged*, not
  modelled — the standard KV formula doesn't apply.
- **Ollama** output parsing is implemented; wiring it as a calibration backend is
  a follow-up. Calibration v1 means `llama-bench`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see `LICENSE`.

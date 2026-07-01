"""On-machine calibration.

Runs a small known GGUF through an existing runtime (llama-bench preferred, then
ollama), measures real prompt-processing and token-generation throughput, and
backs those out into a Calibration whose constants the estimator substitutes for
its spec-sheet guesses.

Why shell out instead of embedding a runtime: the measured constant is
runtime-specific (it absorbs MBU, kernel quality, build flags, thermals). Measuring
through the runtime the user actually runs makes the calibration match their real
experience, and keeps a C++ toolchain out of the install path.

The back-out is the exact inverse of the estimator's forward model: decode is
bandwidth-bound (tok/s x bytes/token = bytes/s) and prefill is compute-bound
(tok/s x 2*params = FLOP/s). So a calibration fed back into the estimator
reproduces the measured throughput.

Parsers and back-out are pure and tested here; the subprocess/download run on a
machine with a runtime installed.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

from .config import EstimatorConfig
# FileReader lives in gguf.py; re-exported here so existing imports
# (`from canirunit.benchmark import FileReader`) keep working.
from .gguf import FileReader, parse_gguf
from .hub import build_model_spec
from .types import Calibration, ModelSpec, Runtime, SystemProfile

__all__ = [
    "BenchResult",
    "BenchModel",
    "DEFAULT_BENCH_MODEL",
    "DEFAULT_MLX_BENCH_REPO",
    "FileReader",
    "calibrate",
    "calibration_from_bench",
    "find_runtime",
    "find_runtime_for",
    "parse_llama_bench_json",
    "parse_llama_bench_text",
    "parse_mlx_lm",
    "parse_ollama_verbose",
]

# llama-bench defaults: prompt-processing over 512 tokens, generation of 128.
# The average context during a tg128 run is ~half the generated length.
_DEFAULT_GEN_CTX = 64


@dataclass
class BenchResult:
    pp_tok_s: Optional[float]   # prompt processing (prefill) throughput
    tg_tok_s: Optional[float]   # token generation (decode) throughput
    raw: str = ""


@dataclass
class BenchModel:
    repo_id: str
    filename: str
    quant: str


# Small (~400 MB), fast to fetch, runs anywhere — a good calibration target.
DEFAULT_BENCH_MODEL = BenchModel(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
    quant="Q4_K_M",
)


# --------------------------------------------------------------------------- #
# Parsers (pure)
# --------------------------------------------------------------------------- #
def parse_llama_bench_json(out: str) -> BenchResult:
    """Parse `llama-bench -o json`. Each row reports one test; a row with n_gen>0
    is the generation test, n_prompt>0 & n_gen==0 the prompt-processing test."""
    data = json.loads(out)
    pp = tg = None
    for row in data:
        n_gen = int(row.get("n_gen", 0) or 0)
        n_prompt = int(row.get("n_prompt", 0) or 0)
        ts = row.get("avg_ts")
        if ts is None:
            continue
        if n_gen > 0:
            tg = float(ts)
        elif n_prompt > 0:
            pp = float(ts)
    return BenchResult(pp_tok_s=pp, tg_tok_s=tg, raw=out)


_TEST_RE = re.compile(r"\b(pp|tg)\d+\b")
_FLOAT_RE = re.compile(r"[\d.]+")


def parse_llama_bench_text(out: str) -> BenchResult:
    """Fallback parser for the markdown-table output of older/odd builds."""
    pp = tg = None
    for line in out.splitlines():
        if "|" not in line:
            continue
        m = _TEST_RE.search(line)
        if not m:
            continue
        cells = [c.strip() for c in line.split("|")]
        ts_cell = next((c for c in reversed(cells) if _FLOAT_RE.match(c)), None)
        if ts_cell is None:
            continue
        value = float(_FLOAT_RE.match(ts_cell).group())
        if m.group(1) == "tg":
            tg = value
        else:
            pp = value
    return BenchResult(pp_tok_s=pp, tg_tok_s=tg, raw=out)


def parse_ollama_verbose(out: str) -> BenchResult:
    """Parse `ollama run --verbose` stats. 'eval rate' is decode; 'prompt eval
    rate' is prefill."""
    pp = tg = None
    for line in out.splitlines():
        s = line.strip()
        m = re.search(r"([\d.]+)\s*tokens/s", s)
        if not m:
            continue
        rate = float(m.group(1))
        if s.startswith("prompt eval rate"):
            pp = rate
        elif s.startswith("eval rate"):
            tg = rate
    return BenchResult(pp_tok_s=pp, tg_tok_s=tg, raw=out)


# Tolerant: mlx_lm's exact wording varies by version. Match the rate that
# directly precedes "tokens-per-sec" / "tok/s" / "tps" — not the prompt-length
# count earlier in the line. The "prompt"/"generation" anchor decides which
# bucket the rate lands in.
_MLX_RATE_RE = re.compile(r"([\d.]+)\s*(?:tokens-per-sec|tok/s|tps)\b", re.IGNORECASE)


def parse_mlx_lm(out: str) -> BenchResult:
    """Parse the throughput summary printed by ``python -m mlx_lm generate``.

    Lines look like (exact wording is version-dependent):
        Prompt: 512 tokens, 1234.5 tokens-per-sec
        Generation: 128 tokens, 88.7 tokens-per-sec
    Older builds may print ``tok/s`` instead of ``tokens-per-sec``.
    """
    pp = tg = None
    for line in out.splitlines():
        m = _MLX_RATE_RE.search(line)
        if not m:
            continue
        value = float(m.group(1))
        low = line.lower()
        if "prompt" in low:
            pp = value
        elif "generation" in low or "decode" in low:
            tg = value
    return BenchResult(pp_tok_s=pp, tg_tok_s=tg, raw=out)


# --------------------------------------------------------------------------- #
# Back-out (pure): BenchResult + bench model -> Calibration
# --------------------------------------------------------------------------- #
def calibration_from_bench(
    bench: BenchResult,
    bench_spec: ModelSpec,
    chip_id: str,
    source: str,
    gen_ctx: int = _DEFAULT_GEN_CTX,
    config: Optional[EstimatorConfig] = None,
    *,
    runtime: Runtime = "gguf",
) -> Calibration:
    """Invert the estimator's forward model to recover its constants.

    decode:  effective_bytes_per_sec = tg_tok_s * (active_weight_bytes + kv@gen_ctx)
    prefill: prefill_flops_per_sec   = pp_tok_s * 2 * active_params

    ``runtime`` tags which runtime family the resulting constants apply to.
    The estimator's calibration guard uses this to refuse cross-application
    (e.g. an MLX-measured constant on a GGUF target).
    """
    from .estimator import kv_cache_bytes  # local import avoids a cycle at import time

    config = config or EstimatorConfig()
    if not bench.tg_tok_s:
        raise ValueError("cannot calibrate without a token-generation measurement")

    kv = kv_cache_bytes(bench_spec, gen_ctx, "f16", config)
    eff_bps = bench.tg_tok_s * (bench_spec.active_weight_bytes + kv)

    pf_flops = None
    # Only trust the prefill anchor if the bench param count is plausible. A near-
    # zero count (e.g. a GGUF without parameter_count and unparsed tensors) would
    # otherwise produce a microscopic FLOP/s and absurd TTFTs — fall back to static.
    if bench.pp_tok_s and bench_spec.decode_active_params > 1_000_000:
        pf_flops = bench.pp_tok_s * 2 * bench_spec.decode_active_params

    return Calibration(
        effective_bytes_per_sec=eff_bps,
        measured_on_chip=chip_id,
        source=source,
        prefill_flops_per_sec=pf_flops,
        runtime=runtime,
    )


# --------------------------------------------------------------------------- #
# Runtime discovery
# --------------------------------------------------------------------------- #
_RUNTIME_BINARIES = [("llama-bench", "llama-bench"), ("ollama", "ollama")]


def find_runtime(which: Callable[[str], Optional[str]] = shutil.which) -> Optional[str]:
    """Preference order is signal quality: llama-bench (purpose-built, machine
    readable) before ollama. Returns None if nothing usable is installed.

    Kept for backward compatibility; the runtime-aware path is
    ``find_runtime_for``.
    """
    for name, binary in _RUNTIME_BINARIES:
        if which(binary):
            return name
    return None


def _mlx_available() -> bool:
    """True iff this machine can run mlx_lm: Apple Silicon + ``mlx_lm`` importable.

    Never import ``mlx_lm`` at module top — it's Apple-only and we don't want
    test runs on Linux/Intel Mac to fail at import time.
    """
    import platform

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import importlib

        importlib.import_module("mlx_lm")
    except ImportError:
        return False
    return True


def find_runtime_for(
    target_runtime: Runtime,
    which: Callable[[str], Optional[str]] = shutil.which,
    mlx_available: Callable[[], bool] = _mlx_available,
) -> Optional[str]:
    """Pick the best installed measurement tool for a target runtime.

    Returns the tool name (``"llama-bench" | "ollama" | "mlx_lm"``) or None.
    ``which`` and ``mlx_available`` are injectable so tests don't depend on
    the host's installed binaries.
    """
    if target_runtime in ("gguf", "ollama"):
        for name, binary in _RUNTIME_BINARIES:
            if which(binary):
                return name
        return None
    if target_runtime == "mlx":
        return "mlx_lm" if mlx_available() else None
    return None


# --------------------------------------------------------------------------- #
# Orchestration (network + subprocess; runs on a machine with a runtime)
# --------------------------------------------------------------------------- #
def _run(args: list) -> Optional[str]:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=600)
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _download(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


def _local_bench_spec(bench: BenchModel, path: str) -> ModelSpec:
    import os

    # need_tensors=True: the bench GGUF may lack general.parameter_count, in which
    # case the param count is summed from the tensor table. The prefill back-out
    # depends on it, and the file is local so a full parse is cheap.
    info = parse_gguf(FileReader(path), need_tensors=True)
    return build_model_spec(bench.repo_id, bench.quant, info, os.path.getsize(path))


# Small (~400 MB), already an MLX-community 4-bit quant. Verified on-device
# step in §12 confirms the repo + the mlx_lm CLI output format.
DEFAULT_MLX_BENCH_REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# Default prompt for mlx_lm — long enough to make pp meaningful, short enough
# to keep the calibration fast. The exact tokens don't matter for throughput.
_DEFAULT_MLX_PROMPT = "The quick brown fox " * 80


def _mlx_snapshot_download(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id)


def _mlx_bench_spec_from_local(model_ref: str, snapshot_path: str) -> ModelSpec:
    """Build a ModelSpec for the calibration target from a local MLX snapshot."""
    import os

    from .source_mlx import build_mlx_spec

    with open(os.path.join(snapshot_path, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    weight_bytes = sum(
        os.path.getsize(os.path.join(snapshot_path, name))
        for name in os.listdir(snapshot_path)
        if name.lower().endswith(".safetensors")
    )
    return build_mlx_spec(model_ref, config, weight_bytes)


def _calibrate_gguf(
    profile: SystemProfile,
    runner: Callable[[list], Optional[str]],
    bench: BenchModel,
    downloader: Callable[[str, str], str],
    config: Optional[EstimatorConfig],
) -> Optional[Calibration]:
    runtime_tool = find_runtime_for("gguf", which=shutil.which)
    if runtime_tool != "llama-bench":
        # Ollama-as-calibration-backend is a follow-up; today only llama-bench
        # is wired up. The parser is ready (`parse_ollama_verbose`).
        return None

    path = downloader(bench.repo_id, bench.filename)
    bench_spec = _local_bench_spec(bench, path)
    out = runner(["llama-bench", "-m", path, "-o", "json", "-p", "512", "-n", "128"])
    if out is None:
        return None
    result = parse_llama_bench_json(out)
    if not result.tg_tok_s:
        return None
    return calibration_from_bench(
        result, bench_spec, profile.chip_id, runtime_tool, config=config, runtime="gguf"
    )


def _calibrate_mlx(
    profile: SystemProfile,
    runner: Callable[[list], Optional[str]],
    mlx_repo: str,
    snapshot_downloader: Callable[[str], str],
    mlx_available: Callable[[], bool],
    config: Optional[EstimatorConfig],
) -> Optional[Calibration]:
    if not mlx_available():
        return None

    snapshot_path = snapshot_downloader(mlx_repo)
    bench_spec = _mlx_bench_spec_from_local(mlx_repo, snapshot_path)
    out = runner([
        "python", "-m", "mlx_lm", "generate",
        "--model", snapshot_path,
        "--prompt", _DEFAULT_MLX_PROMPT,
        "--max-tokens", "128",
    ])
    if out is None:
        return None
    result = parse_mlx_lm(out)
    if not result.tg_tok_s:
        return None
    return calibration_from_bench(
        result, bench_spec, profile.chip_id, "mlx_lm", config=config, runtime="mlx"
    )


def calibrate(
    profile: SystemProfile,
    runner: Callable[[list], Optional[str]] = _run,
    bench: BenchModel = DEFAULT_BENCH_MODEL,
    downloader: Callable[[str, str], str] = _download,
    config: Optional[EstimatorConfig] = None,
    *,
    target_runtime: Runtime = "gguf",
    mlx_repo: str = DEFAULT_MLX_BENCH_REPO,
    mlx_snapshot_downloader: Callable[[str], str] = _mlx_snapshot_download,
    mlx_available: Callable[[], bool] = _mlx_available,
) -> Optional[Calibration]:
    """Run a calibration against the best available runtime.

    Returns None when no runtime is installed or the run fails — the caller
    then falls back to the static estimate. ``target_runtime`` selects which
    backend to use:
      * ``"gguf"``/``"ollama"``: llama-bench, with a small GGUF download.
      * ``"mlx"``: ``python -m mlx_lm generate``, with a small MLX snapshot.

    The resulting Calibration is tagged with the runtime family it applies
    to; the estimator refuses to cross-apply (see ``_calibration_applies``).
    """
    if target_runtime == "mlx":
        return _calibrate_mlx(
            profile,
            runner=runner,
            mlx_repo=mlx_repo,
            snapshot_downloader=mlx_snapshot_downloader,
            mlx_available=mlx_available,
            config=config,
        )
    # gguf and ollama share the llama.cpp physics and tooling; treat both as
    # gguf for calibration. (Ollama's own --verbose path is a follow-up.)
    return _calibrate_gguf(
        profile,
        runner=runner,
        bench=bench,
        downloader=downloader,
        config=config,
    )

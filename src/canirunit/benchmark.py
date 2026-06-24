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
from .gguf import parse_gguf
from .hub import build_model_spec
from .types import Calibration, ModelSpec, SystemProfile

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
) -> Calibration:
    """Invert the estimator's forward model to recover its constants.

    decode:  effective_bytes_per_sec = tg_tok_s * (active_weight_bytes + kv@gen_ctx)
    prefill: prefill_flops_per_sec   = pp_tok_s * 2 * active_params
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
    )


# --------------------------------------------------------------------------- #
# Runtime discovery
# --------------------------------------------------------------------------- #
_RUNTIME_BINARIES = [("llama-bench", "llama-bench"), ("ollama", "ollama")]


def find_runtime(which: Callable[[str], Optional[str]] = shutil.which) -> Optional[str]:
    """Preference order is signal quality: llama-bench (purpose-built, machine
    readable) before ollama. Returns None if nothing usable is installed."""
    for name, binary in _RUNTIME_BINARIES:
        if which(binary):
            return name
    return None


# --------------------------------------------------------------------------- #
# Orchestration (network + subprocess; runs on a machine with a runtime)
# --------------------------------------------------------------------------- #
class FileReader:
    """ByteReader over a local file — for parsing an already-downloaded GGUF."""

    def __init__(self, path: str):
        self.path = path

    def read_range(self, start: int, length: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(start)
            return f.read(length)


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


def calibrate(
    profile: SystemProfile,
    runner: Callable[[list], Optional[str]] = _run,
    bench: BenchModel = DEFAULT_BENCH_MODEL,
    downloader: Callable[[str, str], str] = _download,
    config: Optional[EstimatorConfig] = None,
) -> Optional[Calibration]:
    """Run a calibration against the best available runtime. Returns None when no
    runtime is installed or the run fails — the caller then falls back to the
    static estimate. Currently wires the llama-bench path; the ollama parser is
    ready but its orchestration (model naming, spec lookup) is a follow-up."""
    runtime = find_runtime()
    if runtime != "llama-bench":
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
        result, bench_spec, profile.chip_id, runtime, config=config
    )

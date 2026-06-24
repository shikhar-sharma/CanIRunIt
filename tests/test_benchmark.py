"""Benchmark tests: parsers on captured output, the round-trip back-out invariant,
and runtime discovery."""
from __future__ import annotations

import pytest

from canirunit import EstimatorConfig, ModelSpec, SystemProfile
from canirunit.benchmark import (
    BenchResult,
    calibration_from_bench,
    find_runtime,
    parse_llama_bench_json,
    parse_llama_bench_text,
    parse_ollama_verbose,
)
from canirunit.estimator import decode_tok_s, prefill_tok_s

GiB = 1024 ** 3


# --------------------------------------------------------------------------- #
# Captured sample outputs
# --------------------------------------------------------------------------- #
LLAMA_BENCH_JSON = """
[
  {"model_filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf", "n_prompt": 512, "n_gen": 0, "avg_ts": 1234.5, "stddev_ts": 10.2},
  {"model_filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf", "n_prompt": 0, "n_gen": 128, "avg_ts": 88.7, "stddev_ts": 0.5}
]
"""

LLAMA_BENCH_TEXT = """\
| model            |     size |   params | backend | ngl |  test |              t/s |
| ---------------- | -------: | -------: | ------- | --: | ----: | ---------------: |
| qwen2 0.5B Q4_K_M | 397 MiB |  494 M   | Metal   |  99 | pp512 | 1234.50 ± 10.20 |
| qwen2 0.5B Q4_K_M | 397 MiB |  494 M   | Metal   |  99 | tg128 |   88.70 ± 0.50 |
"""

OLLAMA_VERBOSE = """\
total duration:       2.5s
prompt eval count:    512 token(s)
prompt eval rate:     1234.50 tokens/s
eval count:           128 token(s)
eval rate:            88.70 tokens/s
"""


def test_parse_llama_bench_json():
    r = parse_llama_bench_json(LLAMA_BENCH_JSON)
    assert r.pp_tok_s == 1234.5
    assert r.tg_tok_s == 88.7


def test_parse_llama_bench_text():
    r = parse_llama_bench_text(LLAMA_BENCH_TEXT)
    assert r.pp_tok_s == 1234.5
    assert r.tg_tok_s == 88.7


def test_parse_ollama_verbose():
    r = parse_ollama_verbose(OLLAMA_VERBOSE)
    assert r.pp_tok_s == 1234.5
    assert r.tg_tok_s == 88.7


# --------------------------------------------------------------------------- #
# Round-trip: back-out is the exact inverse of the estimator's forward model
# --------------------------------------------------------------------------- #
@pytest.fixture
def qwen_0_5b():
    """Qwen2.5-0.5B Q4_K_M: dense, 24 layers, 2 KV heads, head_dim 64."""
    return ModelSpec(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF", quant="Q4_K_M",
        total_weight_bytes=397_000_000, active_weight_bytes=397_000_000,
        total_params=494_000_000,
        n_layers=24, n_kv_heads=2, key_length=64, value_length=64,
        native_ctx=32768, architecture="qwen2",
    )


@pytest.fixture
def any_profile():
    # Calibration overrides bandwidth, so the profile's own numbers are irrelevant.
    return SystemProfile(
        total_memory_bytes=16 * GiB, available_memory_bytes=12 * GiB,
        memory_bandwidth_gbs=68.0, accelerator="apple_metal", chip_id="Apple M1",
        storage_free_bytes=100 * GiB, metal_max_working_set_bytes=12 * GiB, peak_flops=2.6e12,
    )


def test_decode_calibration_round_trips(qwen_0_5b, any_profile):
    """A calibration built from a measured tg must reproduce that tg when fed
    back through decode_tok_s at the same context."""
    gen_ctx = 64
    bench = BenchResult(pp_tok_s=1234.5, tg_tok_s=88.7)
    calib = calibration_from_bench(bench, qwen_0_5b, "Apple M1", "llama-bench", gen_ctx=gen_ctx)
    recovered = decode_tok_s(any_profile, qwen_0_5b, gen_ctx, EstimatorConfig(), calibration=calib)
    assert recovered == pytest.approx(88.7, rel=1e-9)


def test_prefill_calibration_round_trips(qwen_0_5b, any_profile):
    bench = BenchResult(pp_tok_s=1234.5, tg_tok_s=88.7)
    calib = calibration_from_bench(bench, qwen_0_5b, "Apple M1", "llama-bench")
    recovered = prefill_tok_s(any_profile, qwen_0_5b, EstimatorConfig(), calibration=calib)
    assert recovered == pytest.approx(1234.5, rel=1e-9)


def test_calibration_requires_tg(qwen_0_5b):
    with pytest.raises(ValueError, match="token-generation"):
        calibration_from_bench(BenchResult(pp_tok_s=100.0, tg_tok_s=None),
                               qwen_0_5b, "Apple M1", "llama-bench")


def test_calibration_without_pp_has_no_prefill_anchor(qwen_0_5b):
    calib = calibration_from_bench(BenchResult(pp_tok_s=None, tg_tok_s=88.7),
                                   qwen_0_5b, "Apple M1", "llama-bench")
    assert calib.effective_bytes_per_sec > 0
    assert calib.prefill_flops_per_sec is None


def test_prefill_anchor_skipped_when_bench_param_count_implausible():
    """Regression: a bench spec with a near-zero param count (no parameter_count
    key, unparsed tensors) must NOT produce a garbage prefill anchor — the cause
    of 260-year TTFTs in the field. Falls back to static prefill instead."""
    broken = ModelSpec(
        repo_id="x", quant="Q4_K_M", total_weight_bytes=491_000_000,
        active_weight_bytes=491_000_000, total_params=0,   # the bug condition
        n_layers=24, n_kv_heads=2, key_length=64, value_length=64,
        native_ctx=32768, architecture="qwen2",
    )
    calib = calibration_from_bench(BenchResult(pp_tok_s=2010.0, tg_tok_s=88.7),
                                   broken, "Apple M1", "llama-bench")
    assert calib.effective_bytes_per_sec > 0      # decode anchor still valid
    assert calib.prefill_flops_per_sec is None     # prefill anchor refused


# --------------------------------------------------------------------------- #
# Runtime discovery
# --------------------------------------------------------------------------- #
def test_find_runtime_prefers_llama_bench():
    which = lambda b: "/usr/bin/" + b if b in ("llama-bench", "ollama") else None
    assert find_runtime(which) == "llama-bench"


def test_find_runtime_falls_back_to_ollama():
    which = lambda b: "/usr/bin/ollama" if b == "ollama" else None
    assert find_runtime(which) == "ollama"


def test_find_runtime_none_when_absent():
    assert find_runtime(lambda b: None) is None

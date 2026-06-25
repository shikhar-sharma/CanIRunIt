"""Tests for the across-runtimes comparison.

Source fetches, resolve, and availability predicates are all injected so the
tests run without network, without the alias table, and without depending on
the host's accelerator.
"""
from __future__ import annotations

import pytest

from canirunit import EstimatorConfig, ModelSpec, SystemProfile
from canirunit.compare import RuntimeComparison, compare


GiB = 1024 ** 3


def m1_profile():
    return SystemProfile(
        total_memory_bytes=16 * GiB, available_memory_bytes=12 * GiB,
        memory_bandwidth_gbs=68.0, accelerator="apple_metal", chip_id="Apple M1",
        storage_free_bytes=100 * GiB, metal_max_working_set_bytes=12 * GiB, peak_flops=2.6e12,
    )


def _llama_spec(repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                runtime="gguf", quant_label="Q4_K_M"):
    return ModelSpec(
        repo_id=repo_id, quant=quant_label,
        total_weight_bytes=4_900_000_000, active_weight_bytes=4_900_000_000,
        total_params=8_030_000_000, n_layers=32, n_kv_heads=8,
        key_length=128, value_length=128, native_ctx=8192, architecture="llama",
        runtime=runtime, quant_label=quant_label,
    )


# A reusable alias entry for the tests.
LLAMA_ENTRY = {
    "display_name": "Llama 3.1 8B Instruct",
    "family": "llama",
    "sources": {
        "gguf":   {"repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "default_quant": "Q4_K_M"},
        "mlx":    {"repo_id": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"},
        "ollama": {"tag": "llama3.1:8b"},
    },
}


class _FakeSource:
    def __init__(self, spec=None, raises=None, runtime="gguf"):
        self.runtime = runtime
        self._spec = spec
        self._raises = raises
        self.calls = []

    def fetch(self, model_ref, quant=None):
        self.calls.append((model_ref, quant))
        if self._raises is not None:
            raise self._raises
        return self._spec


def _source_for_factory(by_runtime: dict):
    def source_for(runtime):
        return by_runtime[runtime]
    return source_for


# --------------------------------------------------------------------------- #
# Happy path: three rows, each from its own source
# --------------------------------------------------------------------------- #
def test_compare_returns_one_row_per_declared_runtime():
    sources = {
        "gguf":   _FakeSource(spec=_llama_spec(runtime="gguf",   quant_label="Q4_K_M"),   runtime="gguf"),
        "mlx":    _FakeSource(spec=_llama_spec(runtime="mlx",    quant_label="4bit-g64"), runtime="mlx"),
        "ollama": _FakeSource(spec=_llama_spec(runtime="ollama", quant_label="Q4_K_M"),   runtime="ollama"),
    }
    rows = compare(
        "llama-3.1-8b-instruct",
        m1_profile(),
        source_for=_source_for_factory(sources),
        resolve_fn=lambda lid: LLAMA_ENTRY,
        is_apple_silicon=lambda: True,
        mlx_available=lambda: True,
    )
    assert {r.runtime for r in rows} == {"gguf", "mlx", "ollama"}
    for r in rows:
        assert r.error is None
        assert r.fit is not None and r.speed is not None
        assert r.available is True
        assert r.quant_label is not None


def test_compare_per_runtime_source_receives_correct_ref():
    sources = {
        "gguf":   _FakeSource(spec=_llama_spec(), runtime="gguf"),
        "mlx":    _FakeSource(spec=_llama_spec(runtime="mlx", quant_label="4bit-g64"), runtime="mlx"),
        "ollama": _FakeSource(spec=_llama_spec(runtime="ollama"), runtime="ollama"),
    }
    compare(
        "x", m1_profile(),
        source_for=_source_for_factory(sources),
        resolve_fn=lambda lid: LLAMA_ENTRY,
        is_apple_silicon=lambda: True,
        mlx_available=lambda: True,
    )
    # Each source was called exactly once with its alias-declared ref.
    assert sources["gguf"].calls == [("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Q4_K_M")]
    assert sources["mlx"].calls == [("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", None)]
    assert sources["ollama"].calls == [("llama3.1:8b", None)]


# --------------------------------------------------------------------------- #
# Error isolation: one runtime fails, others still produce rows
# --------------------------------------------------------------------------- #
def test_compare_isolates_per_runtime_errors():
    sources = {
        "gguf":   _FakeSource(spec=_llama_spec(), runtime="gguf"),
        "mlx":    _FakeSource(raises=ValueError("repo not found"), runtime="mlx"),
        "ollama": _FakeSource(raises=FileNotFoundError("not pulled"), runtime="ollama"),
    }
    rows = compare(
        "x", m1_profile(),
        source_for=_source_for_factory(sources),
        resolve_fn=lambda lid: LLAMA_ENTRY,
        is_apple_silicon=lambda: True,
        mlx_available=lambda: True,
    )
    by_runtime = {r.runtime: r for r in rows}
    assert by_runtime["gguf"].error is None
    assert by_runtime["mlx"].error == "repo not found"
    assert by_runtime["ollama"].error is not None


# --------------------------------------------------------------------------- #
# Availability semantics
# --------------------------------------------------------------------------- #
def test_mlx_available_false_on_non_apple():
    sources = {
        "mlx": _FakeSource(spec=_llama_spec(runtime="mlx", quant_label="4bit-g64"), runtime="mlx"),
    }
    rows = compare(
        "x", m1_profile(),
        source_for=_source_for_factory(sources),
        resolve_fn=lambda lid: {"sources": {"mlx": {"repo_id": "x/m"}}},
        is_apple_silicon=lambda: False,
        mlx_available=lambda: False,
    )
    assert rows[0].available is False
    assert rows[0].available_reason == "not Apple Silicon"
    # Fetch still succeeded — the spec/fit/speed are populated; only `available` is False.
    assert rows[0].spec is not None
    assert rows[0].fit is not None


def test_mlx_available_false_when_mlx_lm_missing():
    sources = {
        "mlx": _FakeSource(spec=_llama_spec(runtime="mlx", quant_label="4bit-g64"), runtime="mlx"),
    }
    rows = compare(
        "x", m1_profile(),
        source_for=_source_for_factory(sources),
        resolve_fn=lambda lid: {"sources": {"mlx": {"repo_id": "x/m"}}},
        is_apple_silicon=lambda: True,
        mlx_available=lambda: False,
    )
    assert rows[0].available is False
    assert rows[0].available_reason == "mlx_lm not installed"


def test_ollama_unavailable_when_not_pulled():
    sources = {
        "ollama": _FakeSource(raises=FileNotFoundError("not pulled"), runtime="ollama"),
    }
    rows = compare(
        "x", m1_profile(),
        source_for=_source_for_factory(sources),
        resolve_fn=lambda lid: {"sources": {"ollama": {"tag": "llama3.1:8b"}}},
        is_apple_silicon=lambda: True,
        mlx_available=lambda: True,
    )
    assert rows[0].available is False
    assert rows[0].available_reason == "ollama model not pulled"


def test_gguf_always_available_for_estimates_when_fetch_ok():
    sources = {
        "gguf": _FakeSource(spec=_llama_spec(), runtime="gguf"),
    }
    rows = compare(
        "x", m1_profile(),
        source_for=_source_for_factory(sources),
        resolve_fn=lambda lid: {"sources": {"gguf": {"repo_id": "x/g"}}},
        is_apple_silicon=lambda: False,  # CUDA box; gguf row should still be available
        mlx_available=lambda: False,
    )
    assert rows[0].available is True


# --------------------------------------------------------------------------- #
# Calibration plumbing — applies only to matching runtime
# --------------------------------------------------------------------------- #
def test_compare_passes_per_runtime_calibration():
    from canirunit import Calibration

    cal = Calibration(
        effective_bytes_per_sec=4.76e10, measured_on_chip="Apple M1",
        source="llama-bench", prefill_flops_per_sec=2e12, runtime="gguf",
    )
    sources = {
        "gguf": _FakeSource(spec=_llama_spec(), runtime="gguf"),
    }
    rows = compare(
        "x", m1_profile(),
        calibration_by_runtime={"gguf": cal},
        source_for=_source_for_factory(sources),
        resolve_fn=lambda lid: {"sources": {"gguf": {"repo_id": "x/g"}}},
        is_apple_silicon=lambda: True,
        mlx_available=lambda: True,
    )
    assert rows[0].speed.confidence == "measured"

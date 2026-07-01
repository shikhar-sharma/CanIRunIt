"""Serialization contract tests.

Pins the field names, unit conventions, and JSON-native typing that the web
frontend consumes. Any drift here is a frontend break.
"""
from __future__ import annotations

import json
import math

import pytest

from canirunit import (
    Calibration,
    EstimatorConfig,
    ModelSpec,
    SystemProfile,
    estimate_fit,
    estimate_speed,
)
from canirunit.compare import RuntimeComparison
from canirunit.serialize import (
    calibration_to_dict,
    comparison_to_dict,
    fit_to_dict,
    memory_curve,
    speed_to_dict,
    spec_to_dict,
    system_to_dict,
)


GiB = 1024 ** 3


# --------------------------------------------------------------------------- #
# Fixtures — mirror the ones in test_estimator.py so behaviour is comparable
# --------------------------------------------------------------------------- #
@pytest.fixture
def cfg():
    return EstimatorConfig()


@pytest.fixture
def m1():
    return SystemProfile(
        total_memory_bytes=16 * GiB,
        available_memory_bytes=12 * GiB,
        memory_bandwidth_gbs=68.0,
        accelerator="apple_metal",
        chip_id="Apple M1",
        storage_free_bytes=100 * GiB,
        metal_max_working_set_bytes=12 * GiB,
        peak_flops=2.6e12,
    )


@pytest.fixture
def cuda_box():
    return SystemProfile(
        total_memory_bytes=64 * GiB,
        available_memory_bytes=48 * GiB,
        memory_bandwidth_gbs=900.0,
        accelerator="cuda",
        chip_id="NVIDIA RTX 4090",
        storage_free_bytes=500 * GiB,
        metal_max_working_set_bytes=None,
        peak_flops=82e12,
    )


@pytest.fixture
def llama_spec():
    return ModelSpec(
        repo_id="meta-llama/Meta-Llama-3-8B",
        quant="Q4_K_M",
        total_weight_bytes=4_900_000_000,
        active_weight_bytes=4_900_000_000,
        total_params=8_030_000_000,
        n_layers=32,
        n_kv_heads=8,
        key_length=128,
        value_length=128,
        native_ctx=8192,
        architecture="llama",
        quant_label="Q4_K_M",
    )


# --------------------------------------------------------------------------- #
# JSON-nativity: every result round-trips through json.dumps
# --------------------------------------------------------------------------- #
def _assert_json_native(payload):
    """Round-trip through the JSON encoder. Catches numpy floats, Decimal,
    dataclass instances, and other non-native values."""
    dumped = json.dumps(payload)
    reloaded = json.loads(dumped)
    assert reloaded == payload


# --------------------------------------------------------------------------- #
# system_to_dict
# --------------------------------------------------------------------------- #
def test_system_to_dict_apple_shape(m1, cfg):
    d = system_to_dict(m1, cfg)
    assert d["chip_id"] == "Apple M1"
    assert d["accelerator"] == "apple_metal"
    assert d["usable_basis"] == "Metal working set"
    assert d["chip_is_known"] is True
    assert d["usable_memory_bytes"] > 0
    # On Apple the hard ceiling is populated (soft wired-limit)
    assert d["hard_usable_memory_bytes"] is not None
    assert d["hard_usable_memory_bytes"] > d["usable_memory_bytes"]
    _assert_json_native(d)


def test_system_to_dict_cuda_hard_ceiling_is_null(cuda_box, cfg):
    d = system_to_dict(cuda_box, cfg)
    assert d["usable_basis"] == "free VRAM"
    assert d["hard_usable_memory_bytes"] is None  # no soft/hard distinction on CUDA
    assert d["metal_max_working_set_bytes"] is None
    _assert_json_native(d)


def test_system_to_dict_unknown_chip_flag():
    quirk = SystemProfile(
        total_memory_bytes=16 * GiB, available_memory_bytes=12 * GiB,
        memory_bandwidth_gbs=68.0, accelerator="apple_metal",
        chip_id="Apple M99 Ultra",  # not in the table
        storage_free_bytes=100 * GiB, metal_max_working_set_bytes=12 * GiB,
        peak_flops=2.6e12,
    )
    d = system_to_dict(quirk)
    assert d["chip_is_known"] is False


# --------------------------------------------------------------------------- #
# spec_to_dict
# --------------------------------------------------------------------------- #
def test_spec_to_dict_keys(llama_spec):
    d = spec_to_dict(llama_spec)
    assert d["runtime"] == "gguf"
    assert d["quant_label"] == "Q4_K_M"
    assert d["is_moe"] is False
    assert d["active_params"] is None  # dense
    _assert_json_native(d)


def test_spec_to_dict_moe_carries_active_params():
    moe = ModelSpec(
        repo_id="x/moe", quant="Q4_K_M",
        total_weight_bytes=26_000_000_000, active_weight_bytes=4_000_000_000,
        total_params=26_000_000_000, active_params=4_000_000_000,
        n_layers=40, n_kv_heads=8, key_length=128, value_length=128,
        native_ctx=8192, architecture="gemma3", is_moe=True,
    )
    d = spec_to_dict(moe)
    assert d["is_moe"] is True
    assert d["active_params"] == 4_000_000_000
    _assert_json_native(d)


# --------------------------------------------------------------------------- #
# fit_to_dict / speed_to_dict via a real estimator pass
# --------------------------------------------------------------------------- #
def test_fit_and_speed_to_dict_from_real_estimator(m1, cfg, llama_spec):
    fit = estimate_fit(m1, llama_spec, cfg)
    speed = estimate_speed(m1, llama_spec, cfg)

    f = fit_to_dict(fit)
    assert f["fits_at_native_ctx"] is True
    assert f["max_ctx_that_fits"] == 8192  # native cap for this model
    assert "weight_bytes" in f["breakdown"]
    assert "kv_bytes_at_native" in f["breakdown"]
    _assert_json_native(f)

    s = speed_to_dict(speed)
    assert s["confidence"] == "estimated"
    assert len(s["points"]) > 0
    for p in s["points"]:
        assert set(p.keys()) == {"ctx", "decode_tok_s", "ttft_s"}
    _assert_json_native(s)


def test_speed_infinity_becomes_null():
    """inf/nan (rare — from zero active_params fallback) must not leak as JSON."""
    from canirunit.types import SpeedPoint, SpeedResult

    speed = SpeedResult(
        points=[SpeedPoint(ctx=2048, decode_tok_s=8.0, ttft_s=float("inf"))],
        confidence="estimated", notes=[],
    )
    d = speed_to_dict(speed)
    assert d["points"][0]["ttft_s"] is None
    _assert_json_native(d)


# --------------------------------------------------------------------------- #
# calibration_to_dict
# --------------------------------------------------------------------------- #
def test_calibration_to_dict_shape():
    cal = Calibration(
        effective_bytes_per_sec=4.76e10, measured_on_chip="Apple M1",
        source="mlx_lm", runtime="mlx", prefill_flops_per_sec=2e12,
    )
    d = calibration_to_dict(cal)
    assert d["runtime"] == "mlx"
    assert d["source"] == "mlx_lm"
    assert d["measured_on_chip"] == "Apple M1"
    _assert_json_native(d)


def test_calibration_to_dict_none_passes_through():
    assert calibration_to_dict(None) is None


# --------------------------------------------------------------------------- #
# comparison_to_dict
# --------------------------------------------------------------------------- #
def test_comparison_to_dict_ok_row(m1, cfg, llama_spec):
    fit = estimate_fit(m1, llama_spec, cfg)
    speed = estimate_speed(m1, llama_spec, cfg)
    row = RuntimeComparison(
        runtime="gguf", spec=llama_spec, fit=fit, speed=speed,
        available=True, available_reason="ok", error=None, quant_label="Q4_K_M",
    )
    d = comparison_to_dict(row)
    assert d["runtime"] == "gguf" and d["available"] is True
    assert d["spec"] is not None and d["fit"] is not None and d["speed"] is not None
    _assert_json_native(d)


def test_comparison_to_dict_error_row_has_nulls_not_missing_keys():
    row = RuntimeComparison(
        runtime="mlx", spec=None, fit=None, speed=None,
        available=False, available_reason="mlx_lm not installed",
        error="repo not found", quant_label=None,
    )
    d = comparison_to_dict(row)
    # The error case must still present the full shape (nulls, not missing).
    assert set(d.keys()) == {
        "runtime", "available", "available_reason", "error",
        "spec", "fit", "speed", "quant_label",
    }
    assert d["spec"] is None and d["fit"] is None and d["speed"] is None
    assert d["error"] == "repo not found"
    _assert_json_native(d)


# --------------------------------------------------------------------------- #
# memory_curve — the KV teaching chart data
# --------------------------------------------------------------------------- #
def test_memory_curve_shape(m1, cfg, llama_spec):
    curve = memory_curve(m1, llama_spec, cfg, kv_quant="f16")
    assert curve["kv_quant"] == "f16"
    assert curve["weight_bytes"] == 4_900_000_000
    assert curve["overhead_bytes"] == cfg.compute_overhead_bytes
    # Apple: hard ceiling populated
    assert curve["hard_usable_bytes"] is not None
    assert curve["hard_usable_bytes"] > curve["usable_bytes"]
    assert curve["native_ctx"] == 8192
    assert len(curve["points"]) >= 2
    _assert_json_native(curve)


def test_memory_curve_total_is_monotonic_in_ctx(m1, cfg, llama_spec):
    """The point of the chart is that memory need grows with ctx. Pin it."""
    curve = memory_curve(m1, llama_spec, cfg, kv_quant="f16")
    totals = [p["total_bytes"] for p in curve["points"]]
    assert totals == sorted(totals)
    # Weight baseline is the minimum possible total (at ctx=0, KV=0, plus overhead).
    assert totals[0] == curve["weight_bytes"] + curve["overhead_bytes"]


def test_memory_curve_kv_quant_shrinks_curve(m1, cfg, llama_spec):
    """q4 KV per element is 1/4 of f16 — every KV sample should be roughly quartered."""
    f16 = memory_curve(m1, llama_spec, cfg, kv_quant="f16")
    q4 = memory_curve(m1, llama_spec, cfg, kv_quant="q4")
    # Pair by ctx (they may not use identical samples if native differs, but here it does).
    f16_by_ctx = {p["ctx"]: p["kv_bytes"] for p in f16["points"]}
    q4_by_ctx = {p["ctx"]: p["kv_bytes"] for p in q4["points"]}
    for ctx in f16_by_ctx:
        if ctx == 0 or ctx not in q4_by_ctx:
            continue
        # 0.5 bytes/elem vs 2.0 -> quarter, allow small integer rounding slack.
        ratio = q4_by_ctx[ctx] / f16_by_ctx[ctx]
        assert 0.24 < ratio < 0.26


def test_memory_curve_cuda_hard_usable_is_null(cuda_box, cfg, llama_spec):
    curve = memory_curve(cuda_box, llama_spec, cfg)
    assert curve["hard_usable_bytes"] is None


def test_memory_curve_respects_custom_ctx_points(m1, cfg, llama_spec):
    curve = memory_curve(m1, llama_spec, cfg, ctx_points=[0, 1024, 4096, 8192])
    ctxs = [p["ctx"] for p in curve["points"]]
    assert ctxs == [0, 1024, 4096, 8192]


def test_memory_curve_zero_native_ctx_is_handled():
    """Defensive: a degenerate spec with native_ctx==0 shouldn't crash the
    sampler. Not a real model, but shape checks may see it."""
    spec = ModelSpec(
        repo_id="x/degenerate", quant="Q4_K_M",
        total_weight_bytes=1, active_weight_bytes=1, total_params=1,
        n_layers=1, n_kv_heads=1, key_length=1, value_length=1,
        native_ctx=0, architecture="llama",
    )
    profile = SystemProfile(
        total_memory_bytes=1 * GiB, available_memory_bytes=1 * GiB,
        memory_bandwidth_gbs=10.0, accelerator="cpu", chip_id="cpu",
        storage_free_bytes=1 * GiB, peak_flops=1e12,
    )
    curve = memory_curve(profile, spec)
    assert curve["native_ctx"] == 0
    assert all(p["ctx"] == 0 for p in curve["points"])


# --------------------------------------------------------------------------- #
# Whole-shebang: end-to-end JSON compilation
# --------------------------------------------------------------------------- #
def test_end_to_end_check_response_shape(m1, cfg, llama_spec):
    """The exact shape /api/check returns. Locks the contract."""
    fit = estimate_fit(m1, llama_spec, cfg)
    speed = estimate_speed(m1, llama_spec, cfg)
    curve = memory_curve(m1, llama_spec, cfg)

    payload = {
        "spec": spec_to_dict(llama_spec),
        "fit": fit_to_dict(fit),
        "speed": speed_to_dict(speed),
        "memory_curve": curve,
    }
    _assert_json_native(payload)
    # Sanity — the frontend uses these paths directly.
    assert payload["fit"]["max_ctx_that_fits"] == 8192
    assert payload["speed"]["confidence"] == "estimated"
    assert payload["memory_curve"]["weight_bytes"] > 0

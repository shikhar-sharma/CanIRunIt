"""Estimator tests. Fixtures are hand-computed and pinned, so a regression in
the math fails loudly rather than silently shipping a wrong verdict.
"""
from __future__ import annotations

import math

import pytest

from canirunit import (
    Calibration,
    EstimatorConfig,
    ModelSpec,
    SystemProfile,
    decode_tok_s,
    estimate_fit,
    estimate_speed,
    kv_cache_bytes,
    max_ctx_that_fits,
)
from canirunit.estimator import usable_memory_bytes

GiB = 1024 ** 3


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def cfg():
    return EstimatorConfig()


@pytest.fixture
def m1():
    """Base M1, 16 GB. 68 GB/s memory bandwidth; Metal working set ~12 GiB."""
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
def llama3_8b_q4():
    """Llama-3-8B at Q4_K_M. Dense, so total == active weight bytes.
    32 layers, 8 KV heads (GQA from 32 query heads), head_dim 128."""
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
    )


# --------------------------------------------------------------------------- #
# KV cache — exact integer fixtures
# --------------------------------------------------------------------------- #
def test_usable_driven_by_working_set_not_instantaneous_free_ram(cfg):
    """Regression: low free RAM at run time must NOT collapse the budget on Apple.
    macOS reclaims memory on load up to the wired limit, so the working set is the
    cap, not psutil's instantaneous available."""
    busy_m1 = SystemProfile(
        total_memory_bytes=16 * GiB,
        available_memory_bytes=4 * GiB,           # lots of apps open
        memory_bandwidth_gbs=68.0, accelerator="apple_metal", chip_id="Apple M1",
        storage_free_bytes=100 * GiB,
        metal_max_working_set_bytes=12 * GiB, peak_flops=2.6e12,
    )
    # 12 GiB working set - 1 GiB headroom = 11 GiB, regardless of the 4 GiB free.
    assert usable_memory_bytes(busy_m1, cfg) == 11 * GiB


def test_usable_uses_available_when_no_working_set(cfg):
    """CUDA/CPU have no Metal working set; there free VRAM / available RAM is the
    real cap."""
    cuda = SystemProfile(
        total_memory_bytes=24 * GiB, available_memory_bytes=8 * GiB,
        memory_bandwidth_gbs=1008.0, accelerator="cuda", chip_id="RTX 4090",
        storage_free_bytes=100 * GiB, metal_max_working_set_bytes=None,
    )
    assert usable_memory_bytes(cuda, cfg) == 8 * GiB - 1 * GiB


def test_busy_machine_still_fits_a_model_that_obviously_runs(cfg):
    """The exact failure from the field run: an 8B on a 16 GB M1 with low free RAM
    was reported as 'max ctx 0'. It must now fit to a useful context."""
    busy_m1 = SystemProfile(
        total_memory_bytes=16 * GiB, available_memory_bytes=4 * GiB,
        memory_bandwidth_gbs=68.0, accelerator="apple_metal", chip_id="Apple M1",
        storage_free_bytes=100 * GiB, metal_max_working_set_bytes=12 * GiB, peak_flops=2.6e12,
    )
    llama31_8b = ModelSpec(
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", quant="Q4_K_M",
        total_weight_bytes=4_900_000_000, active_weight_bytes=4_900_000_000,
        total_params=8_030_000_000, n_layers=32, n_kv_heads=8,
        key_length=128, value_length=128, native_ctx=131072, architecture="llama",
    )
    fit = estimate_fit(busy_m1, llama31_8b, cfg)
    assert fit.max_ctx_that_fits > 40_000           # ~49.7K, not 0
    assert fit.fits_at_native_ctx is False          # full 128K KV won't fit at f16
    assert fit.kv_quant_suggestion is not None       # q4 KV reaches native


def test_kv_cache_exact_f16(llama3_8b_q4, cfg):
    # 32 * 8 * (128+128) * 2 = 131072 bytes/token; * 8192 = 1.0 GiB exactly
    per_token = 32 * 8 * (128 + 128) * 2
    assert per_token == 131072
    assert kv_cache_bytes(llama3_8b_q4, 8192, "f16", cfg) == 131072 * 8192
    assert kv_cache_bytes(llama3_8b_q4, 8192, "f16", cfg) == 1 * GiB


def test_kv_quant_halves_and_quarters(llama3_8b_q4, cfg):
    f16 = kv_cache_bytes(llama3_8b_q4, 8192, "f16", cfg)
    assert kv_cache_bytes(llama3_8b_q4, 8192, "q8", cfg) == f16 // 2
    assert kv_cache_bytes(llama3_8b_q4, 8192, "q4", cfg) == f16 // 4


def test_kv_uses_provided_head_dim_not_ratio(cfg):
    """The Gemma trap: head_dim must come from key/value_length, never from
    hidden/n_heads. Doubling the provided head dim must double the KV exactly."""
    base = dict(
        repo_id="x", quant="q", total_weight_bytes=1, active_weight_bytes=1,
        total_params=1, n_layers=10, n_kv_heads=4, native_ctx=4096, architecture="gemma3",
    )
    small = ModelSpec(key_length=128, value_length=128, **base)
    gemma = ModelSpec(key_length=256, value_length=256, **base)  # Gemma's real head_dim
    assert kv_cache_bytes(gemma, 4096, "f16", cfg) == 2 * kv_cache_bytes(small, 4096, "f16", cfg)


# --------------------------------------------------------------------------- #
# Decode speed — the M1 sanity check from our design discussion
# --------------------------------------------------------------------------- #
def test_decode_m1_llama3_8b_matches_hand_calc(m1, llama3_8b_q4, cfg):
    # eff = 0.7 * 68e9 = 4.76e10 B/s
    # denom@8k = 4.9e9 + 1.0737e9 = 5.9737e9 ; 4.76e10 / 5.9737e9 = 7.97 tok/s
    dec = decode_tok_s(m1, llama3_8b_q4, 8192, cfg)
    assert dec == pytest.approx(7.97, abs=0.05)


def test_decode_decays_with_context(m1, llama3_8b_q4, cfg):
    assert decode_tok_s(m1, llama3_8b_q4, 2048, cfg) > decode_tok_s(m1, llama3_8b_q4, 8192, cfg)


def test_calibration_overrides_bandwidth(m1, llama3_8b_q4, cfg):
    calib = Calibration(
        effective_bytes_per_sec=4.76e10, measured_on_chip="Apple M1", source="llama-bench"
    )
    # Same effective B/s as the static estimate -> identical decode number.
    assert decode_tok_s(m1, llama3_8b_q4, 8192, cfg, calibration=calib) == pytest.approx(
        decode_tok_s(m1, llama3_8b_q4, 8192, cfg), rel=1e-9
    )


def test_mbu_config_override_scales_decode(m1, llama3_8b_q4):
    full = decode_tok_s(m1, llama3_8b_q4, 8192, EstimatorConfig(mbu=0.7))
    half = decode_tok_s(m1, llama3_8b_q4, 8192, EstimatorConfig(mbu=0.35))
    assert half == pytest.approx(full / 2, rel=1e-9)


# --------------------------------------------------------------------------- #
# Fit ceiling + KV-quant suggestion
# --------------------------------------------------------------------------- #
def test_fit_ceiling_and_kv_quant_suggestion(m1, cfg):
    """Weights fit; native context overflows at f16 but fits at q8.
    usable = 12 GiB - 1 GiB headroom = 11811160064
    budget for KV = usable - 10e9 weights - 384 MiB overhead = 1,408,506,880
    f16 @ 131072 B/tok -> 10746 tok ceiling (< 16384 native)
    q8  @  65536 B/tok -> 21492 tok (> 16384) -> suggest q8
    """
    spec = ModelSpec(
        repo_id="x", quant="q",
        total_weight_bytes=10_000_000_000, active_weight_bytes=10_000_000_000,
        total_params=13_000_000_000,
        n_layers=32, n_kv_heads=8, key_length=128, value_length=128,
        native_ctx=16384, architecture="llama",
    )
    assert usable_memory_bytes(m1, cfg) == 11 * GiB  # min(12,12) GiB - 1 GiB headroom
    fit = estimate_fit(m1, spec, cfg, "f16")
    assert fit.fits_at_native_ctx is False
    assert fit.max_ctx_that_fits == 10746
    assert fit.kv_quant_suggestion is not None and "q8" in fit.kv_quant_suggestion
    assert fit.storage_ok is True


def test_weights_too_big_gives_zero_ceiling(m1, cfg):
    spec = ModelSpec(
        repo_id="x", quant="q",
        total_weight_bytes=26_000_000_000, active_weight_bytes=4_000_000_000,
        total_params=26_000_000_000,
        n_layers=40, n_kv_heads=8, key_length=128, value_length=128,
        native_ctx=8192, architecture="gemma3", is_moe=True, active_params=4_000_000_000,
    )
    fit = estimate_fit(m1, spec, cfg)
    assert fit.max_ctx_that_fits == 0
    assert fit.fits_at_native_ctx is False


# --------------------------------------------------------------------------- #
# MoE — fit uses total, decode uses active
# --------------------------------------------------------------------------- #
def test_moe_decode_uses_active_not_total(m1, cfg):
    """26B total / 4B active. Decode must be governed by the 4B active weight,
    so it's fast despite a footprint that won't fit."""
    moe = ModelSpec(
        repo_id="google/gemma-4-moe", quant="Q4_K_M",
        total_weight_bytes=26_000_000_000, active_weight_bytes=4_000_000_000,
        total_params=26_000_000_000,
        n_layers=40, n_kv_heads=8, key_length=128, value_length=128,
        native_ctx=8192, architecture="gemma3", is_moe=True, active_params=4_000_000_000,
    )
    dec = decode_tok_s(m1, moe, 2048, cfg)
    # active path: 4.76e10 / (4e9 + 131072*2048) ~ 11.1 tok/s
    assert dec == pytest.approx(11.15, abs=0.2)
    # If it had (wrongly) used total weight bytes it would be < 2 tok/s.
    assert dec > 5.0


# --------------------------------------------------------------------------- #
# MLA / compressed KV — flag, don't crash
# --------------------------------------------------------------------------- #
def test_mla_is_flagged_not_silently_wrong(m1, cfg):
    mla = ModelSpec(
        repo_id="deepseek-ai/x", quant="Q4_K_M",
        total_weight_bytes=8_000_000_000, active_weight_bytes=8_000_000_000,
        total_params=16_000_000_000,
        n_layers=30, n_kv_heads=128, key_length=128, value_length=128,
        native_ctx=8192, architecture="deepseek2", kv_is_standard=False,
    )
    fit = estimate_fit(m1, mla, cfg)
    assert any("Compressed-KV" in n or "MLA" in n for n in fit.notes)
    # We don't emit a KV-quant suggestion for an architecture we can't model.
    assert fit.kv_quant_suggestion is None


# --------------------------------------------------------------------------- #
# Speed result shape
# --------------------------------------------------------------------------- #
def test_speed_result_is_a_curve_with_confidence(m1, llama3_8b_q4, cfg):
    res = estimate_speed(m1, llama3_8b_q4, cfg)
    assert res.confidence == "estimated"
    assert [p.ctx for p in res.points] == [2048, 8192]  # filtered to <= native 8192
    # decode strictly decreasing across the curve
    decs = [p.decode_tok_s for p in res.points]
    assert decs == sorted(decs, reverse=True)

    calib = Calibration(effective_bytes_per_sec=4.76e10, measured_on_chip="Apple M1", source="llama-bench")
    assert estimate_speed(m1, llama3_8b_q4, cfg, calibration=calib).confidence == "measured"

"""Backward-compat pins: ModelSpec/Calibration can still be built from the v1
argument set, and the new fields default sanely.

Failure here means a runtime extension broke an additive contract and a
downstream construction (existing tests, user code) will break.
"""
from __future__ import annotations

from canirunit import Calibration, ModelSpec


def test_modelspec_v1_args_still_construct():
    spec = ModelSpec(
        repo_id="repo/x",
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
    # Defaults must be backwards-compatible (no observable change in v1 behavior):
    assert spec.runtime == "gguf"
    assert spec.default_kv_bytes_per_element == 2.0
    assert spec.quant_label is None  # callers that need a label set it explicitly
    assert spec.is_moe is False
    assert spec.kv_is_standard is True


def test_calibration_v1_args_still_construct():
    cal = Calibration(
        effective_bytes_per_sec=47.6e9,
        measured_on_chip="Apple M1",
        source="llama-bench",
    )
    assert cal.runtime == "gguf"  # default for the existing gguf-only callers
    assert cal.prefill_flops_per_sec is None

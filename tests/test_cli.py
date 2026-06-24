"""CLI tests: argument parsing, the pure renderer, and run_check orchestration
with injected detect/fetch/calibrate."""
from __future__ import annotations

import pytest

from llmfit import Calibration, ModelSpec, SystemProfile
from llmfit.cli import build_parser, run_check

GiB = 1024 ** 3


def m1_profile(chip="Apple M1"):
    return SystemProfile(
        total_memory_bytes=16 * GiB, available_memory_bytes=12 * GiB,
        memory_bandwidth_gbs=68.0, accelerator="apple_metal", chip_id=chip,
        storage_free_bytes=100 * GiB, metal_max_working_set_bytes=12 * GiB, peak_flops=2.6e12,
    )


def llama_spec():
    return ModelSpec(
        repo_id="meta-llama/Meta-Llama-3-8B", quant="Q4_K_M",
        total_weight_bytes=4_900_000_000, active_weight_bytes=4_900_000_000,
        total_params=8_030_000_000, n_layers=32, n_kv_heads=8,
        key_length=128, value_length=128, native_ctx=8192, architecture="llama",
    )


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def test_parser_check():
    args = build_parser().parse_args(["check", "repo/model", "--quant", "Q5_K_M", "--ctx", "4096", "--calibrate"])
    assert args.command == "check"
    assert args.model == "repo/model"
    assert args.quant == "Q5_K_M"
    assert args.ctx == 4096
    assert args.calibrate is True


def test_parser_defaults():
    args = build_parser().parse_args(["check", "repo/model"])
    assert args.quant == "Q4_K_M" and args.ctx is None and args.calibrate is False


# --------------------------------------------------------------------------- #
# run_check orchestration (injected deps)
# --------------------------------------------------------------------------- #
def test_run_check_fitting_model_static():
    report = run_check(
        "meta-llama/Meta-Llama-3-8B",
        detect_fn=m1_profile,
        fetch_fn=lambda model, quant: llama_spec(),
        calibrate_fn=lambda profile: None,
    )
    assert "meta-llama/Meta-Llama-3-8B" in report
    assert "Apple M1" in report
    assert "Fits at native context (8192):   yes" in report
    assert "[estimated]" in report
    assert "GB/s" in report                 # bandwidth now surfaced
    assert "Metal working set" in report    # usable basis labelled
    assert "8.0 tok/s" in report  # the pinned M1 decode figure at 8k


def test_run_check_calibrated_reports_measured():
    calib = Calibration(effective_bytes_per_sec=4.76e10, measured_on_chip="Apple M1", source="llama-bench",
                        prefill_flops_per_sec=2e12)
    report = run_check(
        "meta-llama/Meta-Llama-3-8B", do_calibrate=True,
        detect_fn=m1_profile,
        fetch_fn=lambda model, quant: llama_spec(),
        calibrate_fn=lambda profile: calib,
    )
    assert "[measured]" in report


def test_run_check_calibrate_unavailable_notes_fallback():
    report = run_check(
        "meta-llama/Meta-Llama-3-8B", do_calibrate=True,
        detect_fn=m1_profile,
        fetch_fn=lambda model, quant: llama_spec(),
        calibrate_fn=lambda profile: None,
    )
    assert "no supported runtime" in report
    assert "[estimated]" in report


def test_run_check_unknown_chip_warns():
    report = run_check(
        "meta-llama/Meta-Llama-3-8B",
        detect_fn=lambda: m1_profile(chip="Apple M9 Ultra"),
        fetch_fn=lambda model, quant: llama_spec(),
        calibrate_fn=lambda profile: None,
    )
    assert "coarse default" in report


def test_run_check_moe_shows_both_footprints():
    moe = ModelSpec(
        repo_id="google/gemma-4-moe", quant="Q4_K_M",
        total_weight_bytes=26_000_000_000, active_weight_bytes=4_000_000_000,
        total_params=26_000_000_000, n_layers=40, n_kv_heads=8,
        key_length=128, value_length=128, native_ctx=8192, architecture="gemma3",
        is_moe=True, active_params=4_000_000_000,
    )
    report = run_check(
        "google/gemma-4-moe",
        detect_fn=m1_profile,
        fetch_fn=lambda model, quant: moe,
        calibrate_fn=lambda profile: None,
    )
    assert "MoE" in report
    assert "Fits at native context (8192):   no" in report  # 26 GB won't fit in 16 GB


def test_run_check_mla_note_surfaces():
    mla = ModelSpec(
        repo_id="deepseek-ai/x", quant="Q4_K_M",
        total_weight_bytes=8_000_000_000, active_weight_bytes=8_000_000_000,
        total_params=16_000_000_000, n_layers=30, n_kv_heads=128,
        key_length=128, value_length=128, native_ctx=8192, architecture="deepseek2",
        kv_is_standard=False,
    )
    report = run_check(
        "deepseek-ai/x",
        detect_fn=m1_profile,
        fetch_fn=lambda model, quant: mla,
        calibrate_fn=lambda profile: None,
    )
    assert "Compressed-KV" in report

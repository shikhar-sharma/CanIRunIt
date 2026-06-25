"""CLI tests: argument parsing, the pure renderer, and run_check orchestration
with injected detect/fetch/calibrate."""
from __future__ import annotations

import pytest

from canirunit import Calibration, ModelSpec, SystemProfile
from canirunit.cli import (
    build_parser,
    format_comparison,
    format_models,
    run_check,
    run_compare,
    run_models,
    run_refresh,
)
from canirunit.compare import RuntimeComparison

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


# --------------------------------------------------------------------------- #
# Argument parsing — new flags / subcommands
# --------------------------------------------------------------------------- #
def test_parser_check_runtime_flag():
    args = build_parser().parse_args(["check", "llama3.1:8b", "--runtime", "ollama"])
    assert args.runtime == "ollama"


def test_parser_check_runtime_default_is_gguf():
    args = build_parser().parse_args(["check", "repo/x"])
    assert args.runtime == "gguf"


def test_parser_check_runtime_rejects_unknown():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["check", "x", "--runtime", "vllm"])


def test_parser_compare():
    args = build_parser().parse_args(["compare", "llama-3.1-8b-instruct", "--calibrate"])
    assert args.command == "compare"
    assert args.logical_id == "llama-3.1-8b-instruct"
    assert args.calibrate is True


def test_parser_refresh_and_models():
    assert build_parser().parse_args(["refresh"]).command == "refresh"
    assert build_parser().parse_args(["models"]).command == "models"


# --------------------------------------------------------------------------- #
# run_check with --runtime: alias resolution + per-runtime fetch
# --------------------------------------------------------------------------- #
def test_run_check_resolves_logical_id_to_gguf_repo():
    """When the model arg matches an alias, run_check should hand the runtime-
    specific repo id (and default_quant) to fetch_fn — not the alias id itself."""
    seen = []
    def fake_fetch(ref, quant):
        seen.append((ref, quant))
        return llama_spec()
    run_check(
        "llama-3.1-8b-instruct",
        detect_fn=m1_profile,
        fetch_fn=fake_fetch,
        calibrate_fn=lambda profile: None,
        resolve_fn=lambda lid: {
            "sources": {"gguf": {"repo_id": "x/llama-gguf", "default_quant": "Q5_K_M"}}
        },
    )
    assert seen == [("x/llama-gguf", "Q5_K_M")]


def test_run_check_unknown_logical_id_passes_arg_through():
    """If the model arg isn't in the alias table, treat it as a runtime-native ref."""
    seen = []
    def fake_fetch(ref, quant):
        seen.append((ref, quant))
        return llama_spec()
    def raise_keyerror(lid):
        raise KeyError("unknown")
    run_check(
        "meta-llama/Some-Other-Model",
        quant="Q4_K_M",
        detect_fn=m1_profile,
        fetch_fn=fake_fetch,
        calibrate_fn=lambda profile: None,
        resolve_fn=raise_keyerror,
    )
    assert seen == [("meta-llama/Some-Other-Model", "Q4_K_M")]


# --------------------------------------------------------------------------- #
# run_compare orchestration
# --------------------------------------------------------------------------- #
def _comparison_row(runtime, spec=None, fit=None, speed=None, available=True,
                    reason="ok", error=None, quant="Q4_K_M"):
    return RuntimeComparison(
        runtime=runtime, spec=spec, fit=fit, speed=speed,
        available=available, available_reason=reason, error=error,
        quant_label=quant,
    )


def test_run_compare_renders_each_runtime_row():
    """run_compare calls compare_fn and renders the result. We inject a fake
    compare_fn so the test exercises CLI plumbing, not estimator math."""
    from canirunit import FitResult, MemoryBreakdown, SpeedPoint, SpeedResult

    spec = llama_spec()
    fit = FitResult(
        max_ctx_that_fits=8192, fits_at_native_ctx=True,
        breakdown=MemoryBreakdown(
            weight_bytes=spec.total_weight_bytes, kv_bytes=1_000_000_000,
            compute_overhead_bytes=384 * 1024 ** 2, headroom_bytes=GiB,
            usable_bytes=12 * GiB,
        ),
        storage_ok=True,
    )
    speed = SpeedResult(
        points=[SpeedPoint(ctx=8192, decode_tok_s=8.0, ttft_s=1.2)],
        confidence="estimated", notes=[],
    )
    rows = [
        _comparison_row("gguf", spec=spec, fit=fit, speed=speed),
        _comparison_row("mlx",  spec=spec, fit=fit, speed=speed, reason="ok", quant="4bit-g64"),
        _comparison_row("ollama", available=False, reason="ollama model not pulled",
                        error="model not pulled", quant=None),
    ]

    report = run_compare(
        "llama-3.1-8b-instruct",
        detect_fn=m1_profile,
        resolve_fn=lambda lid: {"display_name": "Llama 3.1 8B Instruct",
                                "sources": {"gguf": {}, "mlx": {}, "ollama": {}}},
        compare_fn=lambda lid, profile, **kw: rows,
        calibrate_fn=lambda p, r: None,
    )
    assert "Llama 3.1 8B Instruct" in report
    assert "gguf" in report and "mlx" in report and "ollama" in report
    assert "ollama model not pulled" in report
    assert "Q4_K_M" in report and "4bit-g64" in report


def test_run_compare_calibrate_passes_per_runtime_calibrations():
    """When --calibrate is set, run_compare calls calibrate_fn once per runtime
    declared in the alias entry, and hands the resulting per-runtime map to
    compare_fn."""
    called_with = []

    def fake_calibrate(profile, runtime):
        # Pretend gguf calibrates fine; mlx tooling missing.
        if runtime == "gguf":
            return Calibration(
                effective_bytes_per_sec=4.76e10, measured_on_chip="Apple M1",
                source="llama-bench", runtime="gguf",
            )
        return None

    def fake_compare(lid, profile, **kw):
        called_with.append(kw.get("calibration_by_runtime") or {})
        return []  # rendering an empty table is fine here

    run_compare(
        "x", do_calibrate=True,
        detect_fn=m1_profile,
        resolve_fn=lambda lid: {"display_name": "X",
                                "sources": {"gguf": {}, "mlx": {}}},
        compare_fn=fake_compare,
        calibrate_fn=fake_calibrate,
    )
    assert called_with == [{"gguf": fake_calibrate(m1_profile(), "gguf")}]


def test_run_compare_unknown_logical_id_raises():
    def boom(lid):
        raise KeyError("unknown")
    with pytest.raises(KeyError):
        run_compare(
            "nope",
            detect_fn=m1_profile,
            resolve_fn=boom,
        )


# --------------------------------------------------------------------------- #
# run_refresh / run_models
# --------------------------------------------------------------------------- #
def test_run_refresh_reports_success():
    out = run_refresh(refresh_fn=lambda: {"ok": True, "models": 7,
                                          "updated_at": "2026-06-25T00:00:00Z",
                                          "path": "/tmp/x.json"})
    assert "refreshed" in out and "7 models" in out


def test_run_refresh_reports_failure():
    out = run_refresh(refresh_fn=lambda: {"ok": False, "error": "schema mismatch"})
    assert "refresh failed" in out and "schema mismatch" in out


def test_run_models_renders_listing():
    out = run_models(list_fn=lambda: [
        {"id": "llama-3.1-8b-instruct", "display_name": "L",
         "family": "llama", "runtimes": ["gguf", "mlx", "ollama"]},
    ])
    assert "llama-3.1-8b-instruct" in out
    assert "gguf" in out and "mlx" in out

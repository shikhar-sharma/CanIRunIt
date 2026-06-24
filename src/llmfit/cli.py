"""Command-line interface: the specific-model-check flow (v1 MVP).

    llmfit check <repo_id> [--quant Q4_K_M] [--ctx N] [--calibrate]

Wires detect -> hub.fetch_model -> optional calibrate -> estimate, then renders a
verdict. The renderer is pure and tested; the orchestration takes injectable
detect/fetch/calibrate functions so it is testable without network or hardware.
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable, Optional

import requests

from .benchmark import calibrate as _calibrate
from .config import EstimatorConfig
from .detector import chip_is_known, detect as _detect
from .estimator import estimate_fit, estimate_speed
from .hub import fetch_model as _fetch_model
from .types import Calibration, FitResult, ModelSpec, SpeedResult, SystemProfile


def _gb(b: float) -> float:
    return b / 1e9


def _fmt_ttft(s: float) -> str:
    if s == float("inf") or s != s:  # inf or nan
        return "n/a"
    return f"{s:.1f} s"


# --------------------------------------------------------------------------- #
# Pure renderer
# --------------------------------------------------------------------------- #
def format_report(
    profile: SystemProfile,
    spec: ModelSpec,
    fit: FitResult,
    speed: SpeedResult,
    calibration: Optional[Calibration],
    chip_known: bool,
    calib_note: Optional[str] = None,
) -> str:
    L: list[str] = []
    L.append(f"llmfit — {spec.repo_id}  ({spec.quant})")
    basis = (
        "Metal working set" if profile.metal_max_working_set_bytes is not None
        else "free VRAM" if profile.accelerator == "cuda"
        else "available RAM"
    )
    L.append(
        f"Machine: {profile.chip_id}  [{profile.accelerator}]  ·  "
        f"{profile.memory_bandwidth_gbs:.0f} GB/s  ·  "
        f"{_gb(profile.total_memory_bytes):.1f} GB total, "
        f"{_gb(fit.breakdown.usable_bytes):.1f} GB usable ({basis})"
    )
    if not chip_known and profile.accelerator != "cpu":
        L.append("  ! Chip not in the bandwidth table — decode uses a coarse default; --calibrate for real numbers.")
    if calib_note:
        L.append(f"  ! {calib_note}")
    L.append("")

    # FIT
    b = fit.breakdown
    L.append("FIT")
    L.append(f"  Fits at native context ({spec.native_ctx}):   {'yes' if fit.fits_at_native_ctx else 'no'}")
    L.append(f"  Max context that fits:           {fit.max_ctx_that_fits} tokens")
    L.append(
        f"  Storage for weights:             {'ok' if fit.storage_ok else 'INSUFFICIENT'}  "
        f"({_gb(spec.total_weight_bytes):.1f} GB needed, {_gb(profile.storage_free_bytes):.1f} GB free)"
    )
    L.append(
        f"  Memory at native context:        weights {_gb(b.weight_bytes):.1f} + "
        f"KV {_gb(b.kv_bytes):.1f} + overhead {_gb(b.compute_overhead_bytes):.1f} = "
        f"{_gb(b.required_bytes):.1f} GB  (usable {_gb(b.usable_bytes):.1f} GB)"
    )
    if spec.is_moe:
        L.append(
            f"  MoE: fit uses full {_gb(spec.total_weight_bytes):.1f} GB; "
            f"decode uses active {_gb(spec.active_weight_bytes):.1f} GB."
        )
    if fit.kv_quant_suggestion:
        L.append(f"  Suggestion: {fit.kv_quant_suggestion}")
    # Free RAM no longer caps the Apple verdict, but it's worth knowing if it's low.
    if profile.metal_max_working_set_bytes is not None:
        min_to_load = spec.total_weight_bytes + b.compute_overhead_bytes
        if profile.available_memory_bytes < min_to_load:
            L.append(
                f"  Advisory: {_gb(profile.available_memory_bytes):.1f} GB free right now; "
                f"macOS will reclaim memory on load, but close apps if it thrashes."
            )
    for note in fit.notes:
        L.append(f"  Note: {note}")
    L.append("")

    # SPEED
    L.append(f"SPEED  [{speed.confidence}]")
    L.append("  context     decode        time-to-first-token")
    for p in speed.points:
        L.append(f"    {p.ctx:>6}    {p.decode_tok_s:5.1f} tok/s    {_fmt_ttft(p.ttft_s)}")
    if speed.notes:
        L.append("  Notes:")
        for note in speed.notes:
            L.append(f"   - {note}")

    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Orchestration (injectable for testing)
# --------------------------------------------------------------------------- #
def run_check(
    model: str,
    quant: str = "Q4_K_M",
    ctx: Optional[int] = None,
    do_calibrate: bool = False,
    *,
    detect_fn: Callable[[], SystemProfile] = _detect,
    fetch_fn: Callable[..., ModelSpec] = _fetch_model,
    calibrate_fn: Callable[[SystemProfile], Optional[Calibration]] = _calibrate,
    config: Optional[EstimatorConfig] = None,
) -> str:
    config = config or EstimatorConfig()
    profile = detect_fn()
    spec = fetch_fn(model, quant)

    calibration = None
    calib_note = None
    if do_calibrate:
        calibration = calibrate_fn(profile)
        if calibration is None:
            calib_note = "Calibration requested but no supported runtime (llama-bench) found; showing static estimate."

    ctx_points = None
    if ctx:
        base = [c for c in (2048, 8192, ctx, spec.native_ctx) if 0 < c <= spec.native_ctx]
        ctx_points = sorted(set(base))

    fit = estimate_fit(profile, spec, config)
    speed = estimate_speed(profile, spec, config, calibration=calibration, ctx_points=ctx_points)
    return format_report(profile, spec, fit, speed, calibration, chip_is_known(profile), calib_note)


# --------------------------------------------------------------------------- #
# Argument parsing / entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmfit", description="Estimate whether a local GGUF model fits and runs well on this machine.")
    sub = p.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Check a specific model by Hugging Face repo id.")
    check.add_argument("model", help="Hugging Face repo id, e.g. bartowski/Meta-Llama-3-8B-Instruct-GGUF")
    check.add_argument("--quant", default="Q4_K_M", help="Quantization tag to match (default: Q4_K_M)")
    check.add_argument("--ctx", type=int, default=None, help="Context length of interest (tokens)")
    check.add_argument("--calibrate", action="store_true", help="Run an on-machine benchmark for measured speed")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        try:
            report = run_check(args.model, quant=args.quant, ctx=args.ctx, do_calibrate=args.calibrate)
        except (ValueError, KeyError) as e:
            print(f"llmfit: {e}", file=sys.stderr)
            return 1
        except (OSError, requests.exceptions.RequestException) as e:
            # HF HTTP errors subclass OSError (httpx); our range reader uses requests.
            print(
                f"llmfit: could not fetch '{args.model}' from Hugging Face. "
                f"Check the repo id, quant tag, and your connection.\n  ({e})",
                file=sys.stderr,
            )
            return 1
        print(report)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

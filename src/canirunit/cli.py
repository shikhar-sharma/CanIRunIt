"""Command-line interface.

Subcommands:
    canirunit check <model_or_alias> [--quant Q4_K_M] [--ctx N] [--calibrate]
                                     [--runtime {gguf,mlx,ollama}]
    canirunit compare <logical_id> [--calibrate]
    canirunit refresh
    canirunit models

`check` is the v1 flow: detect -> fetch (per runtime) -> optional calibrate ->
estimate -> render. Renderers are pure and tested; orchestration takes
injectable detect/fetch/calibrate/resolve functions so it is testable without
network or hardware. `compare` fans the same profile across every runtime that
has a source for the given logical id.
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from typing import Callable, Optional

import requests

from .aliases import (
    list_models as _list_models,
    refresh as _refresh,
    resolve as _resolve_alias,
)
from .benchmark import calibrate as _calibrate
from .compare import RuntimeComparison, compare as _compare
from .config import EstimatorConfig
from .detector import chip_is_known, detect as _detect
from .estimator import estimate_fit, estimate_speed
from .hub import fetch_model as _fetch_model
from .sources import get_source
from .types import Calibration, FitResult, ModelSpec, Runtime, SpeedResult, SystemProfile


def _gb(b: float) -> float:
    return b / 1e9


def _fmt_ttft(s: float) -> str:
    if s == float("inf") or s != s:  # inf or nan
        return "n/a"
    return f"{s:.1f} s"


# --------------------------------------------------------------------------- #
# Pure renderer — check (unchanged)
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
    L.append(f"canirunit — {spec.repo_id}  ({spec.quant})")
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
    # Two-ceiling model: on Apple the wired-memory cap is soft; macOS extends
    # it via compression/swap. Surface the hard "loads at all" ceiling when
    # it's actually higher than the comfort one.
    if fit.hard_max_ctx_that_fits is not None and fit.hard_max_ctx_that_fits > fit.max_ctx_that_fits:
        L.append(
            f"  Loads (with slowdown past wired limit) to: "
            f"{fit.hard_max_ctx_that_fits} tokens"
        )
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
# Pure renderer — compare
# --------------------------------------------------------------------------- #
_COMPARE_TARGET_CTX = 8192


def _nearest_speed_point(points, target_ctx: int):
    return min(points, key=lambda p: abs(p.ctx - target_ctx))


def format_comparison(
    logical_id: str,
    display_name: str,
    profile: SystemProfile,
    rows: list[RuntimeComparison],
    calib_note: Optional[str] = None,
) -> str:
    L: list[str] = []
    L.append(f"canirunit compare — {logical_id}  ({display_name})")
    basis = (
        "Metal working set" if profile.metal_max_working_set_bytes is not None
        else "free VRAM" if profile.accelerator == "cuda"
        else "available RAM"
    )
    usable_bytes = next(
        (r.fit.breakdown.usable_bytes for r in rows if r.fit is not None),
        profile.available_memory_bytes,
    )
    L.append(
        f"Machine: {profile.chip_id}  [{profile.accelerator}]  ·  "
        f"{profile.memory_bandwidth_gbs:.0f} GB/s  ·  "
        f"{_gb(profile.total_memory_bytes):.1f} GB total, "
        f"{_gb(usable_bytes):.1f} GB usable ({basis})"
    )
    if calib_note:
        L.append(f"  ! {calib_note}")
    L.append("")

    header = (
        f"  {'runtime':<8}  {'quant':<12}  {'fits-native':<11}  "
        f"{'max-ctx':>8}  {'decode@8k':>10}  {'ttft@8k':>9}  availability"
    )
    L.append(header)
    L.append("  " + "-" * (len(header) - 2))
    for r in rows:
        quant = r.quant_label or "-"
        if r.error:
            L.append(
                f"  {r.runtime:<8}  {quant:<12}  {'-':<11}  {'-':>8}  "
                f"{'-':>10}  {'-':>9}  {r.available_reason}: {r.error.splitlines()[0][:60]}"
            )
            continue
        assert r.fit is not None and r.speed is not None and r.spec is not None
        target = min(_COMPARE_TARGET_CTX, r.spec.native_ctx)
        pt = _nearest_speed_point(r.speed.points, target)
        fits = "yes" if r.fit.fits_at_native_ctx else "no"
        L.append(
            f"  {r.runtime:<8}  {quant:<12}  {fits:<11}  "
            f"{r.fit.max_ctx_that_fits:>8}  "
            f"{pt.decode_tok_s:>5.1f} tok/s  {_fmt_ttft(pt.ttft_s):>9}  "
            f"{r.available_reason}"
        )
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Pure renderer — models
# --------------------------------------------------------------------------- #
def format_models(rows: list[dict]) -> str:
    L: list[str] = []
    L.append(f"  {'id':<32}  {'family':<10}  runtimes")
    L.append("  " + "-" * 60)
    for r in rows:
        L.append(f"  {r['id']:<32}  {r['family']:<10}  {', '.join(r['runtimes'])}")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Resolution helper: model arg may be a logical id or a runtime-native ref
# --------------------------------------------------------------------------- #
def _resolve_model_arg(
    model: str,
    runtime: Runtime,
    quant: str,
    resolve_fn: Callable[[str], dict],
) -> tuple[str, str]:
    """If ``model`` matches an alias entry for ``runtime``, return its source
    ref (and the alias's default_quant when GGUF). Otherwise pass through."""
    try:
        entry = resolve_fn(model)
    except KeyError:
        return model, quant
    src_info = (entry.get("sources") or {}).get(runtime)
    if not src_info:
        # Known logical id but no source for this runtime — let the source raise.
        return model, quant
    if runtime in ("gguf", "mlx"):
        ref = src_info.get("repo_id", model)
    else:
        ref = src_info.get("tag", model)
    if runtime == "gguf" and src_info.get("default_quant"):
        quant = src_info["default_quant"]
    return ref, quant


def _default_fetch_for(runtime: Runtime) -> Callable[..., ModelSpec]:
    """The right default `fetch_fn` for `run_check`. GGUF keeps the legacy
    `_fetch_model` for byte-for-byte backward compatibility; other runtimes
    use their SpecSource."""
    if runtime == "gguf":
        return _fetch_model
    return get_source(runtime).fetch


# --------------------------------------------------------------------------- #
# Orchestration: check
# --------------------------------------------------------------------------- #
def run_check(
    model: str,
    quant: str = "Q4_K_M",
    ctx: Optional[int] = None,
    do_calibrate: bool = False,
    *,
    runtime: Runtime = "gguf",
    detect_fn: Callable[[], SystemProfile] = _detect,
    fetch_fn: Optional[Callable[..., ModelSpec]] = None,
    calibrate_fn: Callable[[SystemProfile], Optional[Calibration]] = _calibrate,
    resolve_fn: Callable[[str], dict] = _resolve_alias,
    config: Optional[EstimatorConfig] = None,
) -> str:
    config = config or EstimatorConfig()
    profile = detect_fn()
    ref, quant = _resolve_model_arg(model, runtime, quant, resolve_fn)

    fetch = fetch_fn if fetch_fn is not None else _default_fetch_for(runtime)
    spec = fetch(ref, quant)

    calibration = None
    calib_note = None
    if do_calibrate:
        calibration = calibrate_fn(profile)
        if calibration is None:
            calib_note = (
                "Calibration requested but no supported runtime "
                "(llama-bench / mlx_lm) found; showing static estimate."
            )

    ctx_points = None
    if ctx:
        base = [c for c in (2048, 8192, ctx, spec.native_ctx) if 0 < c <= spec.native_ctx]
        ctx_points = sorted(set(base))

    fit = estimate_fit(profile, spec, config)
    speed = estimate_speed(profile, spec, config, calibration=calibration, ctx_points=ctx_points)
    return format_report(profile, spec, fit, speed, calibration, chip_is_known(profile), calib_note)


# --------------------------------------------------------------------------- #
# Orchestration: compare
# --------------------------------------------------------------------------- #
def run_compare(
    logical_id: str,
    do_calibrate: bool = False,
    *,
    detect_fn: Callable[[], SystemProfile] = _detect,
    compare_fn: Callable[..., list[RuntimeComparison]] = _compare,
    resolve_fn: Callable[[str], dict] = _resolve_alias,
    calibrate_fn: Callable[[SystemProfile, Runtime], Optional[Calibration]] = (
        lambda p, r: _calibrate(p, target_runtime=r)
    ),
    config: Optional[EstimatorConfig] = None,
) -> str:
    config = config or EstimatorConfig()
    profile = detect_fn()
    entry = resolve_fn(logical_id)  # raises KeyError for unknown ids
    display_name = entry.get("display_name", logical_id)

    cal_by_runtime: dict[Runtime, Calibration] = {}
    calib_note = None
    if do_calibrate:
        sources = entry.get("sources", {})
        attempted, missing = 0, []
        for runtime in sources:
            attempted += 1
            cal = calibrate_fn(profile, runtime)
            if cal is not None:
                cal_by_runtime[runtime] = cal
            else:
                missing.append(runtime)
        if attempted and not cal_by_runtime:
            calib_note = (
                "Calibration requested but no supported tooling found for any "
                f"runtime ({', '.join(sources)}); showing static estimates."
            )
        elif missing:
            calib_note = (
                "Static fallback for runtime(s) without calibration tooling: "
                f"{', '.join(missing)}."
            )

    rows = compare_fn(
        logical_id, profile, config=config,
        calibration_by_runtime=cal_by_runtime,
        resolve_fn=resolve_fn,
    )
    return format_comparison(logical_id, display_name, profile, rows, calib_note)


# --------------------------------------------------------------------------- #
# Orchestration: refresh / models
# --------------------------------------------------------------------------- #
def run_refresh(
    *,
    refresh_fn: Callable[..., dict] = _refresh,
) -> str:
    result = refresh_fn()
    if result.get("ok"):
        return (
            f"canirunit: refreshed alias table ({result['models']} models) -> "
            f"{result['path']}  [updated_at {result.get('updated_at')}]"
        )
    return f"canirunit: refresh failed — {result.get('error')}"


def run_models(
    *,
    list_fn: Callable[..., list[dict]] = _list_models,
) -> str:
    return format_models(list_fn())


# --------------------------------------------------------------------------- #
# Orchestration: serve (blocks; returns process exit code)
# --------------------------------------------------------------------------- #
def _run_serve(host: str, port: int, open_browser: bool) -> int:
    try:
        from .server import create_app
    except ImportError:
        print(
            "canirunit: the web UI extras aren't installed.\n"
            "  pip install 'canirunit[ui]'\n"
            "and try `canirunit serve` again.",
            file=sys.stderr,
        )
        return 1
    import uvicorn

    app = create_app()
    url = f"http://{host}:{port}/"
    if open_browser:
        import threading
        import webbrowser

        # Delay opening so uvicorn is listening by the time the browser hits it.
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"canirunit: serving at {url}  (Ctrl-C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="canirunit",
        description="Estimate whether a local model fits and runs well on this machine.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Check a specific model by repo id, ollama tag, or alias id.")
    check.add_argument(
        "model",
        help="HF repo id, Ollama tag (with --runtime ollama), or logical alias id",
    )
    check.add_argument("--quant", default="Q4_K_M", help="Quantization tag to match (GGUF; default: Q4_K_M)")
    check.add_argument("--ctx", type=int, default=None, help="Context length of interest (tokens)")
    check.add_argument("--calibrate", action="store_true", help="Run an on-machine benchmark for measured speed")
    check.add_argument(
        "--runtime",
        choices=("gguf", "mlx", "ollama"),
        default="gguf",
        help="Which runtime to evaluate (default: gguf)",
    )

    cmp_ = sub.add_parser("compare", help="Compare a logical model across all runtimes that have a source.")
    cmp_.add_argument("logical_id", help="Logical model id from the alias table (see `canirunit models`)")
    cmp_.add_argument("--calibrate", action="store_true",
                      help="Calibrate each available runtime first; otherwise show static estimates")

    sub.add_parser("refresh", help="Pull the latest published alias table into the local overlay.")
    sub.add_parser("models", help="List the known model aliases (shipped + any overlay).")

    serve = sub.add_parser("serve", help="Launch the local web UI on http://127.0.0.1:8765/")
    serve.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    serve.add_argument("--no-browser", action="store_true", help="Do not open a browser window")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            calibrate_fn = (
                partial(_calibrate, target_runtime=args.runtime) if args.calibrate else _calibrate
            )
            report = run_check(
                args.model,
                quant=args.quant,
                ctx=args.ctx,
                do_calibrate=args.calibrate,
                runtime=args.runtime,
                calibrate_fn=calibrate_fn,
            )
        elif args.command == "compare":
            report = run_compare(args.logical_id, do_calibrate=args.calibrate)
        elif args.command == "refresh":
            report = run_refresh()
        elif args.command == "models":
            report = run_models()
        elif args.command == "serve":
            return _run_serve(host=args.host, port=args.port, open_browser=not args.no_browser)
        else:
            return 1
    except (ValueError, KeyError) as e:
        print(f"canirunit: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"canirunit: {e}", file=sys.stderr)
        return 1
    except (OSError, requests.exceptions.RequestException) as e:
        print(
            f"canirunit: could not fetch from upstream. "
            f"Check the ref, your connection, and any --runtime flag.\n  ({e})",
            file=sys.stderr,
        )
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

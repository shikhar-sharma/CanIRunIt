"""Across-runtime comparison: 'Can I run X?' answered for every runtime
that has a source.

The estimator math is identical across runtimes; this module fans the same
profile across each runtime's spec, isolates per-runtime failures (one row
failing must not abort the whole table), and surfaces availability — i.e.
whether the user could actually run the model on this machine (tools
installed, model pulled).
"""
from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Callable, Optional

from .aliases import resolve as _resolve
from .config import EstimatorConfig
from .estimator import estimate_fit, estimate_speed
from .sources import get_source
from .types import (
    Calibration,
    FitResult,
    ModelSpec,
    Runtime,
    SpeedResult,
    SystemProfile,
)


@dataclass(frozen=True)
class RuntimeComparison:
    """One row of a compare output.

    ``spec``/``fit``/``speed`` are None when ``error`` is set; otherwise all
    three are populated. ``available`` is independent of whether fetch
    succeeded: a CUDA box can still fetch an MLX spec and compute estimates,
    but can't run the model, so ``available=False, available_reason="not
    Apple Silicon"``.
    """

    runtime: Runtime
    spec: Optional[ModelSpec]
    fit: Optional[FitResult]
    speed: Optional[SpeedResult]
    available: bool
    available_reason: str
    error: Optional[str]
    quant_label: Optional[str]


# --------------------------------------------------------------------------- #
# Default availability predicates (injectable for tests)
# --------------------------------------------------------------------------- #
def _default_is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _default_mlx_available() -> bool:
    if not _default_is_apple_silicon():
        return False
    try:
        import importlib

        importlib.import_module("mlx_lm")
    except ImportError:
        return False
    return True


def _runtime_availability(
    runtime: Runtime,
    fetch_succeeded: bool,
    is_apple_silicon: Callable[[], bool],
    mlx_available: Callable[[], bool],
) -> tuple[bool, str]:
    if runtime == "gguf":
        return (True, "ok") if fetch_succeeded else (False, "fetch failed")
    if runtime == "ollama":
        # Fetch only succeeds when the manifest+blob are present locally.
        return (True, "ok") if fetch_succeeded else (False, "ollama model not pulled")
    if runtime == "mlx":
        if not is_apple_silicon():
            return False, "not Apple Silicon"
        if not mlx_available():
            return False, "mlx_lm not installed"
        return (True, "ok") if fetch_succeeded else (False, "fetch failed")
    return False, f"unknown runtime {runtime!r}"


def _model_ref_for(runtime: Runtime, src_info: dict) -> str:
    """Per-runtime: GGUF and MLX use repo_id, Ollama uses tag."""
    if runtime in ("gguf", "mlx"):
        ref = src_info.get("repo_id")
    else:
        ref = src_info.get("tag")
    if not ref:
        raise KeyError(f"alias entry for runtime {runtime!r} has no model reference")
    return ref


def compare(
    logical_id: str,
    profile: SystemProfile,
    config: Optional[EstimatorConfig] = None,
    calibration_by_runtime: Optional[dict[Runtime, Calibration]] = None,
    *,
    source_for: Callable[[Runtime], object] = get_source,
    resolve_fn: Callable[[str], dict] = _resolve,
    is_apple_silicon: Callable[[], bool] = _default_is_apple_silicon,
    mlx_available: Callable[[], bool] = _default_mlx_available,
) -> list[RuntimeComparison]:
    """Return one ``RuntimeComparison`` per runtime declared in the alias entry.

    Per-runtime failures are caught and surfaced as ``error`` on the
    respective row; other runtimes are unaffected.
    """
    config = config or EstimatorConfig()
    calibration_by_runtime = calibration_by_runtime or {}

    entry = resolve_fn(logical_id)
    sources: dict = entry.get("sources", {})

    rows: list[RuntimeComparison] = []
    # Iterate in a stable order so the rendered table is deterministic.
    for runtime in sorted(sources.keys()):
        src_info = sources[runtime]
        try:
            src = source_for(runtime)
            model_ref = _model_ref_for(runtime, src_info)
            spec = src.fetch(model_ref, src_info.get("default_quant"))
            cal = calibration_by_runtime.get(runtime)
            fit = estimate_fit(profile, spec, config)
            speed = estimate_speed(profile, spec, config, calibration=cal)
            avail, reason = _runtime_availability(
                runtime, True, is_apple_silicon, mlx_available
            )
            rows.append(RuntimeComparison(
                runtime=runtime,
                spec=spec,
                fit=fit,
                speed=speed,
                available=avail,
                available_reason=reason,
                error=None,
                quant_label=spec.quant_label or spec.quant,
            ))
        except Exception as e:  # noqa: BLE001 — surface per-row, never abort
            avail, reason = _runtime_availability(
                runtime, False, is_apple_silicon, mlx_available
            )
            rows.append(RuntimeComparison(
                runtime=runtime,
                spec=None,
                fit=None,
                speed=None,
                available=avail,
                available_reason=reason,
                error=str(e),
                quant_label=None,
            ))
    return rows

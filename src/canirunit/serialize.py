"""JSON serialization contract for the web UI.

Every field name and unit here is a stable public API the frontend depends on.
Keep this file the single source of truth for the API shapes. Rules:

- snake_case keys
- units in the name where ambiguous: ``*_bytes``, ``*_gbs``, ``*_tok_s``, ``*_s``
- every value JSON-native (int/float/str/bool/None/list/dict — no numpy, no
  Decimal, no dataclasses leaking through)
- additive changes only; never rename a field the frontend consumes

The API layer (``server.py``) calls these; the estimator/detector stay pure and
runtime-agnostic.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from .compare import RuntimeComparison
from .config import EstimatorConfig
from .detector import chip_is_known
from .estimator import (
    hard_usable_memory_bytes,
    kv_cache_bytes,
    usable_memory_bytes,
)
from .types import (
    Calibration,
    FitResult,
    ModelSpec,
    SpeedResult,
    SystemProfile,
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _f(x) -> Optional[float]:
    """Coerce to a JSON-native float, mapping inf/nan -> None so the client
    doesn't have to know they exist. Keep as float otherwise."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isinf(v) or math.isnan(v):
        return None
    return v


def _usable_basis(profile: SystemProfile) -> str:
    if profile.metal_max_working_set_bytes is not None:
        return "Metal working set"
    if profile.accelerator == "cuda":
        return "free VRAM"
    return "available RAM"


# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #
def system_to_dict(profile: SystemProfile, config: Optional[EstimatorConfig] = None) -> dict:
    """The machine as the frontend sees it. Includes derived facts the UI
    needs (usable ceiling + its basis label) so the JS never has to branch
    on accelerator."""
    config = config or EstimatorConfig()
    return {
        "chip_id": profile.chip_id,
        "accelerator": profile.accelerator,
        "chip_is_known": bool(chip_is_known(profile)),
        "memory_bandwidth_gbs": _f(profile.memory_bandwidth_gbs),
        "total_memory_bytes": int(profile.total_memory_bytes),
        "available_memory_bytes": int(profile.available_memory_bytes),
        "usable_memory_bytes": int(usable_memory_bytes(profile, config)),
        "hard_usable_memory_bytes": (
            int(hard_usable_memory_bytes(profile, config))
            if profile.metal_max_working_set_bytes is not None
            else None
        ),
        "usable_basis": _usable_basis(profile),
        "storage_free_bytes": int(profile.storage_free_bytes),
        "metal_max_working_set_bytes": (
            int(profile.metal_max_working_set_bytes)
            if profile.metal_max_working_set_bytes is not None
            else None
        ),
    }


# --------------------------------------------------------------------------- #
# Model spec
# --------------------------------------------------------------------------- #
def spec_to_dict(spec: ModelSpec) -> dict:
    return {
        "repo_id": spec.repo_id,
        "runtime": spec.runtime,
        "quant": spec.quant,
        "quant_label": spec.quant_label,
        "architecture": spec.architecture,
        "n_layers": int(spec.n_layers),
        "n_kv_heads": int(spec.n_kv_heads),
        "key_length": int(spec.key_length),
        "value_length": int(spec.value_length),
        "native_ctx": int(spec.native_ctx),
        "total_weight_bytes": int(spec.total_weight_bytes),
        "active_weight_bytes": int(spec.active_weight_bytes),
        "total_params": int(spec.total_params),
        "active_params": int(spec.active_params) if spec.active_params is not None else None,
        "is_moe": bool(spec.is_moe),
        "kv_is_standard": bool(spec.kv_is_standard),
    }


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def fit_to_dict(fit: FitResult) -> dict:
    b = fit.breakdown
    return {
        "max_ctx_that_fits": int(fit.max_ctx_that_fits),
        "hard_max_ctx_that_fits": (
            int(fit.hard_max_ctx_that_fits)
            if fit.hard_max_ctx_that_fits is not None
            else None
        ),
        "fits_at_native_ctx": bool(fit.fits_at_native_ctx),
        "storage_ok": bool(fit.storage_ok),
        "kv_quant_suggestion": fit.kv_quant_suggestion,
        "notes": list(fit.notes),
        "breakdown": {
            "weight_bytes": int(b.weight_bytes),
            "kv_bytes_at_native": int(b.kv_bytes),
            "compute_overhead_bytes": int(b.compute_overhead_bytes),
            "headroom_bytes": int(b.headroom_bytes),
            "usable_bytes": int(b.usable_bytes),
            "required_bytes_at_native": int(b.required_bytes),
        },
    }


# --------------------------------------------------------------------------- #
# Speed
# --------------------------------------------------------------------------- #
def speed_to_dict(speed: SpeedResult) -> dict:
    return {
        "confidence": speed.confidence,
        "points": [
            {"ctx": int(p.ctx), "decode_tok_s": _f(p.decode_tok_s), "ttft_s": _f(p.ttft_s)}
            for p in speed.points
        ],
        "notes": list(speed.notes),
    }


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibration_to_dict(cal: Optional[Calibration]) -> Optional[dict]:
    if cal is None:
        return None
    return {
        "runtime": cal.runtime,
        "source": cal.source,
        "measured_on_chip": cal.measured_on_chip,
        "effective_bytes_per_sec": _f(cal.effective_bytes_per_sec),
        "prefill_flops_per_sec": _f(cal.prefill_flops_per_sec),
    }


# --------------------------------------------------------------------------- #
# Comparison row
# --------------------------------------------------------------------------- #
def comparison_to_dict(row: RuntimeComparison) -> dict:
    return {
        "runtime": row.runtime,
        "available": bool(row.available),
        "available_reason": row.available_reason,
        "error": row.error,
        "quant_label": row.quant_label,
        "spec": spec_to_dict(row.spec) if row.spec is not None else None,
        "fit": fit_to_dict(row.fit) if row.fit is not None else None,
        "speed": speed_to_dict(row.speed) if row.speed is not None else None,
    }


# --------------------------------------------------------------------------- #
# Memory curve — the data behind the KV teaching chart
# --------------------------------------------------------------------------- #
def _log_linear_ctx_points(native_ctx: int, n_points: int = 40) -> list[int]:
    """Sample context values across [0, native_ctx] with denser points at the low
    end (where the curve changes fast) and sparser at the high end (where it
    changes slowly). Log-ish spacing without the log(0) headache."""
    if native_ctx <= 0:
        return [0]
    if n_points < 2:
        return [0, native_ctx]
    xs: list[int] = [0]
    # Log-spaced from 1 upward feels right for KV (linear in ctx but the chart
    # spans several orders of magnitude for large native contexts).
    lo = math.log(1.0)
    hi = math.log(float(native_ctx))
    for i in range(1, n_points):
        t = i / (n_points - 1)
        xs.append(int(round(math.exp(lo + t * (hi - lo)))))
    # Dedup while preserving order.
    seen: set[int] = set()
    out: list[int] = []
    for x in xs:
        if 0 <= x <= native_ctx and x not in seen:
            seen.add(x)
            out.append(x)
    if native_ctx not in seen:
        out.append(native_ctx)
    return out


def memory_curve(
    profile: SystemProfile,
    spec: ModelSpec,
    config: Optional[EstimatorConfig] = None,
    kv_quant: str = "f16",
    ctx_points: Optional[Sequence[int]] = None,
) -> dict:
    """Data for the KV-memory-vs-context chart.

    Returns weights baseline, overhead, both ceilings, and a list of
    ``{ctx, kv_bytes, total_bytes}`` samples. The frontend plots weights as
    a flat baseline and stacks KV on top; comfort and hard ceilings render
    as horizontal lines.
    """
    config = config or EstimatorConfig()
    xs = list(ctx_points) if ctx_points is not None else _log_linear_ctx_points(spec.native_ctx)

    weight_bytes = int(spec.total_weight_bytes)
    overhead_bytes = int(config.compute_overhead_bytes)
    usable = int(usable_memory_bytes(profile, config))
    hard_usable = (
        int(hard_usable_memory_bytes(profile, config))
        if profile.metal_max_working_set_bytes is not None
        else None
    )

    points = []
    for ctx in xs:
        kv = int(kv_cache_bytes(spec, ctx, kv_quant, config))
        total = weight_bytes + kv + overhead_bytes
        points.append({"ctx": int(ctx), "kv_bytes": kv, "total_bytes": total})

    return {
        "kv_quant": kv_quant,
        "weight_bytes": weight_bytes,
        "overhead_bytes": overhead_bytes,
        "usable_bytes": usable,
        "hard_usable_bytes": hard_usable,
        "native_ctx": int(spec.native_ctx),
        "points": points,
    }

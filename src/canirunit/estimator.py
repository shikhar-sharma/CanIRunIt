"""The pure estimator core. No I/O, fully deterministic, unit-testable against
hand-computed fixtures. Everything takes a SystemProfile + ModelSpec (+ optional
Calibration) and returns a result dataclass.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .config import EstimatorConfig
from .types import (
    Calibration,
    FitResult,
    MemoryBreakdown,
    ModelSpec,
    SpeedPoint,
    SpeedResult,
    SystemProfile,
)


# --------------------------------------------------------------------------- #
# KV cache — the load-bearing calculation
# --------------------------------------------------------------------------- #
def kv_cache_bytes(spec: ModelSpec, ctx: int, kv_quant: str, config: EstimatorConfig) -> int:
    """KV cache size at a given context length.

    bytes = ctx * n_layers * n_kv_heads * (key_length + value_length) * bytes_per_elem

    Two gotchas are encoded in the field choices upstream:
      * n_kv_heads (NOT n_heads): GQA makes KV heads a fraction of query heads.
      * key_length/value_length read explicitly — never derived as hidden/n_heads,
        which is wrong for Gemma (head_dim is set independently of that ratio).
    """
    bpe = config.kv_bytes_per_elem(kv_quant)
    per_token = spec.n_layers * spec.n_kv_heads * (spec.key_length + spec.value_length) * bpe
    return int(per_token * ctx)


# --------------------------------------------------------------------------- #
# Memory budget
# --------------------------------------------------------------------------- #
def usable_memory_bytes(profile: SystemProfile, config: EstimatorConfig) -> int:
    """What we can actually spend on weights + KV + compute buffers.

    On Apple the Metal working set is the authoritative ceiling: when a model
    loads, macOS reclaims cache/inactive memory up to the wired limit, so the
    *instantaneous* free RAM must NOT cap the budget (otherwise the verdict swings
    with whatever apps happen to be open). Elsewhere — CUDA free VRAM, or CPU
    available RAM — the available figure is the real cap.
    """
    if profile.metal_max_working_set_bytes is not None:
        cap = profile.metal_max_working_set_bytes
    else:
        cap = profile.available_memory_bytes
    return max(0, cap - config.safety_headroom_bytes)


def _need_at(spec: ModelSpec, ctx: int, kv_quant: str, config: EstimatorConfig) -> int:
    return (
        spec.total_weight_bytes
        + kv_cache_bytes(spec, ctx, kv_quant, config)
        + config.compute_overhead_bytes
    )


def _fits(profile: SystemProfile, spec: ModelSpec, ctx: int, kv_quant: str, config: EstimatorConfig) -> bool:
    return _need_at(spec, ctx, kv_quant, config) <= usable_memory_bytes(profile, config)


def max_ctx_that_fits(
    profile: SystemProfile, spec: ModelSpec, kv_quant: str, config: EstimatorConfig
) -> int:
    """Largest context (<= native_ctx) whose weights + KV + overhead fit.

    Memory need is monotonic increasing in ctx, so we binary-search the ceiling.
    Returns 0 if the weights alone don't fit.
    """
    hi = spec.native_ctx
    if not _fits(profile, spec, 0, kv_quant, config):
        return 0
    if _fits(profile, spec, hi, kv_quant, config):
        return hi
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _fits(profile, spec, mid, kv_quant, config):
            lo = mid
        else:
            hi = mid - 1
    return lo


def estimate_fit(
    profile: SystemProfile,
    spec: ModelSpec,
    config: EstimatorConfig,
    kv_quant: str = "f16",
) -> FitResult:
    notes: list[str] = []
    if not spec.kv_is_standard:
        notes.append(
            "Compressed-KV architecture (e.g. MLA): the standard KV formula "
            "over-estimates and this fit result is unreliable."
        )

    max_ctx = max_ctx_that_fits(profile, spec, kv_quant, config)
    fits_native = max_ctx >= spec.native_ctx

    # If native context doesn't fit at f16, see whether a cheaper KV quant rescues it.
    suggestion: Optional[str] = None
    if not fits_native and kv_quant == "f16" and spec.kv_is_standard:
        for q in ("q8", "q4"):
            if max_ctx_that_fits(profile, spec, q, config) >= spec.native_ctx:
                suggestion = f"{q} KV cache reaches native context ({spec.native_ctx} tokens)"
                break

    breakdown = MemoryBreakdown(
        weight_bytes=spec.total_weight_bytes,
        kv_bytes=kv_cache_bytes(spec, spec.native_ctx, kv_quant, config),
        compute_overhead_bytes=config.compute_overhead_bytes,
        headroom_bytes=config.safety_headroom_bytes,
        usable_bytes=usable_memory_bytes(profile, config),
    )
    storage_ok = profile.storage_free_bytes >= spec.total_weight_bytes

    return FitResult(
        max_ctx_that_fits=max_ctx,
        fits_at_native_ctx=fits_native,
        breakdown=breakdown,
        storage_ok=storage_ok,
        kv_quant_suggestion=suggestion,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Speed — decode (bandwidth-bound) and prefill/TTFT (compute-bound)
# --------------------------------------------------------------------------- #
def _calibration_applies(calibration: Optional[Calibration], spec: ModelSpec) -> bool:
    """A calibration applies if its runtime tag matches the spec's runtime.

    GGUF and Ollama share the llama.cpp physics and tooling, so calibrations
    cross-apply between them. MLX is its own world — an MLX-measured constant
    on a GGUF target (or vice versa) would silently mislead, so we refuse.
    """
    if calibration is None:
        return False
    if calibration.runtime == spec.runtime:
        return True
    return {calibration.runtime, spec.runtime} == {"gguf", "ollama"}


def _pick_calibration(
    calibration: Optional[Calibration], spec: ModelSpec
) -> Optional[Calibration]:
    """Return ``calibration`` iff it applies to ``spec``, else None."""
    return calibration if _calibration_applies(calibration, spec) else None


def effective_bytes_per_sec(
    profile: SystemProfile, config: EstimatorConfig, calibration: Optional[Calibration]
) -> float:
    """The measured constant, when present, replaces the whole MBU x bandwidth
    product — which is the entire payoff of calibration."""
    if calibration is not None:
        return calibration.effective_bytes_per_sec
    bw = profile.memory_bandwidth_gbs or config.default_fallback_bandwidth_gbs
    return config.mbu * bw * 1e9  # GB/s are decimal; 1e9 bytes per GB


def decode_tok_s(
    profile: SystemProfile,
    spec: ModelSpec,
    ctx: int,
    config: EstimatorConfig,
    kv_quant: str = "f16",
    calibration: Optional[Calibration] = None,
) -> float:
    """Decode is memory-bandwidth-bound. KV sits in the denominator, so speed
    decays as context fills. Uses active_weight_bytes (MoE-correct).

    A calibration is only applied if its runtime matches ``spec.runtime``
    (gguf/ollama are interchangeable). Cross-runtime calibrations are
    silently dropped here; ``estimate_speed`` surfaces the fact in a note.
    """
    eff = effective_bytes_per_sec(profile, config, _pick_calibration(calibration, spec))
    denom = spec.active_weight_bytes + kv_cache_bytes(spec, ctx, kv_quant, config)
    return eff / denom if denom > 0 else 0.0


def prefill_tok_s(
    profile: SystemProfile,
    spec: ModelSpec,
    config: EstimatorConfig,
    calibration: Optional[Calibration] = None,
) -> float:
    """Prefill is compute-bound: ~ effective_FLOPS / (2 * active_params).
    Lower confidence than decode; anchored on measured FLOP/s when calibrated.

    Same calibration runtime-guard as ``decode_tok_s``.
    """
    params = max(1, spec.decode_active_params)
    cal = _pick_calibration(calibration, spec)
    if cal is not None and cal.prefill_flops_per_sec:
        return cal.prefill_flops_per_sec / (2 * params)
    flops = (profile.peak_flops or config.default_peak_flops) * config.compute_efficiency
    return flops / (2 * params)


def _default_ctx_points(native_ctx: int) -> list[int]:
    candidates = [2048, 8192, 16384, 32768, native_ctx]
    pts = sorted({c for c in candidates if 0 < c <= native_ctx})
    return pts or [native_ctx]


def estimate_speed(
    profile: SystemProfile,
    spec: ModelSpec,
    config: EstimatorConfig,
    calibration: Optional[Calibration] = None,
    kv_quant: str = "f16",
    ctx_points: Optional[Sequence[int]] = None,
) -> SpeedResult:
    points_ctx = list(ctx_points) if ctx_points is not None else _default_ctx_points(spec.native_ctx)
    effective_cal = _pick_calibration(calibration, spec)
    pf = prefill_tok_s(profile, spec, config, calibration)  # guard re-applied inside

    points = [
        SpeedPoint(
            ctx=ctx,
            decode_tok_s=decode_tok_s(profile, spec, ctx, config, kv_quant, calibration),
            ttft_s=(ctx / pf if pf > 0 else float("inf")),
        )
        for ctx in points_ctx
    ]

    notes: list[str] = []
    if calibration is not None and effective_cal is None:
        notes.append(
            f"Calibration was measured for runtime {calibration.runtime!r} but "
            f"this spec's runtime is {spec.runtime!r}; ignoring the calibration "
            "and falling back to the static estimate."
        )
    if effective_cal is None:
        notes.append("Static estimate; run an on-machine calibration for measured numbers.")
    notes.append("Prefill/TTFT is a rougher estimate than decode.")
    if profile.accelerator == "apple_metal":
        notes.append("Sustained decode may run below a short benchmark due to thermal limits.")

    return SpeedResult(
        points=points,
        confidence="measured" if effective_cal is not None else "estimated",
        notes=notes,
    )


def estimate(
    profile: SystemProfile,
    spec: ModelSpec,
    config: Optional[EstimatorConfig] = None,
    calibration: Optional[Calibration] = None,
    kv_quant: str = "f16",
) -> tuple[FitResult, SpeedResult]:
    """Convenience: fit + speed in one call."""
    config = config or EstimatorConfig()
    return (
        estimate_fit(profile, spec, config, kv_quant),
        estimate_speed(profile, spec, config, calibration, kv_quant),
    )

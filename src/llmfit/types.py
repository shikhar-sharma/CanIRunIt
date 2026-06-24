"""Shared dataclasses — the contracts every other module speaks in.

This module depends on nothing. It is the spine: if these shapes are right,
each module can be built and tested in isolation against them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Accelerator = Literal["apple_metal", "cuda", "cpu"]
KVQuant = Literal["f16", "q8", "q4"]
BenchSource = Literal["llama-bench", "ollama", "static"]
Confidence = Literal["measured", "estimated"]


@dataclass(frozen=True)
class SystemProfile:
    """A normalized view of the machine. Produced by the detector's
    platform-specific backends; consumed by the estimator, which never
    sees a single ``if mac / elif cuda`` branch."""

    total_memory_bytes: int
    available_memory_bytes: int          # usable *now*, not nameplate total
    memory_bandwidth_gbs: float          # decode-speed driver; lookup table or calibrated
    accelerator: Accelerator
    chip_id: str                         # e.g. "Apple M1" — keys the bandwidth table
    storage_free_bytes: int
    # Apple GPU allocation ceiling (recommendedMaxWorkingSetSize, ~0.75 x total).
    # The honest cap on resident weights on Apple Silicon; None elsewhere.
    metal_max_working_set_bytes: Optional[int] = None
    # Effective FP16 throughput for the static prefill fallback. Fuzzy by nature;
    # calibration replaces reliance on it.
    peak_flops: Optional[float] = None


@dataclass(frozen=True)
class ModelSpec:
    """A model at a specific quantization. ``total_*`` answers the fit question,
    ``active_*`` answers the speed question — carried separately so no downstream
    code has to remember which question it is answering (this is the MoE split)."""

    repo_id: str
    quant: str

    total_weight_bytes: int              # sum of ALL shards — the fit cost
    active_weight_bytes: int             # the decode cost; == total for dense models
    total_params: int                    # param COUNT (not bytes) — for prefill

    n_layers: int
    n_kv_heads: int                      # head_count_kv — NOT head_count (GQA matters)
    key_length: int                      # per-head K dim, read explicitly from GGUF
    value_length: int                    # per-head V dim, read explicitly from GGUF
    native_ctx: int

    architecture: str                    # "llama" | "gemma3" | "deepseek2" | ...
    is_moe: bool = False
    active_params: Optional[int] = None  # MoE only: the "A4B" number, drives prefill
    # False for MLA / compressed-KV architectures: the standard KV formula does
    # not apply, so the estimator flags rather than lies.
    kv_is_standard: bool = True

    @property
    def decode_active_params(self) -> int:
        """Param count that governs decode/prefill compute: active experts for
        MoE, full param count for dense."""
        if self.is_moe and self.active_params:
            return self.active_params
        return self.total_params


@dataclass(frozen=True)
class MemoryBreakdown:
    """Auditable components of the fit calculation, so a verdict is never opaque."""

    weight_bytes: int
    kv_bytes: int                        # evaluated at native_ctx
    compute_overhead_bytes: int
    headroom_bytes: int
    usable_bytes: int

    @property
    def required_bytes(self) -> int:
        return self.weight_bytes + self.kv_bytes + self.compute_overhead_bytes


@dataclass(frozen=True)
class FitResult:
    max_ctx_that_fits: int               # the ceiling — the number people actually want
    fits_at_native_ctx: bool
    breakdown: MemoryBreakdown
    storage_ok: bool
    kv_quant_suggestion: Optional[str] = None  # "q8 KV cache reaches native context ..."
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SpeedPoint:
    ctx: int
    decode_tok_s: float
    ttft_s: float                        # time-to-first-token for a prompt of length ctx


@dataclass(frozen=True)
class SpeedResult:
    points: list[SpeedPoint]             # a curve over context, not a single number
    confidence: Confidence
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Calibration:
    """Measured constants from an on-machine micro-benchmark. Each absorbs a pile
    of messy real-world factors (MBU, kernel quality, thermal state) into one number
    the estimator can substitute for its spec-sheet guesses."""

    effective_bytes_per_sec: float       # decode anchor
    measured_on_chip: str
    source: BenchSource
    prefill_flops_per_sec: Optional[float] = None  # prefill anchor (effective FLOP/s)

"""canirunit — estimate which local GGUF models will fit and run well on a machine."""
from __future__ import annotations

from .config import EstimatorConfig
from .benchmark import calibrate
from .detector import detect
from .estimator import (
    decode_tok_s,
    estimate,
    estimate_fit,
    estimate_speed,
    kv_cache_bytes,
    max_ctx_that_fits,
    prefill_tok_s,
    usable_memory_bytes,
)
from .types import (
    Calibration,
    FitResult,
    MemoryBreakdown,
    ModelSpec,
    SpeedPoint,
    SpeedResult,
    SystemProfile,
)

__all__ = [
    "EstimatorConfig",
    "Calibration",
    "FitResult",
    "MemoryBreakdown",
    "ModelSpec",
    "SpeedPoint",
    "SpeedResult",
    "SystemProfile",
    "detect",
    "calibrate",
    "estimate",
    "estimate_fit",
    "estimate_speed",
    "decode_tok_s",
    "prefill_tok_s",
    "kv_cache_bytes",
    "max_ctx_that_fits",
    "usable_memory_bytes",
]

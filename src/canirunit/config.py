"""Tunable static defaults for the estimator.

These are the numbers used *before* an on-machine calibration exists. Holding
them in a config object (rather than module constants) gives calibration a clean
place to override values and keeps the estimator's behaviour explicit and testable.
"""
from __future__ import annotations

from dataclasses import dataclass

# Bytes per element for the KV cache at each quantization level. f16 is exact;
# q8/q4 are close approximations of llama.cpp's quantized KV (q8_0 ~1.06,
# q4 ~0.56) — rounded here and refinable per-build.
KV_BYTES_PER_ELEM = {"f16": 2.0, "q8": 1.0, "q4": 0.5}


@dataclass(frozen=True)
class EstimatorConfig:
    # --- decode (memory-bandwidth-bound) ---
    mbu: float = 0.7                     # memory-bandwidth utilization; 0.6-0.8 typical
    default_fallback_bandwidth_gbs: float = 50.0  # if detector can't supply bandwidth

    # --- prefill (compute-bound, lower confidence) ---
    compute_efficiency: float = 0.25     # fraction of peak FLOPS actually reached
    default_peak_flops: float = 2.0e12   # fallback effective FP16 throughput

    # --- fit ---
    # llama.cpp compute/graph buffers: a modest static guess, overwritten by the
    # runtime's actually-reported buffer size once calibration runs.
    compute_overhead_bytes: int = 384 * 1024 ** 2
    # Deliberate slack left free so the machine stays responsive. Distinct from
    # in-use memory, which is already excluded by available_memory_bytes.
    safety_headroom_bytes: int = 1024 ** 3
    # Reserve for the "loads at all" ceiling on Apple, where the wired memory
    # limit is soft (macOS extends it via compression/swap with a throughput
    # cost). Models the soft cap as roughly all of RAM minus what the OS
    # itself needs to stay alive. Approximate — on-device behaviour varies.
    hard_ceiling_reserve_bytes: int = 3 * 1024 ** 3 // 2  # 1.5 GiB

    # Order in which to try lower KV quant when native context doesn't fit at f16.
    kv_quant_order: tuple = ("f16", "q8", "q4")

    def kv_bytes_per_elem(self, kv_quant: str) -> float:
        return KV_BYTES_PER_ELEM[kv_quant]

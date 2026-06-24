"""Hardware detection: detect() -> SystemProfile.

One interface, platform-specific backends underneath. Everything downstream sees
only a normalized SystemProfile and never branches on platform. Adding a new
platform means writing one backend; nothing else changes.

The probe I/O (sysctl, nvidia-smi, psutil) is isolated behind an injectable
command runner so the parsing and assembly logic is testable without the hardware.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Callable, Optional

import psutil

from .types import SystemProfile

Runner = Callable[[list], Optional[str]]

# --------------------------------------------------------------------------- #
# Hardware tables. These are COARSE reference values, keyed on the reported
# chip/GPU id. Memory bandwidth (GB/s) drives the decode estimate; peak_flops is
# a rough prefill anchor only and is superseded the moment calibration runs.
# --------------------------------------------------------------------------- #
APPLE_BANDWIDTH_GBS = {
    "Apple M1": 68.25, "Apple M1 Pro": 200.0, "Apple M1 Max": 400.0, "Apple M1 Ultra": 800.0,
    "Apple M2": 100.0, "Apple M2 Pro": 200.0, "Apple M2 Max": 400.0, "Apple M2 Ultra": 800.0,
    "Apple M3": 100.0, "Apple M3 Pro": 150.0, "Apple M3 Max": 400.0, "Apple M3 Ultra": 800.0,
    "Apple M4": 120.0, "Apple M4 Pro": 273.0, "Apple M4 Max": 410.0,
}
# Approximate GPU FP16 throughput (TFLOPS). Coarse on purpose.
APPLE_PEAK_TFLOPS = {
    "Apple M1": 4.6, "Apple M1 Pro": 10.6, "Apple M1 Max": 21.2, "Apple M1 Ultra": 42.0,
    "Apple M2": 7.1, "Apple M2 Pro": 13.6, "Apple M2 Max": 27.2, "Apple M2 Ultra": 54.0,
    "Apple M3": 7.0, "Apple M3 Pro": 14.0, "Apple M3 Max": 28.0, "Apple M3 Ultra": 56.0,
    "Apple M4": 9.0, "Apple M4 Pro": 17.0, "Apple M4 Max": 34.0,
}
# Fraction of unified memory the GPU may hold resident, when the OS doesn't tell
# us via iogpu.wired_limit_mb. Approximate; refine against real fit testing.
APPLE_WIRED_FRACTION = 0.75
APPLE_BANDWIDTH_FALLBACK = 100.0

# NVIDIA: matched by substring against the nvidia-smi name. GB/s.
NVIDIA_BANDWIDTH_GBS = {
    "RTX 4090": 1008.0, "RTX 4080": 716.8, "RTX 4070": 504.2, "RTX 4060": 272.0,
    "RTX 3090": 936.2, "RTX 3080": 760.3, "RTX 3070": 448.0, "RTX 3060": 360.0,
    "A100": 1555.0, "H100": 3350.0, "A6000": 768.0, "L40": 864.0, "T4": 320.0,
}
NVIDIA_PEAK_TFLOPS = {
    "RTX 4090": 165.0, "RTX 4080": 97.0, "RTX 4070": 58.0, "RTX 4060": 31.0,
    "RTX 3090": 71.0, "RTX 3080": 59.0, "RTX 3070": 40.0, "RTX 3060": 26.0,
    "A100": 312.0, "H100": 990.0, "A6000": 77.0, "L40": 181.0, "T4": 65.0,
}
NVIDIA_BANDWIDTH_FALLBACK = 400.0

# CPU-only fallback: bandwidth is the binding term and varies widely; a
# conservative dual-channel DDR figure. Coarse.
CPU_BANDWIDTH_GBS = 40.0
CPU_PEAK_TFLOPS = 0.5

_NVIDIA_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,memory.free",
    "--format=csv,noheader,nounits",
]


# --------------------------------------------------------------------------- #
# Pure parsers / lookups
# --------------------------------------------------------------------------- #
def parse_int(raw: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (ValueError, AttributeError):
        return default


def normalize_chip(raw: Optional[str]) -> str:
    return raw.strip() if raw and raw.strip() else "Unknown"


def lookup_substring(table: dict, name: str, fallback):
    for key, val in table.items():
        if key in name:
            return val
    return fallback


def parse_nvidia_smi(out: Optional[str]) -> Optional[tuple[str, int, int]]:
    """(name, total_bytes, free_bytes) from the first GPU line, or None."""
    if not out or not out.strip():
        return None
    name, total_mib, free_mib = (p.strip() for p in out.strip().splitlines()[0].split(","))
    mib = 1024 * 1024
    return name, int(total_mib) * mib, int(free_mib) * mib


# --------------------------------------------------------------------------- #
# Assembly (pure: raw numbers -> SystemProfile)
# --------------------------------------------------------------------------- #
def assemble_apple(total: int, chip: str, wired_mb: int, available: int, disk_free: int) -> SystemProfile:
    wired = wired_mb * 1024 * 1024 if wired_mb else int(total * APPLE_WIRED_FRACTION)
    tflops = APPLE_PEAK_TFLOPS.get(chip)
    return SystemProfile(
        total_memory_bytes=total,
        available_memory_bytes=available,
        memory_bandwidth_gbs=APPLE_BANDWIDTH_GBS.get(chip, APPLE_BANDWIDTH_FALLBACK),
        accelerator="apple_metal",
        chip_id=chip,
        storage_free_bytes=disk_free,
        metal_max_working_set_bytes=wired,
        peak_flops=tflops * 1e12 if tflops else None,
    )


def assemble_nvidia(name: str, total_vram: int, free_vram: int, disk_free: int) -> SystemProfile:
    tflops = lookup_substring(NVIDIA_PEAK_TFLOPS, name, None)
    return SystemProfile(
        total_memory_bytes=total_vram,
        available_memory_bytes=free_vram,           # VRAM is the budget (full offload)
        memory_bandwidth_gbs=lookup_substring(NVIDIA_BANDWIDTH_GBS, name, NVIDIA_BANDWIDTH_FALLBACK),
        accelerator="cuda",
        chip_id=name,
        storage_free_bytes=disk_free,
        metal_max_working_set_bytes=None,
        peak_flops=tflops * 1e12 if tflops else None,
    )


# --------------------------------------------------------------------------- #
# Command runner
# --------------------------------------------------------------------------- #
def _run(args: list) -> Optional[str]:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _home_free() -> int:
    return shutil.disk_usage(os.path.expanduser("~")).free


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class AppleBackend:
    def __init__(self, runner: Runner = _run, mem_available: Optional[int] = None,
                 disk_free: Optional[int] = None):
        self.runner = runner
        self._mem_available = mem_available
        self._disk_free = disk_free

    def is_available(self) -> bool:
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def probe(self) -> SystemProfile:
        total = parse_int(self.runner(["sysctl", "-n", "hw.memsize"]), default=0)
        chip = normalize_chip(self.runner(["sysctl", "-n", "machdep.cpu.brand_string"]))
        wired_mb = parse_int(self.runner(["sysctl", "-n", "iogpu.wired_limit_mb"]), default=0) or 0
        available = self._mem_available if self._mem_available is not None else psutil.virtual_memory().available
        disk_free = self._disk_free if self._disk_free is not None else _home_free()
        return assemble_apple(total, chip, wired_mb, available, disk_free)


class NvidiaBackend:
    def __init__(self, runner: Runner = _run, disk_free: Optional[int] = None):
        self.runner = runner
        self._disk_free = disk_free

    def is_available(self) -> bool:
        return shutil.which("nvidia-smi") is not None and parse_nvidia_smi(self.runner(_NVIDIA_QUERY)) is not None

    def probe(self) -> SystemProfile:
        parsed = parse_nvidia_smi(self.runner(_NVIDIA_QUERY))
        if parsed is None:
            raise RuntimeError("nvidia-smi returned no GPU")
        name, total_vram, free_vram = parsed
        disk_free = self._disk_free if self._disk_free is not None else _home_free()
        return assemble_nvidia(name, total_vram, free_vram, disk_free)


class CpuBackend:
    def is_available(self) -> bool:
        return True

    def probe(self) -> SystemProfile:
        mem = psutil.virtual_memory()
        chip = platform.processor() or platform.machine() or "CPU"
        return SystemProfile(
            total_memory_bytes=mem.total,
            available_memory_bytes=mem.available,
            memory_bandwidth_gbs=CPU_BANDWIDTH_GBS,
            accelerator="cpu",
            chip_id=chip,
            storage_free_bytes=_home_free(),
            metal_max_working_set_bytes=None,
            peak_flops=CPU_PEAK_TFLOPS * 1e12,
        )


def detect(backends=None) -> SystemProfile:
    """Return a SystemProfile from the first applicable backend (accelerator
    first, CPU last)."""
    for backend in backends or [AppleBackend(), NvidiaBackend(), CpuBackend()]:
        if backend.is_available():
            return backend.probe()
    return CpuBackend().probe()


def chip_is_known(profile: SystemProfile) -> bool:
    """Whether the bandwidth came from the table rather than a fallback — the cli
    uses this to warn that the decode estimate is on shakier ground."""
    if profile.accelerator == "apple_metal":
        return profile.chip_id in APPLE_BANDWIDTH_GBS
    if profile.accelerator == "cuda":
        return lookup_substring(NVIDIA_BANDWIDTH_GBS, profile.chip_id, None) is not None
    return False

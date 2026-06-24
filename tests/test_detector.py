"""Detector tests. Parsers and assembly are pure; Apple/NVIDIA probes run against
captured sample command output via an injected runner; CPU runs live."""
from __future__ import annotations

import pytest

from llmfit.detector import (
    AppleBackend,
    CpuBackend,
    NvidiaBackend,
    assemble_apple,
    chip_is_known,
    detect,
    lookup_substring,
    normalize_chip,
    parse_int,
    parse_nvidia_smi,
    APPLE_BANDWIDTH_GBS,
    NVIDIA_BANDWIDTH_GBS,
)

GiB = 1024 ** 3


def fake_runner(mapping):
    return lambda args: mapping.get(tuple(args))


# --------------------------------------------------------------------------- #
# Pure parsers / lookups
# --------------------------------------------------------------------------- #
def test_parse_int():
    assert parse_int("17179869184\n") == 17179869184
    assert parse_int("  0 ") == 0
    assert parse_int(None, default=5) == 5
    assert parse_int("not-a-number", default=7) == 7


def test_normalize_chip():
    assert normalize_chip("Apple M2 Pro\n") == "Apple M2 Pro"
    assert normalize_chip("") == "Unknown"
    assert normalize_chip(None) == "Unknown"


def test_lookup_substring_matches_partial():
    assert lookup_substring(NVIDIA_BANDWIDTH_GBS, "NVIDIA GeForce RTX 4090", 0) == 1008.0
    assert lookup_substring(NVIDIA_BANDWIDTH_GBS, "Some Unknown GPU", 42) == 42


def test_parse_nvidia_smi():
    name, total, free = parse_nvidia_smi("NVIDIA GeForce RTX 4090, 24564, 23000\n")
    assert name == "NVIDIA GeForce RTX 4090"
    assert total == 24564 * 1024 * 1024
    assert free == 23000 * 1024 * 1024
    assert parse_nvidia_smi("") is None
    assert parse_nvidia_smi(None) is None


# --------------------------------------------------------------------------- #
# Apple assembly + probe
# --------------------------------------------------------------------------- #
def test_assemble_apple_uses_fraction_when_no_wired_limit():
    p = assemble_apple(total=16 * GiB, chip="Apple M1", wired_mb=0,
                       available=12 * GiB, disk_free=100 * GiB)
    assert p.chip_id == "Apple M1"
    assert p.memory_bandwidth_gbs == 68.25
    assert p.accelerator == "apple_metal"
    assert p.metal_max_working_set_bytes == int(16 * GiB * 0.75)  # 12 GiB
    assert p.peak_flops is not None


def test_assemble_apple_respects_explicit_wired_limit():
    p = assemble_apple(total=16 * GiB, chip="Apple M1", wired_mb=10240,  # 10 GiB
                       available=12 * GiB, disk_free=100 * GiB)
    assert p.metal_max_working_set_bytes == 10240 * 1024 * 1024


def test_apple_probe_end_to_end_via_fake_runner():
    runner = fake_runner({
        ("sysctl", "-n", "hw.memsize"): "17179869184\n",          # 16 GiB
        ("sysctl", "-n", "machdep.cpu.brand_string"): "Apple M1\n",
        ("sysctl", "-n", "iogpu.wired_limit_mb"): "0\n",
    })
    p = AppleBackend(runner=runner, mem_available=12 * GiB, disk_free=100 * GiB).probe()
    assert p.total_memory_bytes == 16 * GiB
    assert p.chip_id == "Apple M1"
    assert p.memory_bandwidth_gbs == 68.25
    assert p.metal_max_working_set_bytes == 12 * GiB
    assert chip_is_known(p) is True


def test_apple_unknown_chip_falls_back_and_is_flagged():
    p = assemble_apple(total=32 * GiB, chip="Apple M9 Ultra", wired_mb=0,
                       available=24 * GiB, disk_free=100 * GiB)
    assert p.memory_bandwidth_gbs == 100.0          # fallback
    assert p.peak_flops is None                     # unknown -> estimator uses its default
    assert chip_is_known(p) is False


# --------------------------------------------------------------------------- #
# NVIDIA probe
# --------------------------------------------------------------------------- #
def test_nvidia_probe_via_fake_runner():
    from llmfit.detector import _NVIDIA_QUERY
    runner = fake_runner({tuple(_NVIDIA_QUERY): "NVIDIA GeForce RTX 4090, 24564, 23000\n"})
    p = NvidiaBackend(runner=runner, disk_free=500 * GiB).probe()
    assert p.accelerator == "cuda"
    assert p.memory_bandwidth_gbs == 1008.0
    assert p.total_memory_bytes == 24564 * 1024 * 1024
    assert p.available_memory_bytes == 23000 * 1024 * 1024
    assert p.metal_max_working_set_bytes is None
    assert chip_is_known(p) is True


# --------------------------------------------------------------------------- #
# CPU backend runs live; detect() selection
# --------------------------------------------------------------------------- #
def test_cpu_backend_live():
    p = CpuBackend().probe()
    assert p.accelerator == "cpu"
    assert p.total_memory_bytes > 0
    assert p.available_memory_bytes > 0
    assert p.storage_free_bytes > 0


def test_detect_picks_first_available_backend():
    class Always:
        def __init__(self, profile):
            self._p = profile
        def is_available(self):
            return True
        def probe(self):
            return self._p

    sentinel = CpuBackend().probe()
    assert detect(backends=[Always(sentinel)]) is sentinel

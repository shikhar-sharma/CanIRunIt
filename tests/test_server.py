"""FastAPI server tests.

TestClient exercises the real routing + serialization stack; every hardware /
network call is behind a Depends override so nothing hits real hardware or HF.
"""
from __future__ import annotations

from typing import Callable

import pytest
from fastapi.testclient import TestClient

from canirunit import (
    Calibration,
    EstimatorConfig,
    ModelSpec,
    SystemProfile,
)
from canirunit.compare import RuntimeComparison
from canirunit.estimator import estimate_fit, estimate_speed
from canirunit.server import (
    create_app,
    get_alias_lister,
    get_alias_resolver,
    get_compare_fn,
    get_config,
    get_profile,
    get_source_registry,
)


GiB = 1024 ** 3


# --------------------------------------------------------------------------- #
# Test fakes
# --------------------------------------------------------------------------- #
def _m1_profile() -> SystemProfile:
    return SystemProfile(
        total_memory_bytes=16 * GiB,
        available_memory_bytes=12 * GiB,
        memory_bandwidth_gbs=68.0,
        accelerator="apple_metal",
        chip_id="Apple M1",
        storage_free_bytes=100 * GiB,
        metal_max_working_set_bytes=12 * GiB,
        peak_flops=2.6e12,
    )


def _llama_spec(runtime="gguf", quant_label="Q4_K_M") -> ModelSpec:
    return ModelSpec(
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        quant=quant_label,
        total_weight_bytes=4_900_000_000,
        active_weight_bytes=4_900_000_000,
        total_params=8_030_000_000,
        n_layers=32, n_kv_heads=8, key_length=128, value_length=128,
        native_ctx=8192, architecture="llama",
        runtime=runtime, quant_label=quant_label,
    )


class _FakeSource:
    def __init__(self, spec=None, raises=None, runtime="gguf"):
        self.runtime = runtime
        self._spec = spec
        self._raises = raises
        self.calls: list[tuple[str, str | None]] = []

    def fetch(self, model_ref, quant=None):
        self.calls.append((model_ref, quant))
        if self._raises is not None:
            raise self._raises
        return self._spec


def _fake_source_factory(by_runtime: dict) -> Callable[[str], object]:
    def source_for(runtime):
        return by_runtime[runtime]
    return source_for


LLAMA_ENTRY = {
    "display_name": "Llama 3.1 8B Instruct",
    "family": "llama",
    "sources": {
        "gguf":   {"repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "default_quant": "Q4_K_M"},
        "mlx":    {"repo_id": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"},
        "ollama": {"tag": "llama3.1:8b"},
    },
}


@pytest.fixture
def client():
    """Fresh app + client with a stable set of fakes wired in."""
    app = create_app()
    app.dependency_overrides[get_profile] = _m1_profile
    app.dependency_overrides[get_config] = lambda: EstimatorConfig()
    _fake_models = [
        {"id": "llama-3.1-8b-instruct", "display_name": "Llama 3.1 8B Instruct",
         "family": "llama", "runtimes": ["gguf", "mlx", "ollama"]},
    ]
    app.dependency_overrides[get_alias_lister] = lambda: (lambda: _fake_models)
    app.dependency_overrides[get_alias_resolver] = lambda: (
        lambda lid: LLAMA_ENTRY if lid == "llama-3.1-8b-instruct" else
        (_ for _ in ()).throw(KeyError(f"unknown {lid!r}"))
    )
    # Default: gguf source returns a llama spec; mlx and ollama sources per test.
    default_sources = {
        "gguf":   _FakeSource(spec=_llama_spec(runtime="gguf"),   runtime="gguf"),
        "mlx":    _FakeSource(spec=_llama_spec(runtime="mlx",    quant_label="4bit-g64"), runtime="mlx"),
        "ollama": _FakeSource(spec=_llama_spec(runtime="ollama", quant_label="Q4_K_M"),   runtime="ollama"),
    }
    app.dependency_overrides[get_source_registry] = lambda: _fake_source_factory(default_sources)
    yield TestClient(app), app, default_sources


# --------------------------------------------------------------------------- #
# create_app: exists, routes registered, static skipped when no web/ dir
# --------------------------------------------------------------------------- #
def test_create_app_registers_expected_routes():
    app = create_app()
    paths = {r.path for r in app.routes}
    for expected in ("/api/system", "/api/models", "/api/check", "/api/compare"):
        assert expected in paths, f"missing route {expected!r}"


# --------------------------------------------------------------------------- #
# GET /api/system
# --------------------------------------------------------------------------- #
def test_api_system_returns_shape(client):
    tc, _, _ = client
    r = tc.get("/api/system")
    assert r.status_code == 200
    d = r.json()
    assert d["chip_id"] == "Apple M1"
    assert d["accelerator"] == "apple_metal"
    assert d["usable_basis"] == "Metal working set"
    assert d["usable_memory_bytes"] > 0
    assert d["hard_usable_memory_bytes"] is not None  # Apple has two ceilings


# --------------------------------------------------------------------------- #
# GET /api/models
# --------------------------------------------------------------------------- #
def test_api_models_returns_alias_list(client):
    tc, _, _ = client
    r = tc.get("/api/models")
    assert r.status_code == 200
    d = r.json()
    assert "models" in d
    ids = [m["id"] for m in d["models"]]
    assert "llama-3.1-8b-instruct" in ids


# --------------------------------------------------------------------------- #
# POST /api/check
# --------------------------------------------------------------------------- #
def test_api_check_returns_spec_fit_speed_memory_curve(client):
    tc, _, sources = client
    r = tc.post("/api/check", json={"model": "llama-3.1-8b-instruct", "runtime": "gguf"})
    assert r.status_code == 200
    d = r.json()
    for key in ("spec", "fit", "speed", "memory_curve"):
        assert key in d
    # Alias got resolved to the source repo_id before hitting the source.
    assert sources["gguf"].calls == [("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Q4_K_M")]
    # Speed points must be JSON-native and non-empty
    assert len(d["speed"]["points"]) > 0
    assert d["speed"]["confidence"] == "estimated"
    # Memory curve has weights baseline + points list
    assert d["memory_curve"]["weight_bytes"] > 0
    assert len(d["memory_curve"]["points"]) >= 2


def test_api_check_passes_kv_quant_through(client):
    tc, _, _ = client
    r_f16 = tc.post("/api/check", json={"model": "llama-3.1-8b-instruct", "runtime": "gguf", "kv_quant": "f16"})
    r_q4 = tc.post("/api/check", json={"model": "llama-3.1-8b-instruct", "runtime": "gguf", "kv_quant": "q4"})
    assert r_f16.status_code == 200 and r_q4.status_code == 200
    # q4 KV cache is 1/4 of f16 -> memory_curve totals should be smaller at
    # the same ctx.
    f16_points = {p["ctx"]: p["total_bytes"] for p in r_f16.json()["memory_curve"]["points"]}
    q4_points = {p["ctx"]: p["total_bytes"] for p in r_q4.json()["memory_curve"]["points"]}
    for ctx in f16_points:
        if ctx == 0 or ctx not in q4_points:
            continue
        assert q4_points[ctx] <= f16_points[ctx]


def test_api_check_advanced_free_text_still_resolves_alias(client):
    """Free-text input from the Advanced disclosure that happens to match an
    alias id should still resolve — same behaviour as `canirunit check`."""
    tc, _, sources = client
    tc.post("/api/check", json={"model": "llama-3.1-8b-instruct", "runtime": "gguf"})
    assert sources["gguf"].calls[-1][0] == "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"


def test_api_check_non_alias_ref_passes_through(client):
    tc, _, sources = client
    r = tc.post("/api/check", json={"model": "meta-llama/Custom-Weights", "runtime": "gguf", "quant": "Q5_K_M"})
    assert r.status_code == 200
    assert sources["gguf"].calls[-1] == ("meta-llama/Custom-Weights", "Q5_K_M")


def test_api_check_source_raises_returns_404(client):
    tc, app, _ = client
    src = _FakeSource(raises=FileNotFoundError("model not pulled"), runtime="ollama")
    app.dependency_overrides[get_source_registry] = lambda: _fake_source_factory({"ollama": src})
    r = tc.post("/api/check", json={"model": "llama3.1:8b", "runtime": "ollama"})
    assert r.status_code == 404
    assert "model not pulled" in r.json()["detail"]


def test_api_check_network_error_returns_502(client):
    tc, app, _ = client
    src = _FakeSource(raises=OSError("connection reset"), runtime="gguf")
    app.dependency_overrides[get_source_registry] = lambda: _fake_source_factory({"gguf": src})
    r = tc.post("/api/check", json={"model": "some/repo", "runtime": "gguf"})
    assert r.status_code == 502
    assert "upstream" in r.json()["detail"].lower()


def test_api_check_defaults_runtime_to_gguf(client):
    tc, _, sources = client
    r = tc.post("/api/check", json={"model": "llama-3.1-8b-instruct"})
    assert r.status_code == 200
    assert len(sources["gguf"].calls) == 1


def test_api_check_confidence_is_estimated_without_calibration(client):
    """No calibration cache in commit 2 — every /api/check must still be marked
    'estimated'. The calibration commit changes this behaviour additively."""
    tc, _, _ = client
    d = tc.post("/api/check", json={"model": "llama-3.1-8b-instruct", "runtime": "gguf"}).json()
    assert d["speed"]["confidence"] == "estimated"
    assert d["calibration_applied"] is None


# --------------------------------------------------------------------------- #
# POST /api/compare
# --------------------------------------------------------------------------- #
def test_api_compare_returns_rows_for_each_runtime(client):
    tc, _, _ = client
    r = tc.post("/api/compare", json={"logical_id": "llama-3.1-8b-instruct"})
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert {row["runtime"] for row in rows} == {"gguf", "mlx", "ollama"}


def test_api_compare_isolates_per_runtime_errors(client):
    """One runtime failing must not 500 the whole endpoint."""
    tc, app, _ = client
    sources = {
        "gguf":   _FakeSource(spec=_llama_spec(),                                          runtime="gguf"),
        "mlx":    _FakeSource(raises=ValueError("repo not found"),                          runtime="mlx"),
        "ollama": _FakeSource(spec=_llama_spec(runtime="ollama", quant_label="Q4_K_M"),     runtime="ollama"),
    }
    app.dependency_overrides[get_source_registry] = lambda: _fake_source_factory(sources)
    r = tc.post("/api/compare", json={"logical_id": "llama-3.1-8b-instruct"})
    assert r.status_code == 200
    rows = {row["runtime"]: row for row in r.json()["rows"]}
    assert rows["mlx"]["error"] == "repo not found"
    assert rows["mlx"]["fit"] is None
    assert rows["gguf"]["error"] is None
    assert rows["ollama"]["error"] is None


def test_api_compare_unknown_logical_id_404s(client):
    tc, _, _ = client
    r = tc.post("/api/compare", json={"logical_id": "does-not-exist"})
    assert r.status_code == 404

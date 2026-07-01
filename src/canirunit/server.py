"""FastAPI backend for `canirunit serve`.

Design rules (enforced by the spec):
- The server is a thin renderer over the existing Python core. It contains
  ZERO estimation logic. Endpoints return the dict shapes from ``serialize.py``.
- Every hardware/network call is behind a FastAPI ``Depends`` so tests can
  override with fakes and never touch a real machine or Hugging Face.
- Server binds ``127.0.0.1`` only (owned by the CLI ``serve`` command).

Import guard: importing this module requires FastAPI. The ``serve`` CLI
subcommand catches the ImportError and prints a clean install hint, so
``canirunit`` remains usable without the ``[ui]`` extra.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import requests

from . import aliases
from .benchmark import calibrate as _calibrate
from .cli import _resolve_model_arg
from .compare import compare as _compare
from .config import EstimatorConfig
from .detector import detect as _detect
from .estimator import estimate_fit, estimate_speed
from .serialize import (
    calibration_to_dict,
    comparison_to_dict,
    fit_to_dict,
    memory_curve,
    speed_to_dict,
    spec_to_dict,
    system_to_dict,
)
from .sources import get_source
from .types import Calibration, ModelSpec, Runtime, SystemProfile


# --------------------------------------------------------------------------- #
# Dependency injection seams — tests override these via app.dependency_overrides
# --------------------------------------------------------------------------- #
_cached_profile: Optional[SystemProfile] = None


def get_profile() -> SystemProfile:
    """One-shot detection cached for the process lifetime. Tests override."""
    global _cached_profile
    if _cached_profile is None:
        _cached_profile = _detect()
    return _cached_profile


def get_config() -> EstimatorConfig:
    return EstimatorConfig()


def get_source_registry() -> Callable[[Runtime], object]:
    """Returns the runtime->SpecSource lookup. Tests override to inject fakes."""
    return get_source


def get_alias_resolver() -> Callable[[str], dict]:
    return aliases.resolve


def get_alias_lister() -> Callable[[], list[dict]]:
    return aliases.list_models


def get_refresh_fn() -> Callable[[], dict]:
    return aliases.refresh


def get_compare_fn():
    return _compare


# Server-side calibration cache — populated by completed calibrate jobs.
# Keyed by runtime; consumed by /api/check so the client never has to round-
# trip a Calibration object. Process-local; wiped on server restart (the spec
# explicitly rules out cross-invocation persistence).
_last_calibration_by_runtime: dict[Runtime, Calibration] = {}


def get_calibration_cache() -> dict[Runtime, Calibration]:
    return _last_calibration_by_runtime


def get_calibrate_fn():
    return _calibrate


# Job registry. Same DI treatment as the cache so tests get a fresh dict per
# app, and there's no cross-test bleed via module-level state.
_jobs: dict[str, dict] = {}


def get_jobs() -> dict[str, dict]:
    return _jobs


def _default_executor(fn: Callable, *args) -> None:
    """Spawn ``fn(*args)`` in a daemon thread. The daemon flag matters — we
    don't want a hung calibration to keep uvicorn from shutting down."""
    threading.Thread(target=fn, args=args, daemon=True).start()


def get_job_executor():
    return _default_executor


# --------------------------------------------------------------------------- #
# Request bodies
# --------------------------------------------------------------------------- #
class CheckBody(BaseModel):
    model: str
    runtime: Runtime = "gguf"
    quant: Optional[str] = None
    kv_quant: str = "f16"
    ctx_points: Optional[list[int]] = None


class CompareBody(BaseModel):
    logical_id: str


class CalibrateBody(BaseModel):
    runtime: Runtime


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app() -> FastAPI:
    """Build the FastAPI app. Kept as a factory so tests can construct fresh
    instances with their own dependency overrides."""
    app = FastAPI(
        title="canirunit",
        version="0.2.0",
        docs_url=None,  # Local tool — no need for interactive Swagger.
        redoc_url=None,
    )

    _register_api(app)
    _mount_static_if_present(app)
    return app


# --------------------------------------------------------------------------- #
# Static frontend mount (optional — files land in commits 5+)
# --------------------------------------------------------------------------- #
def _mount_static_if_present(app: FastAPI) -> None:
    web_dir = Path(__file__).parent / "web"
    if web_dir.is_dir() and (web_dir / "index.html").exists():
        # html=True makes StaticFiles serve index.html for "/" automatically.
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")


# --------------------------------------------------------------------------- #
# API routes
# --------------------------------------------------------------------------- #
def _register_api(app: FastAPI) -> None:
    @app.get("/api/system")
    def api_system(
        profile: SystemProfile = Depends(get_profile),
        config: EstimatorConfig = Depends(get_config),
    ):
        return system_to_dict(profile, config)

    @app.get("/api/models")
    def api_models(lister=Depends(get_alias_lister)):
        return {"models": lister()}

    @app.post("/api/check")
    def api_check(
        body: CheckBody,
        profile: SystemProfile = Depends(get_profile),
        config: EstimatorConfig = Depends(get_config),
        source_for=Depends(get_source_registry),
        resolver=Depends(get_alias_resolver),
        cal_cache: dict = Depends(get_calibration_cache),
    ):
        # Lookup happens on the spec's runtime AFTER fetch, so it lives in
        # _do_check.
        return _do_check(body, profile, config, source_for, resolver, cal_cache=cal_cache)

    @app.post("/api/compare")
    def api_compare(
        body: CompareBody,
        profile: SystemProfile = Depends(get_profile),
        config: EstimatorConfig = Depends(get_config),
        source_for=Depends(get_source_registry),
        resolver=Depends(get_alias_resolver),
        compare_fn=Depends(get_compare_fn),
    ):
        try:
            rows = compare_fn(
                body.logical_id, profile, config=config,
                source_for=source_for, resolve_fn=resolver,
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"rows": [comparison_to_dict(r) for r in rows]}

    @app.post("/api/calibrate")
    def api_calibrate(
        body: CalibrateBody,
        profile: SystemProfile = Depends(get_profile),
        config: EstimatorConfig = Depends(get_config),
        cal_cache: dict = Depends(get_calibration_cache),
        calibrate_fn=Depends(get_calibrate_fn),
        jobs: dict = Depends(get_jobs),
        executor: Callable = Depends(get_job_executor),
    ):
        job_id = uuid4().hex
        jobs[job_id] = {
            "job_id": job_id,
            "runtime": body.runtime,
            "status": "pending",
            "progress": None,
            "result": None,
            "error": None,
        }
        executor(
            _run_calibration_job,
            jobs, job_id, body.runtime, profile, config, calibrate_fn, cal_cache,
        )
        return {"job_id": job_id, "status": "pending"}

    @app.get("/api/calibrate/{job_id}")
    def api_calibrate_poll(job_id: str, jobs: dict = Depends(get_jobs)):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        return job

    @app.post("/api/refresh")
    def api_refresh(refresh_fn=Depends(get_refresh_fn)):
        """Pull the canonical alias table into the local overlay. Delegates
        to ``aliases.refresh`` which handles the network+atomic-write; failures
        (network/JSON/schema) surface as a 502."""
        result = refresh_fn()
        if not result.get("ok"):
            raise HTTPException(status_code=502, detail=result.get("error", "refresh failed"))
        return {
            "models": result["models"],
            "updated_at": result.get("updated_at"),
            "path": result.get("path"),
        }


# --------------------------------------------------------------------------- #
# Calibration job runner (runs off the request thread)
# --------------------------------------------------------------------------- #
def _run_calibration_job(
    jobs: dict,
    job_id: str,
    runtime: Runtime,
    profile: SystemProfile,
    config: EstimatorConfig,
    calibrate_fn: Callable[..., Optional[Calibration]],
    cal_cache: dict,
) -> None:
    """Runs the (blocking) calibration and updates the job record.

    ``progress`` is a coarse string ('running benchmark' -> 'done') rather
    than a percentage. The spec explicitly allows this in v2 — threading a
    fine-grained callback into ``benchmark.calibrate`` would require
    invasive changes for little UI payoff.
    """
    job = jobs.get(job_id)
    if job is None:
        return
    job["status"] = "running"
    job["progress"] = "downloading bench model / running benchmark"

    try:
        cal = calibrate_fn(profile, target_runtime=runtime, config=config)
    except Exception as e:  # noqa: BLE001 — surface any failure as a job error
        job["status"] = "error"
        job["progress"] = None
        job["error"] = str(e) or e.__class__.__name__
        return

    if cal is None:
        job["status"] = "error"
        job["progress"] = None
        job["error"] = (
            f"no calibration tooling found for runtime {runtime!r} "
            "(install llama-bench for gguf/ollama, or mlx-lm for mlx)"
        )
        return

    _install_calibration(cal_cache, cal)
    job["status"] = "done"
    job["progress"] = "done"
    job["result"] = calibration_to_dict(cal)


def _install_calibration(cache: dict, cal: Calibration) -> None:
    """Store the calibration under every runtime tag it would apply to.

    Mirrors the estimator's cross-runtime rule: gguf and ollama share
    physics so a gguf calibration is also valid for an ollama target. By
    duplicating the entry at store-time, the /api/check read path stays a
    single ``cache.get(spec.runtime)`` — no compatibility branch there.
    """
    cache[cal.runtime] = cal
    if cal.runtime == "gguf":
        cache["ollama"] = cal
    elif cal.runtime == "ollama":
        cache["gguf"] = cal


# --------------------------------------------------------------------------- #
# /api/check core (also reused by the calibration cache path in a later commit)
# --------------------------------------------------------------------------- #
def _do_check(
    body: CheckBody,
    profile: SystemProfile,
    config: EstimatorConfig,
    source_for: Callable[[Runtime], object],
    resolver: Callable[[str], dict],
    cal_cache: Optional[dict] = None,
) -> dict:
    """Runs the same detect -> fetch -> estimate -> serialize sequence as
    ``cli.run_check`` but returns dicts. Does NOT download full models — only
    range-reads GGUF headers or fetches config.json / listings. The calibration
    argument stays None here; the async-calibration commit wires in a cached
    per-runtime calibration."""
    try:
        ref, quant = _resolve_model_arg(
            body.model, body.runtime, body.quant or "Q4_K_M", resolver,
        )
        src = source_for(body.runtime)
        spec: ModelSpec = src.fetch(ref, quant)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=f"model not found: {e}")
    except FileNotFoundError as e:
        # Ollama model not pulled locally.
        raise HTTPException(status_code=404, detail=str(e))
    except (OSError, requests.exceptions.RequestException) as e:
        raise HTTPException(status_code=502, detail=f"upstream fetch failed: {e}")

    # Server-side calibration cache: apply the most recent calibration for
    # this spec's runtime, if any. The estimator's runtime guard makes this
    # safe — a wrong-runtime entry is silently dropped and speed.confidence
    # stays "estimated".
    calibration: Optional[Calibration] = None
    if cal_cache is not None:
        calibration = cal_cache.get(spec.runtime)

    fit = estimate_fit(profile, spec, config)
    speed = estimate_speed(
        profile, spec, config,
        calibration=calibration,
        kv_quant=body.kv_quant,
        ctx_points=body.ctx_points,
    )
    curve = memory_curve(profile, spec, config, kv_quant=body.kv_quant)

    return {
        "spec": spec_to_dict(spec),
        "fit": fit_to_dict(fit),
        "speed": speed_to_dict(speed),
        "memory_curve": curve,
        "calibration_applied": calibration_to_dict(calibration),
    }

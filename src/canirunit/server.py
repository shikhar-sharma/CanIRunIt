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

from pathlib import Path
from typing import Callable, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import requests

from . import aliases
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
from .types import ModelSpec, Runtime, SystemProfile


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


def get_compare_fn():
    return _compare


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
    ):
        return _do_check(body, profile, config, source_for, resolver, calibration=None)

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


# --------------------------------------------------------------------------- #
# /api/check core (also reused by the calibration cache path in a later commit)
# --------------------------------------------------------------------------- #
def _do_check(
    body: CheckBody,
    profile: SystemProfile,
    config: EstimatorConfig,
    source_for: Callable[[Runtime], object],
    resolver: Callable[[str], dict],
    calibration=None,
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

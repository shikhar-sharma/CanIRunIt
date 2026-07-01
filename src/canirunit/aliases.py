"""Alias table: maps a logical model id to per-runtime sources.

Two layers:
  * Shipped: ``src/canirunit/data/aliases.json`` — bundled with the package,
    always present. This is the seed every install starts with.
  * Overlay: ``{config_dir}/canirunit/aliases.json`` — written by ``refresh()``
    when a user pulls the latest published table. Overlay entries override
    shipped entries per ``logical_id``.

Resolution is a lookup, never fuzzy. Fuzzy matching belongs in the
maintainer-side builder (``scripts/build_aliases.py``).
"""
from __future__ import annotations

import json
import os
from importlib import resources
from typing import Callable, Optional

SCHEMA_VERSION = 1

# Canonical URL the refresh subcommand pulls from. Kept in sync with the
# repo's root-level data/aliases.json by the maintainer's release process.
CANONICAL_URL = (
    "https://raw.githubusercontent.com/shikhar-sharma/CanIRunIt/main/data/aliases.json"
)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _shipped_aliases() -> dict:
    """Read the package-bundled aliases.json. Always succeeds for a normal install."""
    with resources.files("canirunit.data").joinpath("aliases.json").open(
        "r", encoding="utf-8"
    ) as f:
        return json.load(f)


def _default_overlay_path() -> str:
    config_home = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(config_home, "canirunit", "aliases.json")


def _read_overlay(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # A corrupt overlay must not poison the shipped view; treat as absent.
        return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_aliases(
    shipped: Optional[dict] = None,
    overlay_path: Optional[str] = None,
) -> dict:
    """Return the merged alias table (overlay over shipped, per logical_id).

    ``shipped`` and ``overlay_path`` are injectable so tests can swap both.
    CLI callers use the defaults: package-bundled shipped + XDG overlay.
    """
    base = shipped if shipped is not None else _shipped_aliases()
    overlay = _read_overlay(overlay_path or _default_overlay_path())

    if not overlay or not isinstance(overlay.get("models"), dict):
        return base

    merged_models = dict(base.get("models", {}))
    merged_models.update(overlay["models"])
    return {
        "schema_version": base.get("schema_version", SCHEMA_VERSION),
        "updated_at": overlay.get("updated_at", base.get("updated_at")),
        "models": merged_models,
    }


def list_models(
    shipped: Optional[dict] = None,
    overlay_path: Optional[str] = None,
) -> list[dict]:
    """Flat listing of known model ids for the `models` CLI command."""
    data = load_aliases(shipped=shipped, overlay_path=overlay_path)
    out = []
    for model_id, entry in sorted(data.get("models", {}).items()):
        out.append(
            {
                "id": model_id,
                "display_name": entry.get("display_name", model_id),
                "family": entry.get("family", ""),
                "runtimes": sorted(entry.get("sources", {}).keys()),
            }
        )
    return out


def resolve(
    logical_id: str,
    shipped: Optional[dict] = None,
    overlay_path: Optional[str] = None,
) -> dict:
    """Return the entry for ``logical_id``. Raises KeyError if unknown."""
    data = load_aliases(shipped=shipped, overlay_path=overlay_path)
    models = data.get("models", {})
    if logical_id not in models:
        raise KeyError(
            f"unknown model id {logical_id!r}; try `canirunit models` to list "
            "known ids, or `canirunit refresh` to pull the latest alias table"
        )
    return models[logical_id]


def _default_http_get(url: str) -> bytes:
    import requests

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content


def refresh(
    url: str = CANONICAL_URL,
    http_get: Optional[Callable[[str], bytes]] = None,
    overlay_path: Optional[str] = None,
) -> dict:
    """Pull the canonical alias table and write it to the overlay path.

    Returns a status dict with ``ok: True/False``. Failures (network, bad
    JSON, schema mismatch) leave any existing overlay intact (atomic
    temp+rename). ``http_get`` and ``overlay_path`` are injectable for tests.
    """
    fetch = http_get or _default_http_get
    target_path = overlay_path or _default_overlay_path()

    try:
        body = fetch(url)
    except Exception as e:  # noqa: BLE001 — network errors come in many shapes
        return {"ok": False, "error": f"network error: {e}"}

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        return {"ok": False, "error": f"bad JSON from {url}: {e}"}

    schema = data.get("schema_version")
    if schema != SCHEMA_VERSION:
        return {
            "ok": False,
            "error": (
                f"schema_version {schema!r} does not match supported "
                f"{SCHEMA_VERSION}; upgrade canirunit or fix the published table"
            ),
        }

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    tmp_path = target_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, target_path)

    return {
        "ok": True,
        "models": len(data.get("models", {})),
        "updated_at": data.get("updated_at"),
        "path": target_path,
    }

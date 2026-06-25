"""Alias table tests.

Inject both ``shipped`` (dict) and ``overlay_path`` (tmp file) so the suite
exercises the merging rules without touching the package-bundled JSON or the
user's real ~/.config.
"""
from __future__ import annotations

import json
import os

import pytest

from canirunit import aliases as A


SHIPPED = {
    "schema_version": 1,
    "updated_at": "2026-01-01T00:00:00Z",
    "models": {
        "alpha": {
            "display_name": "Alpha",
            "family": "test",
            "sources": {
                "gguf": {"repo_id": "x/alpha-gguf", "default_quant": "Q4_K_M"},
                "ollama": {"tag": "alpha:1"},
            },
        },
        "beta": {
            "display_name": "Beta",
            "family": "test",
            "sources": {"gguf": {"repo_id": "x/beta-gguf", "default_quant": "Q4_K_M"}},
        },
    },
}


# --------------------------------------------------------------------------- #
# Shipped table sanity (the file actually ships and parses)
# --------------------------------------------------------------------------- #
def test_shipped_table_parses():
    data = A._shipped_aliases()
    assert data["schema_version"] == A.SCHEMA_VERSION
    assert "models" in data and len(data["models"]) > 0


def test_shipped_models_have_at_least_one_runtime():
    for model_id, entry in A._shipped_aliases()["models"].items():
        srcs = entry.get("sources", {})
        assert srcs, f"{model_id!r} has no sources"
        assert set(srcs).issubset({"gguf", "mlx", "ollama"}), \
            f"{model_id!r} declares unknown runtime keys: {set(srcs)}"


# --------------------------------------------------------------------------- #
# load_aliases: overlay overrides shipped per id
# --------------------------------------------------------------------------- #
def test_load_aliases_no_overlay_returns_shipped(tmp_path):
    overlay = tmp_path / "no_such_overlay.json"
    data = A.load_aliases(shipped=SHIPPED, overlay_path=str(overlay))
    assert data["models"].keys() == SHIPPED["models"].keys()


def test_load_aliases_overlay_overrides_per_id(tmp_path):
    overlay = tmp_path / "aliases.json"
    overlay.write_text(json.dumps({
        "schema_version": 1,
        "updated_at": "2026-06-01T00:00:00Z",
        "models": {
            "alpha": {  # overrides shipped 'alpha'
                "display_name": "Alpha-overridden",
                "family": "test",
                "sources": {"gguf": {"repo_id": "y/alpha-new", "default_quant": "Q6_K"}},
            },
            "gamma": {  # net new
                "display_name": "Gamma",
                "family": "test",
                "sources": {"mlx": {"repo_id": "y/gamma-mlx"}},
            },
        },
    }))
    data = A.load_aliases(shipped=SHIPPED, overlay_path=str(overlay))
    # alpha replaced (not deep-merged) — overlay wins entirely
    assert data["models"]["alpha"]["display_name"] == "Alpha-overridden"
    assert "ollama" not in data["models"]["alpha"]["sources"]
    # beta untouched (not in overlay)
    assert data["models"]["beta"]["display_name"] == "Beta"
    # gamma added
    assert data["models"]["gamma"]["display_name"] == "Gamma"
    # updated_at follows the overlay
    assert data["updated_at"] == "2026-06-01T00:00:00Z"


def test_load_aliases_ignores_corrupt_overlay(tmp_path):
    overlay = tmp_path / "aliases.json"
    overlay.write_text("{ not valid json")
    data = A.load_aliases(shipped=SHIPPED, overlay_path=str(overlay))
    # Falls back to shipped rather than crashing.
    assert data["models"].keys() == SHIPPED["models"].keys()


# --------------------------------------------------------------------------- #
# resolve / list_models
# --------------------------------------------------------------------------- #
def test_resolve_known_id(tmp_path):
    entry = A.resolve("alpha", shipped=SHIPPED, overlay_path=str(tmp_path / "nope.json"))
    assert entry["display_name"] == "Alpha"


def test_resolve_unknown_id_message_mentions_refresh(tmp_path):
    with pytest.raises(KeyError, match="refresh"):
        A.resolve("does-not-exist", shipped=SHIPPED,
                  overlay_path=str(tmp_path / "nope.json"))


def test_list_models_is_sorted_and_includes_runtimes(tmp_path):
    rows = A.list_models(shipped=SHIPPED, overlay_path=str(tmp_path / "nope.json"))
    ids = [r["id"] for r in rows]
    assert ids == sorted(ids)
    alpha = next(r for r in rows if r["id"] == "alpha")
    assert "gguf" in alpha["runtimes"] and "ollama" in alpha["runtimes"]


# --------------------------------------------------------------------------- #
# refresh: injectable http_get; failures must not corrupt the overlay
# --------------------------------------------------------------------------- #
def _ok_payload() -> bytes:
    return json.dumps({
        "schema_version": 1,
        "updated_at": "2026-06-25T00:00:00Z",
        "models": {
            "delta": {"display_name": "Delta", "family": "test",
                      "sources": {"gguf": {"repo_id": "x/delta", "default_quant": "Q4_K_M"}}}
        },
    }).encode("utf-8")


def test_refresh_writes_overlay_and_reports_count(tmp_path):
    overlay = tmp_path / "overlay" / "aliases.json"
    res = A.refresh(
        url="http://example/aliases.json",
        http_get=lambda u: _ok_payload(),
        overlay_path=str(overlay),
    )
    assert res["ok"] is True
    assert res["models"] == 1
    assert os.path.exists(overlay)
    # overlay file is valid JSON and contains the fetched models
    saved = json.loads(overlay.read_text())
    assert "delta" in saved["models"]


def test_refresh_atomic_write_leaves_old_overlay_intact_on_bad_json(tmp_path):
    overlay = tmp_path / "aliases.json"
    overlay.write_text(json.dumps({"schema_version": 1, "models": {"existing": {}}}))
    res = A.refresh(
        url="http://example/aliases.json",
        http_get=lambda u: b"<<not json>>",
        overlay_path=str(overlay),
    )
    assert res["ok"] is False
    assert "bad JSON" in res["error"]
    # Old overlay still readable and unchanged.
    saved = json.loads(overlay.read_text())
    assert saved == {"schema_version": 1, "models": {"existing": {}}}


def test_refresh_rejects_schema_mismatch(tmp_path):
    overlay = tmp_path / "aliases.json"
    bad = json.dumps({"schema_version": 999, "models": {}}).encode()
    res = A.refresh(
        url="http://example/aliases.json",
        http_get=lambda u: bad,
        overlay_path=str(overlay),
    )
    assert res["ok"] is False
    assert "schema_version" in res["error"]
    assert not os.path.exists(overlay)


def test_refresh_network_error_returns_clean_status(tmp_path):
    def boom(url):
        raise ConnectionError("dns")

    overlay = tmp_path / "aliases.json"
    res = A.refresh(
        url="http://example/aliases.json",
        http_get=boom,
        overlay_path=str(overlay),
    )
    assert res["ok"] is False
    assert "network error" in res["error"]
    assert not os.path.exists(overlay)

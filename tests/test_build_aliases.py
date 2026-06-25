"""Lightweight unit tests for the maintainer-side alias builder.

The HF discovery path is out of CI scope (network); only the pure grouping
and key-normalization logic is exercised here. That's the bit that determines
whether GGUF and MLX repos for the same underlying model end up paired.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make scripts/ importable without packaging it.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import build_aliases as B  # noqa: E402


# --------------------------------------------------------------------------- #
# normalize_model_key
# --------------------------------------------------------------------------- #
def test_normalize_strips_author():
    assert B.normalize_model_key("bartowski/Meta-Llama-3.1-8B-Instruct") == "meta-llama-3.1-8b-instruct"


def test_normalize_strips_runtime_suffixes():
    a = B.normalize_model_key("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
    b = B.normalize_model_key("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    assert a == b == "meta-llama-3.1-8b-instruct"


def test_normalize_collapses_separators():
    assert B.normalize_model_key("org/Foo  Bar__baz") == "foo-bar-baz"


def test_normalize_handles_repo_id_without_author():
    assert B.normalize_model_key("Qwen2.5-0.5B-Instruct-GGUF") == "qwen2.5-0.5b-instruct"


# --------------------------------------------------------------------------- #
# group_repos: GGUF and MLX repos of the same model end up paired
# --------------------------------------------------------------------------- #
def test_group_pairs_gguf_and_mlx_for_same_model():
    repos = [
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "bartowski/Qwen2.5-7B-Instruct-GGUF",
    ]
    g = B.group_repos(repos)
    llama = g["meta-llama-3.1-8b-instruct"]
    assert "gguf" in llama and "mlx" in llama
    assert "gguf" in g["qwen2.5-7b-instruct"]
    assert "mlx" not in g["qwen2.5-7b-instruct"]   # no MLX entry given


# --------------------------------------------------------------------------- #
# build_alias_entries: composes the published shape, seeds ollama from the map
# --------------------------------------------------------------------------- #
def test_build_entries_picks_first_alphabetical():
    grouped = {
        "meta-llama-3.1-8b-instruct": {
            "gguf": ["bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                     "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"],
            "mlx":  ["mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"],
        }
    }
    entries = B.build_alias_entries(grouped, {"meta-llama-3.1-8b-instruct": "llama3.1:8b"})
    sources = entries["meta-llama-3.1-8b-instruct"]["sources"]
    # alphabetically first wins (bartowski < lmstudio-community)
    assert sources["gguf"]["repo_id"] == "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    assert sources["gguf"]["default_quant"] == "Q4_K_M"
    assert sources["mlx"]["repo_id"] == "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
    assert sources["ollama"]["tag"] == "llama3.1:8b"


def test_build_entries_skips_groups_without_any_runtime_source():
    """A key with no gguf/mlx repos and no ollama tag should produce no entry."""
    grouped = {"orphan-model": {}}
    assert B.build_alias_entries(grouped, {}) == {}

"""OllamaSource tests.

Build a synthetic Ollama models tree (manifest + GGUF blob) on a tmp dir,
point the source at it via the injected ``models_root``, and assert on the
resulting ModelSpec. No real ~/.ollama is ever touched.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from ggufkit import STRING, U32, U64, build_gguf
from canirunit.source_ollama import (
    GGUF_FILE_TYPE_LABELS,
    OllamaSource,
    _parse_ref,
)


# --------------------------------------------------------------------------- #
# Pure: ref parsing
# --------------------------------------------------------------------------- #
def test_parse_ref_library_default():
    assert _parse_ref("llama3.1:8b") == ("registry.ollama.ai", "library", "llama3.1", "8b")


def test_parse_ref_defaults_tag_to_latest():
    assert _parse_ref("llama3.1") == ("registry.ollama.ai", "library", "llama3.1", "latest")


def test_parse_ref_custom_registry():
    assert _parse_ref("hf.co/user/repo:tag") == ("hf.co", "user", "repo", "tag")


def test_parse_ref_user_namespace_default_registry():
    assert _parse_ref("user/model:v1") == ("registry.ollama.ai", "user", "model", "v1")


# --------------------------------------------------------------------------- #
# End-to-end: synthetic Ollama tree -> ModelSpec
# --------------------------------------------------------------------------- #
def _write_ollama_tree(tmp_path, ref: str, blob_bytes: bytes) -> str:
    """Materialise a minimal Ollama models tree and return the root path."""
    root = tmp_path / "ollama_root"
    digest_hex = hashlib.sha256(blob_bytes).hexdigest()
    digest = f"sha256:{digest_hex}"
    blob_path = root / "blobs" / f"sha256-{digest_hex}"
    blob_path.parent.mkdir(parents=True)
    blob_path.write_bytes(blob_bytes)

    # Parse ref the way the source does so the manifest path matches.
    registry, namespace, name, tag = _parse_ref(ref)
    manifest_path = root / "manifests" / registry / namespace / name / tag
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                "config": {"mediaType": "application/vnd.ollama.image.config", "digest": "sha256:cafebabe"},
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": digest,
                        "size": len(blob_bytes),
                    },
                    {"mediaType": "application/vnd.ollama.image.params", "digest": "sha256:deadbeef"},
                ],
            }
        )
    )
    return str(root)


def _llama_gguf(file_type: int = 15) -> bytes:
    """A small but format-valid GGUF blob mirroring a llama-family model.
    file_type=15 -> Q4_K_M (see GGUF_FILE_TYPE_LABELS)."""
    return build_gguf([
        ("general.architecture", STRING, "llama"),
        ("general.parameter_count", U64, 8_030_000_000),
        ("general.file_type", U32, file_type),
        ("llama.block_count", U32, 32),
        ("llama.context_length", U32, 8192),
        ("llama.embedding_length", U32, 4096),
        ("llama.attention.head_count", U32, 32),
        ("llama.attention.head_count_kv", U32, 8),
        ("llama.attention.key_length", U32, 128),
        ("llama.attention.value_length", U32, 128),
    ])


def test_fetch_resolves_and_tags_runtime(tmp_path):
    blob = _llama_gguf()
    root = _write_ollama_tree(tmp_path, "llama3.1:8b", blob)

    spec = OllamaSource(models_root=root).fetch("llama3.1:8b")

    assert spec.runtime == "ollama"
    assert spec.architecture == "llama"
    assert spec.n_layers == 32
    assert spec.n_kv_heads == 8
    assert spec.key_length == 128 and spec.value_length == 128
    assert spec.native_ctx == 8192
    assert spec.total_weight_bytes == len(blob)
    assert spec.quant_label == "Q4_K_M"


def test_fetch_default_tag_is_latest(tmp_path):
    blob = _llama_gguf()
    root = _write_ollama_tree(tmp_path, "llama3.1:latest", blob)

    # Caller passes no tag; source should default to 'latest'.
    spec = OllamaSource(models_root=root).fetch("llama3.1")
    assert spec.runtime == "ollama"


def test_quant_label_unknown_for_unmapped_file_type(tmp_path):
    blob = _llama_gguf(file_type=999)  # not in the labels map
    root = _write_ollama_tree(tmp_path, "llama3.1:8b", blob)
    spec = OllamaSource(models_root=root).fetch("llama3.1:8b")
    assert spec.quant_label == "unknown"


def test_quant_label_map_has_common_quants():
    # Smoke test the static table: the labels the comparison renderer will print.
    assert GGUF_FILE_TYPE_LABELS[15] == "Q4_K_M"
    assert GGUF_FILE_TYPE_LABELS[17] == "Q5_K_M"
    assert GGUF_FILE_TYPE_LABELS[18] == "Q6_K"
    assert GGUF_FILE_TYPE_LABELS[1] == "F16"


def test_missing_manifest_raises_clean_error(tmp_path):
    root = tmp_path / "empty_ollama_root"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="ollama pull llama3.1:8b"):
        OllamaSource(models_root=str(root)).fetch("llama3.1:8b")


def test_missing_blob_raises_clean_error(tmp_path):
    blob = _llama_gguf()
    root = _write_ollama_tree(tmp_path, "llama3.1:8b", blob)
    # Delete the blob but leave the manifest pointing at it.
    blob_dir = os.path.join(root, "blobs")
    for f in os.listdir(blob_dir):
        os.remove(os.path.join(blob_dir, f))

    with pytest.raises(FileNotFoundError, match="blob missing"):
        OllamaSource(models_root=str(root)).fetch("llama3.1:8b")


def test_manifest_without_model_layer_raises(tmp_path):
    root = tmp_path / "ollama_root"
    manifest_path = root / "manifests" / "registry.ollama.ai" / "library" / "llama3.1" / "8b"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"schemaVersion": 2, "layers": [
            {"mediaType": "application/vnd.ollama.image.params", "digest": "sha256:deadbeef"}
        ]})
    )
    with pytest.raises(FileNotFoundError, match="no 'application/vnd.ollama.image.model' layer"):
        OllamaSource(models_root=str(root)).fetch("llama3.1:8b")

"""Ollama is a packaging skin over GGUF.

Fit and speed math are identical to GGUF — we reuse `parse_gguf` +
`build_model_spec` on Ollama's local blob. Only resolution differs: instead of
listing files on Hugging Face, we read Ollama's local manifest store to find
the underlying GGUF blob.

Storage layout (documented defaults; ``models_root`` overridable for tests):

    {root}/manifests/{registry}/{namespace}/{name}/{tag}     # JSON manifest
    {root}/blobs/sha256-{hex}                                # the GGUF blob

A ref like ``llama3.1:8b`` resolves to namespace=``library``,
name=``llama3.1``, tag=``8b`` on the default ``registry.ollama.ai`` registry.
The ``library`` namespace is the well-trodden path and is fully covered by
tests; refs from other registries (``hf.co/...`` etc.) are handled
best-effort with a comment flagging the limited coverage.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Optional

from .gguf import FileReader, parse_gguf
from .hub import build_model_spec
from .types import ModelSpec

# `general.file_type` enum -> human label. Mirrors llama.cpp's LLAMA_FTYPE_*.
# Not exhaustive — covers the common quants Ollama ships; unknown values fall
# back to "unknown".
GGUF_FILE_TYPE_LABELS: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
}

_DEFAULT_REGISTRY = "registry.ollama.ai"
_DEFAULT_NAMESPACE = "library"
_MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"


def _default_ollama_root() -> str:
    return os.environ.get("OLLAMA_MODELS") or os.path.expanduser("~/.ollama/models")


def _parse_ref(ref: str) -> tuple[str, str, str, str]:
    """Parse an Ollama model reference into (registry, namespace, name, tag).

    Conventions:
      ``llama3.1:8b``         -> (registry.ollama.ai, library, llama3.1, 8b)
      ``llama3.1``            -> (..., library, llama3.1, latest)
      ``hf.co/user/repo:tag`` -> (hf.co, user, repo, tag)
      ``user/model:tag``      -> (registry.ollama.ai, user, model, tag)
    Only the ``library`` case is fully tested in v2.
    """
    name_part, _, tag = ref.partition(":")
    if not tag:
        tag = "latest"

    # Heuristic: a leading segment with a dot is treated as a registry hostname
    # (e.g. hf.co, ghcr.io). Otherwise the first slash separates namespace/name.
    segments = name_part.split("/")
    if len(segments) >= 3 and "." in segments[0]:
        registry = segments[0]
        namespace = segments[1]
        name = "/".join(segments[2:])
    elif len(segments) == 2:
        registry = _DEFAULT_REGISTRY
        namespace, name = segments
    else:
        registry = _DEFAULT_REGISTRY
        namespace = _DEFAULT_NAMESPACE
        name = name_part
    return registry, namespace, name, tag


def _manifest_path(root: str, ref: str) -> str:
    registry, namespace, name, tag = _parse_ref(ref)
    return os.path.join(root, "manifests", registry, namespace, name, tag)


def _model_digest(manifest: dict) -> str:
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == _MODEL_MEDIA_TYPE:
            digest = layer.get("digest")
            if digest:
                return digest
    raise FileNotFoundError(
        f"Ollama manifest has no '{_MODEL_MEDIA_TYPE}' layer (corrupt or unsupported)"
    )


def _blob_path(root: str, digest: str) -> str:
    # Digest looks like 'sha256:abcdef...'; the on-disk filename uses '-' not ':'.
    safe = digest.replace(":", "-")
    return os.path.join(root, "blobs", safe)


class OllamaSource:
    """SpecSource for locally-pulled Ollama models.

    ``models_root`` is injectable so tests never touch a real ``~/.ollama``;
    mirrors the detector's runner-injection pattern.
    """

    runtime = "ollama"

    def __init__(self, models_root: Optional[str] = None):
        self._root = models_root or _default_ollama_root()

    def fetch(self, model_ref: str, quant: Optional[str] = None) -> ModelSpec:
        # The `quant` argument is ignored: which quant ships with an Ollama tag
        # is intrinsic to that tag (e.g. `llama3.1:8b-instruct-q4_K_M`), not
        # chosen by the caller. We surface what we find via quant_label.
        manifest_path = _manifest_path(self._root, model_ref)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Ollama model '{model_ref}' not found locally — "
                f"run `ollama pull {model_ref}` (looked in {self._root})"
            )
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Ollama manifest at {manifest_path} is not valid JSON: {e}") from e

        digest = _model_digest(manifest)
        blob_path = _blob_path(self._root, digest)
        if not os.path.exists(blob_path):
            raise FileNotFoundError(
                f"Ollama model blob missing: manifest references {digest} but "
                f"{blob_path} does not exist (try `ollama pull {model_ref}` again)"
            )

        # Match the GGUF live path: a small read first; reach for tensors only
        # when we actually need them (MoE active-fraction or no parameter_count).
        info = parse_gguf(FileReader(blob_path), need_tensors=False)
        arch = info.metadata.get("general.architecture")
        expert_count = int(info.metadata.get(f"{arch}.expert_count", 0) or 0) if arch else 0
        if expert_count > 1 or "general.parameter_count" not in info.metadata:
            info = parse_gguf(FileReader(blob_path), need_tensors=True)

        file_type = info.metadata.get("general.file_type")
        quant_label = (
            GGUF_FILE_TYPE_LABELS.get(int(file_type), "unknown")
            if file_type is not None
            else "unknown"
        )

        spec = build_model_spec(model_ref, quant_label, info, os.path.getsize(blob_path))
        return replace(spec, runtime="ollama", quant_label=quant_label)

"""Turn a Hugging Face repo + quant into a ModelSpec.

Two layers:
  * Pure mapping/selection (build_model_spec, select_quant_files) — fully tested.
  * Network edge (list_gguf_files, HttpRangeReader, fetch_model) — wired here,
    truly exercised on a machine with network access to huggingface.co.

The size ground-truth is the GGUF file size from the repo listing; the GGUF
header (range-read) supplies architecture params. Card prose is never consulted.
"""
from __future__ import annotations

from typing import Optional

import requests

from .gguf import (
    ByteReader,
    GGUFInfo,
    moe_active_fraction,
    kv_is_standard,
    parse_gguf,
)
from .types import ModelSpec

HF_BASE = "https://huggingface.co"


# --------------------------------------------------------------------------- #
# Pure: metadata -> ModelSpec
# --------------------------------------------------------------------------- #
def _req(meta: dict, arch: str, suffix: str):
    key = f"{arch}.{suffix}"
    if key not in meta:
        raise KeyError(f"GGUF metadata missing required key {key!r}")
    return meta[key]


def build_model_spec(
    repo_id: str, quant: str, gguf: GGUFInfo, total_weight_bytes: int
) -> ModelSpec:
    meta = gguf.metadata
    arch = meta.get("general.architecture")
    if not arch:
        raise KeyError("GGUF metadata missing 'general.architecture'")

    n_layers = int(_req(meta, arch, "block_count"))
    n_heads = int(meta.get(f"{arch}.attention.head_count", 0)) or None
    # MHA fallback: no separate KV head count means KV heads == query heads.
    n_kv_heads = int(meta.get(f"{arch}.attention.head_count_kv", n_heads or 0)) or n_heads
    native_ctx = int(_req(meta, arch, "context_length"))
    embedding = meta.get(f"{arch}.embedding_length")

    # Head dims: read explicitly. Only fall back to embedding/n_heads when the
    # GGUF genuinely omits them — the division is wrong for Gemma, so it is a
    # last resort, never the default.
    key_length = meta.get(f"{arch}.attention.key_length")
    value_length = meta.get(f"{arch}.attention.value_length")
    if key_length is None or value_length is None:
        if not embedding or not n_heads:
            raise KeyError(
                f"cannot determine head_dim for {arch}: no key_length/value_length "
                "and no embedding_length/head_count to fall back on"
            )
        head_dim = int(embedding) // int(n_heads)
        key_length = int(key_length) if key_length is not None else head_dim
        value_length = int(value_length) if value_length is not None else head_dim
    key_length, value_length = int(key_length), int(value_length)

    expert_count = int(meta.get(f"{arch}.expert_count", 0) or 0)
    expert_used = int(meta.get(f"{arch}.expert_used_count", 0) or 0)
    is_moe = expert_count > 1

    total_params = meta.get("general.parameter_count")
    if total_params is None and gguf.tensors is not None:
        total_params = sum(t.n_elements for t in gguf.tensors)
    total_params = int(total_params or 0)

    if is_moe and gguf.tensors is not None:
        frac = moe_active_fraction(gguf.tensors, expert_count, expert_used)
        active_weight_bytes = int(round(total_weight_bytes * frac))
        active_params: Optional[int] = int(round(total_params * frac))
    else:
        active_weight_bytes = total_weight_bytes
        active_params = total_params if is_moe else None

    return ModelSpec(
        repo_id=repo_id,
        quant=quant,
        total_weight_bytes=total_weight_bytes,
        active_weight_bytes=active_weight_bytes,
        total_params=total_params,
        n_layers=n_layers,
        n_kv_heads=int(n_kv_heads),
        key_length=key_length,
        value_length=value_length,
        native_ctx=native_ctx,
        architecture=arch,
        is_moe=is_moe,
        active_params=active_params,
        kv_is_standard=kv_is_standard(arch),
    )


# --------------------------------------------------------------------------- #
# Pure: quant file selection
# --------------------------------------------------------------------------- #
def _first_shard(names: list[str]) -> str:
    for n in sorted(names):
        if "00001-of-" in n:
            return n
    return sorted(names)[0]


def select_quant_files(files: dict[str, int], quant: str) -> tuple[str, int]:
    """Return (file_to_read_header_from, total_bytes_across_shards) for a quant.

    Matches the quant tag case-insensitively against GGUF filenames. total is the
    sum of all matching shards; the header is read from the first shard.
    """
    q = quant.lower()
    matched = {n: s for n, s in files.items() if q in n.lower()}
    if not matched:
        raise ValueError(
            f"no GGUF file matching quant {quant!r}; available: {sorted(files)}"
        )
    return _first_shard(list(matched)), sum(matched.values())


# --------------------------------------------------------------------------- #
# Network edge (exercised on a machine with HF access)
# --------------------------------------------------------------------------- #
def _resolve_url(repo_id: str, filename: str, revision: str = "main") -> str:
    return f"{HF_BASE}/{repo_id}/resolve/{revision}/{filename}"


class HttpRangeReader(ByteReader):
    """Reads byte ranges of a remote file via HTTP Range requests."""

    def __init__(self, url: str, session: Optional[requests.Session] = None):
        self.url = url
        self.session = session or requests.Session()

    def read_range(self, start: int, length: int) -> bytes:
        headers = {"Range": f"bytes={start}-{start + length - 1}"}
        r = self.session.get(self.url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.content


def list_gguf_files(repo_id: str, revision: str = "main") -> dict[str, int]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    return {
        s.rfilename: s.size
        for s in info.siblings
        if s.rfilename.lower().endswith(".gguf") and s.size
    }


def fetch_model(
    repo_id: str,
    quant: str,
    revision: str = "main",
    need_tensors: Optional[bool] = None,
) -> ModelSpec:
    """Live path: list files, pick the quant, range-read the header, map to a spec.

    Reads tensor info (a larger read, past the vocab arrays) only when needed:
    for MoE models, or when parameter_count is absent.
    """
    files = list_gguf_files(repo_id, revision)
    first, total = select_quant_files(files, quant)
    url = _resolve_url(repo_id, first, revision)

    info = parse_gguf(HttpRangeReader(url), need_tensors=False)
    arch = info.metadata.get("general.architecture")
    expert_count = int(info.metadata.get(f"{arch}.expert_count", 0) or 0) if arch else 0

    want_tensors = (
        need_tensors
        if need_tensors is not None
        else (expert_count > 1 or "general.parameter_count" not in info.metadata)
    )
    if want_tensors:
        info = parse_gguf(HttpRangeReader(url), need_tensors=True)

    return build_model_spec(repo_id, quant, info, total)

"""MLX SpecSource.

MLX models on Hugging Face ship as safetensors + a JSON ``config.json``: no
binary header to range-parse, just a small HTTP GET for the config and the
file listing for the weight footprint. The fit/speed math itself is identical
across runtimes — the estimator only consumes ModelSpec.

The HF calls (``hf_hub_download``, the file lister) are injectable so unit
tests run without network. Note that ``fetch`` works on any OS: only running
or calibrating MLX needs Apple Silicon.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

from .gguf import kv_is_standard
from .types import ModelSpec, Runtime


# --------------------------------------------------------------------------- #
# Pure helpers: closed-form param estimate from a config dict
# --------------------------------------------------------------------------- #
def estimate_transformer_params(config: dict) -> tuple[int, Optional[int]]:
    """Closed-form parameter estimate from a HF-transformers ``config.json``.

    Returns ``(total_params, active_params_or_none)``. ``active_params`` is None
    for dense models; for MoE it counts only the experts activated per token.

    This is approximate by design — quantized safetensors store packed tensors,
    so we can't sum shapes; and even unquantized, summing per-tensor sizes
    requires reading every shard's metadata. Prefill is the low-confidence,
    calibration-anchored quantity, so a closed-form estimate is appropriate.
    """
    hidden_size = int(config["hidden_size"])
    n_layers = int(config["num_hidden_layers"])
    n_heads = int(config["num_attention_heads"])
    n_kv_heads = int(config.get("num_key_value_heads", n_heads))
    vocab_size = int(config["vocab_size"])
    intermediate_size = int(config.get("intermediate_size", 4 * hidden_size))
    head_dim = int(config.get("head_dim") or (hidden_size // n_heads))
    tied = bool(config.get("tie_word_embeddings", False))

    embed = vocab_size * hidden_size

    # Attention block
    q_proj = hidden_size * (n_heads * head_dim)
    kv_proj = 2 * hidden_size * (n_kv_heads * head_dim)
    o_proj = (n_heads * head_dim) * hidden_size
    attn = q_proj + kv_proj + o_proj

    # MLP — gated (SwiGLU) when intermediate_size is set: 3 matrices of hidden
    # x intermediate. Non-gated would be 2, but every modern arch we target
    # (llama, qwen2, gemma2/3, mistral) uses gated.
    is_moe, num_experts, experts_per_tok = _moe_config(config)
    if is_moe:
        moe_inter = int(config.get("moe_intermediate_size", intermediate_size))
        total_mlp = num_experts * 3 * hidden_size * moe_inter
        active_mlp = experts_per_tok * 3 * hidden_size * moe_inter
    else:
        total_mlp = 3 * hidden_size * intermediate_size
        active_mlp = total_mlp

    total_per_layer = attn + total_mlp
    active_per_layer = attn + active_mlp

    total = embed + n_layers * total_per_layer + (0 if tied else embed)
    if is_moe:
        active = embed + n_layers * active_per_layer + (0 if tied else embed)
        return total, active
    return total, None


def _moe_config(config: dict) -> tuple[bool, int, int]:
    """Detect MoE and return (is_moe, num_experts, experts_per_tok).

    Field naming varies: Mixtral uses ``num_local_experts``, Qwen MoE uses
    ``num_experts``. Both are accepted; the larger wins (defensive against an
    arch that defines both).
    """
    num_experts = max(
        int(config.get("num_local_experts", 0) or 0),
        int(config.get("num_experts", 0) or 0),
    )
    if num_experts <= 1:
        return False, 0, 0
    experts_per_tok = int(config.get("num_experts_per_tok", 0) or 0)
    if experts_per_tok <= 0:
        # An MoE without an experts-per-token key: treat as dense for active
        # accounting (i.e. fall back; do not divide by zero).
        return False, 0, 0
    return True, num_experts, experts_per_tok


def _quantization_label(config: dict) -> str:
    """MLX records quant inline: ``"quantization": {"group_size": 64, "bits": 4}``.

    Returns the displayable label. Absent quant block -> the repo is
    full-precision; the weight bytes already reflect that.
    """
    q = config.get("quantization")
    if not isinstance(q, dict):
        return "fp16"
    bits = q.get("bits")
    gs = q.get("group_size")
    if bits is None or gs is None:
        return "fp16"
    return f"{int(bits)}bit-g{int(gs)}"


def build_mlx_spec(model_ref: str, config: dict, total_weight_bytes: int) -> ModelSpec:
    """Pure mapping ``config.json`` (+ summed safetensors bytes) -> ModelSpec.

    Kept separate from the network-touching ``MlxSource.fetch`` so tests
    exercise the mapping directly with hand-built config dicts.
    """
    arch = str(config.get("model_type") or "")
    if not arch:
        raise KeyError("MLX config.json missing 'model_type'")

    n_layers = int(config["num_hidden_layers"])
    n_heads = int(config["num_attention_heads"])
    n_kv_heads = int(config.get("num_key_value_heads", n_heads))
    native_ctx = int(config["max_position_embeddings"])

    # head_dim: READ EXPLICITLY when present (the Gemma trap — hidden/n_heads
    # gives the wrong answer). Fall back to the ratio only if config genuinely
    # omits it.
    head_dim = config.get("head_dim")
    if head_dim is None:
        head_dim = int(config["hidden_size"]) // n_heads
    key_length = value_length = int(head_dim)

    is_moe, num_experts, experts_per_tok = _moe_config(config)
    total_params, active_params = estimate_transformer_params(config)

    if is_moe:
        active_fraction = active_params / total_params if total_params else 1.0
        active_weight_bytes = int(round(total_weight_bytes * active_fraction))
    else:
        active_weight_bytes = total_weight_bytes
        active_params = None

    quant_label = _quantization_label(config)

    return ModelSpec(
        repo_id=model_ref,
        # `quant` historically meant the GGUF tag; for MLX nothing sensible
        # maps onto it, so we use quant_label as the human-facing handle and
        # mirror it here so the legacy field is non-empty.
        quant=quant_label,
        total_weight_bytes=total_weight_bytes,
        active_weight_bytes=active_weight_bytes,
        total_params=int(total_params),
        n_layers=n_layers,
        n_kv_heads=int(n_kv_heads),
        key_length=key_length,
        value_length=value_length,
        native_ctx=native_ctx,
        architecture=arch,
        is_moe=is_moe,
        active_params=active_params,
        kv_is_standard=kv_is_standard(arch),
        runtime="mlx",
        quant_label=quant_label,
    )


# --------------------------------------------------------------------------- #
# Network edge (real HF calls — replaced by fakes in tests)
# --------------------------------------------------------------------------- #
def _default_config_loader(model_ref: str) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=model_ref, filename="config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_safetensors_lister(model_ref: str) -> dict[str, int]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_ref, files_metadata=True)
    return {
        s.rfilename: s.size
        for s in info.siblings
        if s.rfilename.lower().endswith(".safetensors") and s.size
    }


class MlxSource:
    """SpecSource for MLX models on Hugging Face.

    `fetch` is pure HF reads — it works on any OS. Running or calibrating MLX
    needs Apple Silicon and ``mlx_lm``; those gates live in benchmark.py and
    compare.py respectively.
    """

    runtime: Runtime = "mlx"

    def __init__(
        self,
        config_loader: Callable[[str], dict] = _default_config_loader,
        file_lister: Callable[[str], dict[str, int]] = _default_safetensors_lister,
    ):
        self._config_loader = config_loader
        self._file_lister = file_lister

    def fetch(self, model_ref: str, quant: Optional[str] = None) -> ModelSpec:
        # quant is ignored for MLX: quant is intrinsic to the repo, not chosen.
        # Surfacing this in a note (vs. raising) keeps the comparison flow
        # uniform across runtimes.
        config = self._config_loader(model_ref)
        files = self._file_lister(model_ref)
        total_weight_bytes = sum(
            size for name, size in files.items() if name.lower().endswith(".safetensors")
        )
        if total_weight_bytes <= 0:
            raise FileNotFoundError(
                f"MLX repo '{model_ref}' has no .safetensors files; not an MLX model"
            )
        return build_mlx_spec(model_ref, config, total_weight_bytes)

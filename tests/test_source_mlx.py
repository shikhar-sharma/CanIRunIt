"""MlxSource tests.

The mapping is pure: hand-built config dicts go in, ModelSpec/asserts come out.
End-to-end tests use ``MlxSource(config_loader=..., file_lister=...)`` with
fakes so the suite never hits Hugging Face.
"""
from __future__ import annotations

import pytest

from canirunit.source_mlx import (
    MlxSource,
    build_mlx_spec,
    estimate_transformer_params,
)


# A reasonable Llama 3.1 8B-ish config (fields the mapper reads).
LLAMA_3_1_8B = {
    "model_type": "llama",
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
    "head_dim": 128,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "tie_word_embeddings": False,
}


# --------------------------------------------------------------------------- #
# Pure mapping
# --------------------------------------------------------------------------- #
def test_dense_llama_mapping():
    spec = build_mlx_spec("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                          LLAMA_3_1_8B | {"quantization": {"group_size": 64, "bits": 4}},
                          total_weight_bytes=4_500_000_000)
    assert spec.runtime == "mlx"
    assert spec.architecture == "llama"
    assert spec.n_layers == 32
    assert spec.n_kv_heads == 8
    assert spec.key_length == 128 and spec.value_length == 128
    assert spec.native_ctx == 131072
    assert spec.quant_label == "4bit-g64"
    assert spec.is_moe is False
    assert spec.active_weight_bytes == spec.total_weight_bytes  # dense
    assert spec.kv_is_standard is True


def test_gemma_head_dim_trap_is_avoided():
    """Gemma 2 9B: hidden=3584, n_heads=16 -> naive ratio = 224.
    Real head_dim is 256. The mapper must take head_dim verbatim, never derive."""
    cfg = {
        "model_type": "gemma2",
        "num_hidden_layers": 42,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "hidden_size": 3584,
        "head_dim": 256,                  # the explicit value
        "intermediate_size": 14336,
        "vocab_size": 256000,
        "max_position_embeddings": 8192,
    }
    spec = build_mlx_spec("mlx-community/gemma-2-9b-it-4bit", cfg, total_weight_bytes=5_000_000_000)
    assert spec.key_length == 256
    assert spec.value_length == 256
    # Pin the trap explicitly so a future "optimisation" doesn't sneak the ratio back in.
    assert 3584 // 16 == 224


def test_head_dim_fallback_when_absent():
    cfg = dict(LLAMA_3_1_8B)
    del cfg["head_dim"]
    spec = build_mlx_spec("repo/x", cfg, total_weight_bytes=1)
    assert spec.key_length == 128  # 4096 / 32


def test_kv_heads_falls_back_to_attention_heads():
    cfg = dict(LLAMA_3_1_8B)
    del cfg["num_key_value_heads"]
    spec = build_mlx_spec("repo/x", cfg, total_weight_bytes=1)
    assert spec.n_kv_heads == cfg["num_attention_heads"]


def test_quantization_label_absent_means_fp16():
    spec = build_mlx_spec("repo/x", LLAMA_3_1_8B, total_weight_bytes=1)
    assert spec.quant_label == "fp16"


def test_deepseek_v2_flagged_as_non_standard_kv():
    """MLA architectures don't use the standard per-head KV formula."""
    cfg = {
        "model_type": "deepseek_v2",
        "num_hidden_layers": 60,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "hidden_size": 5120,
        "head_dim": 128,
        "intermediate_size": 12288,
        "vocab_size": 102400,
        "max_position_embeddings": 163840,
    }
    spec = build_mlx_spec("repo/deepseek-v2", cfg, total_weight_bytes=1)
    assert spec.kv_is_standard is False


def test_moe_split_total_vs_active():
    """Mixtral-style MoE: num_local_experts=8, experts_per_tok=2.
    Active fraction is dominated by the MLP term."""
    cfg = {
        "model_type": "mixtral",
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "head_dim": 128,
        "intermediate_size": 14336,
        "moe_intermediate_size": 14336,
        "vocab_size": 32000,
        "max_position_embeddings": 32768,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
    }
    spec = build_mlx_spec("repo/mixtral", cfg, total_weight_bytes=24_000_000_000)
    assert spec.is_moe is True
    assert spec.active_params is not None
    assert 0 < spec.active_params < spec.total_params
    # active_weight_bytes scales by params ratio
    expected = int(round(24_000_000_000 * (spec.active_params / spec.total_params)))
    assert spec.active_weight_bytes == expected


# --------------------------------------------------------------------------- #
# estimate_transformer_params — hand-computed small config
# --------------------------------------------------------------------------- #
def test_estimate_transformer_params_small_dense():
    """A toy dense config small enough to verify by hand.

    embed         = 100 * 64       = 6400
    per-layer attn = q+kv+o
                  = 64*(8*8)+2*64*(8*8)+(8*8)*64
                  = 4096 + 8192 + 4096 = 16384
    per-layer mlp = 3 * 64 * 128   = 24576
    total         = 6400 + 2*(16384+24576) + 6400 (lm_head, not tied)
                  = 6400 + 81920 + 6400 = 94720
    """
    cfg = {
        "model_type": "llama",
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "hidden_size": 64,
        "head_dim": 8,
        "intermediate_size": 128,
        "vocab_size": 100,
        "max_position_embeddings": 2048,
        "tie_word_embeddings": False,
    }
    total, active = estimate_transformer_params(cfg)
    assert total == 94720
    assert active is None  # dense


def test_estimate_transformer_params_tied_embeddings_drops_lm_head():
    cfg = {
        "model_type": "llama",
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "hidden_size": 64,
        "head_dim": 8,
        "intermediate_size": 128,
        "vocab_size": 100,
        "max_position_embeddings": 2048,
        "tie_word_embeddings": True,
    }
    total, _ = estimate_transformer_params(cfg)
    # one less `embed` term than the previous case.
    assert total == 94720 - 6400


def test_estimate_transformer_params_moe_active_is_smaller():
    cfg = {
        "model_type": "mixtral",
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "hidden_size": 64,
        "head_dim": 8,
        "intermediate_size": 128,
        "moe_intermediate_size": 128,
        "vocab_size": 100,
        "max_position_embeddings": 2048,
        "num_local_experts": 4,
        "num_experts_per_tok": 1,
    }
    total, active = estimate_transformer_params(cfg)
    assert active is not None
    assert active < total
    # 1-of-4 experts -> active MLP is 1/4 of total MLP; attn is full.
    # Sanity: active is well under half of total.
    assert active < total / 2


# --------------------------------------------------------------------------- #
# End-to-end MlxSource.fetch with injected HF fakes (no network)
# --------------------------------------------------------------------------- #
def test_fetch_sums_safetensors_only():
    files = {
        "config.json": 500,
        "model-00001-of-00002.safetensors": 2_000_000_000,
        "model-00002-of-00002.safetensors": 2_500_000_000,
        "tokenizer.json": 1_000_000,
        "README.md": 2000,
    }
    src = MlxSource(
        config_loader=lambda ref: LLAMA_3_1_8B | {"quantization": {"group_size": 32, "bits": 8}},
        file_lister=lambda ref: files,
    )
    spec = src.fetch("mlx-community/whatever")
    assert spec.runtime == "mlx"
    assert spec.total_weight_bytes == 4_500_000_000
    assert spec.quant_label == "8bit-g32"


def test_fetch_no_safetensors_raises():
    src = MlxSource(
        config_loader=lambda ref: LLAMA_3_1_8B,
        file_lister=lambda ref: {"config.json": 500, "tokenizer.json": 1000},
    )
    with pytest.raises(FileNotFoundError, match="no .safetensors"):
        src.fetch("repo/bogus")


def test_fetch_quant_arg_is_ignored():
    """MLX has no GGUF-style quant selector; passing one must not break or change anything."""
    files = {"model.safetensors": 1_000_000_000}
    src = MlxSource(
        config_loader=lambda ref: LLAMA_3_1_8B,
        file_lister=lambda ref: files,
    )
    spec_no_quant = src.fetch("repo/x")
    spec_with_quant = src.fetch("repo/x", quant="Q4_K_M")
    assert spec_no_quant == spec_with_quant

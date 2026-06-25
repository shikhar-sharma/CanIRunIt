"""hub tests: pure mapping and selection, plus an end-to-end parse->spec->estimate."""
from __future__ import annotations

import pytest

from ggufkit import ARRAY, STRING, U32, U64, build_gguf
from canirunit import SystemProfile, estimate
from canirunit.gguf import BytesReader, parse_gguf
from canirunit.hub import GgufSource, build_model_spec, select_quant_files
from canirunit.sources import get_source

GiB = 1024 ** 3


# --------------------------------------------------------------------------- #
# Quant selection / shard summing
# --------------------------------------------------------------------------- #
def test_select_single_file():
    files = {"model-Q4_K_M.gguf": 100, "model-Q8_0.gguf": 200}
    assert select_quant_files(files, "Q4_K_M") == ("model-Q4_K_M.gguf", 100)


def test_select_is_case_insensitive():
    files = {"model-Q4_K_M.gguf": 100}
    assert select_quant_files(files, "q4_k_m") == ("model-Q4_K_M.gguf", 100)


def test_select_sums_shards_and_picks_first():
    files = {
        "m-Q4_K_M-00002-of-00002.gguf": 60,
        "m-Q4_K_M-00001-of-00002.gguf": 50,
    }
    first, total = select_quant_files(files, "Q4_K_M")
    assert first == "m-Q4_K_M-00001-of-00002.gguf"
    assert total == 110


def test_select_no_match_raises():
    with pytest.raises(ValueError, match="no GGUF file matching"):
        select_quant_files({"m-Q8_0.gguf": 1}, "Q4_K_M")


# --------------------------------------------------------------------------- #
# Metadata -> ModelSpec
# --------------------------------------------------------------------------- #
def _spec_from(kv, tensors=(), need_tensors=False, total_bytes=4_900_000_000, quant="Q4_K_M"):
    info = parse_gguf(BytesReader(build_gguf(kv, tensors)), need_tensors=need_tensors)
    return build_model_spec("repo/x", quant, info, total_bytes)


def test_dense_llama_mapping():
    spec = _spec_from([
        ("general.architecture", STRING, "llama"),
        ("general.parameter_count", U64, 8_030_000_000),
        ("llama.block_count", U32, 32),
        ("llama.context_length", U32, 8192),
        ("llama.embedding_length", U32, 4096),
        ("llama.attention.head_count", U32, 32),
        ("llama.attention.head_count_kv", U32, 8),
        ("llama.attention.key_length", U32, 128),
        ("llama.attention.value_length", U32, 128),
    ])
    assert spec.architecture == "llama"
    assert (spec.n_layers, spec.n_kv_heads, spec.native_ctx) == (32, 8, 8192)
    assert spec.key_length == 128 and spec.value_length == 128
    assert spec.is_moe is False
    assert spec.active_weight_bytes == spec.total_weight_bytes  # dense
    assert spec.total_params == 8_030_000_000


def test_gemma_head_dim_is_read_not_derived():
    """The trap: hidden/n_heads = 3584/16 = 224, but Gemma's real head_dim is 256.
    The spec must carry 256."""
    spec = _spec_from([
        ("general.architecture", STRING, "gemma3"),
        ("general.parameter_count", U64, 9_000_000_000),
        ("gemma3.block_count", U32, 42),
        ("gemma3.context_length", U32, 8192),
        ("gemma3.embedding_length", U32, 3584),
        ("gemma3.attention.head_count", U32, 16),
        ("gemma3.attention.head_count_kv", U32, 8),
        ("gemma3.attention.key_length", U32, 256),
        ("gemma3.attention.value_length", U32, 256),
    ])
    assert spec.key_length == 256 and spec.value_length == 256
    assert 3584 // 16 == 224  # what the naive shortcut would have produced


def test_head_dim_fallback_when_absent():
    # No key_length/value_length -> fall back to embedding//head_count = 4096/32 = 128.
    spec = _spec_from([
        ("general.architecture", STRING, "llama"),
        ("general.parameter_count", U64, 7_000_000_000),
        ("llama.block_count", U32, 32),
        ("llama.context_length", U32, 4096),
        ("llama.embedding_length", U32, 4096),
        ("llama.attention.head_count", U32, 32),
        ("llama.attention.head_count_kv", U32, 32),
    ])
    assert spec.key_length == 128 and spec.value_length == 128


def test_mha_fallback_kv_heads_equals_heads():
    # No head_count_kv -> KV heads default to query heads (classic MHA).
    spec = _spec_from([
        ("general.architecture", STRING, "llama"),
        ("general.parameter_count", U64, 7_000_000_000),
        ("llama.block_count", U32, 32),
        ("llama.context_length", U32, 4096),
        ("llama.embedding_length", U32, 4096),
        ("llama.attention.head_count", U32, 32),
        ("llama.attention.key_length", U32, 128),
        ("llama.attention.value_length", U32, 128),
    ])
    assert spec.n_kv_heads == 32


def test_moe_split_total_vs_active():
    """expert_count 8, used 2. Non-expert 10000 params, expert 80000.
    active fraction = (10000 + 0.25*80000)/90000 = 1/3.
    total bytes 900 -> active 300; active_params = 30000."""
    spec = _spec_from(
        [
            ("general.architecture", STRING, "llama"),
            ("llama.block_count", U32, 1),
            ("llama.context_length", U32, 4096),
            ("llama.embedding_length", U32, 100),
            ("llama.attention.head_count", U32, 1),
            ("llama.attention.head_count_kv", U32, 1),
            ("llama.attention.key_length", U32, 100),
            ("llama.attention.value_length", U32, 100),
            ("llama.expert_count", U32, 8),
            ("llama.expert_used_count", U32, 2),
            # tokenizer array present to exercise skip-through on the need_tensors path
            ("tokenizer.ggml.tokens", ARRAY, (STRING, ["a", "bb", "ccc"])),
        ],
        tensors=[
            ("blk.0.attn_q.weight", (100, 100), 0),         # 10000 non-expert
            ("blk.0.ffn_gate_exps.weight", (8, 100, 100), 0),  # 80000 expert
        ],
        need_tensors=True,
        total_bytes=900,
    )
    assert spec.is_moe is True
    assert spec.total_params == 90_000
    assert spec.active_weight_bytes == 300
    assert spec.active_params == 30_000
    assert spec.total_weight_bytes == 900  # fit still uses the full footprint


def test_total_params_from_tensors_when_no_parameter_count():
    """When general.parameter_count is absent, the count must be summed from the
    tensor table (the bench-model condition that broke prefill calibration)."""
    spec = _spec_from(
        [
            ("general.architecture", STRING, "qwen2"),
            ("qwen2.block_count", U32, 1),
            ("qwen2.context_length", U32, 32768),
            ("qwen2.embedding_length", U32, 100),
            ("qwen2.attention.head_count", U32, 2),
            ("qwen2.attention.head_count_kv", U32, 2),
            ("qwen2.attention.key_length", U32, 64),
            ("qwen2.attention.value_length", U32, 64),
            # note: no general.parameter_count
        ],
        tensors=[
            ("token_embd.weight", (100, 1000), 0),       # 100000
            ("blk.0.attn_q.weight", (100, 100), 0),      # 10000
        ],
        need_tensors=True,
        total_bytes=491_000_000,
    )
    assert spec.total_params == 110_000
    assert spec.decode_active_params == 110_000  # dense -> equals total


def test_missing_required_key_raises():
    with pytest.raises(KeyError):
        _spec_from([("general.architecture", STRING, "llama")])  # no block_count etc.


# --------------------------------------------------------------------------- #
# End to end: synthetic GGUF -> spec -> estimate, ties back to the M1 hand-calc
# --------------------------------------------------------------------------- #
def test_end_to_end_spec_feeds_estimator():
    spec = _spec_from([
        ("general.architecture", STRING, "llama"),
        ("general.parameter_count", U64, 8_030_000_000),
        ("llama.block_count", U32, 32),
        ("llama.context_length", U32, 8192),
        ("llama.embedding_length", U32, 4096),
        ("llama.attention.head_count", U32, 32),
        ("llama.attention.head_count_kv", U32, 8),
        ("llama.attention.key_length", U32, 128),
        ("llama.attention.value_length", U32, 128),
    ])
    m1 = SystemProfile(
        total_memory_bytes=16 * GiB, available_memory_bytes=12 * GiB,
        memory_bandwidth_gbs=68.0, accelerator="apple_metal", chip_id="Apple M1",
        storage_free_bytes=100 * GiB, metal_max_working_set_bytes=12 * GiB, peak_flops=2.6e12,
    )
    fit, speed = estimate(m1, spec)
    assert fit.fits_at_native_ctx is True
    # same machine + same model as the estimator's pinned fixture -> ~8 tok/s at 8k
    at_8k = next(p.decode_tok_s for p in speed.points if p.ctx == 8192)
    assert at_8k == pytest.approx(7.97, abs=0.05)


# --------------------------------------------------------------------------- #
# build_model_spec tags runtime + quant_label
# --------------------------------------------------------------------------- #
def test_build_model_spec_tags_gguf_runtime_and_quant_label():
    spec = _spec_from([
        ("general.architecture", STRING, "llama"),
        ("general.parameter_count", U64, 8_030_000_000),
        ("llama.block_count", U32, 32),
        ("llama.context_length", U32, 8192),
        ("llama.embedding_length", U32, 4096),
        ("llama.attention.head_count", U32, 32),
        ("llama.attention.head_count_kv", U32, 8),
        ("llama.attention.key_length", U32, 128),
        ("llama.attention.value_length", U32, 128),
    ], quant="Q5_K_M")
    assert spec.runtime == "gguf"
    assert spec.quant_label == "Q5_K_M"


# --------------------------------------------------------------------------- #
# SpecSource adapter
# --------------------------------------------------------------------------- #
def test_gguf_source_exposes_runtime_attribute():
    src = GgufSource()
    assert src.runtime == "gguf"


def test_get_source_returns_gguf_source():
    src = get_source("gguf")
    assert isinstance(src, GgufSource)


def test_get_source_unknown_runtime_raises():
    with pytest.raises(ValueError, match="unknown runtime"):
        get_source("notaruntime")  # type: ignore[arg-type]

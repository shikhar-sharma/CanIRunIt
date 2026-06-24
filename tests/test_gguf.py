"""GGUF parser tests against synthetic, format-valid byte blobs."""
from __future__ import annotations

import pytest

from ggufkit import ARRAY, F32, STRING, U32, U64, build_gguf
from llmfit.gguf import (
    BytesReader,
    TensorInfo,
    is_expert_tensor,
    moe_active_fraction,
    parse_gguf,
)


def _parse(kv, tensors=(), need_tensors=False):
    return parse_gguf(BytesReader(build_gguf(kv, tensors)), need_tensors=need_tensors)


def test_rejects_non_gguf():
    with pytest.raises(ValueError, match="not a GGUF"):
        parse_gguf(BytesReader(b"\x00\x01\x02\x03" + b"\x00" * 32))


def test_parses_scalar_metadata():
    info = _parse([
        ("general.architecture", STRING, "llama"),
        ("llama.block_count", U32, 32),
        ("llama.context_length", U32, 8192),
        ("llama.attention.head_count_kv", U32, 8),
    ])
    assert info.metadata["general.architecture"] == "llama"
    assert info.metadata["llama.block_count"] == 32
    assert info.metadata["llama.attention.head_count_kv"] == 8
    assert info.tensors is None  # not requested


def test_early_stops_at_tokenizer_without_reading_vocab():
    # A huge token array sits AFTER the hyperparameters. Early-stop means we
    # never decode it — if we did, this test would still pass, but the point is
    # the arch keys are captured from the prefix alone.
    huge_vocab = [f"tok{i}" for i in range(5000)]
    info = _parse([
        ("general.architecture", STRING, "llama"),
        ("llama.block_count", U32, 28),
        ("tokenizer.ggml.model", STRING, "gpt2"),
        ("tokenizer.ggml.tokens", ARRAY, (STRING, huge_vocab)),
        ("llama.context_length", U32, 4096),  # (deliberately after tokenizer)
    ])
    assert info.metadata["llama.block_count"] == 28
    # context_length came after the tokenizer key, so early-stop skips it —
    # the mapping layer falls back when a key is absent.
    assert "llama.context_length" not in info.metadata


def test_skips_unwanted_arrays_when_reading_through(need_tensors=True):
    # With need_tensors=True we do NOT early-stop; the tokenizer string array
    # must be skipped correctly to reach the tensor table beyond it.
    info = _parse(
        [
            ("general.architecture", STRING, "llama"),
            ("llama.block_count", U32, 4),
            ("tokenizer.ggml.tokens", ARRAY, (STRING, ["a", "bb", "ccc"])),
            ("tokenizer.ggml.scores", ARRAY, (F32, [0.1, 0.2, 0.3])),
        ],
        tensors=[("token_embd.weight", (256, 1000), 0)],
        need_tensors=True,
    )
    assert info.metadata["llama.block_count"] == 4
    assert info.tensors is not None
    assert info.tensors[0].name == "token_embd.weight"
    assert info.tensors[0].n_elements == 256 * 1000


def test_tensor_dims_and_element_count():
    info = _parse(
        [("general.architecture", STRING, "llama")],
        tensors=[
            ("blk.0.attn_q.weight", (100, 100), 0),
            ("blk.0.ffn_gate_exps.weight", (8, 100, 100), 0),
        ],
        need_tensors=True,
    )
    by_name = {t.name: t for t in info.tensors}
    assert by_name["blk.0.attn_q.weight"].n_elements == 10_000
    assert by_name["blk.0.ffn_gate_exps.weight"].n_elements == 80_000


def test_is_expert_tensor():
    assert is_expert_tensor("blk.3.ffn_gate_exps.weight")
    assert not is_expert_tensor("blk.3.attn_q.weight")


def test_moe_active_fraction_hand_calc():
    tensors = [
        TensorInfo("blk.0.attn_q.weight", (100, 100), 0),        # 10000 non-expert
        TensorInfo("blk.0.ffn_gate_exps.weight", (8, 100, 100), 0),  # 80000 expert
    ]
    # non_expert + used/count * expert = 10000 + (2/8)*80000 = 30000; /90000 = 1/3
    assert moe_active_fraction(tensors, expert_count=8, expert_used_count=2) == pytest.approx(1 / 3)

"""Pure GGUF binary parser.

Reads a GGUF file's header — magic, metadata key/value store, and (optionally)
the tensor info table — through a swappable byte-range *reader*. That indirection
is the whole point: in tests the reader is backed by in-memory bytes, in
production by HTTP range requests to Hugging Face. The parsing logic is identical
and provable either way.

Format reference: GGUF v2/v3.
  header:   magic u32 | version u32 | tensor_count u64 | metadata_kv_count u64
  metadata: metadata_kv_count x (key:str, value_type:u32, value)
  tensors:  tensor_count x (name:str, n_dims:u32, dims:u64[n_dims], type:u32, offset:u64)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from math import prod
from typing import Optional, Protocol

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# Metadata value type enum
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32 = range(7)
_BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = range(7, 13)

# Byte widths for fixed-size scalar types (used to skip arrays we don't read).
_FIXED_WIDTH = {
    _UINT8: 1, _INT8: 1, _BOOL: 1,
    _UINT16: 2, _INT16: 2,
    _UINT32: 4, _INT32: 4, _FLOAT32: 4,
    _UINT64: 8, _INT64: 8, _FLOAT64: 8,
}
_SCALAR_FMT = {
    _UINT8: "<B", _INT8: "<b", _BOOL: "<?",
    _UINT16: "<H", _INT16: "<h",
    _UINT32: "<I", _INT32: "<i", _FLOAT32: "<f",
    _UINT64: "<Q", _INT64: "<q", _FLOAT64: "<d",
}

# Architectures whose KV cache is compressed (MLA etc.) and so the standard
# per-head formula does not apply. We flag rather than mis-estimate. Includes
# both the GGUF-side names (deepseek2/3) and the HF-transformers model_type
# names (deepseek_v2/v3) so MLX configs hit the same flag.
_NON_STANDARD_KV_ARCHS = {"deepseek2", "deepseek3", "deepseek_v2", "deepseek_v3"}


class ByteReader(Protocol):
    """Reads `length` bytes starting at absolute `start`. May return fewer bytes
    only at end-of-stream."""

    def read_range(self, start: int, length: int) -> bytes: ...


class BytesReader:
    """In-memory reader — used by tests and for already-downloaded files."""

    def __init__(self, data: bytes):
        self._data = data

    def read_range(self, start: int, length: int) -> bytes:
        return self._data[start : start + length]


class FileReader:
    """ByteReader over a local file — for parsing an already-downloaded GGUF.

    Lives here (not in benchmark.py) so source modules can use it without
    creating a benchmark <- source_* dependency.
    """

    def __init__(self, path: str):
        self.path = path

    def read_range(self, start: int, length: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(start)
            return f.read(length)


@dataclass
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    ggml_type: int

    @property
    def n_elements(self) -> int:
        return prod(self.dims) if self.dims else 0


@dataclass
class GGUFInfo:
    metadata: dict[str, object]
    tensors: Optional[list[TensorInfo]]  # None when tensor parsing was skipped


class _Cursor:
    """Sequential forward reader over a ByteReader, fetching in chunks on demand.

    Reads are contiguous from offset 0, so skipping still buffers on the next
    read — the cheapness comes from *stopping early* (we break at the first
    tokenizer.* key, before the multi-megabyte vocab arrays), not from seeking
    past data we've already passed.
    """

    def __init__(self, reader: ByteReader, chunk: int = 1 << 20):
        self.reader = reader
        self.chunk = chunk
        self.pos = 0
        self._buf = bytearray()

    def _ensure(self, upto: int) -> None:
        while len(self._buf) < upto:
            more = self.reader.read_range(len(self._buf), self.chunk)
            if not more:
                raise EOFError("unexpected end of GGUF stream")
            self._buf += more

    def read(self, n: int) -> bytes:
        self._ensure(self.pos + n)
        b = bytes(self._buf[self.pos : self.pos + n])
        self.pos += n
        return b

    def skip(self, n: int) -> None:
        self.pos += n

    def scalar(self, vtype: int):
        fmt = _SCALAR_FMT[vtype]
        return struct.unpack(fmt, self.read(_FIXED_WIDTH[vtype]))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.read(n).decode("utf-8", errors="replace")


def _read_value(cur: _Cursor, vtype: int, *, keep: bool):
    """Read (or, if not `keep`, skip) one metadata value, advancing the cursor."""
    if vtype == _STRING:
        if keep:
            return cur.string()
        n = cur.u64()
        cur.skip(n)
        return None
    if vtype in _SCALAR_FMT:
        v = cur.scalar(vtype)
        return v if keep else None
    if vtype == _ARRAY:
        elem_type = cur.u32()
        count = cur.u64()
        if not keep:
            # Skip the whole array without materializing it.
            if elem_type in _FIXED_WIDTH:
                cur.skip(_FIXED_WIDTH[elem_type] * count)
            elif elem_type == _STRING:
                for _ in range(count):
                    cur.skip(cur.u64())
            else:
                raise ValueError(f"unsupported array elem type {elem_type}")
            return None
        if elem_type == _STRING:
            return [cur.string() for _ in range(count)]
        return [cur.scalar(elem_type) for _ in range(count)]
    raise ValueError(f"unsupported GGUF value type {vtype}")


# Metadata keys we actually consume. Anything else is skipped without keeping.
def _wanted(key: str) -> bool:
    if key in ("general.architecture", "general.parameter_count", "general.file_type"):
        return True
    tail = key.split(".", 1)[-1]
    return tail in (
        "block_count",
        "context_length",
        "embedding_length",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
        "attention.value_length",
        "expert_count",
        "expert_used_count",
    ) or key.endswith((
        ".block_count", ".context_length", ".embedding_length",
        ".attention.head_count", ".attention.head_count_kv",
        ".attention.key_length", ".attention.value_length",
        ".expert_count", ".expert_used_count",
    ))


def parse_gguf(reader: ByteReader, *, need_tensors: bool = False) -> GGUFInfo:
    """Parse a GGUF header.

    By default stops at the first ``tokenizer.*`` key — every model hyperparameter
    llama.cpp writes precedes the tokenizer arrays, so this reads only a small
    prefix. When ``need_tensors`` is True (MoE active-param split, or when
    parameter_count is absent) it parses through to the tensor table, which costs
    reading past the vocab arrays.
    """
    cur = _Cursor(reader)

    magic = cur.u32()
    if magic != GGUF_MAGIC:
        raise ValueError(f"not a GGUF file (magic={magic:#x})")
    version = cur.u32()
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    tensor_count = cur.u64()
    kv_count = cur.u64()

    metadata: dict[str, object] = {}
    parsed_all_metadata = True

    for _ in range(kv_count):
        key = cur.string()
        vtype = cur.u32()
        # Early-stop: the tokenizer section marks the end of model hyperparameters.
        if not need_tensors and key.startswith("tokenizer."):
            parsed_all_metadata = False
            break
        keep = _wanted(key)
        val = _read_value(cur, vtype, keep=keep)
        if keep:
            metadata[key] = val

    tensors: Optional[list[TensorInfo]] = None
    if need_tensors and parsed_all_metadata:
        tensors = []
        for _ in range(tensor_count):
            name = cur.string()
            n_dims = cur.u32()
            dims = tuple(cur.u64() for _ in range(n_dims))
            ggml_type = cur.u32()
            _offset = cur.u64()
            tensors.append(TensorInfo(name=name, dims=dims, ggml_type=ggml_type))

    return GGUFInfo(metadata=metadata, tensors=tensors)


# --------------------------------------------------------------------------- #
# Derived helpers
# --------------------------------------------------------------------------- #
def is_expert_tensor(name: str) -> bool:
    """MoE expert weights are emitted as fused `_exps` tensors by llama.cpp,
    e.g. blk.0.ffn_gate_exps.weight / ffn_up_exps / ffn_down_exps."""
    return "_exps" in name


def moe_active_fraction(
    tensors: list[TensorInfo], expert_count: int, expert_used_count: int
) -> float:
    """Fraction of parameters active per token: all non-expert weights plus the
    used share of expert weights. Approximates the *byte* fraction by the
    *parameter* fraction (experts and the rest are typically the same quant)."""
    expert = sum(t.n_elements for t in tensors if is_expert_tensor(t.name))
    non_expert = sum(t.n_elements for t in tensors if not is_expert_tensor(t.name))
    total = expert + non_expert
    if total == 0 or expert_count == 0:
        return 1.0
    active = non_expert + (expert_used_count / expert_count) * expert
    return active / total


def kv_is_standard(architecture: str) -> bool:
    return architecture not in _NON_STANDARD_KV_ARCHS

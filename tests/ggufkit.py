"""Builds minimal but format-valid GGUF byte blobs for parser tests.

Mirrors the GGUF v3 layout the parser reads, so tests exercise real binary
parsing rather than a mock.
"""
from __future__ import annotations

import struct

# value-type enum (subset we emit)
U32, I32, F32, BOOL, STRING, ARRAY, U64 = 4, 5, 6, 7, 8, 9, 10

_FMT = {U32: "<I", I32: "<i", F32: "<f", BOOL: "<?", U64: "<Q"}


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _value(vtype: int, value) -> bytes:
    if vtype == STRING:
        return _str(value)
    if vtype == ARRAY:
        elem_type, items = value
        out = struct.pack("<I", elem_type) + struct.pack("<Q", len(items))
        for it in items:
            out += _str(it) if elem_type == STRING else struct.pack(_FMT[elem_type], it)
        return out
    return struct.pack(_FMT[vtype], value)


def build_gguf(kv, tensors=()):
    """kv: ordered list of (key, vtype, value). tensors: list of (name, dims, ggml_type)."""
    body = b"".join(
        _str(key) + struct.pack("<I", vtype) + _value(vtype, value)
        for key, vtype, value in kv
    )
    tbody = b""
    for name, dims, ggml_type in tensors:
        tbody += _str(name) + struct.pack("<I", len(dims))
        for d in dims:
            tbody += struct.pack("<Q", d)
        tbody += struct.pack("<I", ggml_type) + struct.pack("<Q", 0)  # type + offset

    header = (
        struct.pack("<I", 0x46554747)   # magic "GGUF"
        + struct.pack("<I", 3)          # version
        + struct.pack("<Q", len(tensors))
        + struct.pack("<Q", len(kv))
    )
    return header + body + tbody

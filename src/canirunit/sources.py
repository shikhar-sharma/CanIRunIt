"""Runtime-agnostic source abstraction.

The estimator only consumes ModelSpec — it does not care whether the spec was
read from a GGUF file header, an MLX config.json, or an Ollama local blob. The
SpecSource Protocol is the seam: one method per runtime that produces a spec.

The registry below lazy-imports the concrete source modules so that, e.g.,
importing the GGUF path never imports MLX-only helpers (and conversely so that
the MLX path never pulls in network code at GGUF import time).
"""
from __future__ import annotations

from typing import Optional, Protocol

from .types import ModelSpec, Runtime


class SpecSource(Protocol):
    """Produces a ModelSpec for a given model reference under a specific runtime.

    The `model_ref` shape is runtime-specific:
      * gguf:   Hugging Face repo id, e.g. "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
      * mlx:    Hugging Face repo id, e.g. "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
      * ollama: Ollama tag, e.g. "llama3.1:8b"
    """

    runtime: Runtime

    def fetch(self, model_ref: str, quant: Optional[str] = None) -> ModelSpec: ...


def get_source(runtime: Runtime) -> SpecSource:
    """Return the SpecSource for a runtime.

    Lazy-imports the source module so that callers that only care about one
    runtime never pay the import cost of the others (and Apple-only deps in
    MLX paths stay out of non-Apple test runs).
    """
    if runtime == "gguf":
        from .hub import GgufSource
        return GgufSource()
    if runtime == "mlx":
        from .source_mlx import MlxSource
        return MlxSource()
    if runtime == "ollama":
        from .source_ollama import OllamaSource
        return OllamaSource()
    raise ValueError(f"unknown runtime {runtime!r}")

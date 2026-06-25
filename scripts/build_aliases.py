"""Maintainer-side helper: best-effort builder for data/aliases.json.

NOT shipped with the package. Run by hand, review the output, edit, commit.

Fuzzy matching is acceptable here because a human reviews before commit;
runtime resolution stays exact (a lookup, not a search). Ollama tags are not
HF-discoverable, so this script leaves them null/TODO for the maintainer to
fill from a small known-tags map or by hand.

Usage:
    python scripts/build_aliases.py                       # dry-run, prints empty skeleton
    python scripts/build_aliases.py --live                # actually call HF; print JSON
    python scripts/build_aliases.py --live --out data/aliases.json   # write to file
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from typing import Iterable, Optional

# Runtime/quant suffixes commonly tacked onto HF repo names. The order matters
# only in that more specific suffixes (e.g. "-Instruct-4bit") are stripped one
# token at a time by the iteration loop.
_STRIP_SUFFIXES = (
    "-gguf", "-mlx",
    "-4bit", "-8bit", "-2bit", "-3bit", "-6bit",
    "-fp16", "-bf16",
)

# Default crawl set. Add more authors / search terms before --live if needed.
_GGUF_AUTHORS = ("bartowski", "lmstudio-community")
_MLX_AUTHORS = ("mlx-community",)

# Ollama tags are not HF-discoverable; this small known-tags map seeds the
# common families. The maintainer fills in the rest by hand.
_OLLAMA_KNOWN_TAGS: dict[str, str] = {
    "meta-llama-3.1-8b-instruct": "llama3.1:8b",
    "llama-3.2-3b-instruct": "llama3.2:3b",
    "qwen2.5-7b-instruct": "qwen2.5:7b",
    "qwen2.5-0.5b-instruct": "qwen2.5:0.5b",
    "gemma-2-9b-it": "gemma2:9b",
    "mistral-7b-instruct-v0.3": "mistral:7b",
    "deepseek-r1-distill-qwen-7b": "deepseek-r1:7b",
}


# --------------------------------------------------------------------------- #
# Pure: key normalization (the bit worth unit-testing)
# --------------------------------------------------------------------------- #
def normalize_model_key(repo_id: str) -> str:
    """Map a HF repo id to a logical model key suitable for pairing GGUF and
    MLX repos of the same underlying model.

    Strips the author/namespace, common runtime suffixes (``-GGUF``, ``-MLX``,
    ``-4bit``, ...), lowercases, and collapses separators.
    """
    name = repo_id.split("/", 1)[-1].lower()

    # Iteratively strip suffixes — one repo may have multiple (e.g. -GGUF, or
    # rare composite -Instruct-4bit-GGUF type names).
    changed = True
    while changed:
        changed = False
        for sfx in _STRIP_SUFFIXES:
            if name.endswith(sfx):
                name = name[: -len(sfx)]
                changed = True

    name = re.sub(r"[\s_]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name


def _is_gguf_repo(repo_id: str) -> bool:
    return repo_id.lower().endswith("-gguf") or "gguf" in repo_id.lower().split("/")[-1]


def _is_mlx_repo(repo_id: str) -> bool:
    # mlx-community prefix is the unambiguous signal; suffix-based detection
    # for stragglers in other orgs.
    return repo_id.lower().startswith("mlx-community/") or re.search(
        r"-\dbit(-g\d+)?$", repo_id.lower()
    ) is not None


def group_repos(repo_ids: Iterable[str]) -> dict[str, dict[str, list[str]]]:
    """Bucket repo ids by normalized key and by runtime.

    Returns ``{key: {"gguf": [...], "mlx": [...]}, ...}`` — only the runtimes
    with at least one repo are present per key.
    """
    groups: dict[str, dict[str, list[str]]] = {}
    for rid in repo_ids:
        key = normalize_model_key(rid)
        bucket = groups.setdefault(key, {})
        if _is_gguf_repo(rid):
            bucket.setdefault("gguf", []).append(rid)
        elif _is_mlx_repo(rid):
            bucket.setdefault("mlx", []).append(rid)
    return groups


def build_alias_entries(
    grouped: dict[str, dict[str, list[str]]],
    ollama_known_tags: Optional[dict[str, str]] = None,
) -> dict:
    """Compose a `models` dict from grouped repos. Each entry picks the first
    candidate per runtime (alphabetical) — the maintainer can edit.
    """
    tags = ollama_known_tags or {}
    out: dict[str, dict] = {}
    for key in sorted(grouped):
        repos = grouped[key]
        sources: dict = {}
        if repos.get("gguf"):
            sources["gguf"] = {"repo_id": sorted(repos["gguf"])[0], "default_quant": "Q4_K_M"}
        if repos.get("mlx"):
            sources["mlx"] = {"repo_id": sorted(repos["mlx"])[0]}
        if key in tags:
            sources["ollama"] = {"tag": tags[key]}
        if not sources:
            continue
        out[key] = {
            "display_name": key.replace("-", " ").title(),
            "family": "TODO",  # maintainer fills
            "sources": sources,
        }
    return out


# --------------------------------------------------------------------------- #
# HF discovery (network — only when --live)
# --------------------------------------------------------------------------- #
def discover_repos(
    gguf_authors: Iterable[str] = _GGUF_AUTHORS,
    mlx_authors: Iterable[str] = _MLX_AUTHORS,
    search_term: str = "",
) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    repos: list[str] = []
    for author in gguf_authors:
        repos.extend(m.id for m in api.list_models(author=author, search=search_term or "GGUF"))
    for author in mlx_authors:
        repos.extend(m.id for m in api.list_models(author=author, search=search_term or ""))
    return repos


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", action="store_true", help="Actually call HF (default: skeleton only)")
    p.add_argument("--out", help="Output path (default: stdout)")
    p.add_argument("--search", default="", help="Optional HF search term to narrow the crawl")
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_argparser().parse_args(argv)
    repo_ids = discover_repos(search_term=args.search) if args.live else []
    grouped = group_repos(repo_ids)
    models = build_alias_entries(grouped, _OLLAMA_KNOWN_TAGS)

    table = {
        "_comment": (
            "Generated by scripts/build_aliases.py. UNVERIFIED — review repo "
            "ids and ollama tags before committing. Ollama tags are seeded "
            "from a small known map; the rest are null and need maintainer fill-in."
        ),
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": models,
    }

    payload = json.dumps(table, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
        print(f"wrote {len(models)} entries to {args.out}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

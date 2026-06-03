#!/usr/bin/env python3
"""Patch pinned safety-eval fork to cap vLLM context length."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAFETY_REPO = PROJECT_ROOT / "external" / "safety-eval-fork"
TARGET_RELATIVE_PATH = Path("src") / "generation_utils.py"
VLLM_MARKER = "trust_remote_code=trust_remote_code,"
PATCH_SNIPPET = "max_model_len=4096,"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch safety-eval fork vLLM max_model_len")
    parser.add_argument("--repo", default=str(DEFAULT_SAFETY_REPO), help="Path to external/safety-eval-fork")
    return parser.parse_args()


def patch_generation_utils(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if PATCH_SNIPPET in text:
        print(f"[skip] already patched: {path}")
        return

    if "max_model_len=" in text:
        raise RuntimeError(f"Refusing to patch file that already sets max_model_len: {path}")

    lines = text.splitlines(keepends=True)
    marker_indexes = [index for index, line in enumerate(lines) if line.strip() == VLLM_MARKER]
    if len(marker_indexes) != 1:
        raise RuntimeError(f"Expected exactly one vLLM trust_remote_code marker in {path}, found {len(marker_indexes)}")

    marker_index = marker_indexes[0]
    indent = lines[marker_index][: len(lines[marker_index]) - len(lines[marker_index].lstrip())]
    newline = "\r\n" if lines[marker_index].endswith("\r\n") else "\n"
    lines.insert(marker_index + 1, f"{indent}{PATCH_SNIPPET}{newline}")

    path.write_text("".join(lines), encoding="utf-8")
    print(f"[patched] {path}")


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    generation_utils = repo / TARGET_RELATIVE_PATH
    if not generation_utils.exists():
        raise FileNotFoundError(f"Missing safety-eval generation utils: {generation_utils}")

    patch_generation_utils(generation_utils)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

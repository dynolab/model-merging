#!/usr/bin/env python3
"""Summarize Qwen-style <think> lengths saved by lm-eval sample logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.common import BENCHMARK_RESULTS_ROOT, PROJECT_ROOT
from common.io import project_path


DEFAULT_SAMPLE_CSV = PROJECT_ROOT / "results" / "benchmark_reasoning_samples.csv"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "results" / "benchmark_reasoning_lengths.csv"

SAMPLE_FIELDS = [
    "config_id",
    "axis",
    "task_id",
    "sample_file",
    "line_number",
    "doc_id",
    "has_think_open",
    "has_think_close",
    "reasoning_char_count",
    "reasoning_word_count",
    "answer_char_count",
    "answer_word_count",
    "output_char_count",
    "output_word_count",
]

SUMMARY_FIELDS = [
    "config_id",
    "axis",
    "task_id",
    "n",
    "tagged_n",
    "tagged_share",
    "mean_reasoning_words",
    "median_reasoning_words",
    "p90_reasoning_words",
    "mean_reasoning_chars",
    "median_reasoning_chars",
    "p90_reasoning_chars",
    "mean_output_words",
    "mean_output_chars",
    "mean_answer_words",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize <think> lengths from lm-eval sample logs")
    parser.add_argument("--results-root", default=str(BENCHMARK_RESULTS_ROOT))
    parser.add_argument(
        "--axis",
        action="append",
        default=None,
        help="Axis to scan. May be repeated. Defaults to every axis with samples_*.jsonl files.",
    )
    parser.add_argument("--samples-output", default=str(DEFAULT_SAMPLE_CSV))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_CSV))
    return parser.parse_args()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(flatten_strings(item))
        return out
    if isinstance(value, list | tuple):
        out: list[str] = []
        for item in value:
            out.extend(flatten_strings(item))
        return out
    return []


def generated_text(sample: dict[str, Any]) -> str:
    for key in (
        "resps",
        "filtered_resps",
        "response",
        "prediction",
        "output",
        "generation",
        "generations",
        "completion",
        "model_output",
    ):
        strings = [s for s in flatten_strings(sample.get(key)) if s]
        if strings:
            return strings[0]
    return ""


def split_reasoning(text: str) -> tuple[str, str, bool, bool]:
    lower = text.lower()
    open_tag = "<think>"
    close_tag = "</think>"
    open_idx = lower.find(open_tag)
    close_idx = lower.find(close_tag)

    has_open = open_idx >= 0
    has_close = close_idx >= 0

    if has_open and has_close and close_idx > open_idx:
        reasoning_start = open_idx + len(open_tag)
        reasoning = text[reasoning_start:close_idx]
        answer = text[close_idx + len(close_tag) :]
    elif has_close:
        reasoning = text[:close_idx]
        answer = text[close_idx + len(close_tag) :]
    elif has_open:
        reasoning = text[open_idx + len(open_tag) :]
        answer = ""
    else:
        reasoning = ""
        answer = text

    return reasoning.strip(), answer.strip(), has_open, has_close


def task_name_from_path(path: Path) -> str:
    match = re.match(r"^samples_(.+)_\d{4}-\d{2}-\d{2}T", path.name)
    if match:
        return match.group(1)
    return path.stem.removeprefix("samples_")


def axis_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    out: set[str] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                out.add(item)
    return out or None


def sample_row(
    *,
    config_id: str,
    axis: str,
    task_id: str,
    sample_file: Path,
    line_number: int | str,
    doc_id: Any,
    text: str,
) -> dict[str, Any]:
    reasoning, answer, has_open, has_close = split_reasoning(text)
    return {
        "config_id": config_id,
        "axis": axis,
        "task_id": task_id,
        "sample_file": project_path(sample_file, PROJECT_ROOT),
        "line_number": line_number,
        "doc_id": doc_id,
        "has_think_open": has_open,
        "has_think_close": has_close,
        "reasoning_char_count": len(reasoning),
        "reasoning_word_count": word_count(reasoning),
        "answer_char_count": len(answer),
        "answer_word_count": word_count(answer),
        "output_char_count": len(text),
        "output_word_count": word_count(text),
    }


def safety_generation_records(value: Any, *, task_hint: str = "safety", path: str = "$") -> list[tuple[str, str, dict[str, Any]]]:
    if isinstance(value, dict):
        task_id = str(value.get("task") or value.get("task_id") or value.get("dataset") or task_hint)
        if generated_text(value):
            return [(path, task_id, value)]
        out: list[tuple[str, str, dict[str, Any]]] = []
        for key, item in value.items():
            child_task = str(key) if isinstance(item, list) else task_id
            out.extend(safety_generation_records(item, task_hint=child_task, path=f"{path}.{key}"))
        return out
    if isinstance(value, list):
        out: list[tuple[str, str, dict[str, Any]]] = []
        for idx, item in enumerate(value):
            out.extend(safety_generation_records(item, task_hint=task_hint, path=f"{path}[{idx}]"))
        return out
    return []


def sample_rows(results_root: Path, axes: set[str] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not results_root.exists():
        return rows

    for model_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        for axis_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            axis = axis_dir.name
            if axes is not None and axis not in axes:
                continue
            for sample_file in sorted(axis_dir.rglob("samples_*.jsonl")):
                task_id = task_name_from_path(sample_file)
                with sample_file.open("r", encoding="utf-8-sig") as f:
                    for line_number, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            sample = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = generated_text(sample)
                        rows.append(
                            sample_row(
                                config_id=model_dir.name,
                                axis=axis,
                                task_id=str(sample.get("task_name") or task_id),
                                sample_file=sample_file,
                                line_number=line_number,
                                doc_id=sample.get("doc_id"),
                                text=text,
                            )
                        )
            for sample_file in sorted(axis_dir.rglob("safety_generations.json")):
                try:
                    data = json.loads(sample_file.read_text(encoding="utf-8-sig"))
                except Exception:
                    continue
                for record_path, task_id, record in safety_generation_records(data):
                    text = generated_text(record)
                    rows.append(
                        sample_row(
                            config_id=model_dir.name,
                            axis=axis,
                            task_id=task_id,
                            sample_file=sample_file,
                            line_number=record_path,
                            doc_id=record.get("id") or record.get("doc_id") or record.get("example_id"),
                            text=text,
                        )
                    )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["config_id"]), str(row["axis"]), str(row["task_id"]))].append(row)
        groups[(str(row["config_id"]), str(row["axis"]), "__all__")].append(row)

    out: list[dict[str, Any]] = []
    for (config_id, axis, task_id), items in sorted(groups.items()):
        reasoning_words = [int(row["reasoning_word_count"]) for row in items]
        reasoning_chars = [int(row["reasoning_char_count"]) for row in items]
        output_words = [int(row["output_word_count"]) for row in items]
        output_chars = [int(row["output_char_count"]) for row in items]
        answer_words = [int(row["answer_word_count"]) for row in items]
        tagged_n = sum(1 for row in items if row["has_think_open"] or row["has_think_close"])
        out.append(
            {
                "config_id": config_id,
                "axis": axis,
                "task_id": task_id,
                "n": len(items),
                "tagged_n": tagged_n,
                "tagged_share": tagged_n / len(items) if items else None,
                "mean_reasoning_words": mean(reasoning_words) if reasoning_words else None,
                "median_reasoning_words": median(reasoning_words) if reasoning_words else None,
                "p90_reasoning_words": percentile(reasoning_words, 0.9),
                "mean_reasoning_chars": mean(reasoning_chars) if reasoning_chars else None,
                "median_reasoning_chars": median(reasoning_chars) if reasoning_chars else None,
                "p90_reasoning_chars": percentile(reasoning_chars, 0.9),
                "mean_output_words": mean(output_words) if output_words else None,
                "mean_output_chars": mean(output_chars) if output_chars else None,
                "mean_answer_words": mean(answer_words) if answer_words else None,
            }
        )
    return out


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = PROJECT_ROOT / results_root

    samples = sample_rows(results_root, axis_filter(args.axis))
    summary = summarize(samples)
    samples_output = Path(args.samples_output)
    summary_output = Path(args.summary_output)
    if not samples_output.is_absolute():
        samples_output = PROJECT_ROOT / samples_output
    if not summary_output.is_absolute():
        summary_output = PROJECT_ROOT / summary_output

    write_csv(samples_output, SAMPLE_FIELDS, samples)
    write_csv(summary_output, SUMMARY_FIELDS, summary)
    print(f"[done] wrote {project_path(samples_output, PROJECT_ROOT)} rows={len(samples)}")
    print(f"[done] wrote {project_path(summary_output, PROJECT_ROOT)} rows={len(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the calibration denylist from the frozen benchmark-input snapshot."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT.parent
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
BENCHMARK_CONFIG = CODE_ROOT / "cfg" / "benchmark" / "benchmark.yaml"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "benchmark" / "benchmark_inputs.jsonl"
DEFAULT_INPUT_MANIFEST = PROJECT_ROOT / "data" / "benchmark" / "benchmark_inputs_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "calibration" / "benchmark_denylist.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "calibration" / "benchmark_denylist_manifest.json"

from common.io import file_sha256, project_path, read_json, read_jsonl, read_yaml, write_jsonl
from common.text import TEXT_MIN_CHARS, normalize_text, sha256_text


def add_row(rows_by_hash: dict[str, dict[str, Any]], row: dict[str, Any]) -> bool:
    text = str(row.get("text", "")).strip()
    if len(text) < TEXT_MIN_CHARS:
        return False
    norm = normalize_text(text)
    if len(norm) < TEXT_MIN_CHARS:
        return False
    h = sha256_text(norm)
    if h in rows_by_hash:
        existing = rows_by_hash[h]
        existing["source_tasks"] = sorted(set(existing.get("source_tasks", [])) | set(row.get("source_tasks", [])) | {str(row.get("source_task"))})
        existing["source_axes"] = sorted(set(existing.get("source_axes", [])) | set(row.get("source_axes", [])) | {str(row.get("source_axis"))})
        return False
    out = dict(row)
    out["normalized_text"] = norm
    out["normalized_sha256"] = h
    rows_by_hash[h] = out
    return True


def main() -> None:
    cfg = read_yaml(BENCHMARK_CONFIG)
    input_rel = cfg.get("benchmark_denylist", {}).get("input_snapshot") or cfg.get("benchmark_inputs", {}).get("output")
    input_manifest_rel = cfg.get("benchmark_denylist", {}).get("input_snapshot_manifest") or cfg.get("benchmark_inputs", {}).get("manifest")
    output_rel = cfg.get("benchmark_denylist", {}).get("output")
    manifest_rel = cfg.get("benchmark_denylist", {}).get("manifest")

    input_path = PROJECT_ROOT / input_rel if input_rel else DEFAULT_INPUT
    input_manifest_path = PROJECT_ROOT / input_manifest_rel if input_manifest_rel else DEFAULT_INPUT_MANIFEST
    output = PROJECT_ROOT / output_rel if output_rel else DEFAULT_OUTPUT
    manifest_path = PROJECT_ROOT / manifest_rel if manifest_rel else DEFAULT_MANIFEST

    if not input_path.exists():
        raise FileNotFoundError(f"Benchmark input snapshot not found: {project_path(input_path, PROJECT_ROOT)}. Run export_benchmark_inputs.py first.")
    if not input_manifest_path.exists():
        raise FileNotFoundError(f"Benchmark input snapshot manifest not found: {project_path(input_manifest_path, PROJECT_ROOT)}. Run export_benchmark_inputs.py first.")

    input_manifest = read_json(input_manifest_path)
    current_config_hash = file_sha256(BENCHMARK_CONFIG)
    warnings: list[str] = []
    if input_manifest.get("benchmark_config_sha256") != current_config_hash:
        warnings.append("benchmark input snapshot was generated from a different benchmark config hash")
    if input_manifest.get("output_sha256") != file_sha256(input_path):
        warnings.append("benchmark input snapshot hash does not match its manifest")
    for warning in warnings:
        print(f"[warning] {warning}")

    rows_by_hash: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(input_path):
        add_row(rows_by_hash, row)

    out_rows = list(rows_by_hash.values())
    task_counts: Counter[str] = Counter()
    for row in out_rows:
        for task in row.get("source_tasks") or [row.get("source_task")]:
            if task:
                task_counts[str(task)] += 1
    out_rows.sort(key=lambda r: (str(r.get("source_axis")), str(r.get("source_task")), str(r.get("source_field")), r["normalized_sha256"]))
    write_jsonl(output, out_rows)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_config": project_path(BENCHMARK_CONFIG, PROJECT_ROOT),
        "benchmark_config_sha256": current_config_hash,
        "benchmark_id": cfg.get("benchmark_id"),
        "input_snapshot": project_path(input_path, PROJECT_ROOT),
        "input_snapshot_sha256": file_sha256(input_path),
        "input_snapshot_manifest": project_path(input_manifest_path, PROJECT_ROOT),
        "input_snapshot_manifest_sha256": file_sha256(input_manifest_path),
        "input_snapshot_summary": {
            "enabled_tasks": input_manifest.get("enabled_tasks"),
            "task_counts": input_manifest.get("task_counts"),
            "num_entries": input_manifest.get("num_entries"),
            "lm_eval_input_sources": input_manifest.get("lm_eval_input_sources"),
            "safety_eval_input_sources": input_manifest.get("safety_eval_input_sources"),
        },
        "task_counts": dict(task_counts),
        "num_unique_entries": len(out_rows),
        "output": project_path(output, PROJECT_ROOT),
        "output_sha256": file_sha256(output),
        "warnings": warnings,
        "normalization": {
            "lowercase": True,
            "collapse_whitespace": True,
            "html_unescape": True,
            "zero_width_removed": True,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[OK] wrote denylist: {project_path(output, PROJECT_ROOT)}")
    print(f"[OK] wrote manifest: {project_path(manifest_path, PROJECT_ROOT)}")
    print(f"[INFO] unique denylist entries: {len(out_rows)}")


if __name__ == "__main__":
    main()

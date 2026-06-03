#!/usr/bin/env python3
"""Freeze best mergekit grid parameters from selection benchmark results."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import yaml

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.common import SELECTION_MAIN_CSV
from benchmark.model_registry import model_registry
from common.io import project_path, read_json, read_yaml


CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT.parent
MERGE_CONFIG = CODE_ROOT / "cfg" / "merge" / "mergekit_baselines.yaml"
BENCHMARK_CONFIG = CODE_ROOT / "cfg" / "benchmark" / "benchmark.yaml"
SUMMARY_CSV = SELECTION_MAIN_CSV
MERGE_MANIFEST_ROOT = PROJECT_ROOT / "outputs" / "merge_manifests"

PRIMARY_METRIC = "Aggregate"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze best mergekit grid parameters from selection benchmark results")
    p.add_argument("baselines", nargs="*", help="baseline id(s); defaults to all baselines with grid")
    return p.parse_args()


def parse_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def best_for_baseline(
    rows: list[dict[str, Any]],
    baseline_id: str,
    spec: dict[str, Any],
    expected_base_model: dict[str, Any] | None,
    expected_source_models: list[dict[str, Any]],
) -> dict[str, Any] | None:
    prefix = baseline_id + "__"
    candidates = [r for r in rows if r.get("Config id") == baseline_id or str(r.get("Config id", "")).startswith(prefix)]
    candidates = [r for r in candidates if str(r.get("Valid run", "")).lower() == "true"]
    scored = []
    for row in candidates:
        primary = parse_float(row.get(PRIMARY_METRIC))
        if primary is None:
            continue
        scored.append((row, primary))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[1])
    row, primary = scored[0]
    config_id = str(row["Config id"])
    manifest_path = MERGE_MANIFEST_ROOT / f"{config_id}.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    run = manifest.get("run") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("baseline_id") != baseline_id
        or manifest.get("mergekit_method") != spec.get("mergekit_method")
        or manifest.get("base_model") != expected_base_model
        or manifest.get("source_models") != expected_source_models
        or not isinstance(run, dict)
        or run.get("exit_code") != 0
    ):
        return None
    params = manifest.get("parameters") if isinstance(manifest, dict) else None
    if not isinstance(params, dict):
        return None
    return {
        "baseline_id": baseline_id,
        "selected_config_id": config_id,
        "selection_score": primary,
        "merge_parameters": params,
    }


def freeze_selected_params(merge_cfg: dict[str, Any], selected: list[dict[str, Any]]) -> None:
    baselines = merge_cfg.get("baselines", {})
    if not isinstance(baselines, dict):
        raise ValueError("merge config must define baselines")
    for row in selected:
        baseline_id = str(row["baseline_id"])
        spec = baselines.get(baseline_id)
        if not isinstance(spec, dict):
            continue
        params = row.get("merge_parameters")
        if not isinstance(params, dict) or not params:
            print(f"[warning] cannot freeze {baseline_id}: selected row has no merge parameters")
            continue
        if "grid" in spec:
            grid = spec.pop("grid")
            spec.setdefault("search_grid", grid)
        spec["parameters"] = params
        spec["frozen_from"] = {
            "selected_config_id": row.get("selected_config_id"),
            "selection_score": row.get("selection_score"),
        }
    with MERGE_CONFIG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merge_cfg, f, sort_keys=False, allow_unicode=False)
    print(f"[done] froze selected parameters in {project_path(MERGE_CONFIG, PROJECT_ROOT)}")


def main() -> int:
    args = parse_args()
    merge_cfg = read_yaml(MERGE_CONFIG)
    baseline_specs = merge_cfg.get("baselines", {})
    if not isinstance(baseline_specs, dict):
        raise ValueError("merge config must define baselines")
    baseline_ids = args.baselines or [str(k) for k, v in baseline_specs.items() if isinstance(v, dict) and "grid" in v]
    if not baseline_ids:
        print("[info] no unfrozen grid baselines to freeze")
        return 0
    missing_ids = [b for b in baseline_ids if b not in baseline_specs]
    if missing_ids:
        raise ValueError("unknown baseline(s): " + ", ".join(missing_ids))
    invalid_ids = [b for b in baseline_ids if not isinstance(baseline_specs.get(b), dict)]
    if invalid_ids:
        raise ValueError("baseline(s) must be mappings: " + ", ".join(invalid_ids))
    non_grid_ids = [b for b in baseline_ids if "grid" not in baseline_specs[b]]
    if non_grid_ids:
        raise ValueError("baseline(s) have no active grid to freeze: " + ", ".join(non_grid_ids))
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"selection benchmark summary not found: {project_path(SUMMARY_CSV, PROJECT_ROOT)}. "
            "Run code/src/benchmark/summarize_benchmark.py --selection first."
        )
    base_alias = str(merge_cfg["base_model"])
    source_aliases = [str(x) for x in merge_cfg["source_models"]]
    benchmark_cfg = read_yaml(BENCHMARK_CONFIG)
    registry = model_registry(benchmark_cfg)
    missing_aliases = [a for a in [base_alias] + source_aliases if a not in registry]
    if missing_aliases:
        raise ValueError("model alias not found in benchmark.yaml: " + ", ".join(missing_aliases))
    expected_base_model = {
        "alias": base_alias,
        "repo": registry[base_alias]["repo"],
        "revision": registry[base_alias]["revision"],
    }
    expected_source_models = [
        {"alias": alias, "repo": registry[alias]["repo"], "revision": registry[alias]["revision"]}
        for alias in source_aliases
    ]

    rows = read_rows(SUMMARY_CSV)
    selected = []
    missing_results = []
    for baseline_id in baseline_ids:
        spec = baseline_specs[baseline_id]
        expected_base = expected_base_model if str(spec.get("mergekit_method")) != "linear" else None
        best = best_for_baseline(
            rows,
            baseline_id,
            spec,
            expected_base,
            expected_source_models,
        )
        if best is None:
            missing_results.append(baseline_id)
            continue
        selected.append(best)

    if missing_results:
        print("[error] no complete valid selection result for: " + ", ".join(missing_results), file=sys.stderr)
        print("[hint] run code/src/merge/run_selection_search.py for those baselines, then summarize selection again.", file=sys.stderr)
        return 1
    freeze_selected_params(merge_cfg, selected)
    for row in selected:
        print(
            "[selected] "
            f"{row['baseline_id']} -> {row['selected_config_id']} "
            f"{PRIMARY_METRIC}={row.get('selection_score')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

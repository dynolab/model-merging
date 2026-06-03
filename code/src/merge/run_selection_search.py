#!/usr/bin/env python3
"""Run disk-safe one-time selection for mergekit grid baselines."""

from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.common import PROJECT_ROOT, SELECTION_CONFIG, SELECTION_RESULTS_ROOT, selection_benchmark_section
from common.io import project_path, read_json, read_yaml
from common.text import safe_id
from run_mergekit_baselines import (
    BENCHMARK_CONFIG,
    MERGE_CONFIG,
    MERGE_MANIFEST,
    MERGEKIT_CONFIG_ROOT,
    OUTPUT_ROOT,
    build_mergekit_config,
    materialize_models,
    run_mergekit,
    selected_baselines,
    write_manifest,
    write_yaml,
)


MERGE_MANIFEST_ROOT = PROJECT_ROOT / "outputs" / "merge_manifests"
BENCHMARK_SCRIPT = "code/src/benchmark/evaluate_benchmark.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge, selection-evaluate, and delete each grid candidate")
    p.add_argument("baselines", nargs="*", help="grid baseline id(s); defaults to all grid baselines")
    p.add_argument("--gpu-ids", default="0", help="CUDA ids for selection eval, e.g. 0 or 0,1")
    return p.parse_args()


def value_id(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(value).replace("-", "m").replace(".", "p")
    return safe_id(str(value))


def candidate_id(baseline_id: str, params: dict[str, Any]) -> str:
    parts = []
    for key in sorted(params):
        if key == "normalize":
            continue
        parts.append(f"{key}{value_id(params[key])}")
    return baseline_id + ("__" + "_".join(parts) if parts else "")


def grid_candidates(baseline_id: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    base_params = dict(spec.get("parameters", {}))
    grid = spec.get("grid")
    if not isinstance(grid, dict):
        raise ValueError(f"{baseline_id} must define grid")
    keys = list(grid.keys())
    values = []
    for key in keys:
        vals = grid[key]
        if not isinstance(vals, list) or not vals:
            raise ValueError(f"{baseline_id}.grid.{key} must be a non-empty list")
        values.append(vals)
    candidates = []
    for combo in itertools.product(*values):
        params = dict(base_params)
        params.update(dict(zip(keys, combo)))
        candidates.append(params)
    return candidates


def safe_rmtree(path: Path, allowed_root: Path) -> None:
    if not path.exists():
        return
    target = path.resolve()
    root = allowed_root.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"refusing to delete outside {root}: {target}") from exc
    if target == root:
        raise ValueError(f"refusing to delete root directory: {target}")
    shutil.rmtree(target)


def archive_merge_manifest(config_id: str, out_dir: Path) -> None:
    src = out_dir / MERGE_MANIFEST
    if not src.exists():
        return
    MERGE_MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, MERGE_MANIFEST_ROOT / f"{config_id}.json")


def read_run_manifest(config_id: str) -> dict[str, Any] | None:
    path = SELECTION_RESULTS_ROOT / config_id / "run_manifest.json"
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def completed_selection_result(config_id: str) -> bool:
    manifest = read_run_manifest(config_id)
    if not isinstance(manifest, dict):
        return False
    archived_manifest = MERGE_MANIFEST_ROOT / f"{config_id}.json"
    if not archived_manifest.exists():
        return False
    try:
        merge_manifest = read_json(archived_manifest)
    except Exception:
        return False
    selection_dir = SELECTION_RESULTS_ROOT / config_id / "selection"
    selection_run = manifest.get("selection_run")
    merge_run = merge_manifest.get("run") if isinstance(merge_manifest, dict) else None
    return (
        manifest.get("exit_code") == 0
        and isinstance(selection_run, dict)
        and selection_run.get("exit_code") == 0
        and isinstance(merge_run, dict)
        and merge_run.get("exit_code") == 0
        and selection_dir.exists()
    )


def run_selection_eval(config_id: str, out_dir: Path, gpu_ids: str) -> int:
    cmd = [
        sys.executable,
        BENCHMARK_SCRIPT,
        "--selection",
        "--model",
        project_path(out_dir, PROJECT_ROOT),
        "--config-id",
        config_id,
        "--gpu-ids",
        gpu_ids,
        "--overwrite",
    ]
    print("[eval] " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False).returncode


def main() -> int:
    args = parse_args()
    merge_cfg = read_yaml(MERGE_CONFIG)
    benchmark_cfg = read_yaml(BENCHMARK_CONFIG)
    selection_cfg = read_yaml(SELECTION_CONFIG)
    selection_benchmark_section(selection_cfg, require_metric=True)
    all_selected = selected_baselines(merge_cfg, args.baselines)
    selected = {k: v for k, v in all_selected.items() if "grid" in v}
    non_grid = sorted(set(all_selected) - set(selected))
    if args.baselines and non_grid:
        raise ValueError("selection search expects grid baseline(s), got: " + ", ".join(non_grid))
    if not selected:
        raise ValueError("no grid baselines selected")

    base_alias = str(merge_cfg["base_model"])
    source_aliases = [str(x) for x in merge_cfg["source_models"]]
    expanded: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
    for baseline_id, spec in selected.items():
        for params in grid_candidates(baseline_id, spec):
            expanded.append((baseline_id, spec, params, candidate_id(baseline_id, params)))
    print(f"[info] selection candidates: {len(expanded)}")

    failures: list[str] = []
    pending: list[tuple[int, str, dict[str, Any], dict[str, Any], str]] = []

    for index, (baseline_id, spec, params, config_id) in enumerate(expanded, start=1):
        out_dir = OUTPUT_ROOT / config_id
        if completed_selection_result(config_id):
            print(f"\n[{index}/{len(expanded)}] {config_id}")
            safe_rmtree(out_dir, OUTPUT_ROOT)
            print("[skip] existing completed selection result")
            continue
        pending.append((index, baseline_id, spec, params, config_id))

    if not pending:
        print("[done] selection search already completed; candidate checkpoints were removed")
        return 0

    aliases = [base_alias] + source_aliases
    model_paths, identities = materialize_models(benchmark_cfg, aliases)

    for index, baseline_id, spec, params, config_id in pending:
        out_dir = OUTPUT_ROOT / config_id
        print(f"\n[{index}/{len(expanded)}] {config_id}")
        safe_rmtree(out_dir, OUTPUT_ROOT)

        mergekit_yaml = build_mergekit_config(
            spec=spec,
            params=params,
            model_paths=model_paths,
            source_aliases=source_aliases,
            base_alias=base_alias,
        )
        mergekit_yaml_path = MERGEKIT_CONFIG_ROOT / f"{config_id}.yaml"
        write_yaml(mergekit_yaml_path, mergekit_yaml)

        result = run_mergekit(mergekit_yaml_path, out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(
            out_dir,
            baseline_id=baseline_id,
            config_id=config_id,
            spec=spec,
            params=params,
            identities=identities,
            source_aliases=source_aliases,
            base_alias=base_alias,
            mergekit_yaml_path=mergekit_yaml_path,
            run_result=result,
        )
        archive_merge_manifest(config_id, out_dir)

        if result["exit_code"] != 0:
            failures.append(config_id)
            print(f"[warning] merge failed for {config_id}; checkpoint will be removed")
            safe_rmtree(out_dir, OUTPUT_ROOT)
            continue

        eval_code = run_selection_eval(config_id, out_dir, args.gpu_ids)
        safe_rmtree(out_dir, OUTPUT_ROOT)
        if eval_code != 0:
            failures.append(config_id)

    if failures:
        print("[done] completed with failures: " + ", ".join(failures))
        return 1
    print("[done] selection search completed; candidate checkpoints were removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

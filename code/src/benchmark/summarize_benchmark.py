#!/usr/bin/env python3
"""Summarize final benchmark results or one-time selection benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.common import (
    BENCHMARK_CONFIG,
    BENCHMARK_FORGETTING_CSV,
    BENCHMARK_MAIN_CSV,
    BENCHMARK_RESULTS_ROOT,
    BENCHMARK_RUNS_JSON,
    BENCHMARK_TASKS_CSV,
    BUNDLES_CONFIG,
    PROJECT_ROOT,
    SELECTION_CONFIG,
    SELECTION_MAIN_CSV,
    SELECTION_RESULTS_ROOT,
    enabled_axes,
    selection_benchmark_section,
    task_entries,
)
from common.io import file_sha256, project_path, read_yaml


def load_jsons(root: Path) -> list[tuple[Path, Any]]:
    out: list[tuple[Path, Any]] = []
    if not root.exists():
        return out
    for path in root.rglob("*.json"):
        name = path.name.lower()
        if name == "run_manifest.json" or "generation" in name or "sample" in name:
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                out.append((path, json.load(f)))
        except Exception:
            continue
    return out


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def get_nested(obj: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = obj
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def get_metric_value(metrics: dict[str, Any], metric_key: str) -> float | None:
    keys = [metric_key]
    if "," not in metric_key:
        keys.append(f"{metric_key},none")
    for key in keys:
        val = metrics.get(key)
        if val is None and "." in key:
            val = get_nested(metrics, key)
        if is_number(val):
            return float(val)
    return None


def normalize_score(score: float | None, task_cfg: dict[str, Any]) -> float | None:
    if score is None:
        return None
    invert = bool(task_cfg.get("invert", False))
    norm = str(task_cfg.get("score_normalization", "none"))
    if norm in {"invert", "lower_is_better_to_higher_is_better"}:
        invert = True
    if task_cfg.get("higher_is_better") is False:
        invert = True
    return 1.0 - score if invert else score


def find_task_score(result_dir: Path, task_cfg: dict[str, Any]) -> tuple[float | None, float | None]:
    task_id = str(task_cfg["task_id"])
    runner_task_name = str(task_cfg.get("runner_task_name") or task_id)
    metric_key = str(task_cfg.get("metric") or "")
    if not metric_key:
        raise ValueError(f"Task {task_id} must define an explicit metric key")

    for _, data in load_jsons(result_dir):
        if isinstance(data, dict) and isinstance(data.get("results"), dict):
            for name in {task_id, runner_task_name}:
                if isinstance(data["results"].get(name), dict):
                    raw = get_metric_value(data["results"][name], metric_key)
                    score = normalize_score(raw, task_cfg)
                    if score is not None:
                        return score, raw
        if isinstance(data, dict):
            for name in {task_id, runner_task_name}:
                if isinstance(data.get(name), dict):
                    raw = get_metric_value(data[name], metric_key)
                    score = normalize_score(raw, task_cfg)
                    if score is not None:
                        return score, raw
    return None, None


def load_manifest(model_dir: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    path = model_dir / "run_manifest.json"
    if not path.exists():
        return {}, [], [f"missing run_manifest.json in {project_path(model_dir, PROJECT_ROOT)}"]
    try:
        with path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        return {}, [], [f"could not read run_manifest.json in {project_path(model_dir, PROJECT_ROOT)}: {exc!r}"]
    if not isinstance(manifest, dict):
        return {}, [], [f"run_manifest.json is not a JSON object: {project_path(path, PROJECT_ROOT)}"]
    return manifest, [], []


def common_manifest_errors(manifest: dict[str, Any], current_config_hash: str) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    if not manifest:
        return warnings, errors
    if manifest.get("exit_code") not in {0, None}:
        errors.append(f"run_manifest exit_code={manifest.get('exit_code')}")
    if manifest.get("benchmark_config_sha256") != current_config_hash:
        errors.append("benchmark_config_sha256 differs from current benchmark config")
    warnings.extend(str(w) for w in manifest.get("warnings", []) or [])

    identity = manifest.get("model_identity", {})
    if isinstance(identity, dict) and identity.get("kind") == "local_checkpoint":
        local_manifest = identity.get("local_model_manifest", {})
        if not isinstance(local_manifest, dict) or not local_manifest.get("manifest"):
            errors.append("local checkpoint has no readable merge_manifest.json")
    return warnings, errors


def validate_final_manifest(manifest: dict[str, Any], current_config_hash: str) -> tuple[list[str], list[str]]:
    warnings, errors = common_manifest_errors(manifest, current_config_hash)
    axis_runs = manifest.get("axis_runs", [])
    if not isinstance(axis_runs, list) or not axis_runs:
        errors.append("no benchmark axes recorded in run_manifest")
    for entry in axis_runs if isinstance(axis_runs, list) else []:
        if isinstance(entry, dict) and entry.get("exit_code") not in {0, None}:
            errors.append(f"axis {entry.get('axis')} previously failed with exit_code={entry.get('exit_code')}")
    return warnings, errors


def validate_selection_manifest(manifest: dict[str, Any], current_config_hash: str) -> tuple[list[str], list[str]]:
    warnings, errors = common_manifest_errors(manifest, current_config_hash)
    selection_run = manifest.get("selection_run")
    if not isinstance(selection_run, dict):
        errors.append("no selection_run recorded in run_manifest")
    elif selection_run.get("exit_code") not in {0, None}:
        errors.append(f"selection benchmark previously failed with exit_code={selection_run.get('exit_code')}")
    return warnings, errors


def summarize_final_model(model_dir: Path, benchmark_cfg: dict[str, Any], current_config_hash: str) -> dict[str, Any]:
    manifest, warnings, errors = load_manifest(model_dir)
    manifest_warnings, manifest_errors = validate_final_manifest(manifest, current_config_hash)
    warnings.extend(manifest_warnings)
    errors.extend(manifest_errors)

    task_scores: dict[str, float | None] = {}
    raw_task_scores: dict[str, float | None] = {}
    axis_scores: dict[str, float | None] = {}
    missing_tasks: list[str] = []
    axes_cfg = enabled_axes(benchmark_cfg)

    for axis_id, axis_cfg in axes_cfg.items():
        scores: list[float] = []
        for task in task_entries(axis_cfg, require_metric=True):
            score, raw = find_task_score(model_dir / axis_id, task)
            key = f"{axis_id}/{task['task_id']}"
            task_scores[key] = score
            raw_task_scores[key] = raw
            if score is not None:
                scores.append(score)
            else:
                missing_tasks.append(key)
        axis_scores[axis_id] = mean(scores) if scores else None

    aggregate_axes = benchmark_cfg.get("aggregate", {}).get("axes", list(axes_cfg.keys()))
    missing_axes = [a for a in aggregate_axes if axis_scores.get(a) is None]
    if missing_tasks:
        errors.append("missing benchmark tasks: " + ", ".join(missing_tasks))
    if missing_axes:
        errors.append("missing aggregate axes: " + ", ".join(missing_axes))
    aggregate = mean(axis_scores[a] for a in aggregate_axes) if not missing_axes and not missing_tasks else None

    return {
        "config_id": model_dir.name,
        "valid_run": not errors,
        "validation_errors": errors,
        "warnings": warnings,
        "task_scores": task_scores,
        "raw_task_scores": raw_task_scores,
        "axis_scores": axis_scores,
        "aggregate": aggregate,
    }


def summarize_selection_model(model_dir: Path, benchmark_cfg: dict[str, Any], current_config_hash: str) -> dict[str, Any]:
    manifest, warnings, errors = load_manifest(model_dir)
    manifest_warnings, manifest_errors = validate_selection_manifest(manifest, current_config_hash)
    warnings.extend(manifest_warnings)
    errors.extend(manifest_errors)

    task_scores: dict[str, float | None] = {}
    scores: list[float] = []
    for task in task_entries(selection_benchmark_section(benchmark_cfg), require_metric=True):
        task_id = str(task["task_id"])
        score, _ = find_task_score(model_dir / "selection", task)
        task_scores[task_id] = score
        if score is not None:
            scores.append(score)

    missing_tasks = [task_id for task_id, score in task_scores.items() if score is None]
    if missing_tasks:
        errors.append("missing selection benchmark tasks: " + ", ".join(missing_tasks))
    aggregate = mean(scores) if scores and not missing_tasks else None

    return {
        "config_id": model_dir.name,
        "valid_run": not errors,
        "validation_errors": errors,
        "warnings": warnings,
        "task_scores": task_scores,
        "aggregate": aggregate,
    }


def bundle_score(summary: dict[str, Any] | None, bundle: dict[str, Any]) -> float | None:
    if summary is None or not summary.get("valid_run", False):
        return None
    values: list[float] = []
    default_source = str(bundle.get("score_source", "task_scores"))
    for task in bundle.get("tasks", []):
        key = f"{task['axis']}/{task['task_id']}"
        source = str(task.get("score_source", default_source))
        if source not in {"task_scores", "raw_task_scores"}:
            raise ValueError(f"Unsupported score_source={source!r} for bundle task {key}")
        value = summary.get(source, {}).get(key)
        if value is None:
            return None
        values.append(float(value))
    return mean(values) if values else None


def add_forgetting(summaries: dict[str, dict[str, Any]], bundles_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bundles = {k: v for k, v in bundles_cfg.get("bundles", {}).items() if v.get("enabled", False)}
    missing = sorted(str(v.get("specialist_id")) for v in bundles.values() if v.get("specialist_id") not in summaries)
    if missing:
        msg = "missing specialist result folders required for forgetting: " + ", ".join(missing)
        for summary in summaries.values():
            summary.setdefault("warnings", []).append(msg)

    for model_id, summary in summaries.items():
        forgetting: dict[str, float | None] = {}
        for bundle_id, bundle in bundles.items():
            specialist_id = str(bundle.get("specialist_id"))
            spec_score = bundle_score(summaries.get(specialist_id), bundle)
            model_score = bundle_score(summary, bundle)
            value = None if spec_score is None or model_score is None else spec_score - model_score
            forgetting[bundle_id] = value
            rows.append(
                {
                    "Config id": model_id,
                    "Domain (i)": bundle_id,
                    "Specialist id": specialist_id,
                    "Specialist score": spec_score,
                    "Merged/model score": model_score,
                    "F_i": value,
                }
            )
        values = [v for v in forgetting.values() if v is not None]
        summary["forgetting"] = forgetting
        summary["max_forgetting"] = max(values) if values else None
        summary["mean_forgetting"] = mean(values) if values else None
    return rows


def notes(row: dict[str, Any]) -> str:
    return "; ".join(row.get("validation_errors", []) + row.get("warnings", []))


def write_final_runs_json(rows: list[dict[str, Any]]) -> None:
    BENCHMARK_RUNS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_RUNS_JSON.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=True, indent=2)


def write_final_tasks_csv(rows: list[dict[str, Any]], benchmark_cfg: dict[str, Any]) -> None:
    axes = list(enabled_axes(benchmark_cfg).keys())
    task_keys = [
        f"{axis_id}/{task['task_id']}"
        for axis_id, axis_cfg in enabled_axes(benchmark_cfg).items()
        for task in task_entries(axis_cfg, require_metric=True)
    ]
    forgetting_keys = sorted({k for row in rows for k in row.get("forgetting", {}).keys()})
    fields = ["config_id", "valid_run", "notes", "aggregate"]
    fields += [f"axis:{a}" for a in axes]
    fields += [f"task:{t}" for t in task_keys]
    fields += [f"raw_task:{t}" for t in task_keys]
    fields += [f"forgetting:{k}" for k in forgetting_keys] + ["max_forgetting", "mean_forgetting"]

    BENCHMARK_TASKS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_TASKS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {
                "config_id": row["config_id"],
                "valid_run": row.get("valid_run"),
                "notes": notes(row),
                "aggregate": row.get("aggregate"),
            }
            for axis in axes:
                flat[f"axis:{axis}"] = row.get("axis_scores", {}).get(axis)
            for task_key in task_keys:
                flat[f"task:{task_key}"] = row.get("task_scores", {}).get(task_key)
                flat[f"raw_task:{task_key}"] = row.get("raw_task_scores", {}).get(task_key)
            for key in forgetting_keys:
                flat[f"forgetting:{key}"] = row.get("forgetting", {}).get(key)
            flat["max_forgetting"] = row.get("max_forgetting")
            flat["mean_forgetting"] = row.get("mean_forgetting")
            writer.writerow(flat)


def write_final_main_csv(rows: list[dict[str, Any]]) -> None:
    axis_labels = {
        "instruction_following": "IF",
        "safety": "Safety",
    }
    axis_ids: list[str] = []
    for row in rows:
        axis_scores = row.get("axis_scores", {})
        if isinstance(axis_scores, dict):
            for axis_id in axis_scores:
                if axis_id not in axis_ids:
                    axis_ids.append(str(axis_id))

    axis_fields = [axis_labels.get(axis_id, axis_id) for axis_id in axis_ids]
    fields = ["Config id", "Valid run", "Notes", "Aggregate"] + axis_fields + ["Max F_i", "Mean F_i"]
    BENCHMARK_MAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_MAIN_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            axis = row.get("axis_scores", {})
            out = {
                "Config id": row.get("config_id"),
                "Valid run": row.get("valid_run"),
                "Notes": notes(row),
                "Aggregate": row.get("aggregate"),
                "Max F_i": row.get("max_forgetting"),
                "Mean F_i": row.get("mean_forgetting"),
            }
            if isinstance(axis, dict):
                for axis_id in axis_ids:
                    out[axis_labels.get(axis_id, axis_id)] = axis.get(axis_id)
            writer.writerow(out)

def write_forgetting_csv(rows: list[dict[str, Any]]) -> None:
    fields = ["Config id", "Domain (i)", "F_i", "Specialist id", "Specialist score", "Merged/model score"]
    BENCHMARK_FORGETTING_CSV.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_FORGETTING_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_selection_csv(rows: list[dict[str, Any]], benchmark_cfg: dict[str, Any]) -> None:
    task_ids = [str(task["task_id"]) for task in selection_benchmark_section(benchmark_cfg, require_metric=True)["tasks"]]
    fields = ["Config id", "Valid run", "Notes", "Aggregate"] + [f"task:{task_id}" for task_id in task_ids]
    SELECTION_MAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SELECTION_MAIN_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {
                "Config id": row.get("config_id"),
                "Valid run": row.get("valid_run"),
                "Notes": notes(row),
                "Aggregate": row.get("aggregate"),
            }
            for task_id in task_ids:
                out[f"task:{task_id}"] = row.get("task_scores", {}).get(task_id)
            writer.writerow(out)


def summarize_final() -> int:
    benchmark_cfg = read_yaml(BENCHMARK_CONFIG)
    current_hash = file_sha256(BENCHMARK_CONFIG)
    if not BENCHMARK_RESULTS_ROOT.exists():
        raise FileNotFoundError(f"results root does not exist: {project_path(BENCHMARK_RESULTS_ROOT, PROJECT_ROOT)}")

    summaries = {
        model_dir.name: summarize_final_model(model_dir, benchmark_cfg, current_hash)
        for model_dir in sorted(p for p in BENCHMARK_RESULTS_ROOT.iterdir() if p.is_dir())
    }
    forgetting_rows = add_forgetting(summaries, read_yaml(BUNDLES_CONFIG))
    rows = [summaries[k] for k in sorted(summaries)]

    write_final_runs_json(rows)
    write_final_tasks_csv(rows, benchmark_cfg)
    write_final_main_csv(rows)
    write_forgetting_csv(forgetting_rows)
    for path in [BENCHMARK_RUNS_JSON, BENCHMARK_TASKS_CSV, BENCHMARK_MAIN_CSV, BENCHMARK_FORGETTING_CSV]:
        print(f"[done] wrote {project_path(path, PROJECT_ROOT)}")
    return 0


def summarize_selection() -> int:
    benchmark_cfg = read_yaml(SELECTION_CONFIG)
    selection_benchmark_section(benchmark_cfg, require_metric=True)
    current_hash = file_sha256(SELECTION_CONFIG)
    if not SELECTION_RESULTS_ROOT.exists():
        raise FileNotFoundError(f"results root does not exist: {project_path(SELECTION_RESULTS_ROOT, PROJECT_ROOT)}")

    rows = [
        summarize_selection_model(model_dir, benchmark_cfg, current_hash)
        for model_dir in sorted(p for p in SELECTION_RESULTS_ROOT.iterdir() if p.is_dir())
    ]
    write_selection_csv(rows, benchmark_cfg)
    print(f"[done] wrote {project_path(SELECTION_MAIN_CSV, PROJECT_ROOT)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark results")
    parser.add_argument("--selection", action="store_true", help="summarize one-time selection benchmark results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(summarize_selection() if args.selection else summarize_final())

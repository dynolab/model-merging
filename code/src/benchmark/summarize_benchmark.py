#!/usr/bin/env python3
"""Summarize outputs from the frozen custom benchmark protocol."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.io import file_sha256, project_path, read_yaml
from benchmark.common import (
    BENCHMARK_CONFIG,
    BUNDLES_CONFIG,
    FORGETTING_CSV,
    MAIN_CSV,
    PROJECT_ROOT,
    RESULTS_ROOT,
    RUNS_JSON,
    TASKS_CSV,
    enabled_axes,
    task_entries,
)


def load_jsons(root: Path) -> list[tuple[Path, Any]]:
    out: list[tuple[Path, Any]] = []
    if not root.exists():
        return out
    for path in root.rglob("*.json"):
        name = path.name.lower()
        if name == "run_manifest.json":
            continue
        if "generation" in name or "sample" in name:
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
    candidate_keys = [metric_key]
    if "," not in metric_key:
        candidate_keys.append(f"{metric_key},none")
    for key in candidate_keys:
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


def find_task_score(axis_dir: Path, task_cfg: dict[str, Any]) -> tuple[float | None, float | None]:
    task_id = str(task_cfg["task_id"])
    runner_task_name = str(task_cfg.get("runner_task_name") or task_id)
    metric_key = str(task_cfg.get("metric") or "")
    if not metric_key:
        raise ValueError(f"Task {task_id} must define an explicit metric key")
    jsons = load_jsons(axis_dir)
    names = {task_id, runner_task_name}
    for path, data in jsons:
        if isinstance(data, dict) and isinstance(data.get("results"), dict):
            for name in names:
                if isinstance(data["results"].get(name), dict):
                    raw_score = get_metric_value(data["results"][name], metric_key)
                    score = normalize_score(raw_score, task_cfg)
                    if score is not None:
                        return score, raw_score
        if isinstance(data, dict):
            for name in names:
                if isinstance(data.get(name), dict):
                    raw_score = get_metric_value(data[name], metric_key)
                    score = normalize_score(raw_score, task_cfg)
                    if score is not None:
                        return score, raw_score
    return None, None


def load_run_manifest(model_dir: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    p = model_dir / "run_manifest.json"
    if not p.exists():
        return {}, [], [f"missing run_manifest.json in {project_path(model_dir, PROJECT_ROOT)}"]
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return {}, [], [f"could not read run_manifest.json in {project_path(model_dir, PROJECT_ROOT)}: {exc!r}"]
    if not isinstance(data, dict):
        return {}, [], [f"run_manifest.json is not a JSON object: {project_path(p, PROJECT_ROOT)}"]
    return data, [], []


def validate_manifest(model_dir: Path, manifest: dict[str, Any], current_config_hash: str) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    validation_errors: list[str] = []
    if not manifest:
        return warnings, validation_errors
    if manifest.get("exit_code") != 0:
        validation_errors.append(f"run_manifest exit_code={manifest.get('exit_code')}")
    for warning in manifest.get("warnings", []) or []:
        warnings.append(str(warning))
    manifest_hash = manifest.get("benchmark_config_sha256")
    if manifest_hash != current_config_hash:
        validation_errors.append("benchmark_config_sha256 differs from current benchmark.yaml")
    axis_runs = manifest.get("axis_runs", [])
    if not isinstance(axis_runs, list) or not axis_runs:
        validation_errors.append("no benchmark axis runs recorded in run_manifest")
    for entry in axis_runs if isinstance(axis_runs, list) else []:
        if isinstance(entry, dict) and entry.get("exit_code") not in {0, None}:
            validation_errors.append(f"axis {entry.get('axis')} previously failed with exit_code={entry.get('exit_code')}")
    identity = manifest.get("model_identity", {})
    if isinstance(identity, dict) and identity.get("kind") == "local_checkpoint":
        local_manifest = identity.get("local_model_manifest", {})
        if not isinstance(local_manifest, dict) or not local_manifest.get("manifest"):
            validation_errors.append("local checkpoint has no readable merge_manifest.json")
    return warnings, validation_errors


def summarize_model(model_dir: Path, benchmark_cfg: dict[str, Any], current_config_hash: str) -> dict[str, Any]:
    manifest, warnings, validation_errors = load_run_manifest(model_dir)
    manifest_warnings, manifest_errors = validate_manifest(model_dir, manifest, current_config_hash)
    warnings.extend(manifest_warnings)
    validation_errors.extend(manifest_errors)
    axes_cfg = enabled_axes(benchmark_cfg)
    task_scores: dict[str, float | None] = {}
    raw_task_scores: dict[str, float | None] = {}
    axis_scores: dict[str, float | None] = {}

    for axis_id, axis_cfg in axes_cfg.items():
        scores: list[float] = []
        axis_dir = model_dir / axis_id
        for task in task_entries(axis_cfg, require_metric=True):
            score, raw_score = find_task_score(axis_dir, task)
            task_key = f"{axis_id}/{task['task_id']}"
            task_scores[task_key] = score
            raw_task_scores[task_key] = raw_score
            if score is not None:
                scores.append(score)
        axis_scores[axis_id] = mean(scores) if scores else None

    aggregate_axes = benchmark_cfg.get("aggregate", {}).get("axes", list(axes_cfg.keys()))
    missing_axes = [a for a in aggregate_axes if axis_scores.get(a) is None]
    aggregate = mean(axis_scores[a] for a in aggregate_axes) if not missing_axes else None

    if missing_axes:
        validation_errors.append("missing aggregate axes: " + ", ".join(missing_axes))

    valid_run = not missing_axes and not validation_errors
    return {
        "config_id": model_dir.name,
        "valid_run": valid_run,
        "validation_errors": validation_errors,
        "warnings": warnings,
        "task_scores": task_scores,
        "raw_task_scores": raw_task_scores,
        "axis_scores": axis_scores,
        "aggregate": aggregate,
        "run_manifest_path": project_path(model_dir / "run_manifest.json", PROJECT_ROOT) if (model_dir / "run_manifest.json").exists() else None,
    }


def bundle_score(summary: dict[str, Any] | None, bundle: dict[str, Any]) -> float | None:
    if summary is None or not summary.get("valid_run", False):
        return None
    vals: list[float] = []
    default_source = str(bundle.get("score_source", "task_scores"))
    for t in bundle.get("tasks", []):
        key = f"{t['axis']}/{t['task_id']}"
        score_source = str(t.get("score_source", default_source))
        if score_source not in {"task_scores", "raw_task_scores"}:
            raise ValueError(f"Unsupported score_source={score_source!r} for bundle task {key}")
        val = summary.get(score_source, {}).get(key)
        if val is not None:
            vals.append(float(val))
    return mean(vals) if vals else None


def add_forgetting(summaries: dict[str, dict[str, Any]], bundles_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    bundles = bundles_cfg.get("bundles", {})
    forgetting_rows: list[dict[str, Any]] = []
    enabled_bundles = {k: v for k, v in bundles.items() if v.get("enabled", False)}
    missing_specialists = sorted(
        str(v.get("specialist_id")) for v in enabled_bundles.values() if v.get("specialist_id") not in summaries
    )
    if missing_specialists:
        msg = "missing specialist result folders required for forgetting: " + ", ".join(missing_specialists)
        for summary in summaries.values():
            summary.setdefault("warnings", []).append(msg)

    for model_id, summary in summaries.items():
        forgetting: dict[str, float | None] = {}
        for bundle_id, bundle in enabled_bundles.items():
            specialist_id = bundle.get("specialist_id")
            specialist_summary = summaries.get(str(specialist_id))
            spec_score = bundle_score(specialist_summary, bundle)
            merged_score = bundle_score(summary, bundle)
            value = None if spec_score is None or merged_score is None else spec_score - merged_score
            forgetting[bundle_id] = value
            forgetting_rows.append(
                {
                    "Config id": model_id,
                    "Domain (i)": bundle_id,
                    "Specialist id": specialist_id,
                    "Specialist score": spec_score,
                    "Merged/model score": merged_score,
                    "F_i": value,
                }
            )
        vals = [v for v in forgetting.values() if v is not None]
        summary["forgetting"] = forgetting
        summary["max_forgetting"] = max(vals) if vals else None
        summary["mean_forgetting"] = mean(vals) if vals else None
    return forgetting_rows


def write_tasks_csv(path: Path, rows: list[dict[str, Any]], benchmark_cfg: dict[str, Any]) -> None:
    axes = list(enabled_axes(benchmark_cfg).keys())
    task_keys: list[str] = []
    for axis_id, axis_cfg in enabled_axes(benchmark_cfg).items():
        for task in task_entries(axis_cfg, require_metric=True):
            task_keys.append(f"{axis_id}/{task['task_id']}")
    forgetting_keys = sorted({k for r in rows for k in r.get("forgetting", {}).keys()})
    fields = ["config_id", "valid_run", "notes", "aggregate"]
    fields += [f"axis:{a}" for a in axes]
    fields += [f"task:{t}" for t in task_keys]
    fields += [f"raw_task:{t}" for t in task_keys]
    fields += [f"forgetting:{k}" for k in forgetting_keys] + ["max_forgetting", "mean_forgetting"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            flat = {
                "config_id": r["config_id"],
                "valid_run": r.get("valid_run"),
                "notes": "; ".join(r.get("validation_errors", []) + r.get("warnings", [])),
                "aggregate": r.get("aggregate"),
            }
            for a in axes:
                flat[f"axis:{a}"] = r.get("axis_scores", {}).get(a)
            for t in task_keys:
                flat[f"task:{t}"] = r.get("task_scores", {}).get(t)
                flat[f"raw_task:{t}"] = r.get("raw_task_scores", {}).get(t)
            for k in forgetting_keys:
                flat[f"forgetting:{k}"] = r.get("forgetting", {}).get(k)
            flat["max_forgetting"] = r.get("max_forgetting")
            flat["mean_forgetting"] = r.get("mean_forgetting")
            writer.writerow(flat)


def write_main_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["Config id", "Valid run", "Notes", "Aggregate", "IF", "Reasoning", "Safety", "Max F_i", "Mean F_i"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            axis = r.get("axis_scores", {})
            writer.writerow(
                {
                    "Config id": r.get("config_id"),
                    "Valid run": r.get("valid_run"),
                    "Notes": "; ".join(r.get("validation_errors", []) + r.get("warnings", [])),
                    "Aggregate": r.get("aggregate"),
                    "IF": axis.get("instruction_following"),
                    "Reasoning": axis.get("reasoning_math"),
                    "Safety": axis.get("safety"),
                    "Max F_i": r.get("max_forgetting"),
                    "Mean F_i": r.get("mean_forgetting"),
                }
            )


def write_forgetting_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["Config id", "Domain (i)", "F_i", "Specialist id", "Specialist score", "Merged/model score"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})


def main() -> int:
    benchmark_cfg = read_yaml(BENCHMARK_CONFIG)
    current_hash = file_sha256(BENCHMARK_CONFIG)
    if not RESULTS_ROOT.exists():
        raise FileNotFoundError(f"results root does not exist: {project_path(RESULTS_ROOT, PROJECT_ROOT)}")

    summaries: dict[str, dict[str, Any]] = {}
    for model_dir in sorted(p for p in RESULTS_ROOT.iterdir() if p.is_dir()):
        summaries[model_dir.name] = summarize_model(model_dir, benchmark_cfg, current_hash)

    invalid = [s for s in summaries.values() if not s.get("valid_run", False)]
    if invalid:
        print("[warning] incomplete benchmark run(s) found; summary will still be written:")
        for s in invalid:
            messages = s.get("validation_errors", []) + s.get("warnings", [])
            print(f"[warning] - {s['config_id']}: {'; '.join(messages) or 'no parseable aggregate'}")

    bundles_cfg = read_yaml(BUNDLES_CONFIG)
    forgetting_rows = add_forgetting(summaries, bundles_cfg)
    rows = [summaries[k] for k in sorted(summaries)]

    RUNS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_JSON.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=True, indent=2)
    write_tasks_csv(TASKS_CSV, rows, benchmark_cfg)
    write_main_csv(MAIN_CSV, rows)
    write_forgetting_csv(FORGETTING_CSV, forgetting_rows)

    print(f"[done] wrote {project_path(RUNS_JSON, PROJECT_ROOT)}")
    print(f"[done] wrote {project_path(TASKS_CSV, PROJECT_ROOT)}")
    print(f"[done] wrote {project_path(MAIN_CSV, PROJECT_ROOT)}")
    print(f"[done] wrote {project_path(FORGETTING_CSV, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

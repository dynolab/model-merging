#!/usr/bin/env python3
"""Build one AIM calibration-ablation table from saved benchmark and activation CSVs.

This script is CPU-only. It does not load checkpoints. It assumes every checkpoint was
already evaluated and activation-scored before being deleted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT.parent

DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "aim_calibration_ablation.csv"
DEFAULT_BENCHMARK_MAIN = PROJECT_ROOT / "results" / "benchmark_main.csv"
DEFAULT_BENCHMARK_FORGETTING = PROJECT_ROOT / "results" / "benchmark_forgetting.csv"
DEFAULT_ACTIVATION_PAIR_SUMMARY = PROJECT_ROOT / "results" / "activation_metrics" / "activation_metric_pair_summary.csv"
DEFAULT_ACTIVATION_SCORES = PROJECT_ROOT / "results" / "activation_metrics" / "activation_metric_scores.csv"
DEFAULT_MERGE_MANIFESTS = PROJECT_ROOT / "results" / "merge_manifests"

AIM_BASELINES = {
    "task_arithmetic_best": "aim_on_task_arithmetic",
    "ties_best": "aim_on_ties",
    "dare_best": "aim_on_dare",
    "della_best": "aim_on_della",
    "breadcrumbs_best": "aim_on_breadcrumbs",
}
AIM_TO_PARENT = {v: k for k, v in AIM_BASELINES.items()}
AIM_CALIBRATIONS = {
    "general",
    "instruction",
    "reasoning",
    "uncensored_refusal_like",
    "mix",
}
DOMAINS = {
    "instruction": {
        "bundle_id": "B_if",
        "specialist_id": "instr_only",
        "activation_probe_subset": "instruction",
        "matched_calibration": "instruction",
    },
    "reasoning_math": {
        "bundle_id": "B_reasoning_math",
        "specialist_id": "reasoning_only",
        "activation_probe_subset": "reasoning",
        "matched_calibration": "reasoning",
    },
    "reasoning_long": {
        "bundle_id": "B_reasoning_long",
        "specialist_id": "instr_only",
        "activation_probe_subset": "reasoning",
        "matched_calibration": "reasoning",
    },
    "uncensored": {
        "bundle_id": "B_uncensored",
        "specialist_id": "uncensored_only",
        "activation_probe_subset": "uncensored_refusal_like",
        "matched_calibration": "uncensored_refusal_like",
    },
}
METRIC_LABELS = {
    "mmd_rbf": "mmd",
    "projected_js": "js",
}
OUTPUT_FIELDS = [
    "parent_id",
    "aim_model_id",
    "aim_calibration",
    "focus_domain",
    "bundle_id",
    "specialist_id",
    "calibration_relation",
    "activation_probe_subset",
    "specialist_bundle_score",
    "parent_bundle_score",
    "aim_bundle_score",
    "delta_bundle_score",
    "parent_F_i",
    "aim_F_i",
    "delta_F_i",
    "parent_aggregate",
    "aim_aggregate",
    "delta_aggregate",
    "parent_if_score",
    "aim_if_score",
    "delta_if_score",
    "parent_reasoning_math_score",
    "aim_reasoning_math_score",
    "delta_reasoning_math_score",
    "parent_reasoning_long_score",
    "aim_reasoning_long_score",
    "delta_reasoning_long_score",
    "parent_safety_score",
    "aim_safety_score",
    "delta_safety_score",
    "parent_to_specialist_mmd",
    "aim_to_specialist_mmd",
    "delta_to_specialist_mmd",
    "parent_to_specialist_js",
    "aim_to_specialist_js",
    "delta_to_specialist_js",
    "parent_to_base_mmd",
    "aim_to_base_mmd",
    "delta_to_base_mmd",
    "parent_to_base_js",
    "aim_to_base_js",
    "delta_to_base_js",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_cell(row.get(key)) for key in OUTPUT_FIELDS})


def format_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    return value


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def first_present(row: dict[str, str], names: list[str]) -> str | None:
    for name in names:
        if name in row:
            return row.get(name)
    return None


def diff(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return after - before


def calibration_relation(aim_calibration: str, focus_domain: str) -> str:
    matched = DOMAINS[focus_domain]["matched_calibration"]
    if aim_calibration == matched:
        return "matched"
    if aim_calibration == "general":
        return "general_far"
    if aim_calibration == "mix":
        return "mixed"
    return "off_domain"


def parse_aim_id_from_name(aim_model_id: str) -> tuple[str | None, str | None]:
    if "__calib_" in aim_model_id:
        base_aim_id, calibration = aim_model_id.split("__calib_", 1)
    else:
        base_aim_id, calibration = aim_model_id, "mix"
    parent_id = AIM_TO_PARENT.get(base_aim_id)
    if parent_id is None or calibration not in AIM_CALIBRATIONS:
        return None, None
    return parent_id, calibration


def load_aim_index(manifest_dir: Path, benchmark_model_ids: set[str]) -> list[dict[str, str]]:
    items: dict[str, dict[str, str]] = {}
    if manifest_dir.exists():
        for path in sorted(manifest_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            method_family = str(data.get("method_family", ""))
            parent_id = data.get("parent_config_id")
            calibration_id = data.get("calibration_id")
            model_id = str(data.get("config_id") or data.get("baseline_id") or path.stem)
            if not parent_id or not calibration_id:
                fallback_parent, fallback_calib = parse_aim_id_from_name(model_id)
                parent_id = parent_id or fallback_parent
                calibration_id = calibration_id or fallback_calib
            if method_family and method_family != "AIM":
                continue
            if not parent_id or not calibration_id or str(calibration_id) not in AIM_CALIBRATIONS:
                continue
            if model_id not in benchmark_model_ids:
                # Keep only AIM checkpoints that have at least benchmark_main presence.
                continue
            items[model_id] = {
                "parent_id": str(parent_id),
                "aim_model_id": model_id,
                "aim_calibration": str(calibration_id),
            }

    # Fallback for cases where manifests were not copied but benchmark rows are present.
    for model_id in sorted(benchmark_model_ids):
        parent_id, calibration = parse_aim_id_from_name(model_id)
        if parent_id and calibration and model_id not in items:
            items[model_id] = {
                "parent_id": parent_id,
                "aim_model_id": model_id,
                "aim_calibration": calibration,
            }
    return [items[key] for key in sorted(items)]


def load_benchmark_main(path: Path) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for row in read_csv_rows(path):
        model_id = str(first_present(row, ["Config id", "config_id", "model_id", "run_id"]) or "").strip()
        if not model_id:
            continue
        out[model_id] = {
            "aggregate": parse_float(first_present(row, ["Aggregate", "aggregate"])),
            "if_score": parse_float(first_present(row, ["IF", "instruction_following", "axis:instruction_following"])),
            "reasoning_math_score": parse_float(first_present(row, ["Reasoning", "reasoning_math", "axis:reasoning_math"])),
            "reasoning_long_score": parse_float(first_present(row, ["reasoning_long", "axis:reasoning_long"])),
            "safety_score": parse_float(first_present(row, ["Safety", "safety", "axis:safety"])),
        }
    return out


def load_forgetting(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_csv_rows(path):
        model_id = str(first_present(row, ["Config id", "config_id", "model_id", "run_id"]) or "").strip()
        bundle_id = str(first_present(row, ["Domain (i)", "domain", "bundle_id"]) or "").strip()
        if not model_id or not bundle_id:
            continue
        out[(model_id, bundle_id)] = {
            "specialist_id": str(first_present(row, ["Specialist id", "specialist_id"]) or "").strip(),
            "specialist_score": parse_float(first_present(row, ["Specialist score", "specialist_score"])),
            "model_score": parse_float(first_present(row, ["Merged/model score", "model_score", "score"])),
            "forgetting": parse_float(first_present(row, ["F_i", "forgetting"])),
        }
    return out


def load_activation_pair_summary(path: Path) -> dict[tuple[str, str, str, str], float]:
    out: dict[tuple[str, str, str, str], float] = {}
    for row in read_csv_rows(path):
        model_id = str(row.get("model_id", "")).strip()
        target_id = str(row.get("target_id", "")).strip()
        subset = str(row.get("calib_subset", "")).strip()
        metric = str(row.get("metric", "")).strip()
        value = parse_float(first_present(row, ["layer_mean_distance", "mean_distance", "value_mean", "mean"]))
        if model_id and target_id and subset and metric and value is not None:
            out[(model_id, target_id, subset, metric)] = value
    return out


def load_activation_scores_as_summary(path: Path) -> dict[tuple[str, str, str, str], float]:
    groups: dict[tuple[str, str, str, str], list[float]] = {}
    for row in read_csv_rows(path):
        model_id = str(row.get("model_id", "")).strip()
        target_id = str(row.get("target_id", "")).strip()
        subset = str(row.get("calib_subset", "")).strip()
        metric = str(row.get("metric", "")).strip()
        value = parse_float(row.get("value"))
        if model_id and target_id and subset and metric and value is not None:
            groups.setdefault((model_id, target_id, subset, metric), []).append(value)
    return {key: sum(values) / len(values) for key, values in groups.items() if values}


def load_activation_summary(pair_summary: Path, raw_scores: Path) -> dict[tuple[str, str, str, str], float]:
    summary = load_activation_pair_summary(pair_summary)
    if summary:
        return summary
    return load_activation_scores_as_summary(raw_scores)


def activation_distance(
    activations: dict[tuple[str, str, str, str], float],
    *,
    model_id: str,
    target_id: str,
    subset: str,
    metric: str,
) -> float | None:
    return activations.get((model_id, target_id, subset, metric))


def axis_fields(prefix: str, values: dict[str, float | None]) -> dict[str, float | None]:
    return {
        f"{prefix}_aggregate": values.get("aggregate"),
        f"{prefix}_if_score": values.get("if_score"),
        f"{prefix}_reasoning_math_score": values.get("reasoning_math_score"),
        f"{prefix}_reasoning_long_score": values.get("reasoning_long_score"),
        f"{prefix}_safety_score": values.get("safety_score"),
    }


def build_rows(
    *,
    aim_items: list[dict[str, str]],
    benchmark: dict[str, dict[str, float | None]],
    forgetting: dict[tuple[str, str], dict[str, Any]],
    activations: dict[tuple[str, str, str, str], float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in aim_items:
        parent_id = item["parent_id"]
        aim_model_id = item["aim_model_id"]
        aim_calibration = item["aim_calibration"]
        parent_axes = benchmark.get(parent_id, {})
        aim_axes = benchmark.get(aim_model_id, {})
        axis_part: dict[str, Any] = {}
        axis_part.update(axis_fields("parent", parent_axes))
        axis_part.update(axis_fields("aim", aim_axes))
        for name in ["aggregate", "if_score", "reasoning_math_score", "reasoning_long_score", "safety_score"]:
            axis_part[f"delta_{name}"] = diff(aim_axes.get(name), parent_axes.get(name))

        for focus_domain, domain_cfg in DOMAINS.items():
            bundle_id = domain_cfg["bundle_id"]
            expected_specialist = domain_cfg["specialist_id"]
            subset = domain_cfg["activation_probe_subset"]
            parent_f = forgetting.get((parent_id, bundle_id), {})
            aim_f = forgetting.get((aim_model_id, bundle_id), {})
            specialist_id = str(aim_f.get("specialist_id") or parent_f.get("specialist_id") or expected_specialist)
            specialist_score = aim_f.get("specialist_score")
            if specialist_score is None:
                specialist_score = parent_f.get("specialist_score")
            parent_score = parent_f.get("model_score")
            aim_score = aim_f.get("model_score")
            parent_forgetting = parent_f.get("forgetting")
            aim_forgetting = aim_f.get("forgetting")

            row: dict[str, Any] = {
                "parent_id": parent_id,
                "aim_model_id": aim_model_id,
                "aim_calibration": aim_calibration,
                "focus_domain": focus_domain,
                "bundle_id": bundle_id,
                "specialist_id": specialist_id,
                "calibration_relation": calibration_relation(aim_calibration, focus_domain),
                "activation_probe_subset": subset,
                "specialist_bundle_score": specialist_score,
                "parent_bundle_score": parent_score,
                "aim_bundle_score": aim_score,
                "delta_bundle_score": diff(aim_score, parent_score),
                "parent_F_i": parent_forgetting,
                "aim_F_i": aim_forgetting,
                "delta_F_i": diff(aim_forgetting, parent_forgetting),
            }
            row.update(axis_part)

            for metric, label in METRIC_LABELS.items():
                parent_spec = activation_distance(
                    activations,
                    model_id=parent_id,
                    target_id=specialist_id,
                    subset=subset,
                    metric=metric,
                )
                aim_spec = activation_distance(
                    activations,
                    model_id=aim_model_id,
                    target_id=specialist_id,
                    subset=subset,
                    metric=metric,
                )
                parent_base = activation_distance(
                    activations,
                    model_id=parent_id,
                    target_id="base",
                    subset=subset,
                    metric=metric,
                )
                aim_base = activation_distance(
                    activations,
                    model_id=aim_model_id,
                    target_id="base",
                    subset=subset,
                    metric=metric,
                )
                row[f"parent_to_specialist_{label}"] = parent_spec
                row[f"aim_to_specialist_{label}"] = aim_spec
                row[f"delta_to_specialist_{label}"] = diff(aim_spec, parent_spec)
                row[f"parent_to_base_{label}"] = parent_base
                row[f"aim_to_base_{label}"] = aim_base
                row[f"delta_to_base_{label}"] = diff(aim_base, parent_base)
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one AIM calibration-ablation CSV from saved result tables")
    parser.add_argument("--benchmark-main", default=str(DEFAULT_BENCHMARK_MAIN))
    parser.add_argument("--benchmark-forgetting", default=str(DEFAULT_BENCHMARK_FORGETTING))
    parser.add_argument("--activation-pair-summary", default=str(DEFAULT_ACTIVATION_PAIR_SUMMARY))
    parser.add_argument("--activation-scores", default=str(DEFAULT_ACTIVATION_SCORES))
    parser.add_argument("--merge-manifests", default=str(DEFAULT_MERGE_MANIFESTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = load_benchmark_main(Path(args.benchmark_main))
    forgetting = load_forgetting(Path(args.benchmark_forgetting))
    activations = load_activation_summary(Path(args.activation_pair_summary), Path(args.activation_scores))
    aim_items = load_aim_index(Path(args.merge_manifests), set(benchmark.keys()))
    rows = build_rows(aim_items=aim_items, benchmark=benchmark, forgetting=forgetting, activations=activations)
    rows.sort(key=lambda row: (str(row["parent_id"]), str(row["aim_calibration"]), str(row["focus_domain"])))
    output = Path(args.output)
    write_csv_rows(output, rows)

    expected_full = len(AIM_BASELINES) * len(AIM_CALIBRATIONS) * len(DOMAINS)
    print(f"[done] wrote {output.relative_to(PROJECT_ROOT) if output.is_relative_to(PROJECT_ROOT) else output}")
    print(f"[summary] rows={len(rows)} expected_full_rows={expected_full}")
    if not activations:
        print("[warning] no activation summary rows found; activation columns are empty")
    if not forgetting:
        print("[warning] no benchmark forgetting rows found; bundle score columns are empty")
    if len(rows) != expected_full:
        print("[warning] incomplete AIM calibration ablation table; this is expected for partial runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

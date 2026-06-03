#!/usr/bin/env python3
"""Run the final benchmark or the one-time selection benchmark for one model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.common import (
    BENCHMARK_CONFIG,
    BENCHMARK_RESULTS_ROOT,
    DEFAULT_GPU_IDS,
    PROJECT_ROOT,
    SELECTION_CONFIG,
    SELECTION_RESULTS_ROOT,
    check_pinned_safety_eval_commit,
    collect_environment,
    enabled_axes,
    parse_gpu_ids,
    safety_eval_root_from_config,
    selection_benchmark_section,
    task_entries,
)
from benchmark.model_registry import materialize_eval_model, resolve_model_request
from benchmark.runners import (
    apply_safety_eval_dataset_pins,
    build_lm_eval_command,
    build_safety_eval_command,
    ensure_openai_import_placeholder,
    ensure_safety_eval_import_paths,
    generate_qwen3_safety_template,
    require_vllm,
    run_cmd,
)
from common.io import file_sha256, project_path, read_yaml


LM_EVAL_TASK_OVERRIDE_KEYS = {
    "batch_size",
    "gen_kwargs",
    "limit",
    "lm_eval_model_args",
    "vllm_batch_size",
    "vllm_model_args",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--run", help="Pinned model alias from benchmark.yaml")
    model_group.add_argument("--model", help="HF repo id pinned in benchmark.yaml, or local checkpoint path")
    parser.add_argument("--config-id", help="Result folder name. Defaults to --run alias or sanitized model path.")
    parser.add_argument("--selection", action="store_true", help="Use benchmark_selection.yaml and results/benchmark_selection")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing result folder before running")
    parser.add_argument("--gpu-ids", default=DEFAULT_GPU_IDS, help="Comma-separated CUDA device ids, e.g. 0 or 0,1")
    return parser.parse_args()


def model_specific_lm_eval_args(
    cfg: dict[str, Any],
    *,
    run_alias: str | None,
    requested_model: str,
    revision: str | None,
) -> dict[str, Any]:
    models = cfg.get("models")
    if not isinstance(models, dict):
        return {}

    spec: dict[str, Any] | None = None
    if run_alias:
        candidate = models.get(run_alias)
        if isinstance(candidate, dict):
            spec = candidate
    else:
        for candidate in models.values():
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("repo")) == requested_model and str(candidate.get("revision")) == str(revision):
                spec = candidate
                break

    if not isinstance(spec, dict):
        return {}
    args = spec.get("lm_eval_model_args", {})
    if args is None:
        return {}
    if not isinstance(args, dict):
        raise ValueError("models.<alias>.lm_eval_model_args must be a mapping when defined")
    return dict(args)


def effective_chat_format(
    cfg: dict[str, Any],
    *,
    run_alias: str | None,
    requested_model: str,
    revision: str | None,
) -> dict[str, Any]:
    chat_format = dict(cfg.get("chat_format", {}) or {})
    global_args = chat_format.get("lm_eval_model_args", {}) or {}
    if not isinstance(global_args, dict):
        raise ValueError("chat_format.lm_eval_model_args must be a mapping when defined")
    merged_args = dict(global_args)
    merged_args.update(
        model_specific_lm_eval_args(
            cfg,
            run_alias=run_alias,
            requested_model=requested_model,
            revision=revision,
        )
    )
    if merged_args:
        chat_format["lm_eval_model_args"] = merged_args
    else:
        chat_format.pop("lm_eval_model_args", None)
    return chat_format


def qwen_enable_thinking_arg(chat_format: dict[str, Any]) -> bool | None:
    args = chat_format.get("lm_eval_model_args", {})
    if not isinstance(args, dict) or "enable_thinking" not in args:
        return None
    value = args["enable_thinking"]
    if not isinstance(value, bool):
        raise ValueError("enable_thinking must be a boolean when defined")
    return value


def chat_format_for_axis(chat_format: dict[str, Any], axis_cfg: dict[str, Any]) -> dict[str, Any]:
    axis_chat_format = dict(chat_format)
    axis_args = axis_cfg.get("lm_eval_model_args", {}) or {}
    if not axis_args:
        return axis_chat_format
    if not isinstance(axis_args, dict):
        raise ValueError("axes.<axis>.lm_eval_model_args must be a mapping when defined")

    merged_args = dict(axis_chat_format.get("lm_eval_model_args", {}) or {})
    merged_args.update(axis_args)
    axis_chat_format["lm_eval_model_args"] = merged_args
    return axis_chat_format


def axis_cfg_for_lm_eval_task(axis_cfg: dict[str, Any], task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    task_axis_cfg = deepcopy(axis_cfg)
    clean_task = {key: deepcopy(value) for key, value in task.items() if key not in LM_EVAL_TASK_OVERRIDE_KEYS}
    task_axis_cfg["tasks"] = [clean_task]

    for key in LM_EVAL_TASK_OVERRIDE_KEYS:
        if key not in task:
            continue
        task_value = deepcopy(task[key])
        if key in {"gen_kwargs", "lm_eval_model_args", "vllm_model_args"}:
            if task_value is None:
                task_axis_cfg.pop(key, None)
                continue
            if not isinstance(task_value, dict):
                raise ValueError(f"Task-level {key} override must be a mapping")
            base_value = task_axis_cfg.get(key, {}) or {}
            if not isinstance(base_value, dict):
                raise ValueError(f"Axis-level {key} must be a mapping when task overrides it")
            merged_value = dict(base_value)
            merged_value.update(task_value)
            task_axis_cfg[key] = merged_value
        else:
            task_axis_cfg[key] = task_value

    return task_axis_cfg, clean_task


def lm_eval_command_group_key(axis_cfg: dict[str, Any]) -> str:
    relevant_keys = [
        "batch_size",
        "gen_kwargs",
        "limit",
        "lm_eval_backend",
        "lm_eval_model_args",
        "log_samples",
        "vllm_batch_size",
        "vllm_model_args",
    ]
    relevant = {key: axis_cfg[key] for key in relevant_keys if key in axis_cfg}
    return json.dumps(relevant, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def lm_eval_task_groups(axis_cfg: dict[str, Any], tasks: list[dict[str, Any]]) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    groups: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    index_by_key: dict[str, int] = {}

    for task in tasks:
        task_axis_cfg, clean_task = axis_cfg_for_lm_eval_task(axis_cfg, task)
        key = lm_eval_command_group_key(task_axis_cfg)
        if key not in index_by_key:
            group_cfg = deepcopy(task_axis_cfg)
            group_cfg["tasks"] = []
            index_by_key[key] = len(groups)
            groups.append((group_cfg, []))

        group_cfg, group_tasks = groups[index_by_key[key]]
        group_cfg["tasks"].append(clean_task)
        group_tasks.append(clean_task)

    return groups


def prepare_safety_runtime(
    cfg: dict[str, Any],
    eval_model_path: str,
    env: dict[str, str],
    chat_format: dict[str, Any],
) -> dict[str, Any]:
    safety_root = safety_eval_root_from_config(cfg)
    require_vllm()
    template_info = generate_qwen3_safety_template(
        Path(eval_model_path),
        enable_thinking=qwen_enable_thinking_arg(chat_format),
    )
    ensure_openai_import_placeholder(env)
    ensure_safety_eval_import_paths(env, safety_root)
    return {
        "root": safety_root,
        "template": template_info,
        "dataset_pins": apply_safety_eval_dataset_pins(cfg, env),
    }


def build_manifest(
    *,
    cfg_path: Path,
    config_id: str,
    requested_model: str,
    model_identity: dict[str, Any],
    cfg: dict[str, Any],
    effective_chat_format: dict[str, Any],
    gpu_ids: str,
    gpu_count: int,
    safety_info: dict[str, Any] | None,
) -> dict[str, Any]:
    if safety_info:
        safety_root = safety_info["root"]
    elif "safety_eval" in cfg:
        safety_root = safety_eval_root_from_config(cfg)
    else:
        safety_root = None
    manifest_warnings = []
    local_manifest = model_identity.get("local_model_manifest")
    if isinstance(local_manifest, dict):
        manifest_warnings.extend(str(w) for w in local_manifest.get("warnings", []) if w)

    manifest = {
        "config_id": config_id,
        "model": requested_model,
        "model_identity": model_identity,
        "benchmark_config": project_path(cfg_path, PROJECT_ROOT),
        "benchmark_config_sha256": file_sha256(cfg_path),
        "benchmark_id": cfg.get("benchmark_id"),
        "chat_format": cfg.get("chat_format", {}),
        "effective_chat_format": effective_chat_format,
        "gpu_ids": gpu_ids,
        "gpu_count": gpu_count,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": collect_environment(gpu_ids, safety_root),
        "warnings": manifest_warnings,
    }
    if safety_info:
        template_info = dict(safety_info["template"])
        template_info["path"] = project_path(template_info["path"], PROJECT_ROOT)
        manifest["safety_eval_template"] = template_info
        manifest["safety_eval_dataset_pins"] = safety_info["dataset_pins"]
    return manifest


def run_final_benchmark(
    *,
    cfg: dict[str, Any],
    chat_format: dict[str, Any],
    eval_model_path: str,
    run_root: Path,
    gpu_count: int,
    env: dict[str, str],
    manifest: dict[str, Any],
    safety_info: dict[str, Any] | None,
) -> int:
    status = 0

    for axis_name, axis_cfg in enabled_axes(cfg).items():
        evaluator = axis_cfg.get("evaluator")
        tasks = task_entries(axis_cfg)
        axis_out = run_root / axis_name
        axis_out.mkdir(parents=True, exist_ok=True)

        if evaluator == "lm_eval":
            task_groups = lm_eval_task_groups(axis_cfg, tasks)
            for group_index, (group_axis_cfg, group_tasks) in enumerate(task_groups, start=1):
                axis_chat_format = chat_format_for_axis(chat_format, group_axis_cfg)
                cmd = build_lm_eval_command(
                    eval_model_path,
                    group_axis_cfg,
                    group_tasks,
                    axis_out,
                    axis_chat_format,
                    gpu_count,
                )
                entry = {
                    "axis": axis_name,
                    "axis_part": group_index,
                    "axis_parts": len(task_groups),
                    "evaluator": evaluator,
                    "tasks": group_tasks,
                    "effective_chat_format": axis_chat_format,
                    "effective_axis_config": {
                        key: group_axis_cfg[key]
                        for key in [
                            "batch_size",
                            "gen_kwargs",
                            "limit",
                            "lm_eval_backend",
                            "lm_eval_model_args",
                            "log_samples",
                            "vllm_batch_size",
                            "vllm_model_args",
                        ]
                        if key in group_axis_cfg
                    },
                }
                ret = run_cmd(cmd, cwd=None, env=env)
                entry["exit_code"] = ret
                manifest.setdefault("axis_runs", []).append(entry)
                if ret != 0:
                    print(f"[error] axis {axis_name} part {group_index} failed with exit code {ret}", file=sys.stderr)
                    status = ret
                    break
            if status != 0:
                break
            continue
        elif evaluator == "safety_eval":
            axis_chat_format = chat_format
            if safety_info is None:
                raise RuntimeError("safety axis enabled but safety runtime was not prepared")
            actual_commit, commit_warnings = check_pinned_safety_eval_commit(cfg, safety_info["root"])
            for warning in commit_warnings:
                print(f"[warning] {warning}")
                manifest.setdefault("warnings", []).append(warning)
            manifest.setdefault("verified_commits", {})["safety_eval"] = {
                "expected": cfg.get("safety_eval", {}).get("pinned_commit"),
                "actual": actual_commit,
            }
            cmd, cwd = build_safety_eval_command(
                eval_model_path,
                axis_cfg,
                tasks,
                axis_out,
                safety_info["root"],
                Path(safety_info["template"]["path"]),
            )
        else:
            raise ValueError(f"Unsupported evaluator for axis {axis_name}: {evaluator}")

        entry = {
            "axis": axis_name,
            "evaluator": evaluator,
            "tasks": tasks,
            "effective_chat_format": axis_chat_format,
        }
        ret = run_cmd(cmd, cwd=cwd, env=env)
        entry["exit_code"] = ret
        manifest.setdefault("axis_runs", []).append(entry)
        if ret != 0:
            print(f"[error] axis {axis_name} failed with exit code {ret}", file=sys.stderr)
            status = ret
            break
    return status


def run_selection_benchmark(
    *,
    cfg: dict[str, Any],
    chat_format: dict[str, Any],
    eval_model_path: str,
    run_root: Path,
    gpu_count: int,
    env: dict[str, str],
    manifest: dict[str, Any],
) -> int:
    section = selection_benchmark_section(cfg, require_metric=True)
    tasks = task_entries(section)
    out_dir = run_root / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_lm_eval_command(eval_model_path, section, tasks, out_dir, chat_format, gpu_count)
    ret = run_cmd(cmd, cwd=None, env=env)
    manifest["selection_run"] = {"evaluator": "lm_eval", "tasks": tasks, "exit_code": ret}
    if ret != 0:
        print(f"[error] selection benchmark failed with exit code {ret}", file=sys.stderr)
    return ret


def result_dir(results_root: Path, config_id: str) -> Path:
    candidate = Path(config_id)
    if candidate.is_absolute() or candidate.name != config_id or config_id in {"", ".", ".."}:
        raise ValueError("--config-id must be a single result folder name, not a path")
    return results_root / config_id


def main() -> int:
    args = parse_args()
    if args.selection and args.run:
        raise ValueError("--selection is for local merge candidates; use --model outputs/merged/<config-id>")

    cfg_path = SELECTION_CONFIG if args.selection else BENCHMARK_CONFIG
    results_root = SELECTION_RESULTS_ROOT if args.selection else BENCHMARK_RESULTS_ROOT
    cfg = read_yaml(cfg_path)
    if args.selection:
        selection_benchmark_section(cfg, require_metric=True)
    else:
        for axis_cfg in enabled_axes(cfg).values():
            task_entries(axis_cfg, require_metric=True)

    gpu_ids = args.gpu_ids
    gpu_id_list = parse_gpu_ids(gpu_ids)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    requested_model, config_id, revision = resolve_model_request(
        cfg,
        run_alias=args.run,
        model=args.model,
        config_id=args.config_id,
    )
    run_root = result_dir(results_root, config_id)
    if run_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Result directory already exists: {project_path(run_root, PROJECT_ROOT)}. Use --overwrite to replace it.")
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    eval_model_path, model_identity = materialize_eval_model(
        requested_model,
        revision,
        materialize_local=not args.selection,
        hash_local_files=not args.selection,
    )
    chat_format = effective_chat_format(
        cfg,
        run_alias=args.run,
        requested_model=requested_model,
        revision=revision,
    )
    safety_enabled = False if args.selection else any(axis.get("evaluator") == "safety_eval" for axis in enabled_axes(cfg).values())
    safety_info = prepare_safety_runtime(cfg, eval_model_path, env, chat_format) if safety_enabled else None
    manifest = build_manifest(
        cfg_path=cfg_path,
        config_id=config_id,
        requested_model=requested_model,
        model_identity=model_identity,
        cfg=cfg,
        effective_chat_format=chat_format,
        gpu_ids=gpu_ids,
        gpu_count=len(gpu_id_list),
        safety_info=safety_info,
    )

    if args.selection:
        status = run_selection_benchmark(
            cfg=cfg,
            chat_format=chat_format,
            eval_model_path=eval_model_path,
            run_root=run_root,
            gpu_count=len(gpu_id_list),
            env=env,
            manifest=manifest,
        )
    else:
        status = run_final_benchmark(
            cfg=cfg,
            chat_format=chat_format,
            eval_model_path=eval_model_path,
            run_root=run_root,
            gpu_count=len(gpu_id_list),
            env=env,
            manifest=manifest,
            safety_info=safety_info,
        )

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["exit_code"] = status
    with (run_root / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)
    print(f"\n[done] manifest: {project_path(run_root / 'run_manifest.json', PROJECT_ROOT)}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

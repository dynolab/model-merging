#!/usr/bin/env python3
"""Run the frozen custom benchmark protocol for one model."""

from __future__ import annotations

import argparse
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

from common.io import file_sha256, project_path, read_yaml
from benchmark.common import (
    BENCHMARK_CONFIG,
    DEFAULT_GPU_IDS,
    PROJECT_ROOT,
    RESULTS_ROOT,
    check_pinned_safety_eval_commit,
    collect_environment,
    enabled_axes,
    parse_gpu_ids,
    safety_eval_root_from_config,
    task_entries,
)
from benchmark.model_registry import materialize_eval_model, resolve_model_request
from benchmark.runners import (
    build_lm_eval_command,
    build_safety_eval_command,
    ensure_openai_import_placeholder,
    ensure_safety_eval_import_paths,
    generate_qwen3_no_think_template,
    require_vllm,
    run_cmd,
    apply_safety_eval_dataset_pins,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen custom benchmark protocol")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--run", help="Pinned model alias from code/cfg/benchmark/benchmark.yaml")
    model_group.add_argument("--model", help="HF repo id pinned in config, or local checkpoint path")
    parser.add_argument("--config-id", help="Result folder name. Defaults to --run alias or sanitized model path.")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing result folder before running")
    parser.add_argument("--gpu-ids", default=DEFAULT_GPU_IDS, help="Comma-separated CUDA device ids, e.g. 0 or 0,1.")
    return parser.parse_args()


def prepare_safety_runtime(cfg: dict[str, Any], eval_model_path: str, env: dict[str, str]) -> dict[str, Any]:
    safety_root = safety_eval_root_from_config(cfg)
    require_vllm()
    template_info = generate_qwen3_no_think_template(Path(eval_model_path))
    ensure_openai_import_placeholder(env)
    ensure_safety_eval_import_paths(env, safety_root)
    return {
        "root": safety_root,
        "template": template_info,
        "dataset_pins": apply_safety_eval_dataset_pins(cfg, env),
    }


def build_manifest(
    *,
    config_id: str,
    requested_model: str,
    model_identity: dict[str, Any],
    cfg: dict[str, Any],
    gpu_ids: str,
    gpu_count: int,
    safety_info: dict[str, Any] | None,
) -> dict[str, Any]:
    safety_root = safety_info["root"] if safety_info else safety_eval_root_from_config(cfg)
    manifest_warnings = []
    local_manifest = model_identity.get("local_model_manifest")
    if isinstance(local_manifest, dict):
        manifest_warnings.extend(str(w) for w in local_manifest.get("warnings", []) if w)
    template_info = None
    if safety_info:
        template_info = dict(safety_info["template"])
        if "path" in template_info:
            template_info["path"] = project_path(template_info["path"], PROJECT_ROOT)
    return {
        "config_id": config_id,
        "model": requested_model,
        "model_identity": model_identity,
        "benchmark_config": project_path(BENCHMARK_CONFIG, PROJECT_ROOT),
        "benchmark_config_sha256": file_sha256(BENCHMARK_CONFIG),
        "benchmark_id": cfg.get("benchmark_id"),
        "chat_format": cfg.get("chat_format", {}),
        "safety_eval_template": template_info,
        "safety_eval_dataset_pins": safety_info["dataset_pins"] if safety_info else None,
        "gpu_ids": gpu_ids,
        "gpu_count": gpu_count,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": collect_environment(gpu_ids, safety_root),
        "warnings": manifest_warnings,
        "axis_runs": [],
    }


def run_axes(
    *,
    cfg: dict[str, Any],
    eval_model_path: str,
    run_root: Path,
    gpu_count: int,
    env: dict[str, str],
    manifest: dict[str, Any],
    safety_info: dict[str, Any] | None,
) -> int:
    status = 0
    chat_format = cfg.get("chat_format", {})
    for axis_name, axis_cfg in enabled_axes(cfg).items():
        evaluator = axis_cfg.get("evaluator")
        tasks = task_entries(axis_cfg)
        task_names = [t["runner_task_name"] for t in tasks]
        axis_out = run_root / axis_name
        axis_out.mkdir(parents=True, exist_ok=True)

        if evaluator == "lm_eval":
            cmd = build_lm_eval_command(eval_model_path, axis_cfg, task_names, axis_out, chat_format, gpu_count)
            cwd = None
        elif evaluator == "safety_eval":
            if safety_info is None:
                raise RuntimeError("internal error: safety axis enabled but safety runtime was not prepared")
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
                task_names,
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
        }
        ret = run_cmd(cmd, cwd=cwd, env=env)
        entry["exit_code"] = ret
        manifest["axis_runs"].append(entry)
        if ret != 0:
            print(f"[error] axis {axis_name} failed with exit code {ret}", file=sys.stderr)
            status = ret
            break
    return status


def main() -> int:
    args = parse_args()
    cfg = read_yaml(BENCHMARK_CONFIG)
    gpu_ids = args.gpu_ids
    gpu_id_list = parse_gpu_ids(gpu_ids)
    command_env = os.environ.copy()
    command_env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    requested_model, config_id, revision = resolve_model_request(
        cfg,
        run_alias=args.run,
        model=args.model,
        config_id=args.config_id,
    )
    run_root = RESULTS_ROOT / config_id
    if run_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Result directory already exists: {project_path(run_root, PROJECT_ROOT)}. Use --overwrite to replace it.")
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    eval_model_path, model_identity = materialize_eval_model(requested_model, revision)
    safety_enabled = any(axis.get("evaluator") == "safety_eval" for axis in enabled_axes(cfg).values())
    safety_info = prepare_safety_runtime(cfg, eval_model_path, command_env) if safety_enabled else None
    manifest = build_manifest(
        config_id=config_id,
        requested_model=requested_model,
        model_identity=model_identity,
        cfg=cfg,
        gpu_ids=gpu_ids,
        gpu_count=len(gpu_id_list),
        safety_info=safety_info,
    )

    status = run_axes(
        cfg=cfg,
        eval_model_path=eval_model_path,
        run_root=run_root,
        gpu_count=len(gpu_id_list),
        env=command_env,
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

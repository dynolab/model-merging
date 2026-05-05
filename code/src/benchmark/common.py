from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from common.io import git_commit, project_path


CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT.parent
BENCHMARK_CONFIG = CODE_ROOT / "cfg" / "benchmark" / "benchmark.yaml"
BUNDLES_CONFIG = CODE_ROOT / "cfg" / "benchmark" / "bundles" / "if_reason_uncensored.yaml"
MATERIALIZED_MODELS_ROOT = PROJECT_ROOT / ".cache" / "benchmark_models"
TEMPLATE_ROOT = PROJECT_ROOT / ".cache" / "benchmark_templates"
SAFETY_DATASET_PIN_ROOT = PROJECT_ROOT / ".cache" / "safety_eval_dataset_pins"
RESULTS_ROOT = PROJECT_ROOT / "results" / "benchmark"
RUNS_JSON = PROJECT_ROOT / "results" / "benchmark_runs.json"
TASKS_CSV = PROJECT_ROOT / "results" / "benchmark_tasks.csv"
MAIN_CSV = PROJECT_ROOT / "results" / "benchmark_main.csv"
FORGETTING_CSV = PROJECT_ROOT / "results" / "benchmark_forgetting.csv"

LM_EVAL_BIN = "lm_eval"
DEFAULT_GPU_IDS = "0"


def parse_gpu_ids(gpu_ids: str) -> list[int]:
    parts = [x.strip() for x in str(gpu_ids).split(",") if x.strip()]
    if not parts:
        raise ValueError("gpu ids must contain at least one id")
    return [int(x) for x in parts]


def safety_eval_root_from_config(cfg: dict[str, Any]) -> Path:
    rel = cfg.get("safety_eval", {}).get("local_root", "external/safety-eval-fork")
    p = Path(str(rel)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def check_pinned_safety_eval_commit(cfg: dict[str, Any], safety_root: Path) -> tuple[str | None, list[str]]:
    expected = cfg.get("safety_eval", {}).get("pinned_commit")
    actual = git_commit(safety_root)
    warnings: list[str] = []
    if not expected:
        warnings.append("benchmark config does not define safety_eval.pinned_commit")
    if actual is None:
        warnings.append(f"could not read git commit for safety-eval root: {project_path(safety_root, PROJECT_ROOT)}")
    elif expected and actual != expected:
        warnings.append(f"safety-eval commit mismatch: expected {expected}, got {actual} at {project_path(safety_root, PROJECT_ROOT)}")
    return actual, warnings


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_version(cmd: str) -> str | None:
    exe = shutil.which(cmd)
    if exe is None:
        return None
    for flag in ("--version", "version"):
        try:
            proc = subprocess.run(
                [exe, flag],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=20,
                check=False,
            )
            out = proc.stdout.strip()
            if out:
                return out.splitlines()[0]
        except Exception:
            continue
    return None


def nvidia_driver_version() -> str | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    versions = sorted({line.strip() for line in proc.stdout.splitlines() if line.strip()})
    return ", ".join(versions) if versions else None


def collect_environment(gpu_ids: str, safety_root: Path) -> dict[str, Any]:
    visible_ids = parse_gpu_ids(gpu_ids)
    env: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "lm_eval_bin": LM_EVAL_BIN,
        "lm_eval_bin_version": command_version(LM_EVAL_BIN),
        "pyyaml": package_version("PyYAML"),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "datasets": package_version("datasets"),
        "huggingface_hub": package_version("huggingface_hub"),
        "lm_eval": package_version("lm_eval"),
        "vllm": package_version("vllm"),
        "safety_eval_commit": git_commit(safety_root),
        "nvidia_driver": nvidia_driver_version(),
    }
    try:
        import torch  # type: ignore

        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        env["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu_count"] = torch.cuda.device_count()
            env["requested_gpu_ids"] = visible_ids
            gpu_details = []
            for idx in visible_ids:
                props = torch.cuda.get_device_properties(idx)
                gpu_details.append(
                    {
                        "gpu_id": idx,
                        "gpu_name": torch.cuda.get_device_name(idx),
                        "gpu_total_memory_gb": round(props.total_memory / (1024**3), 2),
                    }
                )
            env["gpus"] = gpu_details
    except Exception as exc:  # pragma: no cover
        env["torch_probe_error"] = repr(exc)
    return env


def enabled_axes(cfg: dict[str, Any]) -> dict[str, Any]:
    axes = cfg.get("axes")
    if not isinstance(axes, dict):
        raise ValueError("benchmark config must contain mapping field: axes")
    return {name: axis for name, axis in axes.items() if axis.get("enabled", False)}


def task_entries(axis_cfg: dict[str, Any], *, require_metric: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in axis_cfg.get("tasks", []):
        if not isinstance(task, dict):
            raise ValueError(f"Task entries must be mappings: {task!r}")
        task_id = str(task.get("task_id") or task.get("runner_task_name"))
        if not task_id:
            raise ValueError(f"Task entry must define task_id: {task!r}")
        if require_metric and not task.get("metric"):
            raise ValueError(f"Task {task_id} must define metric explicitly in the benchmark config")
        normalized = dict(task)
        normalized["task_id"] = task_id
        normalized["runner_task_name"] = str(task.get("runner_task_name") or task_id)
        if task.get("metric") is not None:
            normalized["metric"] = str(task["metric"])
        out.append(normalized)
    return out

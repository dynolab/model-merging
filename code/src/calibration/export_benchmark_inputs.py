#!/usr/bin/env python3
"""Export one frozen benchmark-input snapshot for calibration exclusion.

The benchmark runners still use native lm-eval and safety-eval for scoring. This
snapshot is an audit/decontamination artifact: it records the benchmark text that
must be excluded from calibration data.
"""

from __future__ import annotations

import json
import os
import sys
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset as hf_load_dataset

CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT.parent
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
BENCHMARK_CONFIG = CODE_ROOT / "cfg" / "benchmark" / "benchmark.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "benchmark" / "benchmark_inputs.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "benchmark" / "benchmark_inputs_manifest.json"

from common.io import file_sha256, git_commit, project_path, read_yaml, write_jsonl
from common.text import TEXT_MIN_CHARS, normalize_text, sha256_text


def add_row(
    rows: list[dict[str, Any]],
    *,
    source_axis: str,
    source_task: str,
    evaluator: str,
    kind: str,
    text: str,
    source_dataset: str | None = None,
    split: str | None = None,
    source_field: str | None = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    text = str(text).strip()
    if len(text) < TEXT_MIN_CHARS:
        return False
    norm = normalize_text(text)
    if len(norm) < TEXT_MIN_CHARS:
        return False
    row = {
        "source_axis": source_axis,
        "source_task": source_task,
        "evaluator": evaluator,
        "kind": kind,
        "source_dataset": source_dataset,
        "split": split,
        "source_field": source_field,
        "text": text,
        "normalized_text": norm,
        "normalized_sha256": sha256_text(norm),
    }
    if extra:
        row["extra"] = extra
    rows.append(row)
    return True


def enabled_tasks(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for axis_name, axis_cfg in cfg.get("axes", {}).items():
        if not isinstance(axis_cfg, dict) or not axis_cfg.get("enabled", False):
            continue
        evaluator = axis_cfg.get("evaluator")
        for task in axis_cfg.get("tasks", []):
            if not isinstance(task, dict):
                raise ValueError(f"task entry must be a mapping: {task!r}")
            task_id = str(task.get("task_id") or task.get("runner_task_name"))
            runner_task_name = str(task.get("runner_task_name") or task_id)
            out.append({
                "axis": axis_name,
                "evaluator": evaluator,
                "task_id": task_id,
                "runner_task_name": runner_task_name,
                "denylist_source": task.get("denylist_source") or task_id,
            })
    return out


def load_hf_dataset_from_source(src: dict[str, Any], split: str):
    dataset_id = src["dataset"]
    name = src.get("name")
    revision = src.get("revision")
    kwargs: dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision
    if name:
        return hf_load_dataset(dataset_id, name, **kwargs)
    return hf_load_dataset(dataset_id, **kwargs)


def collect_lm_eval_source(cfg: dict[str, Any], rows: list[dict[str, Any]], task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    sources = cfg.get("benchmark_input_sources", {})
    source_id = str(task["denylist_source"])
    src = sources.get(source_id)
    if not isinstance(src, dict):
        raise ValueError(f"No benchmark_input_sources.{source_id} entry for task {task['runner_task_name']}")
    splits = src.get("splits") or [src.get("split", "train")]
    text_fields = list(src.get("text_fields", []))
    joined_fields = list(src.get("joined_fields", []))
    if not text_fields and not joined_fields:
        raise ValueError(f"benchmark_input_sources.{source_id} must define text_fields or joined_fields")

    added = 0
    row_counts: dict[str, int] = {}
    for split in splits:
        ds = load_hf_dataset_from_source(src, str(split))
        row_count = 0
        for idx, row in enumerate(ds):
            row_count += 1
            for field in text_fields:
                value = row.get(field)
                if isinstance(value, str) and add_row(
                    rows,
                    source_axis=task["axis"],
                    source_task=task["runner_task_name"],
                    evaluator="lm_eval",
                    kind="benchmark_input_text",
                    text=value,
                    source_dataset=src.get("dataset"),
                    split=str(split),
                    source_field=str(field),
                    extra={"row_index": idx, "revision": src.get("revision"), "name": src.get("name"), "source_id": source_id},
                ):
                    added += 1
            for spec in joined_fields:
                fields = list(spec.get("fields", []))
                values = [row.get(field) for field in fields]
                if all(isinstance(v, str) and v.strip() for v in values):
                    text = str(spec.get("separator", "\n")).join(str(v) for v in values)
                    if add_row(
                        rows,
                        source_axis=task["axis"],
                        source_task=task["runner_task_name"],
                        evaluator="lm_eval",
                        kind=str(spec.get("kind", "benchmark_joined_text")),
                        text=text,
                        source_dataset=src.get("dataset"),
                        split=str(split),
                        source_field="+".join(fields),
                        extra={"row_index": idx, "revision": src.get("revision"), "name": src.get("name"), "source_id": source_id},
                    ):
                        added += 1
        row_counts[str(split)] = row_count
    meta = dict(src)
    meta["row_counts_by_split"] = row_counts
    return added, meta


def safety_root_from_config(cfg: dict[str, Any]) -> Path:
    rel = cfg.get("safety_eval", {}).get("local_root", "external/safety-eval-fork")
    p = Path(str(rel)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def check_safety_commit(cfg: dict[str, Any], root: Path) -> tuple[str | None, list[str]]:
    expected = cfg.get("safety_eval", {}).get("pinned_commit")
    actual = git_commit(root)
    warnings: list[str] = []
    if not expected:
        warnings.append("benchmark config does not define safety_eval.pinned_commit")
    if actual is None:
        warnings.append(f"could not read safety-eval git commit at {project_path(root, PROJECT_ROOT)}")
    elif expected and actual != expected:
        warnings.append(f"safety-eval commit mismatch: expected {expected}, got {actual}")
    return actual, warnings


def get_field(obj: Any, field_path: str) -> Any:
    current = obj
    for part in str(field_path).split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def collect_dataset_revision_pins(cfg: dict[str, Any]) -> list[dict[str, str]]:
    pins: list[dict[str, str]] = []
    for source in cfg.get("safety_eval_input_sources", {}).values():
        if not isinstance(source, dict):
            continue
        for pin in source.get("dataset_revision_overrides", []) or []:
            if not isinstance(pin, dict):
                continue
            dataset = str(pin.get("dataset", "")).strip()
            revision = str(pin.get("revision", "")).strip()
            if not dataset or not revision:
                continue
            pins.append({"dataset": dataset, "name": str(pin.get("name", "")).strip(), "revision": revision})
    return pins


def patch_datasets_load_dataset(pins: list[dict[str, str]]) -> None:
    if not pins:
        return
    import datasets as datasets_module  # type: ignore

    original = datasets_module.load_dataset

    def matches(pin: dict[str, str], path: Any, name: Any) -> bool:
        if str(path) != pin.get("dataset"):
            return False
        pin_name = str(pin.get("name") or "")
        return (not pin_name) or (name is not None and str(name) == pin_name)

    def pinned_load_dataset(path, name=None, *args, **kwargs):
        for pin in pins:
            if matches(pin, path, name) and "revision" not in kwargs:
                kwargs["revision"] = pin["revision"]
                break
        if name is None:
            return original(path, *args, **kwargs)
        return original(path, name, *args, **kwargs)

    datasets_module.load_dataset = pinned_load_dataset


def install_safety_import_stubs() -> None:
    os.environ.setdefault("OPENAI_API_KEY", "dummy-key-for-snapshot-only")
    if "vllm" not in sys.modules:
        vllm_stub = types.ModuleType("vllm")

        class _DummyLLM:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("vllm.LLM was called during benchmark input export")

        class _DummySamplingParams:
            def __init__(self, *args, **kwargs):
                pass

        vllm_stub.LLM = _DummyLLM
        vllm_stub.SamplingParams = _DummySamplingParams
        vllm_stub.RequestOutput = object
        vllm_stub.CompletionOutput = object
        sys.modules["vllm"] = vllm_stub
    if "fastchat" not in sys.modules:
        fastchat_stub = types.ModuleType("fastchat")
        conversation_stub = types.ModuleType("fastchat.conversation")
        model_stub = types.ModuleType("fastchat.model")

        def _dummy(*args, **kwargs):
            raise RuntimeError("FastChat formatting was called during benchmark input export")

        conversation_stub.get_conv_template = _dummy
        model_stub.get_conversation_template = _dummy
        fastchat_stub.conversation = conversation_stub
        fastchat_stub.model = model_stub
        sys.modules["fastchat"] = fastchat_stub
        sys.modules["fastchat.conversation"] = conversation_stub
        sys.modules["fastchat.model"] = model_stub


def configured_safety_source(cfg: dict[str, Any], task_name: str) -> dict[str, Any]:
    source = cfg.get("safety_eval_input_sources", {}).get(task_name, {})
    return source if isinstance(source, dict) else {}


def safety_source_file_hashes(cfg: dict[str, Any], safety_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for task_name, source in cfg.get("safety_eval_input_sources", {}).items():
        if not isinstance(source, dict):
            continue
        for rel in source.get("source_files", []) or []:
            path = safety_root / str(rel)
            out[str(rel)] = {
                "task": task_name,
                "path": project_path(path, PROJECT_ROOT),
                "exists": path.exists(),
                "sha256": file_sha256(path) if path.exists() else None,
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
    return out


def collect_safety_eval_sources(cfg: dict[str, Any], rows: list[dict[str, Any]], safety_tasks: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, Any]]:
    safety_root = safety_root_from_config(cfg)
    if not safety_root.exists():
        raise FileNotFoundError(f"safety-eval root not found: {project_path(safety_root, PROJECT_ROOT)}")
    actual_commit, commit_warnings = check_safety_commit(cfg, safety_root)
    for warning in commit_warnings:
        print(f"[warning] {warning}")
    evaluation_root = safety_root / "evaluation"
    if not evaluation_root.exists():
        raise FileNotFoundError(f"safety-eval evaluation dir not found: {project_path(evaluation_root, PROJECT_ROOT)}")

    pins = collect_dataset_revision_pins(cfg)
    patch_datasets_load_dataset(pins)
    sys.path.insert(0, str(safety_root))
    sys.path.insert(0, str(evaluation_root))
    install_safety_import_stubs()
    from evaluation.tasks import EvalMode, load_evaluation_tasks  # type: ignore

    task_names = [t["runner_task_name"] for t in safety_tasks]
    eval_tasks = load_evaluation_tasks(EvalMode.GENERATION, task_names)
    text_counts: dict[str, int] = {}
    item_counts: dict[str, int] = {}
    task_by_name = {t["runner_task_name"]: t for t in safety_tasks}
    for task_name, eval_task in zip(task_names, eval_tasks):
        task = task_by_name[task_name]
        data = eval_task.data
        item_counts[task_name] = len(data)
        text_counts[task_name] = 0
        source_cfg = configured_safety_source(cfg, task_name)
        for idx, item in enumerate(data):
            text_fields = list(source_cfg.get("text_fields", []))
            if not text_fields:
                raise ValueError(f"safety_eval_input_sources.{task_name}.text_fields must be set")
            for field_path in text_fields:
                text_value = get_field(item, field_path)
                if not isinstance(text_value, str) or not text_value.strip():
                    raise ValueError(f"Task {task_name}, item {idx} missing configured text field: {field_path}")
                if add_row(
                    rows,
                    source_axis=task["axis"],
                    source_task=task_name,
                    evaluator="safety_eval",
                    kind="safety_eval_prompt_text",
                    text=text_value,
                    source_dataset="safety_eval_fork",
                    split=None,
                    source_field=field_path,
                    extra={
                        "source_index": idx,
                        "raw_item_id": item.get("id") if isinstance(item, dict) else None,
                    },
                ):
                    text_counts[task_name] += 1
    meta = {
        "safety_eval_root": project_path(safety_root, PROJECT_ROOT),
        "safety_eval_commit": actual_commit,
        "warnings": commit_warnings,
        "dataset_revision_overrides": pins,
        "source_file_hashes": safety_source_file_hashes(cfg, safety_root),
        "task_item_counts": item_counts,
    }
    return text_counts, meta


def enabled_runner_tasks(cfg: dict[str, Any]) -> list[str]:
    return [t["runner_task_name"] for t in enabled_tasks(cfg)]


def main() -> None:
    cfg = read_yaml(BENCHMARK_CONFIG)
    out_rel = cfg.get("benchmark_inputs", {}).get("output")
    manifest_rel = cfg.get("benchmark_inputs", {}).get("manifest")
    output = PROJECT_ROOT / out_rel if out_rel else DEFAULT_OUTPUT
    manifest_path = PROJECT_ROOT / manifest_rel if manifest_rel else DEFAULT_MANIFEST

    tasks = enabled_tasks(cfg)
    rows: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    lm_source_info: dict[str, Any] = {}
    safety_tasks = [t for t in tasks if t["evaluator"] == "safety_eval"]
    safety_info: dict[str, Any] = {}

    for task in tasks:
        if task["evaluator"] == "lm_eval":
            count, meta = collect_lm_eval_source(cfg, rows, task)
            task_counts[task["runner_task_name"]] += count
            lm_source_info[task["runner_task_name"]] = meta

    if safety_tasks:
        text_counts, safety_info = collect_safety_eval_sources(cfg, rows, safety_tasks)
        for task_name, count in text_counts.items():
            task_counts[task_name] += count

    out_rows = rows
    out_rows.sort(key=lambda r: (str(r.get("source_axis")), str(r.get("source_task")), str(r.get("source_field")), r["normalized_sha256"]))
    write_jsonl(output, out_rows)

    enabled = enabled_runner_tasks(cfg)
    missing = [task for task in enabled if task_counts[task] == 0]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_config": project_path(BENCHMARK_CONFIG, PROJECT_ROOT),
        "benchmark_config_sha256": file_sha256(BENCHMARK_CONFIG),
        "benchmark_id": cfg.get("benchmark_id"),
        "enabled_tasks": enabled,
        "task_counts": dict(task_counts),
        "num_entries": len(out_rows),
        "output": project_path(output, PROJECT_ROOT),
        "output_sha256": file_sha256(output),
        "lm_eval_input_sources": lm_source_info,
        "safety_eval_input_sources": safety_info,
        "normalization": {
            "lowercase": True,
            "collapse_whitespace": True,
            "html_unescape": True,
            "zero_width_removed": True,
        },
        "warnings": list(safety_info.get("warnings", [])) if isinstance(safety_info, dict) else [],
    }
    if missing:
        manifest["warnings"].append("No benchmark input text was collected for: " + ", ".join(missing))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[OK] wrote benchmark input snapshot: {project_path(output, PROJECT_ROOT)}")
    print(f"[OK] wrote manifest: {project_path(manifest_path, PROJECT_ROOT)}")
    print(f"[INFO] benchmark input rows: {len(out_rows)}")
    print("[INFO] task counts:")
    for task in enabled:
        print(f"  - {task}: {task_counts[task]}")
    if missing:
        print("[warning] No benchmark input text was collected for: " + ", ".join(missing))


if __name__ == "__main__":
    main()

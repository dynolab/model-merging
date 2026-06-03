from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from common.io import file_sha256
from common.text import sha256_text

from benchmark.common import LM_EVAL_BIN, PROJECT_ROOT, SAFETY_DATASET_PIN_ROOT, TEMPLATE_ROOT


DEFAULT_BATCH_SIZE = 8
TRUST_REMOTE_CODE = True
QWEN_TEMPLATE_SENTINEL = "__BENCHMARK_INSTRUCTION_PLACEHOLDER__"


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return str(value)


def run_cmd(cmd: list[str], cwd: Path | None, env: dict[str, str] | None = None) -> int:
    print("\n[command]")
    print(shell_join(cmd))
    if cwd is not None:
        print(f"[cwd] {cwd}")
    if env and env.get("CUDA_VISIBLE_DEVICES"):
        print(f"[env] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    return proc.returncode


def generate_qwen3_safety_template(tokenizer_model_path: Path, *, enable_thinking: bool | None) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required to generate the Qwen3 safety-eval template") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_model_path), trust_remote_code=True)
    template_kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": QWEN_TEMPLATE_SENTINEL}],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    if QWEN_TEMPLATE_SENTINEL not in rendered:
        raise RuntimeError("Generated Qwen3 chat template did not contain the sentinel prompt")
    template = rendered.replace(QWEN_TEMPLATE_SENTINEL, "{instruction}")
    TEMPLATE_ROOT.mkdir(parents=True, exist_ok=True)
    template_mode = "no_think" if enable_thinking is False else "think" if enable_thinking is True else "default"
    out = TEMPLATE_ROOT / f"qwen3_{template_mode}__{sha256_text(template)[:12]}.txt"
    out.write_text(template, encoding="utf-8")
    return {
        "source": "generated_from_evaluated_model_tokenizer",
        "path": str(out),
        "sha256": file_sha256(out),
        "enable_thinking": enable_thinking,
        "enable_thinking_arg_present": enable_thinking is not None,
        "add_generation_prompt": True,
    }


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


def apply_safety_eval_dataset_pins(cfg: dict[str, Any], env: dict[str, str]) -> list[dict[str, str]]:
    pins = collect_dataset_revision_pins(cfg)
    if not pins:
        return []
    SAFETY_DATASET_PIN_ROOT.mkdir(parents=True, exist_ok=True)
    sitecustomize = SAFETY_DATASET_PIN_ROOT / "sitecustomize.py"
    patch_code = """import json\nimport os\n\n_pins = json.loads(os.environ.get(\"SAFETY_EVAL_DATASET_PINS_JSON\", \"[]\"))\ntry:\n    import datasets as _datasets\n    _orig_load_dataset = _datasets.load_dataset\n\n    def _matches(pin, path, name):\n        if str(path) != str(pin.get(\"dataset\")):\n            return False\n        pin_name = str(pin.get(\"name\") or \"\")\n        return (not pin_name) or (name is not None and str(name) == pin_name)\n\n    def load_dataset(path, name=None, *args, **kwargs):\n        for pin in _pins:\n            if _matches(pin, path, name) and \"revision\" not in kwargs:\n                kwargs[\"revision\"] = pin[\"revision\"]\n                break\n        if name is None:\n            return _orig_load_dataset(path, *args, **kwargs)\n        return _orig_load_dataset(path, name, *args, **kwargs)\n\n    _datasets.load_dataset = load_dataset\nexcept Exception:\n    pass\n"""
    sitecustomize.write_text(patch_code, encoding="utf-8")
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SAFETY_DATASET_PIN_ROOT) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    env["SAFETY_EVAL_DATASET_PINS_JSON"] = json.dumps(pins, ensure_ascii=True)
    return pins


def require_vllm() -> None:
    try:
        import vllm  # type: ignore

        getattr(vllm, "LLM")
    except Exception as exc:
        raise RuntimeError(
            "The safety-eval axis requires vLLM for this benchmark protocol. "
            "Install a working vLLM build or run in an environment where vLLM is available."
        ) from exc


def prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    old_pythonpath = env.get("PYTHONPATH", "")
    path_s = str(path)
    parts = [p for p in old_pythonpath.split(os.pathsep) if p]
    if path_s not in parts:
        env["PYTHONPATH"] = path_s + (os.pathsep + old_pythonpath if old_pythonpath else "")


def ensure_safety_eval_import_paths(env: dict[str, str], safety_root: Path) -> None:
    paths = [safety_root.resolve(), (safety_root / "evaluation").resolve()]
    for path in reversed(paths):
        prepend_pythonpath(env, path)


def ensure_openai_import_placeholder(env: dict[str, str]) -> None:
    if env.get("OPENAI_API_KEY"):
        return
    env["OPENAI_API_KEY"] = "not-used-by-configured-safety-tasks"


def command_limit(axis_cfg: dict[str, Any], tasks: list[dict[str, Any]]) -> int | None:
    limits = [task.get("limit") for task in tasks if task.get("limit") is not None]
    if axis_cfg.get("limit") is not None:
        limits.append(axis_cfg["limit"])
    if not limits:
        return None
    unique = {int(limit) for limit in limits}
    if len(unique) != 1:
        raise ValueError(
            "Combined benchmark commands require one shared limit; split tasks or use matching task limits."
        )
    return unique.pop()


def build_lm_eval_command(
    eval_model_path: str,
    axis_cfg: dict[str, Any],
    tasks: list[dict[str, Any]],
    axis_out: Path,
    chat_format: dict[str, Any],
    gpu_count: int,
) -> list[str]:
    task_names: list[str] = []
    include_paths: list[str] = []
    for task in tasks:
        runner_task_name = str(task["runner_task_name"])
        runner_task_path = Path(runner_task_name)
        if runner_task_name.endswith((".yaml", ".yml")) or runner_task_path.exists():
            # lm-eval 0.4.x loads project-local YAML tasks reliably via --include_path.
            # Passing the YAML path directly to --tasks can load the file but then fail
            # during task pretty-printing with KeyError('<task_id>').
            if not runner_task_path.is_absolute():
                runner_task_path = PROJECT_ROOT / runner_task_path
            if not runner_task_path.exists():
                raise FileNotFoundError(f"Local lm-eval task YAML not found: {runner_task_path}")
            task_names.append(str(task["task_id"]))
            include_dir = str(runner_task_path.parent)
            if include_dir not in include_paths:
                include_paths.append(include_dir)
        else:
            task_names.append(runner_task_name)

    backend = str(axis_cfg.get("lm_eval_backend", "hf")).strip().lower()
    if backend not in {"hf", "vllm"}:
        raise ValueError(f"Unsupported lm_eval_backend={backend!r}; expected 'hf' or 'vllm'")

    model_args = [f"pretrained={eval_model_path}"]

    if TRUST_REMOTE_CODE:
        model_args.append("trust_remote_code=True")

    chat_args = chat_format.get("lm_eval_model_args", {})
    if isinstance(chat_args, dict):
        for key, value in chat_args.items():
            model_args.append(f"{key}={cli_value(value)}")

    if backend == "hf":
        if gpu_count >= 1 and not any(arg.startswith("parallelize=") for arg in model_args):
            model_args.append("parallelize=True")
        batch_size = str(int(axis_cfg.get("batch_size", DEFAULT_BATCH_SIZE)))
    else:
        # vLLM performs its own scheduling/batching; do not pass the HF-only parallelize flag.
        model_args = [arg for arg in model_args if not arg.startswith("parallelize=")]

        vllm_args = axis_cfg.get("vllm_model_args", {}) or {}
        if not isinstance(vllm_args, dict):
            raise ValueError("vllm_model_args must be a mapping when lm_eval_backend='vllm'")

        existing_keys = {arg.split("=", 1)[0] for arg in model_args if "=" in arg}
        for key, value in vllm_args.items():
            if key in existing_keys:
                continue
            model_args.append(f"{key}={cli_value(value)}")

        batch_size = str(axis_cfg.get("vllm_batch_size", "auto"))

    cmd = [
        LM_EVAL_BIN,
        "--model",
        backend,
        "--model_args",
        ",".join(model_args),
    ]
    for include_path in include_paths:
        cmd.extend(["--include_path", include_path])
    cmd.extend([
        "--tasks",
        ",".join(task_names),
        "--batch_size",
        batch_size,
        "--output_path",
        str(axis_out),
    ])

    limit = command_limit(axis_cfg, tasks)
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if bool(chat_format.get("lm_eval_apply_chat_template", False)):
        cmd.append("--apply_chat_template")

    gen_kwargs = axis_cfg.get("gen_kwargs", {}) or {}
    if gen_kwargs:
        if not isinstance(gen_kwargs, dict):
            raise ValueError("gen_kwargs must be a mapping when defined")
        cmd.extend(
            [
                "--gen_kwargs",
                ",".join(f"{key}={cli_value(value)}" for key, value in gen_kwargs.items()),
            ]
        )

    if bool(axis_cfg.get("log_samples", False)):
        cmd.append("--log_samples")

    return cmd

def build_safety_eval_command(
    eval_model_path: str,
    axis_cfg: dict[str, Any],
    tasks: list[dict[str, Any]],
    axis_out: Path,
    safety_root: Path,
    safety_template_file: Path,
) -> tuple[list[str], Path]:
    task_names = [str(task["runner_task_name"]) for task in tasks]
    eval_py = safety_root / "evaluation" / "eval.py"
    if not eval_py.exists():
        raise FileNotFoundError(f"Cannot find safety-eval entrypoint: {eval_py}")
    if not safety_template_file.exists():
        raise FileNotFoundError(f"Generated safety-eval template file does not exist: {safety_template_file}")
    batch_size = int(axis_cfg.get("batch_size", DEFAULT_BATCH_SIZE))
    cmd = [
        sys.executable,
        "-m",
        "evaluation.eval",
        "generators",
        "--model_name_or_path",
        eval_model_path,
        "--use_vllm",
        "--model_input_template_path_or_name",
        str(safety_template_file),
        "--tasks",
        ",".join(task_names),
        "--report_output_path",
        str((axis_out / "safety_eval.json").resolve()),
        "--save_individual_results_path",
        str((axis_out / "safety_generations.json").resolve()),
        "--batch_size",
        str(batch_size),
    ]
    limit = command_limit(axis_cfg, tasks)
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    return cmd, safety_root

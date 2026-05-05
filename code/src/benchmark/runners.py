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


def run_cmd(cmd: list[str], cwd: Path | None, env: dict[str, str] | None = None) -> int:
    print("\n[command]")
    print(shell_join(cmd))
    if cwd is not None:
        print(f"[cwd] {cwd}")
    if env and env.get("CUDA_VISIBLE_DEVICES"):
        print(f"[env] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    return proc.returncode


def generate_qwen3_no_think_template(tokenizer_model_path: Path) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required to generate the Qwen3 safety-eval template") from exc

    enable_thinking = False
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_model_path), trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": QWEN_TEMPLATE_SENTINEL}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if QWEN_TEMPLATE_SENTINEL not in rendered:
        raise RuntimeError("Generated Qwen3 chat template did not contain the sentinel prompt")
    template = rendered.replace(QWEN_TEMPLATE_SENTINEL, "{instruction}")
    TEMPLATE_ROOT.mkdir(parents=True, exist_ok=True)
    out = TEMPLATE_ROOT / f"qwen3_no_think__{sha256_text(template)[:12]}.txt"
    out.write_text(template, encoding="utf-8")
    return {
        "source": "generated_from_evaluated_model_tokenizer",
        "path": str(out),
        "sha256": file_sha256(out),
        "enable_thinking": enable_thinking,
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
        from vllm import LLM  # noqa: F401
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


def build_lm_eval_command(
    eval_model_path: str,
    axis_cfg: dict[str, Any],
    task_names: list[str],
    axis_out: Path,
    chat_format: dict[str, Any],
    gpu_count: int,
) -> list[str]:
    model_args = [f"pretrained={eval_model_path}"]

    if TRUST_REMOTE_CODE:
        model_args.append("trust_remote_code=True")

    chat_args = chat_format.get("lm_eval_model_args", {})
    if isinstance(chat_args, dict):
        for key, value in chat_args.items():
            value_s = "True" if isinstance(value, bool) and value else "False" if isinstance(value, bool) else str(value)
            model_args.append(f"{key}={value_s}")

    if gpu_count >= 1 and not any(arg.startswith("parallelize=") for arg in model_args):
        model_args.append("parallelize=True")

    batch_size = int(axis_cfg.get("batch_size", DEFAULT_BATCH_SIZE))
    cmd = [
        LM_EVAL_BIN,
        "--model",
        "hf",
        "--model_args",
        ",".join(model_args),
        "--tasks",
        ",".join(task_names),
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(axis_out),
    ]

    if axis_cfg.get("limit") is not None:
        cmd.extend(["--limit", str(axis_cfg["limit"])])

    if bool(chat_format.get("lm_eval_apply_chat_template", False)):
        cmd.append("--apply_chat_template")

    return cmd


def build_safety_eval_command(
    eval_model_path: str,
    axis_cfg: dict[str, Any],
    task_names: list[str],
    axis_out: Path,
    safety_root: Path,
    safety_template_file: Path,
) -> tuple[list[str], Path]:
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
    return cmd, safety_root

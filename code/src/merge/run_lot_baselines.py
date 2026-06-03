#!/usr/bin/env python3
"""Build LOT-style activation-informed merge baselines."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.model_registry import model_registry
from common.io import file_sha256, git_commit, project_path, read_jsonl, read_yaml, resolve_project_path
from merge.lot_merging import (
    LotPaperSolverConfig,
    merge_tensor,
    self_test as lot_math_self_test,
    tensor_nbytes,
)


CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT.parent

BENCHMARK_CONFIG = CODE_ROOT / "cfg" / "benchmark" / "benchmark.yaml"
LOT_CONFIG = CODE_ROOT / "cfg" / "merge" / "lot_baselines.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "merged"
LOT_CONFIG_ROOT = PROJECT_ROOT / "outputs" / "lot_configs"
CALIBRATION_MANIFEST = PROJECT_ROOT / "data" / "calibration" / "calibration_manifest.json"
NON_OVERLAP_REPORT = PROJECT_ROOT / "data" / "calibration" / "non_overlap_report.json"
MERGE_MANIFEST = "merge_manifest.json"

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
INDEX_FILES = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
TOKENIZER_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baselines", nargs="*", help="LOT baseline id(s); defaults to all")
    parser.add_argument("--config", default=str(LOT_CONFIG), help="LOT baseline YAML config")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing LOT output checkpoint")
    parser.add_argument("--device", default=None, help="activation/smoke device; default from config")
    parser.add_argument(
        "--solver-device",
        default=None,
        help="device for LOT solver math (cpu/cuda/auto); default auto picks cuda when available",
    )
    parser.add_argument("--gpu-ids", default=None, help="CUDA_VISIBLE_DEVICES value for Colab pipeline compatibility")
    parser.add_argument("--n-prompts-per-source", type=int, default=None, help="override feature prompt count")
    parser.add_argument("--tokens-per-prompt", type=int, default=None, help="override feature token rows per prompt")
    parser.add_argument("--limit-layers", default=None, help="comma-separated layer ids for smoke runs")
    parser.add_argument("--limit-modules", default=None, help="comma-separated module suffixes for smoke runs")
    parser.add_argument("--skip-smoke", action="store_true", help="skip tokenizer/model smoke validation after writing")
    parser.add_argument("--self-test", action="store_true", help="run lightweight LOT math self-test and exit")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


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


def parse_csv_ints(value: str | None) -> set[int] | None:
    if value is None or not str(value).strip():
        return None
    out = {int(part.strip()) for part in str(value).split(",") if part.strip()}
    if not out:
        return None
    return out


def parse_csv_strings(value: str | None) -> set[str] | None:
    if value is None or not str(value).strip():
        return None
    out = {part.strip() for part in str(value).split(",") if part.strip()}
    if not out:
        return None
    return out


def selected_baselines(cfg: dict[str, Any], requested: list[str]) -> dict[str, dict[str, Any]]:
    baselines = cfg.get("baselines")
    if not isinstance(baselines, dict):
        raise ValueError("LOT config must define baselines")
    out = {str(k): v for k, v in baselines.items() if isinstance(v, dict)}
    if not requested:
        return out
    missing = [name for name in requested if name not in out]
    if missing:
        raise ValueError("unknown LOT baseline(s): " + ", ".join(missing))
    return {name: out[name] for name in requested}


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[str(name)]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def positive_int(value: Any, *, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive, got {out}")
    return out


def resolve_solver_device(name: str | None) -> torch.device:
    """Resolve --solver-device into a concrete torch.device.

    - "auto" (or None): cuda if available, else cpu.
    - "cuda": cuda (errors if not available).
    - "cpu": cpu.
    """
    val = (name or "auto").lower()
    if val == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if val == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--solver-device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if val == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported solver device: {name!r}")


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {project_path(path, PROJECT_ROOT)}")


def materialize_models(benchmark_cfg: dict[str, Any], aliases: list[str]) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    registry = model_registry(benchmark_cfg)
    missing = [alias for alias in aliases if alias not in registry]
    if missing:
        raise ValueError("model alias not found in benchmark.yaml: " + ", ".join(missing))
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required to download pinned model snapshots") from exc

    paths: dict[str, Path] = {}
    identities: dict[str, dict[str, Any]] = {}
    for alias in aliases:
        spec = registry[alias]
        local_path = Path(snapshot_download(repo_id=spec["repo"], revision=spec["revision"])).resolve()
        paths[alias] = local_path
        identities[alias] = {
            "alias": alias,
            "repo": spec["repo"],
            "revision": spec["revision"],
        }
    return paths, identities


def load_calibration_paths(lot_cfg: dict[str, Any]) -> dict[str, Path]:
    calibration_cfg_path = resolve_project_path(str(lot_cfg.get("calibration_config")), PROJECT_ROOT)
    calibration_cfg = read_yaml(calibration_cfg_path)
    output_dir = resolve_project_path(str(calibration_cfg.get("output_dir", "data/calibration")), PROJECT_ROOT)
    paths: dict[str, Path] = {}

    main = calibration_cfg.get("main_corpus", {})
    if isinstance(main, dict) and main.get("output_file"):
        paths["mix"] = output_dir / str(main["output_file"])

    subsets = calibration_cfg.get("ablation_subsets", {})
    if isinstance(subsets, dict):
        for subset_id, subset_cfg in subsets.items():
            if isinstance(subset_cfg, dict) and subset_cfg.get("output_file"):
                paths[str(subset_id)] = output_dir / str(subset_cfg["output_file"])
    return paths


def read_calibration_texts(path: Path, *, text_field: str, limit: int) -> list[str]:
    texts: list[str] = []
    for row in read_jsonl(path):
        text = row.get(text_field)
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
        if len(texts) >= limit:
            break
    if len(texts) < limit:
        raise ValueError(f"not enough rows in {project_path(path, PROJECT_ROOT)}: needed {limit}, got {len(texts)}")
    return texts


def require_calibration_artifacts(paths: list[Path]) -> None:
    require_file(CALIBRATION_MANIFEST, "calibration manifest")
    require_file(NON_OVERLAP_REPORT, "non-overlap report")
    for path in paths:
        require_file(path, "LOT calibration corpus")
        first = next(iter(read_jsonl(path)), None)
        if not first:
            raise ValueError(f"calibration corpus is empty: {project_path(path, PROJECT_ROOT)}")


def load_auto_config(path: Path, *, trust_remote_code: bool) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(path), trust_remote_code=trust_remote_code)


def validate_model_compatibility(
    *,
    paths: dict[str, Path],
    aliases: list[str],
    trust_remote_code: bool,
) -> dict[str, Any]:
    strict_fields = [
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "tie_word_embeddings",
        "rope_theta",
    ]
    informational_fields = [
        "max_position_embeddings",
    ]
    fields = strict_fields + informational_fields
    configs = {alias: load_auto_config(paths[alias], trust_remote_code=trust_remote_code) for alias in aliases}
    base_alias = aliases[0]
    base_cfg = configs[base_alias]
    strict_mismatches: list[str] = []
    informational_mismatches: list[str] = []
    for alias in aliases[1:]:
        cfg = configs[alias]
        for field in strict_fields:
            if getattr(cfg, field, None) != getattr(base_cfg, field, None):
                strict_mismatches.append(
                    f"{alias}.{field}={getattr(cfg, field, None)!r} != "
                    f"{base_alias}.{field}={getattr(base_cfg, field, None)!r}"
                )
        for field in informational_fields:
            if getattr(cfg, field, None) != getattr(base_cfg, field, None):
                informational_mismatches.append(
                    f"{alias}.{field}={getattr(cfg, field, None)!r} != "
                    f"{base_alias}.{field}={getattr(base_cfg, field, None)!r}"
                )
    if strict_mismatches:
        raise ValueError("model config mismatch:\n  - " + "\n  - ".join(strict_mismatches))
    if informational_mismatches:
        print(
            "[warning] non-blocking model config mismatch:\n  - "
            + "\n  - ".join(informational_mismatches),
            file=sys.stderr,
        )
    return {
        "strict_fields": strict_fields,
        "informational_fields": informational_fields,
        "informational_mismatches": informational_mismatches,
        "configs": {
            alias: {field: getattr(cfg, field, None) for field in fields}
            for alias, cfg in configs.items()
        },
    }


def validate_tokenizer(path: Path, *, trust_remote_code: bool) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    info: dict[str, Any] = {
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
    }
    messages = [{"role": "user", "content": "Hello"}]
    for value in [False, True]:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=value,
        )
        if not isinstance(rendered, str) or not rendered.strip():
            raise ValueError(f"empty chat template render for enable_thinking={value}")
        info[f"enable_thinking_{str(value).lower()}_chars"] = len(rendered)
    return info


def copy_checkpoint_metadata(base_path: Path, tokenizer_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in base_path.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(base_path)
        if any(part in {".git", "__pycache__"} for part in rel.parts):
            continue
        if path.name in INDEX_FILES or path.name.endswith(WEIGHT_SUFFIXES):
            continue
        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    for name in TOKENIZER_FILES | {"generation_config.json"}:
        path = tokenizer_path / name
        if path.exists() and path.is_file():
            shutil.copy2(path, output_dir / name)


class SafetensorCheckpoint:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.index_path = self.root / "model.safetensors.index.json"
        self.index_json: dict[str, Any] | None = None
        self.weight_map: dict[str, str] = {}

        if self.index_path.exists():
            self.index_json = json.loads(self.index_path.read_text(encoding="utf-8"))
            raw_map = self.index_json.get("weight_map")
            if not isinstance(raw_map, dict):
                raise ValueError(f"invalid weight_map in {self.index_path}")
            self.weight_map = {str(k): str(v) for k, v in raw_map.items()}
            return

        files = sorted(self.root.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors weights found in {self.root}")
        try:
            from safetensors import safe_open
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required for LOT merging") from exc
        for file_path in files:
            with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    self.weight_map[str(key)] = file_path.name

    @property
    def filenames(self) -> list[str]:
        return sorted(set(self.weight_map.values()))

    def names_in_file(self, filename: str) -> list[str]:
        return sorted(name for name, mapped in self.weight_map.items() if mapped == filename)

    def load_file(self, filename: str) -> dict[str, torch.Tensor]:
        try:
            from safetensors.torch import load_file
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required for LOT merging") from exc
        return load_file(str(self.root / filename), device="cpu")


class SourceTensorCache:
    def __init__(self, checkpoint: SafetensorCheckpoint) -> None:
        self.checkpoint = checkpoint
        self.current_file: str | None = None
        self.current_tensors: dict[str, torch.Tensor] = {}

    def get(self, name: str) -> torch.Tensor:
        filename = self.checkpoint.weight_map[name]
        if filename != self.current_file:
            self.current_tensors = self.checkpoint.load_file(filename)
            self.current_file = filename
        return self.current_tensors[name]

    def clear(self) -> None:
        self.current_file = None
        self.current_tensors = {}


def validate_weight_maps(base: SafetensorCheckpoint, sources: dict[str, SafetensorCheckpoint]) -> None:
    base_names = set(base.weight_map)
    errors: list[str] = []
    for alias, checkpoint in sources.items():
        names = set(checkpoint.weight_map)
        missing = sorted(base_names - names)
        extra = sorted(names - base_names)
        if missing:
            errors.append(f"{alias} missing tensors: {missing[:8]}")
        if extra:
            errors.append(f"{alias} extra tensors: {extra[:8]}")
    if errors:
        raise ValueError("checkpoint tensor map mismatch:\n  - " + "\n  - ".join(errors))


def layer_allowed(module_name: str, allowed_layers: set[int] | None) -> bool:
    if allowed_layers is None:
        return True
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module_name)
    if not match:
        return False
    return int(match.group(1)) in allowed_layers


def module_suffix(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def matching_module(name: str, allowed_modules: set[str], allowed_layers: set[int] | None) -> bool:
    return module_suffix(name) in allowed_modules and layer_allowed(name, allowed_layers)


def extract_features_for_model(
    *,
    model_path: Path,
    tokenizer_path: Path,
    texts: list[str],
    allowed_modules: set[str],
    allowed_layers: set[int] | None,
    dtype: torch.dtype,
    device_name: str,
    max_length: int,
    tokens_per_prompt: int,
    batch_size: int,
    trust_remote_code: bool,
    allowed_norm_modules: set[str] | None = None,
    norm_feature_form: str = "raw",
    rms_eps: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Run forward passes through `model` on `texts` and capture inputs to selected modules.

    Arguments:
        allowed_modules: module suffixes for which to capture the raw input tensor
            (used as X_k in Eq. 9 for matrix-multiplication parameters).
        allowed_norm_modules: module suffixes for normalization layers for which to
            capture features. When `norm_feature_form == "rmsnorm_normalized"`, the
            raw input is divided by its root-mean-square along the last dimension
            (with `rms_eps`) before storing — this recovers the post-normalization,
            pre-scale-multiplication form needed for Eq. 12.
        norm_feature_form: either "raw" (store input as-is) or
            "rmsnorm_normalized" (apply RMSNorm-style normalization before storing).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False

    allowed_norm_modules = allowed_norm_modules or set()
    if norm_feature_form not in {"raw", "rmsnorm_normalized"}:
        raise ValueError(f"unsupported norm_feature_form: {norm_feature_form}")

    captures: dict[str, list[torch.Tensor]] = {}
    state: dict[str, torch.Tensor] = {}
    handles = []

    def _gather(x: torch.Tensor) -> torch.Tensor | None:
        positions = state.get("positions")
        if not isinstance(x, torch.Tensor) or positions is None or x.ndim != 3:
            return None
        batch = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(positions)
        gathered = x[batch, positions]
        return gathered.reshape(-1, gathered.shape[-1])

    def make_linear_hook(module_name: str):
        def hook(_: torch.nn.Module, inputs: tuple[Any, ...], __: Any) -> None:
            if not inputs:
                return
            gathered = _gather(inputs[0])
            if gathered is None:
                return
            captures.setdefault(module_name, []).append(gathered.detach().to("cpu", dtype=torch.float16))

        return hook

    def make_norm_hook(module_name: str, normalize: bool):
        def hook(_: torch.nn.Module, inputs: tuple[Any, ...], __: Any) -> None:
            if not inputs:
                return
            x = inputs[0]
            if normalize:
                x32 = x.to(torch.float32)
                rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + float(rms_eps))
                x = (x32 * rms).to(x.dtype)
            gathered = _gather(x)
            if gathered is None:
                return
            captures.setdefault(module_name, []).append(gathered.detach().to("cpu", dtype=torch.float16))

        return hook

    for name, module in model.named_modules():
        suffix = module_suffix(name)
        if suffix in allowed_modules and layer_allowed(name, allowed_layers):
            handles.append(module.register_forward_hook(make_linear_hook(name)))
        elif suffix in allowed_norm_modules and layer_allowed(name, allowed_layers):
            handles.append(
                module.register_forward_hook(
                    make_norm_hook(name, normalize=norm_feature_form == "rmsnorm_normalized")
                )
            )

    if not handles:
        raise ValueError("no modules matched LOT feature hooks")

    try:
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                encoded = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                lengths = attention_mask.sum(dim=1)
                offsets = torch.arange(tokens_per_prompt, 0, -1, device=device)
                positions = torch.clamp(lengths.unsqueeze(1) - offsets.unsqueeze(0), min=0)
                positions = torch.minimum(positions, torch.full_like(positions, input_ids.shape[1] - 1))
                state["positions"] = positions
                _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                del input_ids, attention_mask, encoded
    finally:
        for handle in handles:
            handle.remove()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {name: torch.cat(chunks, dim=0) for name, chunks in captures.items() if chunks}


def tensor_strategy(
    *,
    tensor_name: str,
    tensor: torch.Tensor,
    linear_modules: set[str],
    norm_cfg: dict[str, Any],
    tensor_cfg: dict[str, Any],
    linear_strategy: str = "lot_paper",
) -> tuple[str, str | None]:
    if not tensor.is_floating_point():
        return str(tensor_cfg.get("non_float", {}).get("strategy", "copy_base")), "non-floating tensor"
    if tensor_name.endswith("embed_tokens.weight"):
        return str(tensor_cfg.get("embeddings", {}).get("strategy", "mean_delta")), "embedding tensor"
    if tensor_name == "lm_head.weight" or tensor_name.endswith(".lm_head.weight"):
        return str(tensor_cfg.get("lm_head", {}).get("strategy", "mean_delta")), "lm_head tensor"
    if tensor_name.endswith(".weight"):
        module_name = tensor_name[: -len(".weight")]
        suffix = module_suffix(module_name)
        if suffix in linear_modules:
            return linear_strategy, None
        if "norm" in suffix or "layernorm" in suffix.lower():
            if bool(norm_cfg.get("enabled", False)):
                include_suffixes = norm_cfg.get("include_suffixes") or []
                include_set = {str(s) for s in include_suffixes}
                if include_set and suffix not in include_set:
                    excluded = str(norm_cfg.get("strategy_excluded", "mean_delta"))
                    return excluded, "norm suffix excluded from LOT"
                return str(norm_cfg.get("strategy", "lot_paper_norm")), "norm LOT requested"
            return str(norm_cfg.get("strategy", "mean_delta")), "norm tensor"
    return str(tensor_cfg.get("unsupported_float", {}).get("strategy", "mean_delta")), "unsupported float tensor"


def save_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    try:
        from safetensors.torch import save_file
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for LOT merging") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={"format": "pt"})


def write_safetensors_index(output_dir: Path, weight_map: dict[str, str], total_size: int) -> None:
    if len(set(weight_map.values())) <= 1:
        return
    index = {"metadata": {"total_size": int(total_size)}, "weight_map": weight_map}
    write_json(output_dir / "model.safetensors.index.json", index)


def merge_checkpoint(
    *,
    base_path: Path,
    source_paths_by_role: dict[str, Path],
    features_by_role: dict[str, dict[str, torch.Tensor]],
    output_dir: Path,
    solver_cfg: LotPaperSolverConfig,
    tensor_cfg: dict[str, Any],
    linear_modules: set[str],
    allowed_layers: set[int] | None,
    linear_strategy: str = "lot_paper",
    solver_device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    base_checkpoint = SafetensorCheckpoint(base_path)
    source_checkpoints = {
        role: SafetensorCheckpoint(path)
        for role, path in source_paths_by_role.items()
    }
    validate_weight_maps(base_checkpoint, source_checkpoints)
    source_caches = {role: SourceTensorCache(checkpoint) for role, checkpoint in source_checkpoints.items()}
    role_order = list(source_paths_by_role)

    tensor_records: list[dict[str, Any]] = []
    strategy_counts: dict[str, int] = {}
    fallback_count = 0
    total_size = 0
    lot_strategies = {"lot_paper", "lot_paper_norm"}
    offload_to_solver_device = solver_device.type != "cpu"

    for filename in base_checkpoint.filenames:
        base_shard = base_checkpoint.load_file(filename)
        out_shard: dict[str, torch.Tensor] = {}
        for name in base_checkpoint.names_in_file(filename):
            base_tensor = base_shard[name]
            source_tensors = [source_caches[role].get(name) for role in role_order]
            strategy, strategy_reason = tensor_strategy(
                tensor_name=name,
                tensor=base_tensor,
                linear_modules=linear_modules,
                norm_cfg=dict(tensor_cfg.get("norm_modules", {}) or {}),
                tensor_cfg=tensor_cfg,
                linear_strategy=linear_strategy,
            )
            if strategy in lot_strategies and not layer_allowed(name, allowed_layers):
                strategy = str(tensor_cfg.get("unsupported_float", {}).get("strategy", "mean_delta"))
                strategy_reason = "outside selected smoke layers"

            module_name = name[: -len(".weight")] if name.endswith(".weight") else name
            feature_rows = [features_by_role.get(role, {}).get(module_name) for role in role_order]
            features = [row for row in feature_rows if row is not None]

            # Only LOT-style strategies benefit from GPU offload. Mean-delta / copy_base
            # are cheap; running them on CPU avoids unnecessary host<->device transfers.
            if offload_to_solver_device and strategy in lot_strategies:
                base_for_solver = base_tensor.to(solver_device, non_blocking=True)
                source_for_solver = [s.to(solver_device, non_blocking=True) for s in source_tensors]
                features_for_solver = [f.to(solver_device, non_blocking=True) for f in features]
            else:
                base_for_solver = base_tensor
                source_for_solver = source_tensors
                features_for_solver = features

            result = merge_tensor(
                tensor_name=name,
                base=base_for_solver,
                source_tensors=source_for_solver,
                features_by_source=features_for_solver,
                strategy=strategy,
                cfg=solver_cfg,
            )
            out_tensor = result.tensor
            if out_tensor.device.type != "cpu":
                out_tensor = out_tensor.to("cpu")
            out_shard[name] = out_tensor
            total_size += tensor_nbytes(out_tensor)
            strategy_counts[result.strategy] = strategy_counts.get(result.strategy, 0) + 1
            if result.fallback:
                fallback_count += 1
            tensor_records.append(
                {
                    "tensor": name,
                    "strategy": result.strategy,
                    "requested_strategy": strategy,
                    "strategy_reason": strategy_reason,
                    "fallback": result.fallback,
                    "fallback_reason": result.reason,
                    "delta_norm": result.delta_norm,
                    "mean_delta_norm": result.mean_delta_norm,
                    "correction_norm": result.correction_norm,
                    "shape": list(base_tensor.shape),
                    "dtype": str(base_tensor.dtype).replace("torch.", ""),
                }
            )
            # Drop GPU-side references for this tensor so the next iteration can reuse memory.
            if offload_to_solver_device and strategy in lot_strategies:
                del base_for_solver, source_for_solver, features_for_solver
        save_safetensors(output_dir / filename, out_shard)
        for cache in source_caches.values():
            cache.clear()
        del base_shard, out_shard
        gc.collect()
        if offload_to_solver_device:
            torch.cuda.empty_cache()

    write_safetensors_index(output_dir, dict(base_checkpoint.weight_map), total_size)
    return {
        "tensor_count": len(tensor_records),
        "strategy_counts": strategy_counts,
        "fallback_count": fallback_count,
        "tensors": tensor_records,
    }


def smoke_validate_checkpoint(
    *,
    output_dir: Path,
    dtype: torch.dtype,
    device_name: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(output_dir), trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    messages = [{"role": "user", "content": "Say OK."}]
    rendered = {}
    for value in [False, True]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=value,
        )
        rendered[str(value).lower()] = len(text)

    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        str(output_dir),
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    encoded = tokenizer("Smoke test.", return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**encoded, use_cache=False)
    logits = outputs.logits
    finite = bool(torch.isfinite(logits).all().item())
    del model, encoded, outputs, logits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not finite:
        raise RuntimeError("smoke validation produced non-finite logits")
    return {"chat_template_chars": rendered, "forward_logits_finite": finite}


def build_lot_baseline(
    *,
    baseline_id: str,
    baseline_spec: dict[str, Any],
    lot_cfg: dict[str, Any],
    lot_cfg_path: Path,
    benchmark_cfg: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    runtime = dict(lot_cfg.get("runtime", {}) or {})
    feature_cfg = dict(lot_cfg.get("features", {}) or {})
    # Per-baseline tensor overrides merge over the global tensor_cfg.
    tensor_cfg = dict(lot_cfg.get("tensors", {}) or {})
    tensors_overrides = baseline_spec.get("tensors_overrides") or {}
    if isinstance(tensors_overrides, dict):
        for key, value in tensors_overrides.items():
            if isinstance(value, dict) and isinstance(tensor_cfg.get(key), dict):
                merged = dict(tensor_cfg[key])
                merged.update(value)
                tensor_cfg[key] = merged
            else:
                tensor_cfg[key] = value
    # Per-baseline solver selection: solver_ref points to a top-level section of lot_cfg.
    solver_ref = str(baseline_spec.get("solver_ref", "solver_paper"))
    solver_raw = dict(lot_cfg.get(solver_ref, {}) or {})
    if not solver_raw:
        raise ValueError(f"{baseline_id}: solver_ref {solver_ref!r} not found in LOT config")
    trust_remote_code = bool(runtime.get("trust_remote_code", True))
    dtype = torch_dtype(str(runtime.get("dtype", "bfloat16")))
    device = str(args.device or runtime.get("device", "cuda"))
    solver_device = resolve_solver_device(args.solver_device or runtime.get("solver_device"))
    input_format = str(feature_cfg.get("input_format", "raw_text"))
    if input_format != "raw_text":
        raise ValueError(f"unsupported LOT feature input_format: {input_format}")
    solver_type = str(solver_raw.get("type", "paper"))
    if solver_type != "paper":
        raise ValueError(f"unsupported LOT solver type: {solver_type}")
    linear_strategy = "lot_paper"
    prompts_per_source = positive_int(
        args.n_prompts_per_source or feature_cfg.get("prompts_per_source", 64),
        name="features.prompts_per_source",
    )
    tokens_per_prompt = positive_int(
        args.tokens_per_prompt or feature_cfg.get("tokens_per_prompt", 1),
        name="features.tokens_per_prompt",
    )
    batch_size = positive_int(feature_cfg.get("batch_size", 1), name="features.batch_size")
    max_length = positive_int(feature_cfg.get("max_length", 512), name="features.max_length")
    allowed_layers = parse_csv_ints(args.limit_layers)
    configured_linear_modules = {str(x) for x in tensor_cfg.get("linear_modules", [])}
    limited_modules = parse_csv_strings(args.limit_modules)
    linear_modules = configured_linear_modules if limited_modules is None else configured_linear_modules & limited_modules
    if not linear_modules:
        raise ValueError("no LOT linear modules selected")

    base_alias = str(lot_cfg["base_model"])
    source_roles = {str(k): str(v) for k, v in lot_cfg.get("source_models", {}).items()}
    tokenizer_alias = str(lot_cfg.get("tokenizer_source", base_alias))
    aliases = [base_alias, tokenizer_alias] + list(source_roles.values())
    aliases = list(dict.fromkeys(aliases))
    model_paths, identities = materialize_models(benchmark_cfg, aliases)
    model_config_compatibility = validate_model_compatibility(
        paths=model_paths,
        aliases=[base_alias] + list(source_roles.values()),
        trust_remote_code=trust_remote_code,
    )
    tokenizer_info = validate_tokenizer(model_paths[tokenizer_alias], trust_remote_code=trust_remote_code)

    calibration_paths = load_calibration_paths(lot_cfg)
    source_calibrations = {
        str(role): str(subset)
        for role, subset in baseline_spec.get("source_calibrations", {}).items()
    }
    missing_roles = sorted(set(source_roles) - set(source_calibrations))
    if missing_roles:
        raise ValueError(f"{baseline_id} missing source_calibrations for: {', '.join(missing_roles)}")
    selected_calibration_paths = []
    for role, subset_id in source_calibrations.items():
        if subset_id not in calibration_paths:
            raise ValueError(f"unknown calibration subset {subset_id!r} for role {role}")
        selected_calibration_paths.append(calibration_paths[subset_id])
    require_calibration_artifacts(selected_calibration_paths)

    output_dir = OUTPUT_ROOT / baseline_id
    resolved_cfg_path = LOT_CONFIG_ROOT / f"{baseline_id}.yaml"
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"LOT output already exists: {project_path(output_dir, PROJECT_ROOT)}. "
                "Use --overwrite to replace it."
            )
        safe_rmtree(output_dir, OUTPUT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg = {
        "baseline_id": baseline_id,
        "lot_config_id": lot_cfg.get("lot_config_id"),
        "baseline": baseline_spec,
        "base_model": base_alias,
        "source_models": source_roles,
        "tokenizer_source": tokenizer_alias,
        "runtime": {
            **runtime,
            "device": device,
            "dtype": str(runtime.get("dtype", "bfloat16")),
            "solver_device": solver_device.type,
        },
        "features": {
            **feature_cfg,
            "prompts_per_source": prompts_per_source,
            "tokens_per_prompt": tokens_per_prompt,
            "limit_layers": sorted(allowed_layers) if allowed_layers is not None else None,
            "limit_modules": sorted(linear_modules),
        },
        "solver": solver_raw,
        "tensors": tensor_cfg,
    }
    write_yaml(resolved_cfg_path, resolved_cfg)

    # Norm-module hooks: enabled only when norm_modules.strategy is one of the LOT-style
    # strategies that actually consume features. For mean_delta fallback no hook is needed.
    norm_cfg_view = dict(tensor_cfg.get("norm_modules", {}) or {})
    norm_lot_enabled = bool(norm_cfg_view.get("enabled", False))
    norm_strategy_active = str(norm_cfg_view.get("strategy", "mean_delta"))
    norm_modules_for_hook: set[str] = set()
    norm_feature_form = "raw"
    if norm_lot_enabled and norm_strategy_active == "lot_paper_norm":
        # Only the explicitly listed RMSNorm suffixes get hooked; final norm is
        # left to the excluded-strategy fallback (mean_delta by default).
        for suf in norm_cfg_view.get("include_suffixes", []) or []:
            norm_modules_for_hook.add(str(suf))
        norm_feature_form = "rmsnorm_normalized"
    rms_eps = float(solver_raw.get("rms_eps", 1.0e-6))

    print(f"[features] collecting LOT activations for {baseline_id}")
    features_by_role: dict[str, dict[str, torch.Tensor]] = {}
    feature_manifest: dict[str, Any] = {}
    for role, alias in source_roles.items():
        subset_id = source_calibrations[role]
        calibration_file = calibration_paths[subset_id]
        texts = read_calibration_texts(
            calibration_file,
            text_field=str(feature_cfg.get("text_field", "text")),
            limit=prompts_per_source,
        )
        print(f"[features] {role}/{alias} subset={subset_id} prompts={len(texts)}")
        features = extract_features_for_model(
            model_path=model_paths[alias],
            tokenizer_path=model_paths[tokenizer_alias],
            texts=texts,
            allowed_modules=linear_modules,
            allowed_layers=allowed_layers,
            dtype=dtype,
            device_name=device,
            max_length=max_length,
            tokens_per_prompt=tokens_per_prompt,
            batch_size=batch_size,
            trust_remote_code=trust_remote_code,
            allowed_norm_modules=norm_modules_for_hook,
            norm_feature_form=norm_feature_form,
            rms_eps=rms_eps,
        )
        features_by_role[role] = features
        feature_manifest[role] = {
            "model_alias": alias,
            "calibration_id": subset_id,
            "calibration_file": project_path(calibration_file, PROJECT_ROOT),
            "calibration_file_sha256": file_sha256(calibration_file),
            "module_count": len(features),
            "rows_per_module_min": min((int(t.shape[0]) for t in features.values()), default=0),
            "rows_per_module_max": max((int(t.shape[0]) for t in features.values()), default=0),
        }

    print(f"[merge] writing LOT checkpoint {baseline_id} (solver_device={solver_device.type})")
    copy_checkpoint_metadata(model_paths[base_alias], model_paths[tokenizer_alias], output_dir)
    pinv_rcond_raw = solver_raw.get("pinv_rcond", None)
    max_delta_norm_ratio_raw = solver_raw.get("max_delta_norm_ratio", None)
    solver_cfg = LotPaperSolverConfig(
        output_scale=float(solver_raw.get("output_scale", 1.0)),
        rms_eps=float(solver_raw.get("rms_eps", 1.0e-6)),
        div_eps=float(solver_raw.get("div_eps", 1.0e-10)),
        pinv_rcond=None if pinv_rcond_raw is None else float(pinv_rcond_raw),
        fallback_on_nonfinite=bool(solver_raw.get("fallback_on_nonfinite", True)),
        max_delta_norm_ratio=(
            None if max_delta_norm_ratio_raw is None else float(max_delta_norm_ratio_raw)
        ),
    )
    merge_stats = merge_checkpoint(
        base_path=model_paths[base_alias],
        source_paths_by_role={role: model_paths[alias] for role, alias in source_roles.items()},
        features_by_role=features_by_role,
        output_dir=output_dir,
        solver_cfg=solver_cfg,
        tensor_cfg=tensor_cfg,
        linear_modules=linear_modules,
        allowed_layers=allowed_layers,
        linear_strategy=linear_strategy,
        solver_device=solver_device,
    )

    smoke_info = None
    if not args.skip_smoke:
        print(f"[smoke] validating {baseline_id}")
        smoke_info = smoke_validate_checkpoint(
            output_dir=output_dir,
            dtype=dtype,
            device_name=device,
            trust_remote_code=trust_remote_code,
        )

    manifest = {
        "config_id": baseline_id,
        "baseline_id": baseline_id,
        "method_family": baseline_spec.get("method_family", "LOT_paper"),
        "variant": baseline_spec.get("variant"),
        "lot_config": project_path(lot_cfg_path, PROJECT_ROOT),
        "lot_config_sha256": file_sha256(lot_cfg_path),
        "resolved_lot_config": project_path(resolved_cfg_path, PROJECT_ROOT),
        "resolved_lot_config_sha256": file_sha256(resolved_cfg_path),
        "base_model": identities[base_alias],
        "source_models": [
            {"role": role, **identities[alias]}
            for role, alias in source_roles.items()
        ],
        "model_config_compatibility": model_config_compatibility,
        "tokenizer_source": identities[tokenizer_alias],
        "tokenizer_validation": tokenizer_info,
        "calibration_config": project_path(resolve_project_path(str(lot_cfg.get("calibration_config")), PROJECT_ROOT), PROJECT_ROOT),
        "calibration_config_sha256": file_sha256(resolve_project_path(str(lot_cfg.get("calibration_config")), PROJECT_ROOT)),
        "calibration_manifest": project_path(CALIBRATION_MANIFEST, PROJECT_ROOT),
        "calibration_manifest_sha256": file_sha256(CALIBRATION_MANIFEST),
        "non_overlap_report": project_path(NON_OVERLAP_REPORT, PROJECT_ROOT),
        "non_overlap_report_sha256": file_sha256(NON_OVERLAP_REPORT),
        "source_calibrations": feature_manifest,
        "feature_settings": resolved_cfg["features"],
        "solver": solver_raw,
        "tensor_settings": tensor_cfg,
        "merge_stats": merge_stats,
        "smoke_validation": smoke_info,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(PROJECT_ROOT),
        "run": {"exit_code": 0},
    }
    write_json(output_dir / MERGE_MANIFEST, manifest)
    print(f"[done] LOT baseline completed: {baseline_id}")


def main() -> int:
    args = parse_args()
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    if args.self_test:
        lot_math_self_test()
        print("LOT self-test OK")
        return 0

    lot_cfg_path = resolve_project_path(args.config, PROJECT_ROOT)
    lot_cfg = read_yaml(lot_cfg_path)
    benchmark_cfg = read_yaml(BENCHMARK_CONFIG)
    baselines = selected_baselines(lot_cfg, args.baselines)
    failures: list[str] = []

    for baseline_id, spec in baselines.items():
        try:
            build_lot_baseline(
                baseline_id=baseline_id,
                baseline_spec=spec,
                lot_cfg=lot_cfg,
                lot_cfg_path=lot_cfg_path,
                benchmark_cfg=benchmark_cfg,
                args=args,
            )
        except Exception as exc:
            failures.append(baseline_id)
            output_dir = OUTPUT_ROOT / baseline_id
            if not isinstance(exc, FileExistsError):
                output_dir.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "config_id": baseline_id,
                    "baseline_id": baseline_id,
                    "method_family": spec.get("method_family", "LOT_paper") if isinstance(spec, dict) else "LOT_paper",
                    "variant": spec.get("variant") if isinstance(spec, dict) else None,
                    "lot_config": project_path(lot_cfg_path, PROJECT_ROOT),
                    "lot_config_sha256": file_sha256(lot_cfg_path) if lot_cfg_path.exists() else None,
                    "created_at_utc": now_utc(),
                    "run": {
                        "exit_code": 1,
                        "error": repr(exc),
                        "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-80:]),
                    },
                }
                write_json(output_dir / MERGE_MANIFEST, manifest)
            print(f"[error] LOT failed for {baseline_id}: {exc!r}", file=sys.stderr)

    if failures:
        print("[done] LOT completed with failures: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

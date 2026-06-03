#!/usr/bin/env python3
"""Compare activation-similarity metrics for phase-one merged checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.model_registry import model_registry, resolve_model_arg, snapshot_download
from common.io import file_sha256, project_path, read_jsonl, read_yaml, resolve_project_path


CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_CONFIG = CODE_ROOT / "cfg" / "analysis" / "activation_metrics.yaml"
PROJECTION_CACHE: dict[tuple[int, int, int, str], torch.Tensor] = {}

SCORES_HEADER = [
    "model_id",
    "model_kind",
    "calib_subset",
    "target_id",
    "layer",
    "metric",
    "value",
    "n_prompts",
]

FORGETTING_HEADER = SCORES_HEADER + [
    "Domain (i)",
    "F_i",
    "Specialist id",
    "Specialist score",
    "Merged/model score",
]

SUMMARY_FIELDS = [
    "n_layers",
    "n_prompts",
    "layer_mean_distance",
    "layer_median_distance",
    "layer_std_distance",
    "layer_min_distance",
    "layer_max_distance",
    "min_layer",
    "max_layer",
]

PAIR_SUMMARY_KEYS = [
    "model_id",
    "model_kind",
    "metric",
    "calib_subset",
    "target_id",
]
PAIR_SUMMARY_HEADER = PAIR_SUMMARY_KEYS + SUMMARY_FIELDS

FORGETTING_SUMMARY_KEYS = [
    "model_id",
    "model_kind",
    "metric",
    "Domain (i)",
    "calib_subset",
    "target_id",
    "Specialist id",
    "F_i",
    "Specialist score",
    "Merged/model score",
]
FORGETTING_SUMMARY_HEADER = FORGETTING_SUMMARY_KEYS + SUMMARY_FIELDS


@dataclass(frozen=True)
class RuntimeConfig:
    cfg: dict[str, Any]
    cfg_path: Path
    output_dir: Path
    reference_cache_dir: Path
    model_scores_dir: Path
    layers: list[int] | None
    n_prompts: int
    batch_size: int
    max_length: int
    text_field: str
    pooling: str
    normalization: str
    device: str
    torch_dtype: torch.dtype
    trust_remote_code: bool


def load_runtime_config(config_path: Path) -> RuntimeConfig:
    cfg = read_yaml(config_path)
    output_dir = resolve_project_path(cfg.get("output_dir", "results/activation_metrics"), PROJECT_ROOT)
    extraction = cfg.get("activation_extraction", {})
    model_loading = cfg.get("model_loading", {})
    layers_cfg = extraction.get("layers", "all")
    if layers_cfg == "all":
        layers = None
    else:
        layers = [int(x) for x in layers_cfg]
    dtype_name = str(model_loading.get("torch_dtype", "bfloat16"))
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported torch dtype: {dtype_name}")
    normalization = str(extraction.get("normalization", "l2"))
    if normalization != "l2":
        raise ValueError(f"unsupported activation normalization: {normalization}")
    return RuntimeConfig(
        cfg=cfg,
        cfg_path=config_path,
        output_dir=output_dir,
        reference_cache_dir=output_dir / "reference_cache",
        model_scores_dir=output_dir / "model_scores",
        layers=layers,
        n_prompts=int(extraction.get("n_prompts_per_subset", 128)),
        batch_size=int(extraction.get("batch_size", 1)),
        max_length=int(extraction.get("max_length", 512)),
        text_field=str(extraction.get("text_field", "text")),
        pooling=str(extraction.get("pooling", "last_non_padding_token")),
        normalization=normalization,
        device=str(model_loading.get("device", "cuda")),
        torch_dtype=dtype,
        trust_remote_code=bool(model_loading.get("trust_remote_code", True)),
    )


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


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


def benchmark_config_path(cfg: dict[str, Any]) -> Path:
    return resolve_project_path(str(cfg.get("benchmark_config", "code/cfg/benchmark/benchmark.yaml")), PROJECT_ROOT)


def resolve_run_alias(alias: str, cfg: dict[str, Any]) -> str:
    bench_cfg = read_yaml(benchmark_config_path(cfg))
    registry = model_registry(bench_cfg)
    if alias not in registry:
        raise ValueError(f"unknown benchmark model alias {alias!r}")
    spec = registry[alias]
    return snapshot_download(spec["repo"], spec["revision"])


def resolve_reference_targets(rt: RuntimeConfig) -> dict[str, str]:
    out: dict[str, str] = {}
    for target_id, spec in rt.cfg.get("reference_targets", {}).items():
        if "run_alias" not in spec:
            raise ValueError(f"reference target {target_id} must define run_alias")
        out[str(target_id)] = resolve_run_alias(str(spec["run_alias"]), rt.cfg)
    if not out:
        raise ValueError("activation metrics config must define reference_targets")
    return out


def tokenizer_source(rt: RuntimeConfig) -> str:
    alias = str(rt.cfg.get("model_loading", {}).get("tokenizer_alias", "base"))
    return resolve_run_alias(alias, rt.cfg)


def load_tokenizer(rt: RuntimeConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source(rt),
        trust_remote_code=rt.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_path: str, rt: RuntimeConfig):
    from transformers import AutoModelForCausalLM

    device = torch.device(rt.device if torch.cuda.is_available() or rt.device == "cpu" else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        resolve_model_arg(model_path),
        torch_dtype=rt.torch_dtype,
        trust_remote_code=rt.trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model, device


def selected_layers(model: Any, rt: RuntimeConfig) -> list[int]:
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    if rt.layers is None:
        return list(range(n_layers))
    missing = [x for x in rt.layers if x < 0 or x >= n_layers]
    if missing:
        raise ValueError(f"requested layer(s) outside model range 0..{n_layers - 1}: {missing}")
    return rt.layers


def extract_last_token_hidden(
    *,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    rt: RuntimeConfig,
) -> tuple[np.ndarray, list[int]]:
    if rt.pooling != "last_non_padding_token":
        raise ValueError(f"unsupported pooling: {rt.pooling}")

    device = next(model.parameters()).device
    layers = selected_layers(model, rt)
    layer_chunks: list[list[np.ndarray]] = [[] for _ in layers]

    with torch.inference_mode():
        for start in range(0, len(texts), rt.batch_size):
            batch_texts = texts[start : start + rt.batch_size]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=rt.max_length,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            last_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.shape[0], device=device)
            for out_idx, layer in enumerate(layers):
                hidden = outputs.hidden_states[layer + 1][batch_indices, last_indices, :]
                layer_chunks[out_idx].append(hidden.detach().to("cpu", dtype=torch.float16).numpy())
            del outputs, input_ids, attention_mask, encoded

    arrays = [np.concatenate(chunks, axis=0) for chunks in layer_chunks]
    activations = np.stack(arrays, axis=0)
    return activations, layers


def cache_path(cache_root: Path, model_id: str, subset_id: str) -> Path:
    return cache_root / model_id / f"{subset_id}.npz"


def save_activation_cache(path: Path, *, activations: np.ndarray, layers: list[int], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        activations=activations.astype(np.float16, copy=False),
        layers=np.array(layers, dtype=np.int16),
        metadata=json.dumps(metadata, ensure_ascii=True),
    )


def load_activation_cache(path: Path) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing activation cache: {project_path(path, PROJECT_ROOT)}")
    with np.load(path, allow_pickle=False) as data:
        activations = data["activations"]
        layers = [int(x) for x in data["layers"].tolist()]
        metadata = json.loads(str(data["metadata"].item()))
    return activations, layers, metadata


def normalize_activations(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), p=2, dim=-1, eps=1e-12)


def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    return torch.clamp(x2 + y2 - 2.0 * (x @ y.T), min=0.0)


def median_positive_sqdist(z: torch.Tensor) -> float:
    d = pairwise_sq_dists(z, z)
    vals = d[d > 1e-12]
    if vals.numel() == 0:
        return 1.0
    return float(torch.median(vals).detach().cpu())


def metric_tensor(x_np: np.ndarray, device: torch.device | str) -> torch.Tensor:
    return normalize_activations(torch.as_tensor(x_np, device=device))


def mmd_rbf_tensors(x: torch.Tensor, y: torch.Tensor, scales: list[float]) -> float:
    z = torch.cat([x, y], dim=0)
    base_sq = max(median_positive_sqdist(z), 1e-12)
    d_xx = pairwise_sq_dists(x, x)
    d_yy = pairwise_sq_dists(y, y)
    d_xy = pairwise_sq_dists(x, y)
    values = []
    for scale in scales:
        sigma_sq = base_sq * float(scale) * float(scale)
        denom = max(2.0 * sigma_sq, 1e-12)
        k_xx = torch.exp(-d_xx / denom).mean()
        k_yy = torch.exp(-d_yy / denom).mean()
        k_xy = torch.exp(-d_xy / denom).mean()
        values.append(k_xx + k_yy - 2.0 * k_xy)
    out = torch.stack(values).mean()
    return float(torch.clamp(out, min=0.0).detach().cpu())


def projection_matrix(hidden_size: int, n_projections: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(hidden_size, n_projections)).astype(np.float32)
    norms = np.linalg.norm(mat, axis=0, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


def projection_tensor(hidden_size: int, n_projections: int, seed: int, device: torch.device | str) -> torch.Tensor:
    device_obj = torch.device(device)
    key = (hidden_size, n_projections, seed, str(device_obj))
    cached = PROJECTION_CACHE.get(key)
    if cached is not None:
        return cached
    tensor = torch.as_tensor(projection_matrix(hidden_size, n_projections, seed), device=device_obj)
    PROJECTION_CACHE[key] = tensor
    return tensor


def projected_js_tensors(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_projections: int,
    bins: int,
    epsilon: float,
    seed: int,
) -> float:
    proj = projection_tensor(x.shape[1], n_projections, seed, x.device)
    xp = x @ proj
    yp = y @ proj
    vals: list[float] = []
    for j in range(n_projections):
        combined = torch.cat([xp[:, j], yp[:, j]])
        lo = float(combined.min().detach().cpu())
        hi = float(combined.max().detach().cpu())
        if math.isclose(lo, hi):
            vals.append(0.0)
            continue
        p = torch.histc(xp[:, j], bins=bins, min=lo, max=hi).float() + float(epsilon)
        q = torch.histc(yp[:, j], bins=bins, min=lo, max=hi).float() + float(epsilon)
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        js = 0.5 * torch.sum(p * torch.log(p / m)) + 0.5 * torch.sum(q * torch.log(q / m))
        vals.append(float(js.detach().cpu()))
    return float(np.mean(vals)) if vals else 0.0


def metric_configs(rt: RuntimeConfig) -> dict[str, dict[str, Any]]:
    metrics = {k: v for k, v in rt.cfg.get("metrics", {}).items() if isinstance(v, dict) and v.get("enabled", True)}
    if "mmd_rbf" in metrics and str(metrics["mmd_rbf"].get("bandwidth", "median")) != "median":
        raise ValueError("mmd_rbf currently supports only bandwidth: median")
    return metrics


def compute_metric_rows(
    *,
    model_id: str,
    model_kind: str,
    subset_id: str,
    target_id: str,
    candidate: np.ndarray,
    target: np.ndarray,
    layers: list[int],
    rt: RuntimeConfig,
    metric_device: str,
) -> list[dict[str, Any]]:
    if candidate.shape != target.shape:
        raise ValueError(
            f"shape mismatch for {model_id} vs {target_id} on {subset_id}: "
            f"{candidate.shape} != {target.shape}"
        )
    metrics = metric_configs(rt)
    out: list[dict[str, Any]] = []
    n_prompts = int(candidate.shape[1])
    mmd_cfg = metrics.get("mmd_rbf")
    js_cfg = metrics.get("projected_js")
    for layer_index, layer_id in enumerate(layers):
        x = candidate[layer_index]
        y = target[layer_index]
        x_t = metric_tensor(x, metric_device)
        y_t = metric_tensor(y, metric_device)
        if mmd_cfg is not None:
            value = mmd_rbf_tensors(
                x_t,
                y_t,
                scales=[float(v) for v in mmd_cfg.get("scales", [1.0])],
            )
            out.append(
                score_row(
                    model_id,
                    model_kind,
                    subset_id,
                    target_id,
                    layer_id,
                    "mmd_rbf",
                    value,
                    n_prompts,
                )
            )
        if js_cfg is not None:
            value = projected_js_tensors(
                x_t,
                y_t,
                n_projections=int(js_cfg.get("n_projections", 64)),
                bins=int(js_cfg.get("bins", 32)),
                epsilon=float(js_cfg.get("epsilon", 1.0e-8)),
                seed=int(js_cfg.get("projection_seed", 1234)),
            )
            out.append(
                score_row(
                    model_id,
                    model_kind,
                    subset_id,
                    target_id,
                    layer_id,
                    "projected_js",
                    value,
                    n_prompts,
                )
            )
        del x_t, y_t
    return out


def score_row(
    model_id: str,
    model_kind: str,
    subset_id: str,
    target_id: str,
    layer: int,
    metric: str,
    value: float,
    n_prompts: int,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "model_kind": model_kind,
        "calib_subset": subset_id,
        "target_id": target_id,
        "layer": layer,
        "metric": metric,
        "value": value,
        "n_prompts": n_prompts,
    }


def write_score_file(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORES_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in SCORES_HEADER})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def combine_model_scores(rt: RuntimeConfig) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not rt.model_scores_dir.exists():
        return rows
    for path in sorted(rt.model_scores_dir.glob("*.csv")):
        rows.extend(read_csv_rows(path))
    return rows


def require_reference_caches(rt: RuntimeConfig) -> None:
    missing: list[str] = []
    for target_id in rt.cfg.get("reference_targets", {}):
        for subset_id in rt.cfg.get("calibration_subsets", {}):
            path = cache_path(rt.reference_cache_dir, str(target_id), str(subset_id))
            if not path.exists():
                missing.append(project_path(path, PROJECT_ROOT))
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "missing activation reference cache. Run build-reference-cache first:\n"
            f"  - {joined}"
        )


def build_reference_cache(args: argparse.Namespace) -> int:
    rt = load_runtime_config(Path(args.config))
    rt.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(rt)
    references = resolve_reference_targets(rt)
    subsets = rt.cfg.get("calibration_subsets", {})
    if not subsets:
        raise ValueError("activation metrics config must define calibration_subsets")

    for target_id, model_path in references.items():
        target_cache_paths = [
            cache_path(rt.reference_cache_dir, target_id, str(subset_id))
            for subset_id in subsets
        ]
        if not args.overwrite and all(path.exists() for path in target_cache_paths):
            print(f"[skip] reference caches exist for {target_id}")
            continue
        model, _ = load_model(model_path, rt)
        try:
            for subset_id, subset_cfg in subsets.items():
                out_path = cache_path(rt.reference_cache_dir, target_id, str(subset_id))
                if out_path.exists() and not args.overwrite:
                    print(f"[skip] reference cache exists: {project_path(out_path, PROJECT_ROOT)}")
                    continue
                subset_path = resolve_project_path(str(subset_cfg["file"]), PROJECT_ROOT)
                texts = read_calibration_texts(subset_path, text_field=rt.text_field, limit=rt.n_prompts)
                activations, layers = extract_last_token_hidden(model=model, tokenizer=tokenizer, texts=texts, rt=rt)
                metadata = {
                    "target_id": target_id,
                    "model_path": model_path,
                    "calib_subset": subset_id,
                    "calib_file": project_path(subset_path, PROJECT_ROOT),
                    "calib_file_sha256": file_sha256(subset_path),
                    "n_prompts": rt.n_prompts,
                    "max_length": rt.max_length,
                    "layers": layers,
                    "created_at_utc": now_utc(),
                }
                save_activation_cache(out_path, activations=activations, layers=layers, metadata=metadata)
                print(f"[cache] {target_id}/{subset_id}: {project_path(out_path, PROJECT_ROOT)}")
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.score_references:
        for model_id in references:
            score_cached_model(
                rt,
                model_id=model_id,
                model_kind="reference",
                activation_cache_root=rt.reference_cache_dir,
                overwrite=args.overwrite,
                metric_device=args.metric_device,
            )
        summarize(args)
    write_manifest(rt)
    return 0


def score_cached_model(
    rt: RuntimeConfig,
    *,
    model_id: str,
    model_kind: str,
    activation_cache_root: Path,
    overwrite: bool,
    metric_device: str,
) -> None:
    score_path = rt.model_scores_dir / f"{model_id}.csv"
    if score_path.exists() and not overwrite:
        print(f"[skip] score file exists: {project_path(score_path, PROJECT_ROOT)}")
        return
    rows: list[dict[str, Any]] = []
    references = rt.cfg.get("reference_targets", {})
    for subset_id in rt.cfg.get("calibration_subsets", {}):
        cand, layers, _ = load_activation_cache(cache_path(activation_cache_root, model_id, str(subset_id)))
        for target_id in references:
            target, target_layers, _ = load_activation_cache(cache_path(rt.reference_cache_dir, str(target_id), str(subset_id)))
            if layers != target_layers:
                raise ValueError(f"layer mismatch for {model_id} vs {target_id} on {subset_id}")
            rows.extend(
                compute_metric_rows(
                    model_id=model_id,
                    model_kind=model_kind,
                    subset_id=str(subset_id),
                    target_id=str(target_id),
                    candidate=cand,
                    target=target,
                    layers=layers,
                    rt=rt,
                    metric_device=metric_device,
                )
            )
            del target
        del cand
    write_score_file(score_path, rows)
    print(f"[scores] {model_id}: {project_path(score_path, PROJECT_ROOT)} rows={len(rows)}")


def score_model(args: argparse.Namespace) -> int:
    rt = load_runtime_config(Path(args.config))
    score_path = rt.model_scores_dir / f"{args.config_id}.csv"
    if score_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"activation score file already exists: {project_path(score_path, PROJECT_ROOT)}. "
            "Use --overwrite to replace it."
        )
    require_reference_caches(rt)

    tokenizer = load_tokenizer(rt)
    model, _ = load_model(args.model, rt)
    rows: list[dict[str, Any]] = []
    references = rt.cfg.get("reference_targets", {})
    try:
        for subset_id, subset_cfg in rt.cfg.get("calibration_subsets", {}).items():
            subset_path = resolve_project_path(str(subset_cfg["file"]), PROJECT_ROOT)
            texts = read_calibration_texts(subset_path, text_field=rt.text_field, limit=rt.n_prompts)
            activations, layers = extract_last_token_hidden(model=model, tokenizer=tokenizer, texts=texts, rt=rt)
            print(f"[candidate] {args.config_id}/{subset_id}: activations={activations.shape}")
            for target_id in references:
                target, target_layers, _ = load_activation_cache(cache_path(rt.reference_cache_dir, str(target_id), str(subset_id)))
                if layers != target_layers:
                    raise ValueError(f"layer mismatch for {args.config_id} vs {target_id} on {subset_id}")
                rows.extend(
                    compute_metric_rows(
                        model_id=args.config_id,
                        model_kind="candidate",
                        subset_id=str(subset_id),
                        target_id=str(target_id),
                        candidate=activations,
                        target=target,
                        layers=layers,
                        rt=rt,
                        metric_device=args.metric_device,
                    )
                )
                del target
            del activations
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_score_file(score_path, rows)
    print(f"[scores] {args.config_id}: {project_path(score_path, PROJECT_ROOT)} rows={len(rows)}")
    write_manifest(rt)
    return 0


def join_forgetting(raw_rows: list[dict[str, str]], rt: RuntimeConfig) -> list[dict[str, Any]]:
    bench_paths = rt.cfg.get("benchmark_results", {})
    forgetting_path = resolve_project_path(str(bench_paths.get("forgetting", "results/benchmark_forgetting.csv")), PROJECT_ROOT)
    if not forgetting_path.exists():
        print(f"[warning] missing forgetting CSV: {project_path(forgetting_path, PROJECT_ROOT)}")
        return []

    domain_to_subset = {str(k): str(v) for k, v in rt.cfg.get("forgetting_domains", {}).items()}
    if not domain_to_subset:
        raise ValueError("activation metrics config must define forgetting_domains")

    by_pair: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in raw_rows:
        by_pair.setdefault(
            (
                str(row["model_id"]),
                str(row["target_id"]),
                str(row["calib_subset"]),
            ),
            []
        ).append(row)
    joined: list[dict[str, Any]] = []
    for frow in read_csv_rows(forgetting_path):
        domain = str(frow.get("Domain (i)", ""))
        if domain not in domain_to_subset:
            raise ValueError(f"no forgetting_domains mapping for {domain!r}")
        key = (
            str(frow.get("Config id", "")),
            str(frow.get("Specialist id", "")),
            domain_to_subset[domain],
        )
        for row in by_pair.get(key, []):
            out = dict(row)
            for col in ["Domain (i)", "F_i", "Specialist id", "Specialist score", "Merged/model score"]:
                out[col] = frow.get(col)
            joined.append(out)
    return joined


def parse_distance(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid activation distance value: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"non-finite activation distance value: {value!r}")
    return out


def summarize_layer_distances(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[tuple[int, float, str]]] = {}
    for row in rows:
        value = parse_distance(row.get("value"))
        try:
            layer = int(str(row.get("layer", "")))
        except ValueError as exc:
            raise ValueError(f"invalid layer value in activation row: {row.get('layer')!r}") from exc
        key = tuple(str(row.get(k, "")) for k in group_keys)
        groups.setdefault(key, []).append((layer, value, str(row.get("n_prompts", ""))))

    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        values = np.array([value for _, value, _ in items], dtype=np.float64)
        min_layer, min_value, _ = min(items, key=lambda item: (item[1], item[0]))
        max_layer, max_value, _ = min(items, key=lambda item: (-item[1], item[0]))
        n_prompts = sorted({prompts for _, _, prompts in items})
        row = {name: key[idx] for idx, name in enumerate(group_keys)}
        row.update(
            {
                "n_layers": len({layer for layer, _, _ in items}),
                "n_prompts": n_prompts[0] if len(n_prompts) == 1 else ";".join(n_prompts),
                "layer_mean_distance": float(np.mean(values)),
                "layer_median_distance": float(np.median(values)),
                "layer_std_distance": float(np.std(values)),
                "layer_min_distance": min_value,
                "layer_max_distance": max_value,
                "min_layer": min_layer,
                "max_layer": max_layer,
            }
        )
        out.append(row)
    return out


def summarize(args: argparse.Namespace) -> int:
    rt = load_runtime_config(Path(args.config))
    rt.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = combine_model_scores(rt)
    scores_path = rt.output_dir / "activation_metric_scores.csv"
    write_csv_rows(scores_path, raw_rows, SCORES_HEADER)
    if not raw_rows:
        print("[warning] no model score CSVs found")

    pair_summary_rows = summarize_layer_distances(raw_rows, PAIR_SUMMARY_KEYS)
    pair_summary_path = rt.output_dir / "activation_metric_pair_summary.csv"
    write_csv_rows(pair_summary_path, pair_summary_rows, PAIR_SUMMARY_HEADER)

    forgetting_rows = join_forgetting(raw_rows, rt)
    forgetting_path = rt.output_dir / "activation_metric_forgetting.csv"
    write_csv_rows(forgetting_path, forgetting_rows, FORGETTING_HEADER)

    forgetting_summary_rows = summarize_layer_distances(forgetting_rows, FORGETTING_SUMMARY_KEYS)
    forgetting_summary_path = rt.output_dir / "activation_metric_forgetting_summary.csv"
    write_csv_rows(forgetting_summary_path, forgetting_summary_rows, FORGETTING_SUMMARY_HEADER)

    write_manifest(rt)
    for path in [
        scores_path,
        pair_summary_path,
        forgetting_path,
        forgetting_summary_path,
    ]:
        if path.exists():
            print(f"[done] wrote {project_path(path, PROJECT_ROOT)}")
    return 0


def write_manifest(rt: RuntimeConfig) -> None:
    calib = {}
    for subset_id, subset_cfg in rt.cfg.get("calibration_subsets", {}).items():
        path = resolve_project_path(str(subset_cfg["file"]), PROJECT_ROOT)
        calib[subset_id] = {
            "file": project_path(path, PROJECT_ROOT),
            "sha256": file_sha256(path) if path.exists() else None,
        }
    manifest = {
        "analysis_id": rt.cfg.get("analysis_id"),
        "created_or_updated_at_utc": now_utc(),
        "config": project_path(rt.cfg_path, PROJECT_ROOT),
        "config_sha256": file_sha256(rt.cfg_path),
        "benchmark_config": project_path(benchmark_config_path(rt.cfg), PROJECT_ROOT),
        "benchmark_config_sha256": file_sha256(benchmark_config_path(rt.cfg)),
        "calibration_subsets": calib,
        "activation_extraction": rt.cfg.get("activation_extraction", {}),
        "metrics": rt.cfg.get("metrics", {}),
        "forgetting_domains": rt.cfg.get("forgetting_domains", {}),
    }
    write_json(rt.output_dir / "run_manifest.json", manifest)


def self_test(_: argparse.Namespace) -> int:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 32)).astype(np.float32)
    y = x.copy()
    z = rng.normal(loc=2.0, size=(16, 32)).astype(np.float32)
    x_t = metric_tensor(x, "cpu")
    y_t = metric_tensor(y, "cpu")
    z_t = metric_tensor(z, "cpu")
    same_mmd = mmd_rbf_tensors(x_t, y_t, [0.5, 1.0, 2.0])
    diff_mmd = mmd_rbf_tensors(x_t, z_t, [0.5, 1.0, 2.0])
    same_js = projected_js_tensors(x_t, y_t, n_projections=8, bins=8, epsilon=1.0e-8, seed=1)
    diff_js = projected_js_tensors(x_t, z_t, n_projections=8, bins=8, epsilon=1.0e-8, seed=1)
    print("same_mmd:", same_mmd)
    print("diff_mmd:", diff_mmd)
    print("same_js:", same_js)
    print("diff_js:", diff_js)
    assert same_mmd <= diff_mmd, "MMD self-test failed"
    assert same_js <= diff_js, "projected JS self-test failed"
    print("self-test OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="activation metrics YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ref = sub.add_parser("build-reference-cache", help="cache base/specialist hidden states")
    p_ref.add_argument("--overwrite", action="store_true", help="overwrite existing reference caches and scores")
    p_ref.add_argument("--score-references", dest="score_references", action="store_true", default=True, help="also score reference models")
    p_ref.add_argument("--no-score-references", dest="score_references", action="store_false", help="only build caches")
    p_ref.add_argument("--metric-device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_ref.set_defaults(func=build_reference_cache)

    p_score = sub.add_parser("score-model", help="score one local checkpoint against reference caches")
    p_score.add_argument("--model", required=True, help="local checkpoint path")
    p_score.add_argument("--config-id", required=True, help="model id used in output tables")
    p_score.add_argument("--overwrite", action="store_true", help="overwrite existing model score file")
    p_score.add_argument("--metric-device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_score.add_argument("--gpu-ids", default=None, help="accepted for Colab pipeline compatibility; use CUDA_VISIBLE_DEVICES externally")
    p_score.set_defaults(func=score_model)

    p_sum = sub.add_parser("summarize", help="combine raw score files and add domain-matched forgetting rows")
    p_sum.set_defaults(func=summarize)

    p_test = sub.add_parser("self-test", help="run lightweight metric checks")
    p_test.set_defaults(func=self_test)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if getattr(args, "gpu_ids", None):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

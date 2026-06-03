#!/usr/bin/env python3
"""Build mergekit baseline checkpoints from the phase-one model set."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.model_registry import model_registry
from common.io import project_path, read_yaml


CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT.parent
BENCHMARK_CONFIG = CODE_ROOT / "cfg" / "benchmark" / "benchmark.yaml"
MERGE_CONFIG = CODE_ROOT / "cfg" / "merge" / "mergekit_baselines.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "merged"
MERGEKIT_CONFIG_ROOT = PROJECT_ROOT / "outputs" / "mergekit_configs"

MERGEKIT_RANDOM_SEED = 1
MERGEKIT_DTYPE = "bfloat16"
MERGE_MANIFEST = "merge_manifest.json"

MODEL_PARAMETER_KEYS = {"weight", "density", "epsilon", "gamma"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mergekit baselines")
    p.add_argument("baselines", nargs="*", help="baseline id(s); defaults to all baselines")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing checkpoint output")
    return p.parse_args()


def selected_baselines(merge_cfg: dict[str, Any], requested: list[str]) -> dict[str, dict[str, Any]]:
    baselines = merge_cfg.get("baselines")
    if not isinstance(baselines, dict):
        raise ValueError("merge config must define baselines")
    out = {str(k): v for k, v in baselines.items() if isinstance(v, dict)}
    if not requested:
        return out
    missing = [b for b in requested if b not in out]
    if missing:
        raise ValueError("unknown baseline(s): " + ", ".join(missing))
    return {b: out[b] for b in requested}


def baseline_variant(spec: dict[str, Any]) -> str:
    if "random" in spec:
        return "random_profile"
    return "fixed"


def uses_base_model(spec: dict[str, Any]) -> bool:
    return str(spec.get("mergekit_method")) != "linear"


def candidate_id(baseline_id: str, variant: str, params: dict[str, Any]) -> str:
    if variant == "fixed":
        return baseline_id
    if variant == "random_profile":
        return f"{baseline_id}__seed{params['seed']}"
    raise ValueError(f"unsupported baseline variant for final build: {variant}")


def random_profile(value: Any, rng: random.Random, n: int) -> list[float]:
    if isinstance(value, list) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
        lo, hi = float(value[0]), float(value[1])
        return [round(rng.uniform(lo, hi), 6) for _ in range(n)]
    if isinstance(value, list) and value:
        return [rng.choice(value) for _ in range(n)]
    if isinstance(value, (int, float)):
        return [float(value) for _ in range(n)]
    raise ValueError(f"random profile value must be a range/list/scalar, got {value!r}")


def random_candidates(baseline_id: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    base_params = dict(spec.get("parameters", {}))
    random_cfg = spec.get("random")
    if not isinstance(random_cfg, dict):
        raise ValueError(f"{baseline_id} must define random")
    seeds = random_cfg.get("seeds", [])
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"{baseline_id}.random.seeds must be a non-empty list")
    n = int(random_cfg.get("profile_points", 0))
    if n <= 0:
        raise ValueError(f"{baseline_id}.random.profile_points must be positive")
    candidates = []
    for seed in seeds:
        rng = random.Random(int(seed))
        params = dict(base_params)
        params.update({"seed": int(seed), "profile_points": n})
        for key, value in random_cfg.items():
            if key in {"seeds", "profile_points"}:
                continue
            params[key] = random_profile(value, rng, n) if key in MODEL_PARAMETER_KEYS else value
        candidates.append(params)
    return candidates


def candidates_for_baseline(baseline_id: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    if "grid" in spec:
        raise ValueError(
            f"{baseline_id} still has an unfrozen grid. "
            "Run code/src/merge/run_selection_search.py, then code/src/merge/select_best_merge.py."
        )
    variant = baseline_variant(spec)
    if variant == "fixed":
        return [dict(spec.get("parameters", {}))]
    return random_candidates(baseline_id, spec)


def split_parameters(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model_params: dict[str, Any] = {}
    global_params: dict[str, Any] = {}
    for key, value in params.items():
        if key in {"seed", "profile_points"}:
            continue
        if key in MODEL_PARAMETER_KEYS:
            model_params[key] = value
        else:
            global_params[key] = value
    return model_params, global_params


def materialize_models(benchmark_cfg: dict[str, Any], aliases: list[str]) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    registry = model_registry(benchmark_cfg)
    missing = [a for a in aliases if a not in registry]
    if missing:
        raise ValueError("model alias not found in benchmark.yaml: " + ", ".join(missing))
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required to download pinned model snapshots") from exc

    paths: dict[str, str] = {}
    identities: dict[str, dict[str, Any]] = {}
    for alias in aliases:
        spec = registry[alias]
        local_path = snapshot_download(repo_id=spec["repo"], revision=spec["revision"])
        paths[alias] = str(Path(local_path).resolve())
        identities[alias] = {"alias": alias, "repo": spec["repo"], "revision": spec["revision"]}
    return paths, identities


def build_mergekit_config(
    *,
    spec: dict[str, Any],
    params: dict[str, Any],
    model_paths: dict[str, str],
    source_aliases: list[str],
    base_alias: str,
) -> dict[str, Any]:
    model_params, global_params = split_parameters(params)
    if spec["mergekit_method"] == "linear" and "weight" not in model_params:
        model_params["weight"] = round(1.0 / len(source_aliases), 12)
    cfg: dict[str, Any] = {
        "merge_method": spec["mergekit_method"],
        "models": [{"model": model_paths[alias], "parameters": dict(model_params)} for alias in source_aliases],
        "dtype": MERGEKIT_DTYPE,
    }
    if uses_base_model(spec):
        cfg["base_model"] = model_paths[base_alias]
    if global_params:
        cfg["parameters"] = global_params
    return cfg


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def run_mergekit(config_path: Path, output_dir: Path) -> dict[str, Any]:
    try:
        from mergekit.config import MergeConfiguration  # type: ignore
        from mergekit.merge import run_merge  # type: ignore
        from mergekit.options import MergeOptions  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mergekit is required to build baseline checkpoints") from exc

    config_source = config_path.read_text(encoding="utf-8")
    merge_config = MergeConfiguration.model_validate(yaml.safe_load(config_source))
    options = MergeOptions(
        trust_remote_code=True,
        random_seed=MERGEKIT_RANDOM_SEED,
    )
    try:
        options.apply_global_options()
        run_merge(merge_config, str(output_dir), options=options, config_source=config_source)
        return {"exit_code": 0}
    except Exception as exc:
        tb = traceback.format_exc().splitlines()
        return {
            "exit_code": 1,
            "error": repr(exc),
            "traceback_tail": "\n".join(tb[-80:]),
        }


def write_manifest(
    output_dir: Path,
    *,
    baseline_id: str,
    config_id: str,
    spec: dict[str, Any],
    params: dict[str, Any],
    identities: dict[str, dict[str, Any]],
    source_aliases: list[str],
    base_alias: str,
    mergekit_yaml_path: Path,
    run_result: dict[str, Any],
) -> None:
    run_info = {"exit_code": run_result["exit_code"]}
    if run_result["exit_code"] != 0:
        for key in ("error", "traceback_tail"):
            if key in run_result:
                run_info[key] = run_result[key]
    manifest = {
        "config_id": config_id,
        "baseline_id": baseline_id,
        "method_family": spec.get("method_family"),
        "mergekit_method": spec.get("mergekit_method"),
        "variant": baseline_variant(spec),
        "parameters": params,
        "mergekit_random_seed": MERGEKIT_RANDOM_SEED,
        "base_model": identities.get(base_alias) if uses_base_model(spec) else None,
        "source_models": [identities[a] for a in source_aliases],
        "mergekit_yaml": project_path(mergekit_yaml_path, PROJECT_ROOT),
        "run": run_info,
    }
    with (output_dir / MERGE_MANIFEST).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)


def main() -> int:
    args = parse_args()
    merge_cfg = read_yaml(MERGE_CONFIG)
    benchmark_cfg = read_yaml(BENCHMARK_CONFIG)
    selected = selected_baselines(merge_cfg, args.baselines)
    base_alias = str(merge_cfg["base_model"])
    source_aliases = [str(x) for x in merge_cfg["source_models"]]

    expanded: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
    for baseline_id, spec in selected.items():
        variant = baseline_variant(spec)
        for params in candidates_for_baseline(baseline_id, spec):
            config_id = candidate_id(baseline_id, variant, params)
            expanded.append((baseline_id, spec, params, config_id))
    print(f"[info] merge candidates: {len(expanded)}")

    existing_outputs = []
    for _, _, _, config_id in expanded:
        out_dir = OUTPUT_ROOT / config_id
        if out_dir.exists():
            existing_outputs.append(out_dir)
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "output already exists: "
            + ", ".join(project_path(path, PROJECT_ROOT) for path in existing_outputs)
            + ". Use --overwrite to replace it."
        )

    aliases = [base_alias] + source_aliases
    model_paths, identities = materialize_models(benchmark_cfg, aliases)
    failures: list[str] = []
    for baseline_id, spec, params, config_id in expanded:
        out_dir = OUTPUT_ROOT / config_id
        if out_dir.exists():
            shutil.rmtree(out_dir)

        mergekit_yaml = build_mergekit_config(
            spec=spec,
            params=params,
            model_paths=model_paths,
            source_aliases=source_aliases,
            base_alias=base_alias,
        )
        mergekit_yaml_path = MERGEKIT_CONFIG_ROOT / f"{config_id}.yaml"
        write_yaml(mergekit_yaml_path, mergekit_yaml)

        print(f"[run] {config_id}")
        result = run_mergekit(mergekit_yaml_path, out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(
            out_dir,
            baseline_id=baseline_id,
            config_id=config_id,
            spec=spec,
            params=params,
            identities=identities,
            source_aliases=source_aliases,
            base_alias=base_alias,
            mergekit_yaml_path=mergekit_yaml_path,
            run_result=result,
        )
        if result["exit_code"] != 0:
            failures.append(config_id)
            print(f"[warning] mergekit failed for {config_id}; see {project_path(out_dir / MERGE_MANIFEST, PROJECT_ROOT)}")

    if failures:
        print("[done] completed with failures: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

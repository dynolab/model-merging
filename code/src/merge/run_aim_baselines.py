#!/usr/bin/env python3
"""Run AIM on frozen `_best` merge checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.model_registry import model_registry
from common.io import file_sha256, git_commit, project_path, read_json, read_jsonl, read_yaml


CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT.parent

BENCHMARK_CONFIG = CODE_ROOT / "cfg" / "benchmark" / "benchmark.yaml"
MERGE_CONFIG = CODE_ROOT / "cfg" / "merge" / "mergekit_baselines.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "merged"
AIM_REPO = PROJECT_ROOT / "external" / "ActivationInformedMerging"
AIM_PATCH_SCRIPT = CODE_ROOT / "patches" / "activation_informed_merging" / "apply_local_calibration_patch.py"

CALIBRATION_ROOT = PROJECT_ROOT / "data" / "calibration"
CALIBRATION_MANIFEST = PROJECT_ROOT / "data" / "calibration" / "calibration_manifest.json"
NON_OVERLAP_REPORT = PROJECT_ROOT / "data" / "calibration" / "non_overlap_report.json"

AIM_CALIBRATIONS = {
    "general": CALIBRATION_ROOT / "calib_general.jsonl",
    "instruction": CALIBRATION_ROOT / "calib_instruction.jsonl",
    "reasoning": CALIBRATION_ROOT / "calib_reasoning.jsonl",
    "uncensored_refusal_like": CALIBRATION_ROOT / "calib_uncensored_refusal_like.jsonl",
    "mix": CALIBRATION_ROOT / "calib_mix.jsonl",
}
AIM_OMEGA = 0.4
AIM_DEVICE = "cuda"
AIM_DTYPE = "bfloat16"
CALIBRATION_TEXT_FIELD = "text"

AIM_BASELINES = {
    "task_arithmetic_best": "aim_on_task_arithmetic",
    "ties_best": "aim_on_ties",
    "dare_best": "aim_on_dare",
    "della_best": "aim_on_della",
    "breadcrumbs_best": "aim_on_breadcrumbs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AIM on one frozen best merge checkpoint with one calibration corpus. "
            "One invocation creates exactly one AIM checkpoint."
        )
    )
    parser.add_argument("baseline", help="single parent `_best` baseline id, e.g. ties_best")
    parser.add_argument(
        "--calibration",
        required=True,
        choices=sorted(AIM_CALIBRATIONS),
        help="single AIM calibration id",
    )
    parser.add_argument("--device", default=AIM_DEVICE, help="device for AIM calibration forward passes")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing AIM output checkpoint")
    return parser.parse_args()


def selected_aim_baseline(requested: str) -> tuple[str, str]:
    reverse = {v: k for k, v in AIM_BASELINES.items()}
    if requested in AIM_BASELINES:
        return requested, AIM_BASELINES[requested]
    if requested in reverse:
        return reverse[requested], requested
    raise ValueError(f"unknown AIM baseline: {requested}")


def selected_calibration(calibration_id: str) -> Path:
    try:
        return AIM_CALIBRATIONS[calibration_id]
    except KeyError as exc:
        raise ValueError(f"unknown AIM calibration id: {calibration_id}") from exc


def aim_output_id(base_output_id: str, calibration_id: str) -> str:
    if calibration_id == "mix":
        return base_output_id
    return f"{base_output_id}__calib_{calibration_id}"


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


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {project_path(path, PROJECT_ROOT)}")


def require_patched_aim_repo() -> None:
    perform_aim = AIM_REPO / "performAIM.py"
    utils = AIM_REPO / "MergeModels" / "ActivationMerging" / "_utils.py"
    require_file(perform_aim, "AIM performAIM.py")
    require_file(utils, "AIM _utils.py")
    perform_text = perform_aim.read_text(encoding="utf-8")
    utils_text = utils.read_text(encoding="utf-8")
    if (
        "--calibration_file" not in perform_text
        or "--device" not in perform_text
        or "merged_tokenizer.save_pretrained(args.save_path)" not in perform_text
    ):
        raise RuntimeError(
            "ActivationInformedMerging is not patched. Run "
            f"`python {project_path(AIM_PATCH_SCRIPT, PROJECT_ROOT)}` first."
        )
    if (
        "calibration_file=None" not in utils_text
        or "base_model.config.use_cache = False" not in utils_text
        or "base_model = base_model.to(device)" not in utils_text
    ):
        raise RuntimeError(
            "ActivationInformedMerging _utils.py is not patched for local calibration + GPU-safe device execution. "
            f"Run `python {project_path(AIM_PATCH_SCRIPT, PROJECT_ROOT)}` first."
        )


def require_calibration(calibration_file: Path) -> None:
    require_file(calibration_file, "calibration corpus")
    require_file(CALIBRATION_MANIFEST, "calibration manifest")
    require_file(NON_OVERLAP_REPORT, "non-overlap report")
    for row in read_jsonl(calibration_file):
        text = row.get(CALIBRATION_TEXT_FIELD)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"first calibration row must contain a non-empty {CALIBRATION_TEXT_FIELD!r} field"
            )
        return
    raise ValueError(f"calibration corpus is empty: {project_path(calibration_file, PROJECT_ROOT)}")


def require_parent_checkpoint(parent_id: str, merge_cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    spec = merge_cfg.get("baselines", {}).get(parent_id)
    if not isinstance(spec, dict):
        raise ValueError(f"parent baseline is not defined in merge config: {parent_id}")
    if "grid" in spec:
        raise ValueError(
            f"{parent_id} still has an unfrozen grid. Run selection search and select_best_merge.py first."
        )
    parent_dir = OUTPUT_ROOT / parent_id
    manifest_path = parent_dir / "merge_manifest.json"
    require_file(manifest_path, f"{parent_id} merge manifest")
    if not checkpoint_complete(parent_dir):
        raise RuntimeError(f"{parent_id} is not a complete local HF checkpoint: {project_path(parent_dir, PROJECT_ROOT)}")
    return parent_dir, read_json(manifest_path)


def base_model_snapshot(merge_cfg: dict[str, Any], benchmark_cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    base_alias = str(merge_cfg["base_model"])
    registry = model_registry(benchmark_cfg)
    if base_alias not in registry:
        raise ValueError(f"base alias {base_alias!r} is not defined in benchmark config")
    spec = registry[base_alias]
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required to materialize the pinned base model") from exc
    local_path = snapshot_download(repo_id=spec["repo"], revision=spec["revision"])
    identity = {"alias": base_alias, "repo": spec["repo"], "revision": spec["revision"]}
    return str(Path(local_path).resolve()), identity


def checkpoint_complete(checkpoint_dir: Path) -> bool:
    if not (checkpoint_dir / "config.json").exists():
        return False
    has_weights = any(checkpoint_dir.glob("*.safetensors")) or any(checkpoint_dir.glob("pytorch_model*.bin"))
    has_tokenizer = (
        (checkpoint_dir / "tokenizer_config.json").exists()
        and (
            (checkpoint_dir / "tokenizer.json").exists()
            or (checkpoint_dir / "tokenizer.model").exists()
            or (checkpoint_dir / "vocab.json").exists()
        )
    )
    return has_weights and has_tokenizer


def run_aim(parent_dir: Path, output_dir: Path, base_path: str, calibration_file: Path, device: str) -> int:
    cmd = [
        sys.executable,
        str((AIM_REPO / "performAIM.py").resolve()),
        "--merged_model",
        str(parent_dir.resolve()),
        "--pretrained_model_name",
        base_path,
        "--omega",
        str(AIM_OMEGA),
        "--save_path",
        str(output_dir.resolve()),
        "--calibration_file",
        str(calibration_file.resolve()),
        "--calibration_text_field",
        CALIBRATION_TEXT_FIELD,
        "--device",
        device,
    ]
    print("[aim] " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(AIM_REPO), check=False).returncode


def write_aim_manifest(
    output_dir: Path,
    *,
    output_id: str,
    parent_id: str,
    parent_dir: Path,
    parent_manifest: dict[str, Any],
    base_identity: dict[str, Any],
    calibration_id: str,
    calibration_file: Path,
    device: str,
) -> None:
    manifest = {
        "config_id": output_id,
        "baseline_id": output_id,
        "method_family": "AIM",
        "variant": "aim_post_merge",
        "calibration_id": calibration_id,
        "parent_config_id": parent_id,
        "parent_checkpoint": project_path(parent_dir, PROJECT_ROOT),
        "parent_manifest": project_path(parent_dir / "merge_manifest.json", PROJECT_ROOT),
        "parent_manifest_sha256": file_sha256(parent_dir / "merge_manifest.json"),
        "base_model": base_identity,
        "source_models": parent_manifest.get("source_models"),
        "calibration_corpus": project_path(calibration_file, PROJECT_ROOT),
        "calibration_corpus_sha256": file_sha256(calibration_file),
        "calibration_manifest": project_path(CALIBRATION_MANIFEST, PROJECT_ROOT),
        "calibration_manifest_sha256": file_sha256(CALIBRATION_MANIFEST),
        "non_overlap_report": project_path(NON_OVERLAP_REPORT, PROJECT_ROOT),
        "non_overlap_report_sha256": file_sha256(NON_OVERLAP_REPORT),
        "aim_repo": project_path(AIM_REPO, PROJECT_ROOT),
        "aim_commit": git_commit(AIM_REPO),
        "aim_patch_script": project_path(AIM_PATCH_SCRIPT, PROJECT_ROOT),
        "aim_patch_script_sha256": file_sha256(AIM_PATCH_SCRIPT),
        "omega": AIM_OMEGA,
        "device": device,
        "dtype": AIM_DTYPE,
        "run": {"exit_code": 0},
    }
    with (output_dir / "merge_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)


def main() -> int:
    args = parse_args()
    merge_cfg = read_yaml(MERGE_CONFIG)
    benchmark_cfg = read_yaml(BENCHMARK_CONFIG)

    parent_id, base_output_id = selected_aim_baseline(str(args.baseline))
    calibration_id = str(args.calibration)
    calibration_file = selected_calibration(calibration_id)

    require_patched_aim_repo()
    require_calibration(calibration_file)

    parent_dir, parent_manifest = require_parent_checkpoint(parent_id, merge_cfg)
    output_id = aim_output_id(base_output_id, calibration_id)
    output_dir = OUTPUT_ROOT / output_id

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"AIM output already exists: {project_path(output_dir, PROJECT_ROOT)}. "
                "Use --overwrite to replace it."
            )
        safe_rmtree(output_dir, OUTPUT_ROOT)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    base_path, base_identity = base_model_snapshot(merge_cfg, benchmark_cfg)

    print(f"[run] {parent_id} -> {output_id} calibration={calibration_id} device={args.device}")
    exit_code = run_aim(parent_dir, output_dir, base_path, calibration_file, str(args.device))
    if exit_code != 0:
        print(f"[done] AIM failed for {output_id} with exit code {exit_code}")
        return exit_code
    if not checkpoint_complete(output_dir):
        print(f"[done] AIM output is incomplete: {project_path(output_dir, PROJECT_ROOT)}")
        return 1

    write_aim_manifest(
        output_dir,
        output_id=output_id,
        parent_id=parent_id,
        parent_dir=parent_dir,
        parent_manifest=parent_manifest,
        base_identity=base_identity,
        calibration_id=calibration_id,
        calibration_file=calibration_file,
        device=str(args.device),
    )

    print(f"[done] AIM baseline completed: {output_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

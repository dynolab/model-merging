from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from common.io import file_sha256, read_json
from common.text import safe_id

from benchmark.common import MATERIALIZED_MODELS_ROOT


LOCAL_MODEL_MANIFEST = "merge_manifest.json"
LOCAL_CHECKPOINT_HASH_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".json", ".model", ".txt")


def is_local_model_path(model: str) -> bool:
    return Path(model).expanduser().exists()


def resolve_model_arg(model: str) -> str:
    p = Path(model).expanduser()
    if p.exists():
        return str(p.resolve())
    return model


def model_registry(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = cfg.get("models")
    if not isinstance(models, dict):
        raise ValueError("benchmark config must define models aliases")
    out: dict[str, dict[str, Any]] = {}
    for alias, spec in models.items():
        if not isinstance(spec, dict):
            raise ValueError(f"models.{alias} must be a mapping")
        repo = str(spec.get("repo", "")).strip()
        revision = str(spec.get("revision", "")).strip()
        if not repo or not revision:
            raise ValueError(f"models.{alias} must define repo and revision")
        out[str(alias)] = {"repo": repo, "revision": revision, **spec}
    return out


def resolve_model_request(
    cfg: dict[str, Any],
    *,
    run_alias: str | None,
    model: str | None,
    config_id: str | None,
) -> tuple[str, str, str | None]:
    if bool(run_alias) == bool(model):
        raise ValueError("Specify exactly one of --run or --model.")

    if run_alias:
        registry = model_registry(cfg)
        if run_alias not in registry:
            raise ValueError(f"Unknown --run alias {run_alias!r}. Available aliases: {', '.join(sorted(registry))}")
        spec = registry[run_alias]
        return spec["repo"], config_id or run_alias, spec["revision"]

    assert model is not None
    if is_local_model_path(model):
        return model, config_id or safe_id(model), None

    registry = model_registry(cfg)
    matches = [spec for spec in registry.values() if spec["repo"] == model]
    if not matches:
        raise ValueError(f"HF model {model!r} is not pinned in models. Use --run for aliases or add it to benchmark.yaml.")
    return model, config_id or safe_id(model), matches[0]["revision"]


def link_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        out_dir = dst / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (out_dir / d).mkdir(exist_ok=True)
        for f in files:
            source_file = root_path / f
            target = out_dir / f
            try:
                os.symlink(source_file.resolve(), target)
            except OSError:
                shutil.copy2(source_file, target)


def hash_local_checkpoint_files(model_path: str) -> dict[str, dict[str, Any]]:
    root = Path(model_path).expanduser().resolve()
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel_path = path.relative_to(root)
        rel = rel_path.as_posix()
        if any(part in {".git", "__pycache__"} for part in rel_path.parts):
            continue
        if path.name == LOCAL_MODEL_MANIFEST or path.name.endswith(LOCAL_CHECKPOINT_HASH_SUFFIXES):
            out[rel] = {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return out


def load_local_model_manifest(model_path: str) -> tuple[dict[str, Any], str | None]:
    path = Path(model_path).expanduser().resolve()
    manifest_path = path / LOCAL_MODEL_MANIFEST
    if not manifest_path.exists():
        return {"manifest": None, "warnings": [f"local checkpoint does not contain {LOCAL_MODEL_MANIFEST}"]}, None
    manifest_sha256 = file_sha256(manifest_path)
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:
        return {"manifest": None, "warnings": [f"could not read {LOCAL_MODEL_MANIFEST}: {exc!r}"]}, manifest_sha256
    return {"manifest": manifest, "warnings": []}, manifest_sha256


def snapshot_download(repo_id: str, revision: str) -> str:
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required for pinned HF model snapshots") from exc
    return hf_snapshot_download(repo_id=repo_id, revision=revision)


def materialize_eval_model(
    model_arg: str,
    configured_revision: str | None,
    *,
    materialize_local: bool = True,
    hash_local_files: bool = True,
) -> tuple[str, dict[str, Any]]:
    local_manifest = None
    local_checkpoint_hashes = None
    if is_local_model_path(model_arg):
        source_path = Path(resolve_model_arg(model_arg))
        local_manifest, local_manifest_sha256 = load_local_model_manifest(model_arg)
        local_checkpoint_hashes = hash_local_checkpoint_files(model_arg) if hash_local_files else None
        if not materialize_local:
            return str(source_path), {
                "kind": "local_checkpoint",
                "input_model": model_arg,
                "configured_revision": configured_revision,
                "local_model_manifest": local_manifest,
                "local_checkpoint_files_sha256": local_checkpoint_hashes,
            }
        if local_manifest_sha256:
            local_identity = local_manifest_sha256[:12]
        elif (source_path / "config.json").exists():
            local_identity = file_sha256(source_path / "config.json")[:12]
        else:
            local_identity = "no_manifest"
        identity_key = safe_id(model_arg) + "__" + local_identity
        model_source_kind = "local_checkpoint"
    else:
        if not configured_revision:
            raise ValueError(f"HF model {model_arg!r} must have a pinned revision")
        source_path = Path(snapshot_download(model_arg, configured_revision)).resolve()
        identity_key = safe_id(model_arg) + "__" + configured_revision[:12]
        model_source_kind = "hf_repo"

    identity: dict[str, Any] = {
        "kind": model_source_kind,
        "input_model": model_arg,
        "configured_revision": configured_revision,
    }
    if model_source_kind == "local_checkpoint":
        identity["local_model_manifest"] = local_manifest
        identity["local_checkpoint_files_sha256"] = local_checkpoint_hashes

    MATERIALIZED_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    eval_dir = MATERIALIZED_MODELS_ROOT / identity_key
    link_tree(source_path, eval_dir)
    return str(eval_dir), identity

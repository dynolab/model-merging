import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer


CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT.parent
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.io import file_sha256, project_path, read_jsonl, read_yaml, resolve_project_path, try_read_json, write_jsonl
from common.text import TEXT_MIN_CHARS, clean_text, normalize_text, sha256_text, word_ngrams


LOG_EVERY_ROWS = 500


class BenchmarkDenylist:
    def __init__(
        self,
        denylist_path: Path,
        *,
        substring_match: bool = True,
        near_duplicate_enabled: bool = True,
        ngram_size: int = 13,
        jaccard_threshold: float = 0.8,
    ) -> None:
        self.denylist_path = denylist_path
        self.substring_match = substring_match
        self.near_duplicate_enabled = near_duplicate_enabled
        self.ngram_size = ngram_size
        self.jaccard_threshold = jaccard_threshold

        self.hashes: Set[str] = set()
        self.texts: List[str] = []
        self.text_sources: List[str] = []

        self.ngram_sets: List[Set[Tuple[str, ...]]] = []
        self.ngram_index: Dict[Tuple[str, ...], List[int]] = defaultdict(list)

        self._load()

    def _load(self) -> None:
        if not self.denylist_path.exists():
            raise FileNotFoundError(f"Benchmark denylist not found: {project_path(self.denylist_path, PROJECT_ROOT)}")

        for row in read_jsonl(self.denylist_path):
            text = row.get("normalized_text") or row.get("text")

            if not text:
                continue

            norm = normalize_text(text, strip_html=True)

            if len(norm) < TEXT_MIN_CHARS:
                continue

            h = sha256_text(norm)
            self.hashes.add(h)

            source_task = row.get("source_task", "unknown")

            idx = len(self.texts)
            self.texts.append(norm)
            self.text_sources.append(source_task)

            ngrams = word_ngrams(norm, self.ngram_size)
            self.ngram_sets.append(ngrams)

            for ng in ngrams:
                self.ngram_index[ng].append(idx)

    def check(self, normalized_candidate: str) -> Optional[Dict[str, Any]]:
        candidate_hash = sha256_text(normalized_candidate)

        if candidate_hash in self.hashes:
            return {
                "reason": "benchmark_exact_hash",
                "matched_task": None,
            }

        if self.substring_match:
            for deny_text, source_task in zip(self.texts, self.text_sources):
                if deny_text in normalized_candidate or normalized_candidate in deny_text:
                    return {
                        "reason": "benchmark_substring",
                        "matched_task": source_task,
                    }

        if not self.near_duplicate_enabled:
            return None

        candidate_ngrams = word_ngrams(normalized_candidate, self.ngram_size)

        if not candidate_ngrams:
            return None

        candidate_ids: Counter[int] = Counter()

        for ng in candidate_ngrams:
            for idx in self.ngram_index.get(ng, []):
                candidate_ids[idx] += 1

        for idx, overlap in candidate_ids.items():
            deny_ngrams = self.ngram_sets[idx]

            if not deny_ngrams:
                continue

            union = len(candidate_ngrams) + len(deny_ngrams) - overlap

            if union <= 0:
                continue

            jaccard = overlap / union

            if jaccard >= self.jaccard_threshold:
                return {
                    "reason": "benchmark_near_duplicate",
                    "matched_task": self.text_sources[idx],
                    "jaccard": jaccard,
                }

        return None


def extract_user_messages(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        role = str(obj.get("role", obj.get("from", ""))).lower()
        content = obj.get("content", obj.get("value", obj.get("text")))

        if role in {"user", "human"} and isinstance(content, str):
            yield content

        for value in obj.values():
            yield from extract_user_messages(value)

    elif isinstance(obj, list):
        for value in obj:
            yield from extract_user_messages(value)


def extract_wildjailbreak_prompt(row: Dict[str, Any], source_cfg: Dict[str, Any]) -> List[str]:
    data_type = str(row.get("data_type", ""))
    allowed_data_types = set(source_cfg.get("data_types", []))

    if allowed_data_types and data_type not in allowed_data_types:
        return []

    if data_type.startswith("adversarial"):
        value = row.get("adversarial") or row.get("vanilla")
    elif data_type.startswith("vanilla"):
        value = row.get("vanilla")
    else:
        value = row.get("adversarial") or row.get("vanilla")

    if isinstance(value, str) and value.strip():
        return [value]

    return []


def extract_texts_from_row(row: Dict[str, Any], source_cfg: Dict[str, Any]) -> List[str]:
    extraction = source_cfg.get("extraction")

    if extraction == "user_messages_only":
        return list(extract_user_messages(row))

    if extraction == "wildjailbreak_prompts":
        return extract_wildjailbreak_prompt(row, source_cfg)

    text_field = source_cfg.get("text_field", "text")
    value = row.get(text_field)

    if isinstance(value, str):
        return [value]

    return []


def load_streaming_dataset(source_cfg: Dict[str, Any]):
    dataset_id = source_cfg["dataset"]
    name = source_cfg.get("name")
    split = source_cfg.get("split", "train")
    revision = source_cfg.get("revision")

    kwargs = {
        "split": split,
        "streaming": True,
    }

    if revision:
        kwargs["revision"] = revision

    extra_kwargs = source_cfg.get("load_dataset_kwargs", {})
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    print(f"[INFO] loading source dataset={dataset_id}, name={name}, split={split}")

    if name:
        return load_dataset(dataset_id, name, **kwargs)

    return load_dataset(dataset_id, **kwargs)


def make_example_id(calibration_id: str, source_name: str, idx: int) -> str:
    safe_source = re.sub(r"[^a-zA-Z0-9_]+", "_", source_name).strip("_").lower()
    return f"{calibration_id}_{safe_source}_{idx:06d}"


def safe_source_config(source_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in source_cfg.items()
        if key not in {"hf_token", "token", "access_token"}
    }


def build_calibration_data(config_path: Path) -> None:
    cfg = read_yaml(config_path)
    config_hash = file_sha256(config_path)

    calibration_id = cfg["calibration_id"]

    output_dir = resolve_project_path(cfg["output_dir"], PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    strip_html = bool(cfg.get("strip_html", True))
    shuffle_seed = int(cfg.get("shuffle_seed", 42))
    log_every = LOG_EVERY_ROWS

    limits = cfg.get("limits", {})
    min_tokens = int(limits.get("min_tokens", 64))
    max_tokens = int(limits.get("max_tokens", 512))

    tokenizer_repo = cfg["base_model"]
    tokenizer_revision = cfg.get("base_revision")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_repo,
        revision=tokenizer_revision,
        trust_remote_code=True,
    )

    exclusion_cfg = cfg["benchmark_exclusion"]
    benchmark_denylist_path = resolve_project_path(exclusion_cfg["benchmark_denylist"], PROJECT_ROOT)
    if not benchmark_denylist_path.exists():
        raise FileNotFoundError(f"Benchmark denylist not found: {project_path(benchmark_denylist_path, PROJECT_ROOT)}")
    denylist_manifest_path = benchmark_denylist_path.with_name("benchmark_denylist_manifest.json")
    denylist_manifest = try_read_json(denylist_manifest_path)
    if not denylist_manifest:
        raise FileNotFoundError(f"Benchmark denylist manifest not found or unreadable: {project_path(denylist_manifest_path, PROJECT_ROOT)}")
    if denylist_manifest.get("output_sha256") != file_sha256(benchmark_denylist_path):
        raise RuntimeError("Benchmark denylist hash does not match its manifest. Rebuild the denylist.")

    near_cfg = exclusion_cfg.get("near_duplicate", {})

    benchmark_denylist = BenchmarkDenylist(
        benchmark_denylist_path,
        substring_match=bool(exclusion_cfg.get("substring_match", True)),
        near_duplicate_enabled=bool(near_cfg.get("enabled", True)),
        ngram_size=int(near_cfg.get("ngram_size", 13)),
        jaccard_threshold=float(near_cfg.get("jaccard_threshold", 0.8)),
    )


    sources_cfg = cfg["sources"]
    main_cfg = cfg["main_corpus"]
    ablation_cfg = cfg.get("ablation_subsets", {})

    required_by_source: Dict[str, int] = {}

    for source_name, source_cfg in sources_cfg.items():
        required_by_source[source_name] = int(source_cfg["n_examples"])

    for _, subset_cfg in ablation_cfg.items():
        source_name = subset_cfg["source"]
        required_by_source[source_name] = max(
            required_by_source.get(source_name, 0),
            int(subset_cfg["n_examples"]),
        )

    accepted_by_source: Dict[str, List[Dict[str, Any]]] = {
        source_name: []
        for source_name in sources_cfg
    }

    seen_corpus_hashes: Set[str] = set()

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_id": calibration_id,
        "config": project_path(config_path, PROJECT_ROOT),
        "config_sha256": config_hash,
        "benchmark_denylist": project_path(benchmark_denylist_path, PROJECT_ROOT),
        "benchmark_denylist_sha256": file_sha256(benchmark_denylist_path),
        "tokenizer": {
            "hf_repo": tokenizer_repo,
            "revision": tokenizer_revision,
        },
        "limits": {
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
        },
        "filter_counts": {
            "seen": 0,
            "empty_or_too_short_chars": 0,
            "duplicate_calibration_text": 0,
            "too_short_tokens": 0,
            "too_long_tokens": 0,
            "benchmark_exact_hash": 0,
            "benchmark_substring": 0,
            "benchmark_near_duplicate": 0,
            "accepted": 0,
        },
        "per_source": {},
        "warnings": [],
    }
    warnings: List[str] = report["warnings"]

    for source_name, source_cfg in sources_cfg.items():
        source_id = source_cfg["dataset"]
        source_revision = source_cfg.get("revision")
        target_n = required_by_source[source_name]

        report["per_source"][source_name] = {
            "dataset": source_id,
            "name": source_cfg.get("name"),
            "revision": source_revision,
            "split": source_cfg.get("split", "train"),
            "target_examples": target_n,
            "accepted": 0,
            "seen_rows": 0,
            "seen_text_candidates": 0,
        }

        ds = load_streaming_dataset(source_cfg)
        local_idx = 0

        for row in ds:
            report["per_source"][source_name]["seen_rows"] += 1
            if report["per_source"][source_name]["seen_rows"] % log_every == 0:
                print(
                    f"[INFO] source={source_name} "
                    f"seen_rows={report['per_source'][source_name]['seen_rows']} "
                    f"seen_texts={report['per_source'][source_name]['seen_text_candidates']} "
                    f"accepted={report['per_source'][source_name]['accepted']} "
                    f"target={target_n}"
                )

            texts = extract_texts_from_row(row, source_cfg)

            for raw_text in texts:
                report["filter_counts"]["seen"] += 1
                report["per_source"][source_name]["seen_text_candidates"] += 1

                cleaned = clean_text(raw_text, strip_html=strip_html)
                normalized = normalize_text(cleaned, strip_html=strip_html)

                if len(normalized) < TEXT_MIN_CHARS:
                    report["filter_counts"]["empty_or_too_short_chars"] += 1
                    continue

                normalized_hash = sha256_text(normalized)

                if normalized_hash in seen_corpus_hashes:
                    report["filter_counts"]["duplicate_calibration_text"] += 1
                    continue

                token_ids = tokenizer.encode(cleaned, add_special_tokens=False)
                token_count = len(token_ids)

                if token_count < min_tokens:
                    report["filter_counts"]["too_short_tokens"] += 1
                    continue

                if token_count > max_tokens:
                    report["filter_counts"]["too_long_tokens"] += 1
                    continue

                deny_match = benchmark_denylist.check(normalized)

                if deny_match is not None:
                    reason = deny_match["reason"]
                    report["filter_counts"][reason] += 1
                    continue

                seen_corpus_hashes.add(normalized_hash)
                local_idx += 1

                example = {
                    "id": make_example_id(calibration_id, source_name, local_idx),
                    "text": cleaned,
                    "source": source_id,
                    "source_name": source_cfg.get("name"),
                    "source_revision": source_revision,
                    "source_group": source_name,
                    "num_tokens": token_count,
                    "normalized_sha256": normalized_hash,
                }

                accepted_by_source[source_name].append(example)

                report["filter_counts"]["accepted"] += 1
                report["per_source"][source_name]["accepted"] += 1

                if len(accepted_by_source[source_name]) >= target_n:
                    break

            if len(accepted_by_source[source_name]) >= target_n:
                break

        if len(accepted_by_source[source_name]) < target_n:
            warning = (
                f"not enough accepted examples for source={source_name}: "
                f"needed {target_n}, got {len(accepted_by_source[source_name])}"
            )
            warnings.append(warning)
            print(f"[warning] {warning}")

    main_rows: List[Dict[str, Any]] = []

    for source_name, source_cfg in sources_cfg.items():
        n = int(source_cfg["n_examples"])
        main_rows.extend(accepted_by_source[source_name][:n])

    random.Random(shuffle_seed).shuffle(main_rows)

    main_output = output_dir / main_cfg["output_file"]
    write_jsonl(main_output, main_rows)

    ablation_outputs: Dict[str, Dict[str, Any]] = {}

    for subset_name, subset_cfg in ablation_cfg.items():
        source_name = subset_cfg["source"]
        n = int(subset_cfg["n_examples"])
        output_file = output_dir / subset_cfg["output_file"]

        rows = accepted_by_source[source_name][:n]
        write_jsonl(output_file, rows)

        ablation_outputs[subset_name] = {
            "file": project_path(output_file, PROJECT_ROOT),
            "num_examples": len(rows),
            "sha256": file_sha256(output_file),
            "source": source_name,
        }

    manifest_path = output_dir / "calibration_manifest.json"
    report_path = output_dir / "non_overlap_report.json"

    output_files = {
        "main": {
            "file": project_path(main_output, PROJECT_ROOT),
            "num_examples": len(main_rows),
            "sha256": file_sha256(main_output),
        },
        "ablation_subsets": ablation_outputs,
    }

    benchmark_source_info: Dict[str, Any] = {
        "benchmark_config_sha256": denylist_manifest.get("benchmark_config_sha256"),
        "input_snapshot": denylist_manifest.get("input_snapshot"),
        "input_snapshot_sha256": denylist_manifest.get("input_snapshot_sha256"),
        "input_snapshot_summary": denylist_manifest.get("input_snapshot_summary"),
        "task_counts": denylist_manifest.get("task_counts"),
    }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_id": calibration_id,
        "config": project_path(config_path, PROJECT_ROOT),
        "config_sha256": config_hash,
        "tokenizer": {
            "hf_repo": tokenizer_repo,
            "revision": tokenizer_revision,
        },
        "token_limits": {
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
        },
        "main_corpus": output_files["main"],
        "ablation_subsets": output_files["ablation_subsets"],
        "sources": {
            source_name: safe_source_config(source_cfg)
            for source_name, source_cfg in sources_cfg.items()
        },
        "source_balance": {
            source_name: int(source_cfg["n_examples"])
            for source_name, source_cfg in sources_cfg.items()
        },
        "actual_source_counts": {
            source_name: len(accepted_by_source[source_name])
            for source_name in sources_cfg
        },
        "benchmark_exclusion": {
            "benchmark_denylist": project_path(benchmark_denylist_path, PROJECT_ROOT),
            "benchmark_denylist_sha256": file_sha256(benchmark_denylist_path),
            "benchmark_denylist_manifest": project_path(denylist_manifest_path, PROJECT_ROOT),
            "benchmark_denylist_manifest_sha256": file_sha256(denylist_manifest_path) if denylist_manifest_path.exists() else None,
            "benchmark_sources": benchmark_source_info,
            "exact_hash_match": True,
            "substring_match": bool(exclusion_cfg.get("substring_match", True)),
            "near_duplicate": near_cfg,
        },
        "preprocessing": {
            "strip_html": strip_html,
            "shuffle_seed": shuffle_seed,
        },
        "non_overlap_report": project_path(report_path, PROJECT_ROOT),
        "warnings": warnings,
    }

    report["outputs"] = output_files

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] wrote main calibration corpus: {project_path(main_output, PROJECT_ROOT)}")
    print(f"[OK] wrote manifest: {project_path(manifest_path, PROJECT_ROOT)}")
    print(f"[OK] wrote non-overlap report: {project_path(report_path, PROJECT_ROOT)}")
    print("[INFO] accepted examples:")
    for source_name, rows in accepted_by_source.items():
        print(f"  - {source_name}: {len(rows)}")
    print("[INFO] filter counts:")
    for key, value in report["filter_counts"].items():
        print(f"  - {key}: {value}")


def main() -> None:
    config = CODE_ROOT / "cfg" / "calibration" / "calib_mix.yaml"
    build_calibration_data(config)


if __name__ == "__main__":
    main()

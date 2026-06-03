#!/usr/bin/env python3
"""Patch upstream ActivationInformedMerging for the local phase-one protocol.

The upstream files are compacted into very long lines, so a conventional
unified diff is hard to audit. This script performs narrow, checked rewrites:

- `performAIM.py` gets local calibration and device CLI arguments.
- `relax_on_merged()` in `_utils.py` gets local JSONL calibration, configurable
  device support, Qwen-compatible model loading options, and honest load failures.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AIM_REPO = PROJECT_ROOT / "external" / "ActivationInformedMerging"


PATCHED_PERFORM_AIM = '''import argparse

from MergeModels.ActivationMerging import relax_on_merged


argparser = argparse.ArgumentParser()
argparser.add_argument("--merged_model", type=str, help="Path to the merged model.")
argparser.add_argument(
    "--pretrained_model_name",
    type=str,
    default="unsloth/llama-2-13b",
    help="Name or path of the pretrained/base model to use for relaxation.",
)
argparser.add_argument("--omega", type=float, default=0.4, help="Value of omega for relaxation. Default: 0.4")
argparser.add_argument("--save_path", type=str, help="Path to save the relaxed model.")
argparser.add_argument("--calibration_file", type=str, default=None, help="Local JSONL calibration file.")
argparser.add_argument("--calibration_text_field", type=str, default="text", help="Text field in the calibration JSONL.")
argparser.add_argument("--device", type=str, default="cuda", help="Device for calibration forward passes. Default: cuda")
args = argparser.parse_args()

print(f"Relaxing {args.merged_model}...")
merged_model, merged_tokenizer = relax_on_merged(
    args.merged_model,
    pretrained_model_name=args.pretrained_model_name,
    omega=args.omega,
    calibration_file=args.calibration_file,
    calibration_text_field=args.calibration_text_field,
    device=args.device,
)
print(f"Saving to {args.save_path}...")
merged_model.save_pretrained(args.save_path)
merged_tokenizer.save_pretrained(args.save_path)
'''


PATCHED_RELAX_ON_MERGED = '''def relax_on_merged(
    merged_model_name,
    pretrained_model_name,
    omega=0.2,
    nonlinear_scaling=lambda x: x,
    calibration_file=None,
    calibration_text_field="text",
    device="cuda",
    **kwargs,
):
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        base_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)
        base_config = AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Model {pretrained_model_name} could not be loaded.")
        print(f"Reason: {e}")
        raise

    try:
        merged_model = AutoModelForCausalLM.from_pretrained(
            merged_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        merged_tokenizer = AutoTokenizer.from_pretrained(merged_model_name, trust_remote_code=True)
        merged_config = AutoConfig.from_pretrained(merged_model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Model {merged_model_name} could not be loaded.")
        print(f"Reason: {e}")
        raise

    logger = logging.getLogger(__name__)
    align_tokenizers_and_embeddings(
        pretrained_model=base_model,
        pretrained_tokenizer=base_tokenizer,
        pretrained_config=base_config,
        finetuned_models=[merged_model],
        finetuned_tokenizers=[merged_tokenizer],
        finetuned_configs=[merged_config],
        logger=logger,
    )

    print("loading calibration dataset...")
    if calibration_file:
        dataset = load_dataset("json", data_files=str(calibration_file), split="train")
        if calibration_text_field != "text":
            if calibration_text_field not in dataset.column_names:
                raise ValueError(f"Calibration field {calibration_text_field!r} is not present in {calibration_file}")
            dataset = dataset.rename_column(calibration_text_field, "text")
    else:
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=42)

    print("getting calibration features...")
    base_model.eval()
    base_model.config.use_cache = False
    if hasattr(base_model, "generation_config"):
        base_model.generation_config.use_cache = False
    base_model = base_model.to(device)
    pretrained_scale_dict = get_calib_feat(base_model, base_tokenizer, dataset)
    base_model = base_model.to("cpu")
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    base_layer_dict = {}
    for name, param in base_model.named_modules():
        if hasattr(param, "weight") and "embed" not in name:
            base_layer_dict[name] = param

    merged_layer_dict = {}
    for name, param in merged_model.named_modules():
        if hasattr(param, "weight") and "embed" not in name:
            merged_layer_dict[name] = param

    final_weight_dict = {}
    for name, param in merged_model.named_modules():
        if hasattr(param, "weight"):
            if "embed" in name:
                final_weight_dict[name] = param.weight.data
            else:
                base_importance = torch.abs(pretrained_scale_dict[name])
                base_importance = base_importance / torch.max(base_importance)
                base_importance = nonlinear_scaling(base_importance)
                total_delta = merged_layer_dict[name].weight.data - base_layer_dict[name].weight.data
                relaxation_factor = 1 - (base_importance * (1 - omega))
                delta_final = total_delta * relaxation_factor
                final_weight_dict[name] = (base_layer_dict[name].weight.data + delta_final).to(torch.bfloat16)

    for name, param in tqdm(merged_model.named_modules(), total=sum(1 for _ in merged_model.named_modules())):
        if name in final_weight_dict:
            if torch.allclose(param.weight.data, final_weight_dict[name]):
                print(f"{name} has not changed")
            param.weight.data = final_weight_dict[name]
        elif hasattr(param, "weight"):
            print(f"{name} not in final_weight_dict")

    return merged_model, merged_tokenizer

'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch external ActivationInformedMerging for local calibration + configurable-device AIM")
    parser.add_argument("--repo", default=str(DEFAULT_AIM_REPO), help="Path to external/ActivationInformedMerging")
    return parser.parse_args()


def patch_perform_aim(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if (
        "--calibration_file" in text
        and "--device" in text
        and 'default="cuda"' in text
        and "merged_tokenizer.save_pretrained(args.save_path)" in text
    ):
        print(f"[skip] already patched: {path}")
        return
    if "relax_on_merged" not in text or "merged_model.save_pretrained(args.save_path)" not in text:
        raise RuntimeError(f"Unexpected performAIM.py contents: {path}")
    path.write_text(PATCHED_PERFORM_AIM, encoding="utf-8")
    print(f"[patched] {path}")


def patch_utils(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "calibration_file=None" in text and 'device="cuda"' in text and "base_model = base_model.to(device)" in text:
        print(f"[skip] already patched: {path}")
        return
    start = text.find("def relax_on_merged(")
    end = text.find("def get_calib_dataset", start)
    if start < 0 or end < 0:
        raise RuntimeError(f"Could not find relax_on_merged() boundaries in {path}")
    if 'load_dataset("mit-han-lab/pile-val-backup", split="validation")' not in text[start:end]:
        raise RuntimeError(f"Unexpected relax_on_merged() body in {path}")
    patched = text[:start] + PATCHED_RELAX_ON_MERGED + text[end:]
    path.write_text(patched, encoding="utf-8")
    print(f"[patched] {path}")


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    perform_aim = repo / "performAIM.py"
    utils = repo / "MergeModels" / "ActivationMerging" / "_utils.py"
    missing = [path for path in (perform_aim, utils) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing AIM file(s): " + ", ".join(str(path) for path in missing))
    patch_perform_aim(perform_aim)
    patch_utils(utils)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

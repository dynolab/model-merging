 # MergeKit Baselines

Run commands from the project root.

## Setup

```powershell
python -m pip install -r code/requirements.txt
hf auth login
```

Config:

```text
code/cfg/merge/mergekit_baselines.yaml
```

## One-Time Search

Disk-safe selection search builds one candidate, evaluates it on the separate
selection benchmark, then deletes the large checkpoint before moving to the next
candidate:

```powershell
python code/src/merge/run_selection_search.py ties_best --gpu-ids 0,1
```

The selection benchmark is configured in `code/cfg/benchmark/benchmark_selection.yaml`.
For the current phase it uses `arc_challenge`.

After the search finishes:

```powershell
python code/src/benchmark/summarize_benchmark.py --selection
python code/src/merge/select_best_merge.py ties_best
```

The disk-safe search keeps small reproducibility artifacts:

```text
outputs/mergekit_configs/<config-id>.yaml
outputs/merge_manifests/<config-id>.json
results/benchmark_selection/<config-id>/
```

The selector writes the chosen parameters back to `code/cfg/merge/mergekit_baselines.yaml`.
It selects the valid candidate with the highest `Aggregate` in `results/benchmark_selection_main.csv`.

## Build Frozen Baselines

This command is only for fixed baselines and grid baselines after `select_best_merge.py`
has frozen their selected parameters. It refuses unfrozen grid baselines so it does
not accidentally create every candidate checkpoint at once.

```powershell
python code/src/merge/run_mergekit_baselines.py ties_best --overwrite
```

After freeze, the same baseline id builds the selected checkpoint, for example `outputs/merged/ties_best/`.

All baselines:

```powershell
python code/src/merge/run_mergekit_baselines.py --overwrite
```

Outputs:

```text
outputs/mergekit_configs/<config-id>.yaml
outputs/merged/<config-id>/
outputs/merged/<config-id>/merge_manifest.json
```

## AIM on Frozen Best Baselines

AIM is an external baseline run after `_best` checkpoints are frozen and built.
Use the official implementation as an external dependency, then apply the local
protocol patch for benchmark-disjoint calibration and configurable-device execution:

```powershell
mkdir external
git clone https://github.com/ahnobari/ActivationInformedMerging external/ActivationInformedMerging
git -C external/ActivationInformedMerging checkout e7eccd573d010da13dc6f795a23a36f9822851c9
python code/patches/activation_informed_merging/apply_local_calibration_patch.py
```

Prerequisites before running AIM:

```text
outputs/merged/task_arithmetic_best/merge_manifest.json
outputs/merged/ties_best/merge_manifest.json
outputs/merged/dare_best/merge_manifest.json
outputs/merged/della_best/merge_manifest.json
outputs/merged/breadcrumbs_best/merge_manifest.json
data/calibration/calib_mix.jsonl
data/calibration/calib_general.jsonl
data/calibration/calib_instruction.jsonl
data/calibration/calib_reasoning.jsonl
data/calibration/calib_uncensored_refusal_like.jsonl
data/calibration/calibration_manifest.json
data/calibration/non_overlap_report.json
```

AIM uses GPU by default (`--device cuda`). Build AIM checkpoints:

```powershell
python code/src/merge/run_aim_baselines.py ties_best --calibration mix --overwrite
python code/src/merge/run_aim_baselines.py ties_best --calibration general --overwrite
python code/src/merge/run_aim_baselines.py ties_best --calibration instruction --overwrite
python code/src/merge/run_aim_baselines.py ties_best --calibration reasoning --overwrite
python code/src/merge/run_aim_baselines.py ties_best --calibration uncensored_refusal_like --overwrite
```

Output ids:

```text
outputs/merged/aim_on_<family>__calib_general/
outputs/merged/aim_on_<family>__calib_instruction/
outputs/merged/aim_on_<family>__calib_reasoning/
outputs/merged/aim_on_<family>__calib_uncensored_refusal_like/
outputs/merged/aim_on_<family>/  # mix calibration
outputs/merged/<aim-id>/merge_manifest.json
```

## LOT Baselines

LOT is tracked as an external activation-informed baseline under the same
benchmark, forgetting, and calibration non-overlap protocol as the other
runs.

Config:

```text
code/cfg/merge/lot_baselines.yaml
```

Runnable variants are:

```text
lot_paper_domain_matched             # each specialist uses its domain-aligned calibration subset
lot_paper_mix                        # all specialists use the generic calibration mix
lot_paper_general                    # all specialists use the general calibration subset
lot_paper_instruction                # all specialists use the instruction calibration subset
lot_paper_reasoning                  # all specialists use the reasoning calibration subset
lot_paper_uncensored_refusal_like    # all specialists use the uncensored/refusal-like subset
```

Build LOT checkpoints:

```powershell
python code/src/merge/run_lot_baselines.py lot_paper_domain_matched --gpu-ids 0 --overwrite
python code/src/merge/run_lot_baselines.py lot_paper_mix --gpu-ids 0 --overwrite
python code/src/merge/run_lot_baselines.py lot_paper_general --gpu-ids 0 --overwrite
python code/src/merge/run_lot_baselines.py lot_paper_instruction --gpu-ids 0 --overwrite
python code/src/merge/run_lot_baselines.py lot_paper_reasoning --gpu-ids 0 --overwrite
python code/src/merge/run_lot_baselines.py lot_paper_uncensored_refusal_like --gpu-ids 0 --overwrite
```

LOT outputs:

```text
outputs/lot_configs/<lot-id>.yaml
outputs/merged/<lot-id>/
outputs/merged/<lot-id>/merge_manifest.json
```

# Reproduce AIMerge (AIM)

## Implementation

| File | Description |
|------|-------------|
| `code/patches/activation_informed_merging/apply_local_calibration_patch.py` | local adaptation of upstream AIM: benchmark-disjoint calibration + configurable device |
| `code/src/merge/run_aim_baselines.py` | builds the 25 AIM checkpoints (each `_best` parent, five calibration subsets, omega 0.4) |
| `ahnobari/ActivationInformedMerging` @ `e7eccd573d010da13dc6f795a23a36f9822851c9` | pinned upstream source that was reproduced |

## Benchmark runs

| File | Description |
|------|-------------|
| `results/benchmark/aim_on_task_arithmetic/`, `results/benchmark/aim_on_task_arithmetic__calib_general/`, `results/benchmark/aim_on_task_arithmetic__calib_instruction/`, `results/benchmark/aim_on_task_arithmetic__calib_reasoning/`, `results/benchmark/aim_on_task_arithmetic__calib_uncensored_refusal_like/` | AIM on task_arithmetic_best, five calibrations |
| `results/benchmark/aim_on_ties/`, `results/benchmark/aim_on_ties__calib_general/`, `results/benchmark/aim_on_ties__calib_instruction/`, `results/benchmark/aim_on_ties__calib_reasoning/`, `results/benchmark/aim_on_ties__calib_uncensored_refusal_like/` | AIM on ties_best, five calibrations |
| `results/benchmark/aim_on_dare/`, `results/benchmark/aim_on_dare__calib_general/`, `results/benchmark/aim_on_dare__calib_instruction/`, `results/benchmark/aim_on_dare__calib_reasoning/`, `results/benchmark/aim_on_dare__calib_uncensored_refusal_like/` | AIM on dare_best, five calibrations |
| `results/benchmark/aim_on_della/`, `results/benchmark/aim_on_della__calib_general/`, `results/benchmark/aim_on_della__calib_instruction/`, `results/benchmark/aim_on_della__calib_reasoning/`, `results/benchmark/aim_on_della__calib_uncensored_refusal_like/` | AIM on della_best, five calibrations |
| `results/benchmark/aim_on_breadcrumbs/`, `results/benchmark/aim_on_breadcrumbs__calib_general/`, `results/benchmark/aim_on_breadcrumbs__calib_instruction/`, `results/benchmark/aim_on_breadcrumbs__calib_reasoning/`, `results/benchmark/aim_on_breadcrumbs__calib_uncensored_refusal_like/` | AIM on breadcrumbs_best, five calibrations |
| `results/aim_calibration_ablation.csv` | AIM accuracy/forgetting per calibration subset |

# Run calibration-free baselines

| File | Description |
|------|-------------|
| `results/benchmark/base/`, `results/benchmark/instr_only/`, `results/benchmark/reasoning_only/`, `results/benchmark/uncensored_only/` | reference runs: base and the three specialists |
| `results/benchmark/task_arithmetic_best/`, `results/benchmark/ties_best/`, `results/benchmark/dare_best/`, `results/benchmark/della_best/`, `results/benchmark/breadcrumbs_best/` | the five `_best` merges (4 axes + `run_manifest.json` each) |
| `results/benchmark_selection/`, `results/benchmark_selection_main.csv` | selection-grid runs and `arc_challenge` scores |
| `outputs/mergekit_configs/`, `outputs/merge_manifests/` | grid candidates' merge configs and manifests |
| `results/logs/` | per-model eval logs |

# Analysis

Post-run analysis over `results/`: activation-space distances (MMD / JS), the AIM calibration ablation, and reasoning lengths. Run all commands from the project root.

Config:

```text
code/cfg/analysis/activation_metrics.yaml
```

## Activation Metrics

Distances (MMD / JS) of each merged model's activations to the base and the three specialists.

Cache the base/specialist activations (run once):

```powershell
python code/src/analysis/activation_metric_comparison.py build-reference-cache
```

Score one merged checkpoint against those cached references (run once per merged model):

```powershell
python code/src/analysis/activation_metric_comparison.py score-model --model outputs/merged/<config-id> --config-id <config-id>
```

Combine all per-model scores into the summary tables:

```powershell
python code/src/analysis/activation_metric_comparison.py summarize
```

## AIM Calibration Ablation

One table comparing AIM across the five calibration subsets (general / instruction / reasoning / uncensored / mix):

```powershell
python code/src/analysis/summarize_aim_calibration_ablation.py
```

## Reasoning Lengths

Length of the `<think>` reasoning blocks per model, from the lm-eval samples:

```powershell
python code/src/analysis/summarize_reasoning_lengths.py
```

## Outputs

```text
results/activation_metrics/
results/aim_calibration_ablation.csv
results/benchmark_reasoning_lengths.csv
results/benchmark_reasoning_samples.csv
```

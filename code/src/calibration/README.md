# Calibration

Build an unlabeled calibration corpus that does not overlap the benchmark. Run all commands from the project root.

## Prerequisites

```powershell
python -m pip install -r code/requirements.txt
hf auth login
```

Clone the pinned safety-eval fork before exporting safety benchmark inputs:

```powershell
mkdir external
git clone https://github.com/nouhadziri/safety-eval-fork external/safety-eval-fork
cd external/safety-eval-fork
git checkout 2920bb85a8a8390144b9256f697395f81b94822e
cd ../..
```

Configs:

```text
code/cfg/benchmark/benchmark.yaml
code/cfg/calibration/calib_mix.yaml
```

You also need Hugging Face access to `allenai/wildguardmix` and `allenai/wildjailbreak`.

## Run

```powershell
python code/src/calibration/export_benchmark_inputs.py
python -u code/src/calibration/build_benchmark_denylist.py
python code/src/calibration/build_calibration_data.py
```

## Outputs

Benchmark exclusion snapshot:

```text
data/benchmark/benchmark_inputs.jsonl
data/benchmark/benchmark_inputs_manifest.json
```

Benchmark denylist:

```text
data/calibration/benchmark_denylist.jsonl
data/calibration/benchmark_denylist_manifest.json
```

Calibration datasets:

```text
data/calibration/calib_mix.jsonl
data/calibration/calib_general.jsonl
data/calibration/calib_instruction.jsonl
data/calibration/calib_reasoning.jsonl
data/calibration/calib_uncensored_refusal_like.jsonl
```

Audit artifacts:

```text
data/calibration/calibration_manifest.json
data/calibration/non_overlap_report.json
```

Calibration candidates are filtered against the benchmark denylist by exact normalized hash, substring match, and 13-gram near-duplicate match.

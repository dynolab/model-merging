# Benchmark

Run the custom benchmark protocol for one model and summarize completed runs. Run all commands from the project root.

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
```

You also need Hugging Face access to `allenai/wildguardmix` and `allenai/wildjailbreak`.

## Run One Model

Pinned aliases from `benchmark.yaml`:

```powershell
python code/src/benchmark/evaluate_benchmark.py --run base --gpu-ids 0 --overwrite
python code/src/benchmark/evaluate_benchmark.py --run instr_only --gpu-ids 0 --overwrite
python code/src/benchmark/evaluate_benchmark.py --run reasoning_only --gpu-ids 0 --overwrite
python code/src/benchmark/evaluate_benchmark.py --run uncensored_only --gpu-ids 0 --overwrite
```

Local merged checkpoint:

```powershell
python code/src/benchmark/evaluate_benchmark.py --model outputs/merged/my_merge --config-id my_merge --gpu-ids 0,1 --overwrite
```


Each run writes:

```text
results/benchmark/<config-id>/
results/benchmark/<config-id>/run_manifest.json
```

`exit_code: 0` in `run_manifest.json` means the run completed.

## Summarize

After the needed source/specialist/merged runs finish:

```powershell
python code/src/benchmark/summarize_benchmark.py
```

Outputs:

```text
results/benchmark_runs.json
results/benchmark_tasks.csv
results/benchmark_main.csv
results/benchmark_forgetting.csv
```

The summarizer marks a run as `valid_run: false` when the run failed, only some axes finished, the benchmark config changed since the run, `run_manifest.json` is missing or a local checkpoint has a missing or unreadable `merge_manifest.json`.
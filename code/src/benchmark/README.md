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
code/cfg/benchmark/benchmark_selection.yaml
```

You also need Hugging Face access to the safety datasets used by the pinned safety-eval fork, including `allenai/wildguardmix` and `allenai/wildjailbreak`.

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
python code/src/analysis/summarize_reasoning_lengths.py
```

Outputs:

```text
results/benchmark_runs.json
results/benchmark_tasks.csv
results/benchmark_main.csv
results/benchmark_forgetting.csv
results/benchmark_reasoning_lengths.csv
results/benchmark_reasoning_samples.csv
```

The summarizer marks a run as `valid_run: false` when the run failed, only some axes or configured tasks finished, the benchmark config changed since the run, `run_manifest.json` is missing or a local checkpoint has a missing or unreadable `merge_manifest.json`.

## Selection Benchmark

Use this only for one-time `_best` grid selection. The task list lives in
`code/cfg/benchmark/benchmark_selection.yaml`; for the current phase it uses
`arc_challenge`, separate from the final IF/reasoning/safety axes.

Run selection through the disk-safe merge runner so only one large candidate
checkpoint exists at a time:

```powershell
python code/src/merge/run_selection_search.py --gpu-ids 0,1
python code/src/benchmark/summarize_benchmark.py --selection
python code/src/merge/select_best_merge.py
```

Selection summary writes:

```text
results/benchmark_selection_main.csv
```

It contains `Aggregate`, the mean over the selection tasks, plus one column per selection task.

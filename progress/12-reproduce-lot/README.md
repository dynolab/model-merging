# Reproduce LOT

## Implementation

| File | Description |
|------|-------------|
| `code/src/merge/lot_merging.py` | implementation of LOT, adapted for Qwen3-8B |
| `code/src/merge/run_lot_baselines.py` | builds the LOT checkpoints (six calibration modes) |
| `code/cfg/merge/lot_baselines.yaml` | the six calibration modes |

## Benchmark runs

| File | Description |
|------|-------------|
| `results/benchmark/lot_paper_domain_matched/`, `results/benchmark/lot_paper_mix/`, `results/benchmark/lot_paper_general/`, `results/benchmark/lot_paper_instruction/`, `results/benchmark/lot_paper_reasoning/`, `results/benchmark/lot_paper_uncensored_refusal_like/` | LOT, six calibration modes; hidden states 192 × 64 |
| `results/benchmark/lot_paper_domain_matched__v1/`, `results/benchmark/lot_paper_mix__v1/`, `results/benchmark/lot_paper_general__v1/`, `results/benchmark/lot_paper_instruction__v1/`, `results/benchmark/lot_paper_reasoning__v1/`, `results/benchmark/lot_paper_uncensored_refusal_like__v1/` | same six modes, earlier collection; hidden states 64 × 128 |

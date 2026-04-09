# Phase one

Produce a detailed spec for this template and each of deliverables (once started). Define research ojectives and questions.

## Deliverables

- [`spec.md`](spec.md): phase-one bindings (baselines, data, benchmark, runs, reporting, AIM reproduction, metric comparison, ablations). Provided plan is also a draft for actual specs.

## Canonical problem definition

Definitions of merge setting, evaluation, forgetting, and reporting conventions remain in [`../01-model-merging-spec/spec.md`](../01-model-merging-spec/spec.md).

# Phase one: execution plan (template)

This document is the **phase-one experiment template**: artifact locations, run matrices, and completion checklists. It does **not** replace the canonical problem definition—see [Scope and references](#scope-and-references).

**Phase-one “done”** means: baselines and data artifacts are selected and recorded; the benchmark and calibration corpora are frozen in configs; calibration-free baselines are evaluated on the benchmark; accuracy and forgetting are reported in the agreed format; AIM (Activation-Informed Merging / “AIMerge” in internal naming) is reproduced on the same harness; activation-similarity metrics are compared under a controlled design; layer-linearity and merge-order ablations are specified and run (where applicable); and pointers to logs/tables exist for a short results report.

## Scope and references

Canonical definitions and protocol live in [`../01-model-merging-spec/spec.md`](../01-model-merging-spec/spec.md). Phase one **inherits by reference**:

- Merge setting, layer-adaptive hypothesis, and compositional ablations (including calibration corpus rules).
- Benchmark axes (MergeBench-style), accuracy aggregation, and **forgetting** \(F_i\), max/mean forgetting.
- Reporting columns, environment/reproducibility expectations, and main baseline families.


## Objectives (mapped to phase-one work)

Each item below maps to a section and to [Deliverables and exit criteria](#deliverables-and-exit-criteria).

| # | Objective | Primary section |
|---|-----------|-----------------|
| 1 | Select and collect baselines | [Baseline selection and inventory](#baseline-selection-and-inventory) |
| 2 | Collect calibration data | [Calibration data (collection)](#calibration-data-collection) |
| 3 | Collect benchmark | [Benchmark (collection)](#benchmark-collection) |
| 4 | Run calibration-free baselines on benchmark | [Protocol: calibration-free baselines](#protocol-calibration-free-baselines), [Runs to execute](#runs-to-execute) |
| 5 | Report on accuracy and forgetting | [Reporting: accuracy and forgetting](#reporting-accuracy-and-forgetting) |
| 6 | Reproduce AIMerge (AIM) on the benchmark | [Reproduce AIMerge (AIM)](#reproduce-aimerge-aim) |
| 7 | Compare metrics of activation similarity | [Activation similarity metrics (comparison study)](#activation-similarity-metrics-comparison-study) |
| 8 | Ablate: do layer merges affect results linearly? | [Ablations (A): layer merging linearity](#ablations-a-layer-merging-linearity) |
| 9 | Ablate: merging order (final scheme + baselines if possible) | [Ablations (B): merge order](#ablations-b-merge-order) |


## Baseline selection and inventory

Select baselines consistent with [Main baselines](../01-model-merging-spec/spec.md#main-baselines) in the parent spec (global merge, uniform/random layer profiles, single-source specialists, AIM on the same underlying merge, external methods as budget allows). Record one row per baseline **before** large eval runs.

| Baseline id | Method family | Global vs layer-wise | Requires calibration? | HF repo + revision | License | Maps to parent baseline # / note |
|-------------|---------------|----------------------|-------------------------|-------------------|---------|-----------------------------------|
| `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

Notes:

- “Requires calibration?” = uses unlabeled forward-pass alignment or activation-guided tuning **before** benchmark eval (if no, eligible for [calibration-free protocol](#protocol-calibration-free-baselines)).


## Calibration data (collection)

Collect unlabeled set of calibration data.  Provide details of data selection and preprocessing (e.g. balance of target tasks, sourses, etc). 


## Benchmark (collection)
Collect becnchmark data and write test scripts. per [Benchmarks](../01-model-merging-spec/spec.md#benchmarks) in the parent spec: MergeBench-oriented task set; minimum axes are instruction-following, reasoning (incl. math), safety. Select appropriate subsets of MergeBench to balance expressivity of results and compute for each model version test.

| Field | Value |
|-------|--------|
| MergeBench fork URL + pinned commit | `[TBD]` |
| Frozen task list / config path in repo | `[TBD]` |
| Optional axes (coding, multilingual, …) | `[TBD]` |
| Specialist domain bundles \(B_i\) config path (for forgetting) | `[TBD]` |


## Reporting: accuracy and forgetting

Collect scores for each baseline.

Follow parent [How results are reported](../01-model-merging-spec/spec.md#how-results-are-reported). Phase-one **minimum**:

- Aggregate downstream score (formula stated).
- Per-axis scores (IF, reasoning, safety; optional axes if run).
- Forgetting: \(F_i\) per domain, \(\max_i F_i\), mean \(F_i\).
- Environment block per paper-ready run.

### Stub: main results (fill after runs)

| Config id | Aggregate | IF | Reasoning | Safety | Max \(F_i\) | Mean \(F_i\) |
|-----------|-----------|-----|-----------|--------|-------------|-------------|
| | | | | | | |

### Stub: per-domain forgetting (optional detail)

| Config id | Domain \(i\) | \(F_i\) |
|-----------|----------------|--------|
| | | |


## Reproduce AIMerge (AIM)

Reproduce AIMerge results (activation-informed layer-wise merging with calibration) for our tasks. This is the minimal result we want to get. Use the same benchmarking setting as for baselines. 

**Naming:** “AIMerge” in checklists refers to **Activation-Informed Merging (AIM)** as in the parent spec and upstream [ActivationInformedMerging](https://github.com/ahnobari/ActivationInformedMerging). If your implementation uses a different codebase, replace the link and pin that revision instead.

## Activation similarity metrics (comparison study)

Compare discrepancy metrics used for activation alignment (e.g. MMD, Jensen–Shannon, others) under **controlled** conditions: same merge family and search budget where possible; vary metric (and optionally calibration corpus). If result is better than original AIM, we will use it as result of phase one.

| Metric | Definition pointer | Used at stage (per-layer score / global / post-merge) | Relative cost notes |
|--------|---------------------|------------------------------------------------------|---------------------|
| MMD | `[TBD]` | `[TBD]` | `[TBD]` |
| Jensen–Shannon | `[TBD]` | `[TBD]` | `[TBD]` |
| `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

**Hypothesis (fill):** `[TBD]` — e.g. whether metric choice correlates with downstream aggregate and forgetting.

**Experimental design summary:** `[TBD]`


## Ablations
Provide ablation study of your method. Give suggestion based on results.

### (A) Layer merging linearity

**Question:** Do merges applied to increasing subsets of layers produce **approximately linear** changes in aggregate score and forgetting, or are there strong interaction effects?

| Field | Value |
|-------|--------|
| Granularity (layer vs block indices) | `[TBD]` |
| Subset schedule (e.g. first \(k\) layers, every \(k\)-th layer, random subsets) | `[TBD]` |
| Metrics logged (aggregate, per-axis, \(F_i\)) | `[TBD]` |
| Linear baseline to compare against (e.g. predicted score from single-layer effects) | `[TBD]` |

### (B) Merge order

**Question:** How sensitive are results to **merge order** for the final scheme and for each baseline where the algorithm is order-dependent (e.g. greedy layer order, TIES-style, chain-of-merges-style)?

| Field | Value |
|-------|--------|
| Methods / baselines where order applies | `[TBD]` |
| Order variants (greedy, fixed ascending, descending, random seeds, …) | `[TBD]` |
| Methods where order is **not** applicable (mark N/A) | `[TBD]` |

| Config id | Method | Order description | Aggregate | Max \(F_i\) | Notes |
|-----------|--------|-------------------|-----------|-------------|-------|
| `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |


## Deliverables and exit criteria

Use this checklist to close phase one.

- [ ] Baselines selected and rows completed in [Baseline selection and inventory](#baseline-selection-and-inventory).
- [ ] Calibration corpora collected; hashes and non-overlap verification recorded in [Calibration data (collection)](#calibration-data-collection).
- [ ] Benchmark and bundles frozen; paths recorded in [Benchmark (collection)](#benchmark-collection).
- [ ] Calibration-free baselines executed per [Protocol: calibration-free baselines](#protocol-calibration-free-baselines); [Runs to execute](#runs-to-execute) matrix complete.
- [ ] Accuracy and forgetting reported per [Reporting: accuracy and forgetting](#reporting-accuracy-and-forgetting); stub tables filled or linked artifacts attached.
- [ ] AIM / AIMerge reproduction complete per [Reproduce AIMerge (AIM)](#reproduce-aimerge-aim).
- [ ] Activation metric comparison documented per [Activation similarity metrics](#activation-similarity-metrics-comparison-study).
- [ ] Layer-linearity ablation documented per [(A)](#ablations-a-layer-merging-linearity).
- [ ] Merge-order ablation documented per [(B)](#ablations-b-merge-order).
- [ ] Short report pointers: paths to logs, tables, and figures `[TBD]`.

# Model + merge evaluation spec

This task defines the merge setting (what we merge, how we adapt per layer) and the evaluation protocol for merged dense LMs. Hugging Face IDs and exact task IDs live in experiment configs.


## Abstract

Global merge recipes for same-base dense LMs are often suboptimal. We study layer-adaptive merging: per layer (or block), combine base weights and specialist deltas with standard families (task arithmetic, TIES, DARE, interpolation), and choose layer-wise hyperparameters on a small benchmark-disjoint unlabeled set by minimizing a single activation-level discrepancy to a target distribution (e.g. MMD or Jensen–Shannon), with benchmarks reserved for validation. Evaluation follows MergeBench on instruction-following, reasoning (incl. math), and safety, plus per-specialist forgetting on pinned task bundles (max/mean across domains). Baselines include the best global merge of the same family, uniform and random layer profiles, single-source specialists, and AIM on top of the same underlying merge. Success means better aggregates or matched aggregates with lower worst forgetting, plus gains over random/uniform selection and reported ablations (merge order, prefix for scoring, calibration domain).

## Model choice

The model family is a dense Qwen3 decoder-only stack, one parameter scale per wave (chosen by compute; document N params + max context in logs). The rationale is that same-base specialists are weight-comparable, and Qwen3 matches our target hardware and hub availability for IF / reasoning / safety-flavored finetunes.

### Checkpoint roles

| Role | Role in merge |
|------|----------------|
| Base | \(W_{\text{base}}\); \(\Delta W = W_{\text{fine}} - W_{\text{base}}\); anchor for task arithmetic. |
| Instruction-following | Chat/instruct finetune; IF benchmark bundle. |
| Reasoning | Math/reasoning finetune; reasoning / math bundle. |
| Uncensored | Weaker-refusal finetune (same base); still evaluated on safety tasks. |

Constraints: same tokenizer and base checkpoint for every source; merges are full weight-space ops. Combine several non-base roles vs base; single-source rows use each finetune alone.

Selection: pick published Hugging Face finetunes (no new specialist training). Per wave, use one finetune per active domain (e.g. IF + reasoning only). Record repo ID, revision, license, and brief justification. Expand domains in later waves if needed.

---

## Merge setting (problem)

The goal is one merged model from two or more same-base finetunes using layer-wise merge strategies (no single global recipe for the entire network).

The hypothesis is to choose per-layer parameters by calibration-time alignment of merged-layer activations (or layer outputs) to a target distribution (convex mix of sources, anchor, or moment summaries), with one primary discrepancy per experiment (e.g. MMD, JS). Downstream evaluation validates this choice and is not assumed sufficient on its own.

Compositional ablations:

1. Local per-layer search vs global.
2. Rank candidates at layer ℓ with layers &lt;ℓ at base vs under the true merged prefix (it means that we merge all the layers till ℓ and then evaluate the model)
3. Sensitivity to greedy merge order.
4. Calibration corpus: generic task-agnostic unlabeled mix vs target-domain-aligned unlabeled subsets (see Calibration data).

Out of scope for v1: different architecture or tokenizer; merge-via-full-retrain as the main method; MoE / quant / multimodal unless added later.

### Calibration data

1. Corpus: a small, unlabeled dataset (raw text and/or unlabeled prompt-like snippets). Labels are not used for calibration objectives. Use forward passes only (activations / layer outputs) for alignment or candidate scoring—not supervised fine-tuning of merge weights.
2. Benchmark non-interference: calibration material must not overlap benchmark evaluation. Exclude benchmark test prompts, few-shot pools, and canonical dev sets that `lm-eval` / MergeBench tasks draw from. Freeze calibration corpus identity in configs (source ID, snapshot or content hash, preprocessing). Record how non-overlap was verified (e.g. checklist + versioned denylist, or citation of an officially separate calibration pack).
3. Default path: one task-agnostic, diverse small unlabeled mix, still benchmark-disjoint.
4. Ablation: repeat calibration on target-aligned small unlabeled subsets—text curated to resemble evaluation domains (e.g. instruction-like prose, math/reasoning-like text, appropriately neutral general text) without labeled benchmark tasks and without any instances used for benchmark scoring. Compare downstream aggregate, per-axis scores, and forgetting to the generic-calibration run to quantify sensitivity of layer-wise alignment to calibration domain.
### Parameterization and baselines

Per layer (or block), \(W_{\text{merged}}^{(\ell)} = W_{\text{base}}^{(\ell)} + g_\ell(\Delta W_1^{(\ell)},\ldots)\) with linear task arithmetic, TIES-style per layer, DARE-style per layer, and two-source interpolation. Optional: group attention + MLP as one unit.

The global baseline is one shared hyperparameter profile over all layers (same method family) to isolate granularity.

Adaptation uses search / grid, greedy layer order, and calibration divergence minimization on the small unlabeled corpus defined under Calibration data, plus an optional light validation signal (perplexity or small MC set) to correlate with distribution choice.

## Accuracy evaluation

### Benchmarks

1. Primary organizer: [MergeBench](https://github.com/uiuctml/MergeBench) — task categories, `lm-eval` / coding / safety harness patterns, reporting norms. Fork the repo; pin upstream updates deliberately. Exact task IDs are frozen in configs when runs begin.
2. Minimum axes: instruction-following, reasoning (incl. mathematics), safety.
3. Optional when budget allows: coding, multilingual (MergeBench-style multi-axis coverage).
4. Diagnostics: held-out perplexity or log-likelihood—not a substitute for the axes above.

### Forgetting vs specialists

TODO: check paper https://arxiv.org/abs/2507.23311 to give better foregetting definition.

For each non-base source \(i\) included in the merge, assign a domain bundle \(B_i\): the fixed subset of MergeBench (or config) tasks that represent that specialist’s target behavior (e.g. IF-heavy tasks for the instruct model, math/reasoning tasks for the reasoning model). Pin bundles in configs.

Per-domain score: mean (or specified aggregate) score of model \(M\) on \(B_i\) is \(S(M, B_i)\).

Reference models:

- Specialist upper bound: \(S(M_i^{\text{spec}}, B_i)\) — specialist \(i\) alone on its own bundle.
- Merged model: \(S(M^{\text{merge}}, B_i)\).

Forgetting (per domain \(i\)):

\[
F_i = S(M_i^{\text{spec}}, B_i) - S(M^{\text{merge}}, B_i)
\]

Report \(F_i\) per included specialist and summarize as follows:

1. Max forgetting: \(\max_i F_i\) (worst regression vs that specialist on its bundle).
2. Mean forgetting: average of \(F_i\) over included domains.

Sign convention: higher score means better on the benchmark; positive \(F_i\) means the merged model forgot capability relative to the specialist on that bundle. If a benchmark uses “lower is better”, define an inverted score in configs and keep \(F_i\) defined so positive still means “worse than specialist”.

Optional: relative drop \(F_i / S(M_i^{\text{spec}}, B_i)\) when \(S > 0\) and comparable across tasks.

## How results are reported

Results should fit a main table plus small supporting tables (per-axis, forgetting, ablations).

### Main table

One row per merge configuration (and seed if multiple). Columns / fields:

1. Config id: consistent scheme `base_model + merge_method + variant + seed_or_calib_id` (e.g. `qwen3-4b-layerwise-mmd-calibA-run1`).
2. Method: global vs layer-adaptive; family (task arithmetic / TIES / DARE / …).
3. Aggregate downstream score (state formula: e.g. macro mean over IF + reasoning + safety).
4. Per-axis scores (IF, reasoning, safety, and optional coding / multilingual).
5. Forgetting: \(F_i\) per domain, max \(F_i\), mean \(F_i\).
6. Notes: OOM, eval fallback, instability.

### Supporting tables

1. Per-task or per-category breakdown for MergeBench-style tasks (required for debugging regressions), at least for merged vs best baseline.
2. Forgetting table: rows = merge configs, columns = specialist domain \(i\), cells = \(F_i\) (and optionally relative drop).
3. Ablations: local vs global scoring, merge order, distribution metric choice—separate small table or appendix rows.

### Minimal reporting conventions

1. Environment block for every paper-ready run: GPU model + VRAM, CUDA / driver, PyTorch, `lm-eval` / harness versions, exact model revisions (HF commit hash) for base and each specialist.
2. Aggregation: state `mean over N seeds` or `median over N calibration draws` where applicable; never leave a single number unexplained.
3. Bundles: path or name of the config that defines each \(B_i\) and the MergeBench task list revision.
4. Calibration: corpus name/version, size (documents or tokens), preprocessing, and non-overlap verification (or pointer to denylist / verification artifact).

---

## Protocol / reproducibility

Fix decoding settings (temperature, top-p, max tokens) across all models on a given benchmark unless the task requires task-specific settings (then document per task).

Run stochastic merges or calibration ≥ 3 times when variance matters; report median primary metrics (keep raw logs).

Success criteria (any counts as progress):

1. Layer-adaptive beats best global merge on stated aggregate, or matches it with lower max forgetting (\(\max_i F_i\)) or better worst-axis downstream score; when AIM baselines are run, include them in this comparison (same underlying merge recipe before AIM).
2. Distribution-based selection correlates with downstream vs random / uniform layer profiles.
3. Compositional ablations reported (local score vs true prefix; merge order).
4. Modest but consistent gains across seeds / calibration draws count.

### Main baselines

1. Best global task-arithmetic / TIES / DARE (fair search budget vs adaptive).
2. Uniform per-layer coefficients if distinct from (1).
3. Each single-source specialist + naive full-weight average if applicable.
. Random layer-wise profile (matched search cardinality).
5. [Activation-Informed Merging (AIM)](https://github.com/ahnobari/ActivationInformedMerging) — activation-guided consensus refinement on top of a merged checkpoint: uses the base model, a task-agnostic calibration set, and weight prioritization informed by activations (continual-learning / compression-style signal). Run AIM after the same underlying merge family as in (1)-(2) (e.g. DARE+TIES, task arithmetic) so comparisons isolate AIM vs our layer-wise strategy.
6. [Chain of Merges](https://openreview.net/pdf?id=Q0ANR30XFh): include this method as an additional external baseline under the same evaluation protocol and reporting format.
7. [LOT Merging](https://openreview.net/pdf?id=0KOfAUiHua): track as related work now; promote to an executable baseline when public code is available, then evaluate under the same protocol and reporting format.

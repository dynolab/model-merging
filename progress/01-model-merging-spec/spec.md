# Model + merge evaluation spec

This task defines the **merge setting** (what we merge, how we adapt per layer) and the **evaluation protocol** for merged dense LMs. Hugging Face IDs and exact task IDs live in experiment configs.

---

## Model choice

- **Family:** dense **Qwen3** decoder-only, **one parameter scale** per wave (chosen by compute; document N params + max context in logs).
- **Rationale:** Same-base specialists are weight-comparable; Qwen3 matches our target hardware and hub availability for IF / reasoning / safety-flavored finetunes.

### Checkpoint roles

| Role | Role in merge |
|------|----------------|
| **Base** | \(W_{\text{base}}\); \(\Delta W = W_{\text{fine}} - W_{\text{base}}\); anchor for task arithmetic. |
| **Instruction-following** | Chat/instruct finetune; IF benchmark bundle. |
| **Reasoning** | Math/reasoning finetune; reasoning / math bundle. |
| **Uncensored** | Weaker-refusal finetune (same base); still evaluated on **safety** tasks. |

**Constraints.** Same tokenizer and base checkpoint for every source; merges are **full weight-space** ops. Combine several non-base roles vs base; **single-source** rows use each finetune alone.

**Selection.** Pick published Hugging Face finetunes (no new specialist training). Per wave: **one finetune per active domain** (e.g. IF + reasoning only). Record: repo ID, revision, license, brief justification. Expand domains in later waves if needed.

---

## Merge setting (problem)

- **Goal:** one merged model from **two or more** same-base finetunes using **layer-wise** merge strategies (no single global recipe for the entire network).
- **Hypothesis:** choose per-layer parameters by **calibration-time alignment** of merged-layer activations (or layer outputs) to a target distribution (convex mix of sources, anchor, or moment summaries); one primary discrepancy per experiment (e.g. MMD, JS). Downstream eval validates—not assumed sufficient alone.
- **Ablations (compositional):** (1) local per-layer search vs global; (2) rank candidates at layer ℓ with layers **&lt;ℓ** at base vs under the **true** merged prefix; (3) sensitivity to **greedy merge order**.
- **Out of scope (v1):** different arch/tokenizer; merge-via-full-retrain as the main method; MoE / quant / multimodal unless added later.

### Parameterization and baselines

- **Per layer (or block):** \(W_{\text{merged}}^{(\ell)} = W_{\text{base}}^{(\ell)} + g_\ell(\Delta W_1^{(\ell)},\ldots)\) — linear task arithmetic, **TIES**-style per layer, **DARE**-style per layer, two-source interpolation. Optional: group attention + MLP as one unit.
- **Global baseline:** one shared hyperparameter profile over **all** layers (same method family) to isolate **granularity**.
- **Adaptation:** search / grid; greedy layer order; calibration divergence minimization; optional light validation signal (perplexity or small MC set) to correlate with distribution choice.

---

## Accuracy evaluation

### Benchmarks

- **Primary organizer:** **[MergeBench](https://github.com/uiuctml/MergeBench)** — task categories, `lm-eval` / coding / safety harness patterns, reporting norms. Fork the repo; pin upstream updates deliberately. Exact task IDs **frozen in configs** when runs begin.
- **Minimum axes:** instruction-following, reasoning (incl. mathematics), **safety**.
- **Optional when budget allows:** coding, multilingual (MergeBench-style multi-axis coverage).
- **Diagnostics:** held-out perplexity or log-likelihood — **not** a substitute for the axes above.

### Forgetting vs specialists
 
TODO: check paper https://arxiv.org/abs/2507.23311 to give better foregetting definition.

For each **non-base source** \(i\) included in the merge, assign a **domain bundle** \(B_i\) — the fixed subset of MergeBench (or config) tasks that represent that specialist’s target behavior (e.g. IF-heavy tasks for the instruct model, math/reasoning tasks for the reasoning model). Pin bundles in configs.

**Per-domain score.** Mean (or specified aggregate) score of model \(M\) on \(B_i\): \(S(M, B_i)\).

**References.**

- **Specialist upper bound:** \(S(M_i^{\text{spec}}, B_i)\) — specialist \(i\) alone on its own bundle.
- **Merged model:** \(S(M^{\text{merge}}, B_i)\).

**Forgetting (per domain \(i\)).**

\[
F_i = S(M_i^{\text{spec}}, B_i) - S(M^{\text{merge}}, B_i)
\]

Report \(F_i\) **per included specialist** and summarize:

- **Max forgetting:** \(\max_i F_i\) (worst regression vs that specialist on its bundle).
- **Mean forgetting:** average of \(F_i\) over included domains.

**Sign.** Higher score = better on the benchmark; **positive** \(F_i\) means the merged model **forgot** capability relative to the specialist on that bundle. If a benchmark uses “lower is better”, define an inverted score in configs and keep \(F_i\) defined so positive still means “worse than specialist”.

**Optional:** relative drop \(F_i / S(M_i^{\text{spec}}, B_i)\) when \(S > 0\) and comparable across tasks.

---

## How results are reported

Results should fit a **main table** plus small supporting tables (per-axis, forgetting, ablations).

### Main table

One row per **merge configuration** (and seed if multiple):

- **Config id:** consistent scheme `base_model + merge_method + variant + seed_or_calib_id` (e.g. `qwen3-4b-layerwise-mmd-calibA-run1`).
- **Method:** global vs layer-adaptive; family (task arithmetic / TIES / DARE / …).
- **Aggregate downstream score** (state formula: e.g. macro mean over IF + reasoning + safety).
- **Per-axis scores** (IF, reasoning, safety, and optional coding / multilingual).
- **Forgetting:** \(F_i\) per domain, **max** \(F_i\), **mean** \(F_i\).
- **Notes:** OOM, eval fallback, instability.

### Supporting tables

- **Per-task or per-category breakdown** for MergeBench-style tasks (required for debugging regressions), at least for merged vs best baseline.
- **Forgetting table:** rows = merge configs, columns = specialist domain \(i\), cells = \(F_i\) (and optionally relative drop).
- **Ablations:** local vs global scoring, merge order, distribution metric choice — separate small table or appendix rows.

### Minimal reporting conventions

- **Environment block** for every paper-ready run: GPU model + VRAM, CUDA / driver, PyTorch, `lm-eval` / harness versions, **exact model revisions** (HF commit hash) for base and each specialist.
- **Aggregation:** state `mean over N seeds` or `median over N calibration draws` where applicable; never leave a single number unexplained.
- **Bundles:** path or name of the config that defines each \(B_i\) and the MergeBench task list revision.

---



## Protocol / reproducibility

- Fix **decoding** settings (temperature, top-p, max tokens) across all models on a given benchmark unless the task requires task-specific settings (then document per task).
- Run stochastic merges or calibration **≥ 3** times when variance matters; report **median** primary metrics (keep raw logs).
- Success criteria (any counts as progress):
  1. Layer-adaptive **beats best global** merge on stated aggregate, **or** matches it with **lower max forgetting** (\(\max_i F_i\)) or better **worst-axis** downstream score; when **AIM** baselines are run, include them in this comparison (same underlying merge recipe before AIM).
  2. Distribution-based selection **correlates** with downstream vs random / uniform layer profiles.
  3. Compositional ablations reported (local score vs true prefix; merge order).
  4. Modest but **consistent** gains across seeds / calibration draws count.

### Main baselines

1. Best **global** task-arithmetic / TIES / DARE (fair search budget vs adaptive).
2. **Uniform** per-layer coefficients if distinct from (1).
3. Each **single-source** specialist + naive full-weight average if applicable.
4. **Random** layer-wise profile (matched search cardinality).
5. **[Activation-Informed Merging (AIM)](https://github.com/ahnobari/ActivationInformedMerging)** — activation-guided consensus refinement on top of a merged checkpoint: uses the **base** model, a **task-agnostic calibration** set, and weight prioritization informed by activations (continual-learning / compression–style signal). Run AIM **after** the same underlying merge family as in (1)–(2) (e.g. DARE+TIES, task arithmetic) so comparisons isolate AIM vs our **layer-wise** strategy.

---

## One-line summary

**Qwen3** same-base specialists, **per-layer** task-arithmetic-style merges with **calibration-time activation alignment**, evaluated on **MergeBench-style** tasks with **per-domain forgetting** vs each included specialist, plus standard baselines and compositional ablations.

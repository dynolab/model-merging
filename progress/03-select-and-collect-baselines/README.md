## Baseline selection and inventory

All merges combine the three single-source specialists over the shared base. The
merged checkpoints we produced are released on the [Hugging Face Hub](https://huggingface.co/libvm/models) (repo prefix `mm-cand-`). The four
reference models below are external sources, not our uploads.

### Reference / source models (external)

| Config id | Role | HF source @ revision | License |
|-----------|------|----------------------|---------|
| `base` | Shared base; reference, not a merge | [`Qwen/Qwen3-8B-Base`](https://huggingface.co/Qwen/Qwen3-8B-Base) @ `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` | Apache-2.0 |
| `instr_only` | Specialist (instruction); reference for `B_if` | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) @ `b968826d9c46dd6066d109eabc6255188de91218` | Apache-2.0 |
| `reasoning_only` | Specialist (math reasoning); reference for `B_reasoning_math` | [`OpenDataArena/Qwen3-8B-ODA-Math-460k`](https://huggingface.co/OpenDataArena/Qwen3-8B-ODA-Math-460k) @ `8e8758ee23e5d959dd844b3bb5c5344219544795` | CC-BY-NC-4.0 |
| `uncensored_only` | Specialist (weak refusal); reference for `B_uncensored` | [`mlabonne/Qwen3-8B-abliterated`](https://huggingface.co/mlabonne/Qwen3-8B-abliterated) @ `30c72fa348f37c72d12ecbb259068ddee98aa9ed` | Apache-2.0 |

### Merged models we released

Global mergekit baselines (`_best` = parameters selected from a fixed grid via `arc_challenge`):

| Config id | Method | Calibration | HF model |
|-----------|--------|-------------|----------|
| `task_arithmetic_best` | Task Arithmetic (global) | none | [`libvm/mm-cand-task_arithmetic_best`](https://huggingface.co/libvm/mm-cand-task_arithmetic_best) |
| `ties_best` | TIES (global) | none | [`libvm/mm-cand-ties_best`](https://huggingface.co/libvm/mm-cand-ties_best) |
| `dare_best` | DARE (global) | none | [`libvm/mm-cand-dare_best`](https://huggingface.co/libvm/mm-cand-dare_best) |
| `della_best` | DELLA (global) | none | [`libvm/mm-cand-della_best`](https://huggingface.co/libvm/mm-cand-della_best) |
| `breadcrumbs_best` | Breadcrumbs (global) | none | [`libvm/mm-cand-breadcrumbs_best`](https://huggingface.co/libvm/mm-cand-breadcrumbs_best) |

AIM - Activation-Informed Merging applied on top of each `_best` parent (global, activation-rescaled), one checkpoint per calibration subset:

| Config id | Parent | Calibration | HF model |
|-----------|--------|-------------|----------|
| `aim_on_task_arithmetic` | task_arithmetic_best | mix | [`libvm/mm-cand-aim_on_task_arithmetic`](https://huggingface.co/libvm/mm-cand-aim_on_task_arithmetic) |
| `aim_on_task_arithmetic__calib_general` | task_arithmetic_best | general | [`libvm/mm-cand-aim_on_task_arithmetic__calib_general`](https://huggingface.co/libvm/mm-cand-aim_on_task_arithmetic__calib_general) |
| `aim_on_task_arithmetic__calib_instruction` | task_arithmetic_best | instruction | [`libvm/mm-cand-aim_on_task_arithmetic__calib_instruction`](https://huggingface.co/libvm/mm-cand-aim_on_task_arithmetic__calib_instruction) |
| `aim_on_task_arithmetic__calib_reasoning` | task_arithmetic_best | reasoning | [`libvm/mm-cand-aim_on_task_arithmetic__calib_reasoning`](https://huggingface.co/libvm/mm-cand-aim_on_task_arithmetic__calib_reasoning) |
| `aim_on_task_arithmetic__calib_uncensored_refusal_like` | task_arithmetic_best | uncensored/refusal-like | [`libvm/mm-cand-aim_on_task_arithmetic__calib_uncensored_refusal_like`](https://huggingface.co/libvm/mm-cand-aim_on_task_arithmetic__calib_uncensored_refusal_like) |
| `aim_on_ties` | ties_best | mix | [`libvm/mm-cand-aim_on_ties`](https://huggingface.co/libvm/mm-cand-aim_on_ties) |
| `aim_on_ties__calib_general` | ties_best | general | [`libvm/mm-cand-aim_on_ties__calib_general`](https://huggingface.co/libvm/mm-cand-aim_on_ties__calib_general) |
| `aim_on_ties__calib_instruction` | ties_best | instruction | [`libvm/mm-cand-aim_on_ties__calib_instruction`](https://huggingface.co/libvm/mm-cand-aim_on_ties__calib_instruction) |
| `aim_on_ties__calib_reasoning` | ties_best | reasoning | [`libvm/mm-cand-aim_on_ties__calib_reasoning`](https://huggingface.co/libvm/mm-cand-aim_on_ties__calib_reasoning) |
| `aim_on_ties__calib_uncensored_refusal_like` | ties_best | uncensored/refusal-like | [`libvm/mm-cand-aim_on_ties__calib_uncensored_refusal_like`](https://huggingface.co/libvm/mm-cand-aim_on_ties__calib_uncensored_refusal_like) |
| `aim_on_dare` | dare_best | mix | [`libvm/mm-cand-aim_on_dare`](https://huggingface.co/libvm/mm-cand-aim_on_dare) |
| `aim_on_dare__calib_general` | dare_best | general | [`libvm/mm-cand-aim_on_dare__calib_general`](https://huggingface.co/libvm/mm-cand-aim_on_dare__calib_general) |
| `aim_on_dare__calib_instruction` | dare_best | instruction | [`libvm/mm-cand-aim_on_dare__calib_instruction`](https://huggingface.co/libvm/mm-cand-aim_on_dare__calib_instruction) |
| `aim_on_dare__calib_reasoning` | dare_best | reasoning | [`libvm/mm-cand-aim_on_dare__calib_reasoning`](https://huggingface.co/libvm/mm-cand-aim_on_dare__calib_reasoning) |
| `aim_on_dare__calib_uncensored_refusal_like` | dare_best | uncensored/refusal-like | [`libvm/mm-cand-aim_on_dare__calib_uncensored_refusal_like`](https://huggingface.co/libvm/mm-cand-aim_on_dare__calib_uncensored_refusal_like) |
| `aim_on_della` | della_best | mix | [`libvm/mm-cand-aim_on_della`](https://huggingface.co/libvm/mm-cand-aim_on_della) |
| `aim_on_della__calib_general` | della_best | general | [`libvm/mm-cand-aim_on_della__calib_general`](https://huggingface.co/libvm/mm-cand-aim_on_della__calib_general) |
| `aim_on_della__calib_instruction` | della_best | instruction | [`libvm/mm-cand-aim_on_della__calib_instruction`](https://huggingface.co/libvm/mm-cand-aim_on_della__calib_instruction) |
| `aim_on_della__calib_reasoning` | della_best | reasoning | [`libvm/mm-cand-aim_on_della__calib_reasoning`](https://huggingface.co/libvm/mm-cand-aim_on_della__calib_reasoning) |
| `aim_on_della__calib_uncensored_refusal_like` | della_best | uncensored/refusal-like | [`libvm/mm-cand-aim_on_della__calib_uncensored_refusal_like`](https://huggingface.co/libvm/mm-cand-aim_on_della__calib_uncensored_refusal_like) |
| `aim_on_breadcrumbs` | breadcrumbs_best | mix | [`libvm/mm-cand-aim_on_breadcrumbs`](https://huggingface.co/libvm/mm-cand-aim_on_breadcrumbs) |
| `aim_on_breadcrumbs__calib_general` | breadcrumbs_best | general | [`libvm/mm-cand-aim_on_breadcrumbs__calib_general`](https://huggingface.co/libvm/mm-cand-aim_on_breadcrumbs__calib_general) |
| `aim_on_breadcrumbs__calib_instruction` | breadcrumbs_best | instruction | [`libvm/mm-cand-aim_on_breadcrumbs__calib_instruction`](https://huggingface.co/libvm/mm-cand-aim_on_breadcrumbs__calib_instruction) |
| `aim_on_breadcrumbs__calib_reasoning` | breadcrumbs_best | reasoning | [`libvm/mm-cand-aim_on_breadcrumbs__calib_reasoning`](https://huggingface.co/libvm/mm-cand-aim_on_breadcrumbs__calib_reasoning) |
| `aim_on_breadcrumbs__calib_uncensored_refusal_like` | breadcrumbs_best | uncensored/refusal-like | [`libvm/mm-cand-aim_on_breadcrumbs__calib_uncensored_refusal_like`](https://huggingface.co/libvm/mm-cand-aim_on_breadcrumbs__calib_uncensored_refusal_like) |

LOT - module-wise activation-informed direct merge, calibration is set per specialist:

| Config id | Calibration | HF model | Prompts × tokens |
|-----------|-------------|----------|-------------|
| `lot_paper_domain_matched` | domain-matched (per-specialist) | [`libvm/mm-cand-lot_paper_domain_matched`](https://huggingface.co/libvm/mm-cand-lot_paper_domain_matched) | 64 × 128 |
| `lot_paper_mix` | mix | [`libvm/mm-cand-lot_paper_mix`](https://huggingface.co/libvm/mm-cand-lot_paper_mix) | 64 × 128 |
| `lot_paper_general` | general | [`libvm/mm-cand-lot_paper_general`](https://huggingface.co/libvm/mm-cand-lot_paper_general) | 64 × 128 |
| `lot_paper_instruction` | instruction | [`libvm/mm-cand-lot_paper_instruction`](https://huggingface.co/libvm/mm-cand-lot_paper_instruction) | 64 × 128 |
| `lot_paper_reasoning` | reasoning | [`libvm/mm-cand-lot_paper_reasoning`](https://huggingface.co/libvm/mm-cand-lot_paper_reasoning) | 64 × 128 |
| `lot_paper_uncensored_refusal_like` | uncensored/refusal-like | [`libvm/mm-cand-lot_paper_uncensored_refusal_like`](https://huggingface.co/libvm/mm-cand-lot_paper_uncensored_refusal_like) | 64 × 128 |
| `v3-lot_paper_domain_matched` | domain-matched (per-specialist) | [`libvm/mm-cand-v3-lot_paper_domain_matched`](https://huggingface.co/libvm/mm-cand-v3-lot_paper_domain_matched) | 192 × 64 |
| `v3-lot_paper_mix` | mix | [`libvm/mm-cand-v3-lot_paper_mix`](https://huggingface.co/libvm/mm-cand-v3-lot_paper_mix) | 192 × 64 |
| `v3-lot_paper_general` | general | [`libvm/mm-cand-v3-lot_paper_general`](https://huggingface.co/libvm/mm-cand-v3-lot_paper_general) | 192 × 64 |
| `v3-lot_paper_instruction` | instruction | [`libvm/mm-cand-v3-lot_paper_instruction`](https://huggingface.co/libvm/mm-cand-v3-lot_paper_instruction) | 192 × 64 |
| `v3-lot_paper_reasoning` | reasoning | [`libvm/mm-cand-v3-lot_paper_reasoning`](https://huggingface.co/libvm/mm-cand-v3-lot_paper_reasoning) | 192 × 64 |
| `v3-lot_paper_uncensored_refusal_like` | uncensored/refusal-like | [`libvm/mm-cand-v3-lot_paper_uncensored_refusal_like`](https://huggingface.co/libvm/mm-cand-v3-lot_paper_uncensored_refusal_like) | 192 × 64 |

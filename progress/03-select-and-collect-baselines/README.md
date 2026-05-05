## Baseline selection and inventory


| Baseline id | Method family | Global vs layer-wise | Requires calibration? | HF repo + revision | License | Maps to parent baseline # / note |
|-------------|---------------|----------------------|------------------------|-------------------|---------|----------------------------------|
| `base` | base checkpoint | N/A | no | `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4` | Apache-2.0 | Reference; not a merge baseline |
| `instr_only` | single-source specialist | N/A | no | `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` | Apache-2.0 | Parent: single-source specialist; evaluated with `enable_thinking=False`; reference for `B_if` |
| `reasoning_only` | single-source specialist | N/A | no | `OpenDataArena/Qwen3-8B-ODA-Math-460k@8e8758ee23e5d959dd844b3bb5c5344219544795` | CC-BY-NC-4.0 | Parent: single-source specialist; reference for `B_reasoning` |
| `uncensored_only` | single-source specialist | N/A | no | `mlabonne/Qwen3-8B-abliterated@30c72fa348f37c72d12ecbb259068ddee98aa9ed` | Apache-2.0 | Parent: single-source specialist; reference for `B_uncensored` |
| `naive_avg` | full-weight average | global | no | local checkpoint; file hashes recorded; `merge_manifest.json` recorded when present | N/A | Parent: naive full-weight average |
| `task_arithmetic_best` | task arithmetic | global | no | local checkpoint; file hashes recorded; `merge_manifest.json` recorded when present | N/A | Parent: best global merge; parameters selected from a fixed grid |
| `ties_best` | TIES | global | no | local checkpoint; file hashes recorded; `merge_manifest.json` recorded when present | N/A | Parent: best global merge; parameters selected from a fixed grid |
| `dare_best` | DARE | global | no | local checkpoint; file hashes recorded; `merge_manifest.json` recorded when present | N/A | Parent: best global merge; parameters selected from a fixed grid |
| `della_best` | DELLA | global | no | local checkpoint; file hashes recorded; `merge_manifest.json` recorded when present | N/A | Additional baseline; global parameters selected from a fixed grid |
| `breadcrumbs_best` | Breadcrumbs | global | no | local checkpoint; file hashes recorded; `merge_manifest.json` recorded when present | N/A | Additional baseline; global parameters selected from a fixed grid |
| `task_arithmetic_random` | task arithmetic | layer-wise | no | local checkpoint; file hashes recorded; `merge_manifest.json` recorded when present | N/A | Parent: random layer-wise profile; parameters sampled from a fixed grid; seed fixes the profile |
| `ties_random` | TIES | layer-wise | no | local checkpoint; file hashes recorded; `merge_manifest.json` recorded when present | N/A | Parent: random layer-wise profile; parameters sampled from a fixed grid; seed fixes the profile |
| `dare_random` | DARE | layer-wise | no | local checkpoint; file hashes recorded; `merge_manifest.json` recorded when present | N/A | Parent: random layer-wise profile; parameters sampled from a fixed grid; seed fixes the profile |
| `della_random` | DELLA | layer-wise | no | local checkpoint; file hashes recorded; `merge_manifest.json` recorded when present | N/A | Additional random layer-wise baseline; parameters sampled from a fixed grid; seed fixes the profile |
| `breadcrumbs_random` | Breadcrumbs | layer-wise | no | local checkpoint; file hashes recorded; `merge_manifest.json` recorded when present | N/A | Additional random layer-wise baseline; parameters sampled from a fixed grid; seed fixes the profile |
| `aim_on_task_arithmetic` | AIM | global | yes | local checkpoint; file hashes recorded; `merge_manifest.json` recorded when present | N/A | Parent: AIM on same underlying merge; after `task_arithmetic_best` |
| `aim_on_ties` | AIM | global | yes | local checkpoint; file hashes recorded; `merge_manifest.json` recorded when present | N/A | Parent: AIM on same underlying merge; after `ties_best` |
| `aim_on_dare` | AIM | global | yes | local checkpoint; file hashes recorded; `merge_manifest.json` recorded when present | N/A | Parent: AIM on same underlying merge; after `dare_best` |
| `aim_on_della` | AIM | global | yes | local checkpoint; file hashes recorded; `merge_manifest.json` recorded when present | N/A | Additional AIM baseline; after `della_best` |
| `aim_on_breadcrumbs` | AIM | global | yes | local checkpoint; file hashes recorded; `merge_manifest.json` recorded when present | N/A | Additional AIM baseline; after `breadcrumbs_best` |
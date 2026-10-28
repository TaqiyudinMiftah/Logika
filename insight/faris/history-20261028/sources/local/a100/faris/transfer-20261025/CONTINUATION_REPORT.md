# Continuation outcome — 25 October 2026

**No new candidate passed the submission rule. The 0.54 local KS target remains unmet.** The best observed public score is still V25's **0.56822**. Five further fixed hypotheses completed after the user requested continuation; no submission was made after the daily reset. The authenticated Kaggle readback still reports three available slots.

The highest new local point is **0.531505932**. That is a research observation, not a promoted improvement: it is only **0.000102127** above A, loses against its matched control on the fresh Kumo seed, and wins only two of five folds for the two-seed mean. The adjusted conditional interval for its gain is **[-0.000896073, +0.001131765]**. The remaining point-score gap to 0.54 is **0.008494068**.

## What the new tests established

| Test | Pooled OOF KS | Decision |
|---|---:|---|
| H012: correction fitted after the final mixture | 0.529905572, two-seed mean | Rejected; below its matched control and B. |
| H013: class-balanced loss for H012 | 0.528845312, two-seed mean | Rejected. |
| H014: shared-application bureau measurement model | 0.530497857 original; 0.529382164 fresh | Measurement diagnostics improved, default prediction did not reliably improve. |
| H015: balanced Kumo context | 0.527676778 original; 0.529070860 fresh | Both lose against full-context controls. Same-size proportional controls also fail to support a balance gain. |
| H016: direct smooth-KS objective in the 56k spline branch | 0.531505932, two-seed mean | Tiny point gain fails the frozen stability requirements. |

All tests retain the original five outer folds and pooled, tie-aware KS over 70,000 applicants. The metric was checked against SciPy. No validation labels fit the model predicting that row. All labels have historical development exposure; this is **not independent confirmation**. Conditional paired bootstraps do not account for the entire adaptive search history or full-data refitting.

The train-versus-test population diagnostic used membership labels only. Its five-fold AUC was **0.501717429**, with conditional interval **[0.498147158, 0.505587791]**. It found no useful separation under the fixed classifier and measured features. It does not rule out target-conditional shift or establish the actual public/private sampling scheme. No domain-weighted model was launched from this result.

The score-sampling illustration shows that close model rankings can reverse on smaller samples. It uses existing OOF rows with replacement; the actual public sample size is unknown. None of its resampled scores counts as a new local achievement or an estimate of the actual Kaggle score.

## Verified artifacts and submissions

- H015's fresh reconstruction reproduced all 20,000 raw and 14,000 final validation predictions exactly: [replay audit](balanced_context_replay_fold0.audit.json).
- All five H016 pipelines were rebuilt from their retained spline updates, original encoders, Kumo predictions and correction heads. They reproduce the scored OOF predictions to numerical precision: [pipeline replay](spline_direct_ks/PIPELINE_REPLAY_AUDIT.json).
- Every new fitted component is retained. There are now **335 serialized artifact files**, up from 245 before this continuation. This is a file count, not 335 independent experiments.
- [Continuation manifest](CONTINUATION_RETENTION_MANIFEST.json) and [local delivery audit](CONTINUATION_LOCAL_DELIVERY_AUDIT.json) record the exact archive and mirror. Some historical feature tables, contexts and the frozen checkpoint remain at recorded PC2 paths. The archive is not a standalone raw-data distribution.
- The first H016 service launch failed before fitting because of its working directory. The retry completed all five fits; a service-variable expansion caused their one-line stdout messages to share a log. Per-fold fitted artifacts, optimizer diagnostics, runtimes and completion hashes survived. See [incident record](H016_LAUNCH_INCIDENT.json).

The latest authenticated submission history still identifies:

| File | Public KS | Reference |
|---|---:|---:|
| V25 | **0.56822** | 57444414 |
| B, maturity-only | 0.56818 | 57604838 |
| Earlier root submission | 0.56678 | 57482053 |
| A, periodicity plus two Kumo seeds | 0.56656 | 57605161 |

The exact scored V25 download is preserved as [submission_best_public_V25_056822_verified.csv](/home/parasite/Project/Competition/logika-2026/submission_best_public_V25_056822_verified.csv), SHA-256 `e7f0748c6c8ded776f0a6a03bdebcd0787cf6c0057d0b078c31196fb34f58efe`. No final Kaggle selection was changed. Private scores are unknown.

## Why this batch ends here

All five fixed proposals failed their own rules. Increasing the number of small score-driven variants would use the same repeatedly exposed labels. The inherited availability of some repayment snapshot values at decision time remains unresolved, so the development gate does not admit broader search.

The [research-analyzer skill](/home/parasite/.codex/skills/research-analyzer/SKILL.md) states: **“Failed, stale, or inconclusive evidence blocks search.”** That requirement limits further search here; it does not mean the existing public incumbent has been proved invalid. The user already authorized submissions, so no additional permission is needed for a future candidate that passes its checks.

The continuation began at **05:48 UTC / 12:48 WIB**, with the original **07:48 UTC** training deadline and **08:48 UTC** finalization deadline preserved. The wall-clock budget was not exhausted. This batch ends on negative evidence and the unresolved broad-search gate, with all completed artifacts retained and no further job queued. [Status](CONTINUATION_STATUS.json) records the exact finalization time and current target state.

Further useful work needs a concrete new source of signal with verified information timing, or a resolved validation concern. A local 0.54 would remain development evidence; it would not itself guarantee a public or private leaderboard gain.

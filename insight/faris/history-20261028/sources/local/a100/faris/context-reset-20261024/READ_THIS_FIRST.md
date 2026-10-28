# Logika: decision brief, 24 October 2026

The user's aim is a better Kaggle result; 0.54 local pooled OOF KS is the desired research milestone. These are related objectives, but the latest evidence shows that choosing the highest local point estimate has not reliably selected the best public submission.

## What is established

Seven named submissions in the user's latest Kaggle results match version IDs and filenames in the retained teammate experiment log. Public scores below are user-reported in this chat, not independently authenticated here. Local results come from the 21 October snapshot; submitted bytes were not checked this turn.

| Recipe | Historical pooled OOF KS | Reported public KS |
|---|---:|---:|
| V25 | 0.53028466 | **0.56822** |
| V21 | 0.53017088 | 0.56579 |
| V90 | 0.52965289 | 0.56551 |
| V19 | 0.53048532 | 0.56533 |
| V40 | 0.53057010 | 0.56464 |
| V80 | **0.53073692** | 0.56430 |
| V71 | 0.53012059 | 0.56404 |

V80 exceeds V25 locally by 0.00045226 but trails publicly by 0.00392. V19 and V40 also exceed V25 locally and trail publicly. This is evidence of ranking reversals among closely scored candidates. Seven selected, related submissions cannot establish that validation is useless, identify distribution shift, or prove V25 is best on the private test.

The user also reports `submission.csv` at 0.56678 but is unsure which model produced it. Leave it unmatched. Other reported files are `submission_large_basket_domain_both.csv` at 0.56088 and `submission_large_cre_next_select64.csv` at 0.56082; their local recipe metrics were not joined in this inspection.

## Current research recipe

Local saved predictions were re-scored in this inspection using tied-score boundaries; they reproduce the retained pooled KS values within 1e-12.

| Recipe | Pooled OOF KS | Mean fold KS |
|---|---:|---:|
| Repaired V25 | 0.53021854 | 0.53336073 |
| Maturity replacement | 0.53063610 | 0.53341234 |
| Periodicity plus maturity | **0.53096653** | 0.53263488 |

The current research lead is 0.00074799 above the repaired baseline and 0.00903347 below the target. Periodicity improves two of five folds versus maturity. Its retained adjusted conditional interval for KS improvement versus the repaired baseline includes zero: approximately [-0.0024075, 0.0040557]. It also lowers AUC. Neither the mean-fold statistic nor AUC replaces the chosen pooled KS objective; they explain why the small pooled gain is not strong evidence of a general improvement.

The current root CSV hash still matches the handoff: `922788d13cd8ec53b7cbd732496a9200150a499fd4363455414e2cc6230a6b06`. Its full-data refit has no outer validation, and its public score remains unidentified. Historical V25 and repaired V25 must remain distinct.

## Recommended next decision

1. Preserve V25 as the best observed public reference and periodicity+maturity as the local point-score reference. Keep both; neither warrants an unconditional generalization claim.
2. Before opening another broad search, compare V25, maturity, and periodicity under a small predeclared robustness check. Use the same evaluation roles for every recipe and vary the context/correction assignment or fitting seed in matched runs. This tests training-role sensitivity without changing the target's original fold definition. If a fixed alternate outer split is also used, report it separately. Previously used labels do not become a fresh holdout after reshuffling.
3. Diagnose repeatable ranking errors using predictions generated without the inspected rows' labels. Start with a few fixed, interpretable slices such as history coverage and repayment maturity. Existing full OOF can generate exploratory hypotheses; any learned correction, slice selection, or routing must be developed inside outer training roles and evaluated outside them. Do not fit a global residual model to OOF labels and call its score validation.
4. Run one intervention linked to an observed failure. Require an incremental ranking effect beyond existing inputs. A better measurement forecast alone is insufficient: several completed studies already demonstrated this. The prepared bureau-occasion study is unfinished, but its measurement-screen gain does not make it a leading route to 0.54.
5. Keep pooled KS primary. Record AUC, log loss, fold behavior, and paired uncertainty as diagnostics rather than silently imposing a requirement that every metric improve. A reproducible 0.54 on the locked protocol would be stronger evidence than the current tiny differences, but still would not guarantee a Kaggle increase.

This review does not establish a noise ceiling or a method guaranteed to reach 0.54. The immediate uncertainty is whether candidate improvements survive changes in fitting roles, followed by which remaining errors have observable predictors. More model count by itself does not answer either question.

## Evidence and inspection scope

- [Recomputed local diagnostics](OOF_DIAGNOSTIC.json): source hash, class counts, pooled and fold metrics, and current CSV hash.
- [Public/local comparison](PUBLIC_LOCAL_COMPARISON.json): named recipe metadata, retained artifact hashes, user-reported public results, and unmatched submissions.
- Source experiment log: `../feature-only-20261021/friend-refresh-20261021/EXPERIMENT_MASTER_LOG.json`.
- Source local predictions and uncertainty: `../shopping-periodicity-fe-20261021/OOF.npz` and `result.json`.
- Broader historical context: project-root `HANDOFF_LOGIKA_20261023.md` and the user-pasted handoff.

This is a new decision brief, not a replacement or modification of historical experiment records. Only retained local evidence was read and metrics recomputed. No training, model replacement, remote process action, or submission was performed.

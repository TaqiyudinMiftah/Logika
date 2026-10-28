# Final champion-anchored candidate

The delivered CSV contains **95% of the preserved 0.56188 public champion and 5% of one elastic addition**. Its exact historical source is `pc2_job_000312`. The prior 0.55353 replacement is not the anchor. Decision branch: `fixed_secondary_candidate`. Evidence status: **Exploratory candidate; interval includes losses**. This candidate is not assigned a new public score or promoted to proven champion.

The addition uses source blocks `application, profile, previous, repayment, bureau, account, shopping`, view `v3`, encoding `onehot`, and reduction `none`. All fit-dependent preprocessing and feature selection are retained with the model.

## What was reviewed

- 22,016 completed aligned prediction records passed independent score/row checks. These are records, not a claim of that many statistically independent experiments.
- Earlier fixed and nested ensemble scoreboards and the historical Logika inventory remain referenced with source hashes.
- The older depth-search ledger includes 329 attempts, of which 315 complete aligned screening predictions were included in the audit.
- Descriptive comparisons of every aligned recipe used the same 5%/10%/20% additions. Only three compatible family representatives were frozen for the actual nested benchmark; the broader descriptive screen is not presented as validation.

## Primary nested procedure: failed consistency gate

All 70,000 labels participated in a three-fold outer cross-fit. For each outer training partition, separate inner fits selected among only the frozen additions and 5%/10% weights. The outer fold never supplied labels to that fold's selection. The adaptive procedure improved one outer fold and selected the champion fallback in the other two. Its mean-fold KS change was +0.000481546, but it failed the predeclared two-fold consistency gate. **That failure is retained and not relabelled a success.**

## Fixed-recipe secondary benchmark

The same three configurations and two weights had already been frozen before benchmark fitting. Their six fixed combinations were compared after the primary result, using their complete fold-local outer predictions. No new model configuration or weight was introduced. The selected fixed combination improved all three folds; it is a different candidate from the variable adaptive selector. Selection among the six uses the highest Bonferroni-adjusted lower bootstrap bound among stable positive-fold contrasts. This is explicitly a **post-primary secondary analysis**, conditional on historical shortlist discovery.

| Outer fold | Champion KS | Selected fixed candidate KS | Change |
|---|---:|---:|---:|
| 1 | 0.512798006 | 0.513248533 | +0.000450527 |
| 2 | 0.524955447 | 0.525060252 | +0.000104805 |
| 3 | 0.523682468 | 0.524278402 | +0.000595934 |
| Mean | 0.520478641 | 0.520862396 | +0.000383755 |

The fixed candidate's matched mean-fold improvement is **+0.000383755 KS**, with conditional six-comparison familywise 95% interval **[-0.000850009, +0.001469538]**. Pooled OOF KS is separately **0.519521001 versus 0.519727615**; it mixes outputs of separately fitted ensembles. These figures describe the selected fixed recipe, not the failed primary selector.

**Historical labels already influenced the feature definitions and descriptive shortlist. This is not untouched private validation.** The 4,000 paired bootstrap replicates stratify by outer fold and target; Bonferroni adjustment covers the six fixed comparisons conditionally. It does not erase historical adaptive-selection bias or full-refit variation. A positive point estimate does not guarantee a higher Kaggle score; if the interval includes losses, the candidate remains exploratory. No separate confirmation-set tuning or public leaderboard probing was used.

## Final fit and verification

The fixed secondary comparison selected `pc2_job_000312` at weight `0.05`. That one addition was fitted on all 70,000 labels without changing its feature definition, parameters or weight after fitting. The champion keeps its exact previous five fitted models and preprocessing. Reloading those models reproduced all 30,000 original predictions to the original CSV's rounding precision.

The final new model and all champion models were reloaded to replay every test row. The CSV matches the official sample order, has 30,000 unique IDs and finite probabilities in range. Verification independently reconstructed every fit/query split and label hash, and confirmed exclusion of outer-validation rows from all corresponding inner fits.

- New trained fit jobs retained: **97** (nested validation fits plus the one final addition).
- Retained model/transformer files, including preserved champion artifacts: **115**.
- Submission SHA-256: `1ba5b8677549f5d05fdda6d63fa5f978e731247275a74ff91f3bdd39ac068036`.
- Root promotion is recorded separately in `delivery.json`; no competition upload is performed.

## Reproduction

On PC2, from `/mnt/home/logika-faris-exploration/submission-robust-20261018`, using the preserved exploration environment:

```sh
../exploration-12h-20261017/.venv/bin/python benchmark.py
../exploration-12h-20261017/.venv/bin/python assess_benchmark.py
../exploration-12h-20261017/.venv/bin/python analyze_fixed.py
../exploration-12h-20261017/.venv/bin/python build_submission.py --fixed-secondary
../exploration-12h-20261017/.venv/bin/python build_submission.py --replay-only
../exploration-12h-20261017/.venv/bin/python verify_run.py
```

The original two-hour build deadline is preserved. Completed fits are hash-checked before reuse; the existing deadline is never reset. Inference-only replay remains available after the deadline.

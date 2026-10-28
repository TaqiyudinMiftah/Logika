# Champion-anchored submission research, 18 October 2026

The user requests one further competition prediction file after the previous research-derived replacement scored 0.55353, below the preserved 0.56188 public champion. Public scores are user-reported. This finite build does not upload files or restart an expired campaign.

## Evidence review

`review_all.py` audits every saved completed single-pipeline prediction from the two earlier meta-experiment waves and the latest exploration campaign, including recovered foundation-model runs. It records the full historical fixed/nested-ensemble inventories. `review_supplementary.py` reconciles the older depth-search runs and the earlier Logika evidence with their original validation protocols. Every verified aligned prediction receives the same predeclared descriptive 5%, 10%, and 20% champion-addition contrasts. These exhaustive contrasts are discovery evidence, **not independent validation**.

`diagnose_refit.py` compares the saved development-trained and full-data-fitted versions of the failed candidate on identical unlabeled test rows. It does not score those rows or assert which fit is superior.

## Small nested benchmark

The shortlist contains at most three distinct model families, selected by the worst-fold 5%/10% addition contrast among compatible completed recipes. Elastic-net and ordinary logistic regression count as one family for this shortlist. Recipe definitions and code hashes are frozen in `benchmark_plan.json` before additional training. Unavailable or unsupported nested model implementations receive explicit exclusions rather than fabricated results. A preliminary preparation manifest that used only 10% discovery contrasts was superseded before any benchmark fitting; it is retained as `benchmark_plan.pretraining-v1.json`.

The benchmark uses all 70,000 labeled applications, with three stratified outer folds (seed 20261019). Inside each outer training partition, three inner folds generate training-only predictions for the five champion recipes and every shortlisted addition. Feature selection, preprocessing, encoders, and reductions are fitted within each individual fit. Outer-validation applicants are excluded from all corresponding inner fits and from addition/weight selection.

The fixed selection rule considers only 5% and 10% probability additions. It requires improvement in at least two inner folds, no inner-fold loss worse than 0.002, pooled change no worse than -0.003, and positive mean-fold change minus half its standard deviation. Otherwise it selects the champion. The selected inner recipe is refitted on the outer training data and evaluated on the excluded outer fold. No outer-fold result changes this rule. The final all-label choice applies this same rule to a full-population cross-fit.

The primary metric is mean matched within-outer-fold KS: each fold measures ranking from a single fitted model ensemble. Pooled OOF KS mixes outputs of different fitted ensembles and is reported separately; its difference can reflect inter-fit probability-scale changes. This distinction was declared before new benchmark training, after the exhaustive descriptive review found some additions improved every within-fold KS while reducing pooled KS.

The predeclared aggregate promotion gate requires positive mean matched outer-fold KS change, two of three improved outer folds, no outer-fold loss worse than 0.003, pooled change no worse than -0.003, and a nonzero final addition. A failed gate is retained as a negative result, not silently redefined as a win. Bootstrap uncertainty remains conditional and must accompany point estimates.

**Historical labels have already informed feature definitions and the shortlist. This cannot produce an honestly untouched test set, and the benchmark must not be presented as an unbiased private-leaderboard estimate.** Actual nested refitting prevents the new addition/weight selector from accessing its outer-validation labels; it does not erase historical selection bias.

## Final prediction boundary

The anchor is the exact previously delivered all-label champion, including its existing fitted preprocessing and seed realization. Before any new CSV is accepted, its models must replay the preserved champion CSV. An eligible addition is fitted on all labels, saved with its preprocessing, and replayed independently. CSV output must match the official sample IDs, contain 30,000 finite probabilities in range, and match saved model aggregation. The previous champion and the failed replacement remain backed up.

All training artifacts and failed/partial attempts are retained. Only this build's processes are managed, with explicit resource limits and an absolute two-hour safety deadline preserved across restart. No unrelated process is stopped.

## Execution

On PC2, the project resides in `/mnt/home/logika-faris-exploration/submission-robust-20261018`, using the existing locked exploration environment. Run the scripts in this order:

```sh
../exploration-12h-20261017/.venv/bin/python review_all.py
../exploration-12h-20261017/.venv/bin/python review_supplementary.py
../exploration-12h-20261017/.venv/bin/python prepare_benchmark.py
../exploration-12h-20261017/.venv/bin/python benchmark.py
```

The files under `evidence/`, `review_summary.json`, `benchmark_result.json`, and the eventual final-build verification state determine what actually completed. Catalogue registration, successful CSV formatting, and higher reused-validation scores are not substitutes for predictive evidence.

## Recorded primary failure and explicit secondary branch

The primary adaptive selector failed its consistency gate: one improved outer fold and two unchanged champion fallbacks. `benchmark_result.json` remains unchanged with `promotion_gate_passed=false`. Its all-label choice is not automatically treated as validated.

The six fixed combinations of the already frozen three candidates and two weights are a separate, post-primary analysis. `analyze_fixed.py` computes paired bootstrap intervals stratified by outer fold/target and applies Bonferroni adjustment across those six comparisons. It selects the highest adjusted lower bound among candidates with positive changes in every fold. This controls only these six comparisons conditionally, not the historical discovery search. An interval including losses is explicitly labelled exploratory.

The user-requested CSV can be built from this fixed secondary candidate with the explicit `--fixed-secondary` branch. This does not rewrite the failed primary gate, claim an unbiased holdout result, promote a new proven champion, or upload anything. `evidence/fixed_secondary_protocol.json`, `fixed_comparison.json`, and the final model index retain the decision path.

```sh
../exploration-12h-20261017/.venv/bin/python analyze_fixed.py
../exploration-12h-20261017/.venv/bin/python finish.py --fixed-secondary
```

## Completed candidate delivery

The requested candidate CSV is complete and installed at the repository root. It is 95% of the preserved public-0.56188 champion plus 5% of `pc2_job_000312`, a regularized logistic recipe with the bureau feature view. The selected final model has 128 inputs, including one added score/report-age interaction; the remaining selected inputs come from the existing feature foundation.

All 97 fit jobs and 115 model/transformer files were retained and independently verified after download. The final 30,000-row CSV passed model replay, sample ordering and reaggregation checks. The primary nested gate remains failed; the secondary fixed candidate has positive mean changes in all three folds but an adjusted interval including losses and slightly lower pooled KS. This is a requested exploratory candidate, not a verified new champion. See `REPORT.md`, `fixed_comparison.json`, `verification.json` and `delivery.json`. No upload was performed.

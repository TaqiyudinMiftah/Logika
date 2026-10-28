# Final research-derived submission, 18 October 2026

**Public-result update:** the user reports **0.55353**, below the prior champion's **0.56188** by **0.00835**. The candidate's higher reused-validation result did not transfer. The prior pipeline remains the best observed public result. See `public_result.json` and [the validation-selection audit](generalization-audit/REVIEW.md). Successful build/replay checks below establish artifact integrity, not improved generalization.

The user authorized a final refit and competition prediction file, followed by a small improvement pass. Eleven structural recipes were declared before the new validation comparisons. This is a finite submission build, not a restart of an expired campaign or a new broad experiment search.

Selected recipe: **equal probability mean of EBM, the nested ridge stack, account elastic-net logistic regression, and bureau TabM** (`without_pc2_job_000853_mean`). The redundant logistic-stack output was removed. A source-availability router and numerically unstable actuarial scorecard changes were not incorporated.

| Evaluation | Selected recipe KS | Matched champion KS |
|---|---:|---:|
| Original development OOF | 0.526342709 | 0.525771439 |
| Different training seed | 0.523146266 | 0.521594385 |
| Second CV partition | 0.520970844 | 0.519098639 |
| Reused 14,000-row local validation | 0.532316133 | 0.530694365 |

The previously measured five-stream probability mean scored 0.532120700 on that validation split. The selected four-stream version has the highest observed reused-validation KS among the eleven declared candidates that met the development gates. Ten candidates passed those gates. The gate requires positive aggregate improvement on all three development evaluations, improvement in at least two original folds, and no original-fold loss worse than 0.005.

**The local validation partition has historical reuse and now explicitly participates in recipe selection. Its score is not an unbiased private-test estimate or a Kaggle private leaderboard score.** No competition upload or leaderboard probing is included.

## Pipeline

| Output stream | Configuration | Final behavior |
|---|---|---|
| EBM | `aaf1faad20c743c8877c`, `representation_17` | Frequency encoding, rank transform, true-gain selection of 128 columns, appended grouped PCA, additive boosting with interactions |
| Ridge stack | `nested_11` | Ridge(alpha=100) fitted on three-fold cross-fitted predictions of three shared base recipes |
| Account elastic net | `2f6c6c38fc4c0efb1c4c`, `account_00` | C=0.005, L1 ratio 0.75, class weight 3, account deficits/volatility/change/recovery view |
| TabM | `b8ed3935d1e518adb6fd`, `bureau_13` | Width 256, four blocks, dropout 0.1, LR 1e-4, 16 members, 128 selected features, inner-only epoch selection |

The ridge stack's three base recipes are:

- `fd1fd49493df80697827`: preserved champion income-gap logistic recipe.
- `d2f4719a69d44e93a849`: preserved champion LightGBM/SVD recipe.
- `eb3419a4bf925b93ca53`: repayment elastic-net recipe, C=0.05, L1 ratio 0, class weight 3.

There are six unique base predictors and one ridge head at inference. Training retains nine additional inner-fold base fits, TabM's epoch-selection fit, and produced checkpoints. Nested base models are trained once and reused rather than separately refitted for duplicate heads. The meta head sees only cross-fitted predictions for its training applications. No test labels exist or are used.

## Build and verification

The build uses the frozen campaign factories and identical tested environment on PC2. `research_source/` preserves those factories. Source hashes, the base bank, additional feature arrays, train/test files, and official sample are verified before fitting. Point-in-time feature semantics are inherited from these hash-verified feature banks.

Before final training, the new inference code reloads retained development-trained models and reproduces all 14,000 validation predictions. The chosen recipe is then refitted on all 70,000 labeled applications, including the previously reserved validation rows, and predicts the 30,000 competition applications.

Before completion, every final model and transformer is independently reloaded, all 30,000 predictions are replayed, and the CSV is read back and checked for exact official ID order, uniqueness, finite probabilities, range, and agreement with model replay. `artifacts/complete.json` exists only after those checks pass.

The immutable original submission is backed up locally in `evidence/previous_submission.csv`. Its SHA-256 is `be059078ade95bc6758522e8d72849e3f793e57380e2c083e05c77c64335e192`, associated with user-reported public score 0.56188. No original campaign models or results are removed.

## Reproduction on PC2

Directory: `/mnt/home/logika-faris-exploration/submission-final-20261018`.

Environment: `/mnt/home/logika-faris-exploration/exploration-12h-20261017/.venv/bin/python`.

```sh
# Freeze selection and verify source/input hashes; already completed for this build.
../exploration-12h-20261017/.venv/bin/python prepare.py

# Fit the frozen recipe, retain all weights, write and verify the CSV.
../exploration-12h-20261017/.venv/bin/python run.py

# Inference-only verification after completion; does not retrain.
../exploration-12h-20261017/.venv/bin/python run.py --replay-only
```

The final model index uses paths relative to this project. For relocated inference, supply the identical feature bank, extras, and sample paths with `--bank`, `--extras`, and `--sample` together with `--replay-only`. Full refits use the input paths pinned in `plan.json`.

The finite build has its own original two-hour safety deadline, four single-thread CPU workers and one GPU worker using two CPU threads. The dedicated user service has an eight-core CPU quota, 24 GiB aggregate RAM ceiling, zero swap, and a two-hour maximum runtime. TabM uses one RTX 3060 with a 3 GiB allocator limit; unrelated workloads are preserved. Resuming completed fit tasks verifies their requests and hashes. Partial attempts are retained separately.

## Artifacts

- `candidate_plan.json`: declared eleven-recipe selection space.
- `evidence/recipe_comparison.json`: all eligible development and reused-validation results.
- `evidence/inference_replay.json`: retained-model validation replay.
- `fits/`: full and inner-fold fitted models, preprocessing, logs, requests, predictions and hashes.
- `artifacts/shared_stack/`: training-only inner OOF predictions and fitted ridge head.
- `artifacts/pipeline_index.json`: retained inference graph.
- `artifacts/submission.csv`: final competition prediction file.
- `artifacts/verification.json`: independent all-row replay and CSV validation.
- `artifacts/complete.json`: terminal success, model hashes, score provenance and submission hash.

## Completed delivery

The full 70,000-row refit completed successfully. All 30,000 test rows were replayed after reloading the saved models, with exact component agreement and CSV rounding error at most 1.12e-16. All 31 retained model/checkpoint files passed SHA-256 verification after download. An independent local verifier also checked sample order, component reaggregation and tie-aware development/validation KS.

The verified artifact was atomically installed as the repository-root `submission.csv`; the previous file remains backed up. New submission SHA-256: `e809cc0100b39ce2bd6ddc15ac7f28fc30f840bf483c3e8777af2e37e7f478c7`. See `local_verification.json` and `delivery.json`. No competition upload was performed.

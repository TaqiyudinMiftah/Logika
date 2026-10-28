# LOGIKA UI 2026 experiment log

Primary objective: robust pooled OOF KS > 0.55. Scores use a 0–1 scale. This log is cumulative. Historical source snapshots and failures remain preserved.

## Original temporal reference

- ID: B000, `reports/baseline_validation.json`.
- Hypothesis: point-in-time aggregation gives a reproducible credit-risk baseline.
- Features: 81 application/profile/previous/bureau/account/order features, repayment excluded.
- Model: LightGBM, learning rate 0.03, 31 leaves, min child 40, L2 2, row fraction 0.8, column fraction 0.85, seed 42, selected 532 iterations.
- Validation: latest 20 percent by decision timestamp, 14,000 rows. KS 0.4242133954, AUC 0.7785503093. No fold mean/SD applies to this single chronological holdout.
- Finding: establishes a working chronological reference. Not comparable directly to shuffled five-fold scores.
- Next: establish identical full OOF folds.

## E001 — matched-fold LightGBM reference

- Run: `e001-lgbm-all-20261013T082101Z-0b5b92`.
- Hypothesis: reproduce a reference under the team's target × decision-month five-fold assignment.
- Changes: nested inner selection and outer refit, same 81 features.
- Exact resolved configuration: run `config.json`. Feature names/hashes, source snapshot, splits and models are in the run directory.
- Validation: 5 outer folds, seed 2026, 80/20 inner selection using seed 2026 + fold, AUC early stopping. All 70,000 rows have OOF predictions.
- Mean fold KS: 0.4234041564, sample SD: 0.0106568397. Pooled OOF KS: 0.4208966794, AUC: 0.7815638991.
- Exact per-fold scores: run `fold_metrics.json`. These are the authoritative full-precision records, retained alongside this log.
- Comparison: first complete OOF reference.
- Finding: pipeline verified, objective not achieved.
- Next: model-family and credit-history-feature comparisons.

## GPU integration check

- ID: `setup-catboost-smoke`, plus synthetic 20-iteration CUDA smoke.
- Hypothesis: CatBoost binary wheel and GPU pipeline work on this shared A100.
- Configuration: 20 iterations, two folds on the real 81-feature table, plus a 128-row synthetic CUDA fit.
- Outcome: execution and output-format checks passed. This is not a competitive CV result and is excluded from the modeling leaderboard.

## E005 pilot — Kumo execution and resource check

- Run: `e005-kumo-large-pilot-20261013T085811Z-4e37b6`.
- Hypothesis: pinned Kumo Large can run the intended context on this account.
- Configuration: frozen Large, checkpoint v1.0.1, SDK commit 329a707bbcf9ca147aa45a63fef8b2029f4ee37b, BF16, 16,384 target/month-stratified outer-training context rows, two sequential estimators, 81 features, batch 256, seed 2027.
- Validation: runtime-only subset, 512 outer-validation and 128 test queries. No reported competition KS, no full submission.
- Outcome: passed, peak combined RSS 3.27 GiB, peak reserved CUDA 2.59 GiB. All source/fold/checkpoint hashes and probability orientation verified.
- Next: complete five-fold execution.

## E005 — Kumo Large on 81 features

- Run: `e005-kumo-large-20261013T090703Z-5eda51`.
- Hypothesis: pretrained Kumo improves the original 81-feature view.
- Configuration: same frozen pilot recipe, five folds, 16,384 context rows per outer fold, seed 2026 + fold. No gradient updates.
- Fold KS: 0.3899947561, 0.3788321571, 0.4242870049, 0.3893521095, 0.3776445221.
- Mean KS: 0.3920221099, sample SD: 0.0189271789. Pooled OOF KS: 0.3888580213, AUC: 0.7634956050.
- Comparison with E001: mean KS difference -0.0313820465. Models consume different numbers of training labels, 16,384 versus 56,000.
- Finding: failed improvement hypothesis for this recipe. Keep as measured reference, not the best model. Full 70,000 OOF and 30,000 test predictions and five-fold average independently verified.
- Resources: peak combined RSS 3.47 GiB, cumulative warm fold runtime 259.1 seconds.
- Next: feature audit and expanded credit view before expensive model tuning.

## E002 — original-view CatBoost GPU

- Hypothesis: categorical boosting improves the 81-feature view with the same outer folds.
- Configuration: `configs/experiments/e002_catboost_all.json`, depth 6, learning rate 0.04, L2 5, max 2,000 iterations, 100-round inner early stopping, seed 2026, four CPU threads.
- Validation: same nested five-fold design as E001.
- Status: completed on A100 as e002-catboost-all-20261013T100323Z-5e1835, log `artifacts/logs/e002-research.log`. Confirm live PID and allocated run directory before resuming or restarting.
- Pooled KS 0.4196180668, mean KS 0.4218920414, sample SD 0.0077347703, AUC 0.7812649112. It did not improve E001 on the original view. Full artifacts remain on A100.

## Research journal — 2026-10-13

- Inspected current code, all dataset schemas/counts, notebooks, configurations, previous results, and both team source snapshots.
- Live Kaggle rules/evaluation access failed. Retained the authenticated prior rule snapshot and enforced its decision-time/external-data restrictions. Official evaluation code has not been supplied.
- Key hypothesis: missing repayment history and mixing provider-specific bureau signals explains much of the feature gap. Team E10/E11 results are references only and have not been counted as our scores.
- Planned next IDs: E010 audited credit features + LightGBM, E011 audited credit features + CatBoost, E012 repayment ablation, E013 bureau denoising, E014 compact-view Kumo. Final IDs/configurations must be recorded before their execution.

## Leaderboard policy

The current best verified pooled OOF KS is E001 at 0.4208966794. Only full, verified OOF runs qualify. Record all unsuccessful runs, resource failures and screening results with their scope. A single-fold score or a tuned inner score cannot replace the full OOF metric. Maintain machine-readable records under `configs/research/` and use these to update the leaderboard.

## Preparation attempt P001 — expanded cache memory

The first credit_v2 build passed its causal-source checks but failed when materializing the wide final join under a 768 MiB DuckDB limit. No feature cache was promoted and no model score was recorded. Error evidence is in artifacts/credit_v2_build.log. Retry P002 uses an explicit 1536 MiB DuckDB buffer budget. This changes execution capacity, not features or row eligibility.

## E010 — audited historical credit view

Hypothesis: repayment snapshots, matured delinquency rates, provider-specific bureau history and cash-flow variability explain the baseline feature gap. Configuration: configs/experiments/e010_lgbm_credit.json, 633 features, LightGBM 31 leaves, minimum child 70, L2 5, learning rate .03, 3000 maximum iterations with inner AUC stopping. Five matched nested folds completed. Pooled OOF KS 0.5072062987, AUC 0.8336081422, sample fold SD 0.0098427533. Improvement over original pooled OOF KS: +0.0863096193. Exact fold scores/configuration/source/cache hashes are retained in the run. Finding: substantial improvement, target not yet achieved. Next: bureau smoothing and a repayment ablation, then compact Kumo.

## E013 — provider smoothing

Hypothesis: provider-specific robust/recency-weighted summaries reduce repeated bureau noise. Adds 168 bureau summaries to credit_v2, yielding 801 features. The model and validation recipe match E010. Dispatched locally. Metrics pending full completion.

## E012 — repayment ablation

Hypothesis: historical delinquency drives most of E010 improvement. Configuration removes only credit_repay_ and credit_loan_ predictors and their prefixed ratios, retaining other credit features and E010 hyperparameters. Pending execution.

## Verified leaderboard generated from artifacts

### e041-fixed-catboost-contract-blend-20261014T173313Z-11ec39

- Full OOF KS: 0.5113838817, mean KS: 0.5142305736, sample SD: 0.0116212025.
- Fold KS: 0.5122968284, 0.5028303016, 0.5281085993, 0.5036203817, 0.5242967570.
- Configuration: `{"base_directories": ["/home/giptarch/Projects/logika-ui-26/artifacts/a100_runs/e011-catboost-credit-20261013T110907Z-1e2930", "/home/giptarch/Projects/logika-ui-26/artifacts/experiments/e030-lgbm-contract-credit-20261013T120201Z-9a615c"], "feature_view": "base_models", "model": "fixed_blend", "n_splits": 5, "name": "e041-fixed-catboost-contract-blend", "weights": [0.5, 0.5]}`.
- Artifacts: `artifacts/experiments/e041-fixed-catboost-contract-blend-20261014T173313Z-11ec39`. OOF/source/submission verified.

### e011-catboost-credit-20261013T110907Z-1e2930

- Full OOF KS: 0.5107802079, mean KS: 0.5135619641, sample SD: 0.0127761944.
- Fold KS: 0.5117464418, 0.5061692727, 0.5244026222, 0.4973164797, 0.5281750040.
- Configuration: `{"device": "gpu", "early_stopping_rounds": 150, "exclude_features": [], "exclude_prefixes": [], "feature_view": "credit_v2", "gpu_devices": "0", "gpu_min_free_gib": 8, "gpu_ram_part": 0.08, "inner_fraction": 0.2, "max_iterations": 3000, "memory_limit": "1536MB", "model": "catboost", "n_splits": 5, "name": "e011-catboost-credit", "parameters": {"depth": 6, "l2_leaf_reg": 5.0, "learning_rate": 0.04}, "seed": 2026, "threads": 4}`.
- Artifacts: `artifacts/a100_runs/e011-catboost-credit-20261013T110907Z-1e2930`. OOF/source/submission verified.

### e040-fixed-credit-contract-blend-20261013T123208Z-225895

- Full OOF KS: 0.5097719433, mean KS: 0.5116537340, sample SD: 0.0109755511.
- Fold KS: 0.5104956266, 0.4967934408, 0.5246627655, 0.5066040950, 0.5197127420.
- Configuration: `{"base_directories": ["/home/giptarch/Projects/logika-ui-26/artifacts/experiments/e010-lgbm-credit-20261013T101651Z-b145c8", "/home/giptarch/Projects/logika-ui-26/artifacts/experiments/e030-lgbm-contract-credit-20261013T120201Z-9a615c"], "feature_view": "base_models", "model": "fixed_blend", "n_splits": 5, "name": "e040-fixed-credit-contract-blend", "weights": [0.5, 0.5]}`.
- Artifacts: `artifacts/experiments/e040-fixed-credit-contract-blend-20261013T123208Z-225895`. OOF/source/submission verified.

### e030-lgbm-contract-credit-20261013T120201Z-9a615c

- Full OOF KS: 0.5089375072, mean KS: 0.5111472354, sample SD: 0.0129982849.
- Fold KS: 0.5099187185, 0.4936620766, 0.5265028903, 0.5048529363, 0.5207995551.
- Configuration: `{"device": "cpu", "early_stopping_rounds": 150, "exclude_features": [], "exclude_prefixes": [], "feature_view": "credit_v4", "gpu_devices": "0", "gpu_min_free_gib": 8, "gpu_ram_part": 0.08, "inner_fraction": 0.2, "max_iterations": 3000, "memory_limit": "1536MB", "model": "lightgbm", "n_splits": 5, "name": "e030-lgbm-contract-credit", "parameters": {"colsample_bytree": 0.8, "learning_rate": 0.03, "min_child_samples": 70, "num_leaves": 31, "reg_lambda": 5.0, "subsample": 0.85, "subsample_freq": 1}, "seed": 2026, "threads": 4}`.
- Artifacts: `artifacts/experiments/e030-lgbm-contract-credit-20261013T120201Z-9a615c`. OOF/source/submission verified.

### e031-lgbm-repayment-specialists-20261014T170401Z-e4d66b

- Full OOF KS: 0.5073556588, mean KS: 0.5095920342, sample SD: 0.0088497286.
- Fold KS: 0.5112276202, 0.4964501394, 0.5191895872, 0.5058585797, 0.5152342446.
- Configuration: `{"device": "cpu", "early_stopping_rounds": 150, "exclude_features": [], "exclude_prefixes": [], "feature_view": "credit_v4", "gpu_devices": "0", "gpu_min_free_gib": 8, "gpu_ram_part": 0.08, "inner_fraction": 0.2, "max_iterations": 3000, "memory_limit": "1536MB", "model": "segmented_lightgbm", "n_splits": 5, "name": "e031-lgbm-repayment-specialists", "parameters": {"colsample_bytree": 0.8, "learning_rate": 0.03, "min_child_samples": 70, "num_leaves": 31, "reg_lambda": 5.0, "subsample": 0.85, "subsample_freq": 1}, "seed": 2026, "threads": 2}`.
- Artifacts: `artifacts/experiments/e031-lgbm-repayment-specialists-20261014T170401Z-e4d66b`. OOF/source/submission verified.

### e010-lgbm-credit-20261013T101651Z-b145c8

- Full OOF KS: 0.5072062987, mean KS: 0.5090048421, sample SD: 0.0098427533.
- Fold KS: 0.5050157610, 0.4976467812, 0.5209715643, 0.5038509015, 0.5175392027.
- Configuration: `{"device": "cpu", "early_stopping_rounds": 150, "exclude_features": [], "exclude_prefixes": [], "feature_view": "credit_v2", "gpu_devices": "0", "gpu_min_free_gib": 8, "gpu_ram_part": 0.08, "inner_fraction": 0.2, "max_iterations": 3000, "memory_limit": "1536MB", "model": "lightgbm", "n_splits": 5, "name": "e010-lgbm-credit", "parameters": {"colsample_bytree": 0.8, "learning_rate": 0.03, "min_child_samples": 70, "num_leaves": 31, "reg_lambda": 5.0, "subsample": 0.85, "subsample_freq": 1}, "seed": 2026, "threads": 4}`.
- Artifacts: `artifacts/experiments/e010-lgbm-credit-20261013T101651Z-b145c8`. OOF/source/submission verified.

### e013-lgbm-smooth-20261013T105912Z-b2e3a6

- Full OOF KS: 0.5066121206, mean KS: 0.5095922295, sample SD: 0.0095383562.
- Fold KS: 0.5108743190, 0.4990139868, 0.5215271393, 0.5010460127, 0.5154996898.
- Configuration: `{"device": "cpu", "early_stopping_rounds": 150, "exclude_features": [], "exclude_prefixes": [], "feature_view": "smooth_v3", "gpu_devices": "0", "gpu_min_free_gib": 8, "gpu_ram_part": 0.08, "inner_fraction": 0.2, "max_iterations": 3000, "memory_limit": "1536MB", "model": "lightgbm", "n_splits": 5, "name": "e013-lgbm-smooth", "parameters": {"colsample_bytree": 0.8, "learning_rate": 0.03, "min_child_samples": 70, "num_leaves": 31, "reg_lambda": 5.0, "subsample": 0.85, "subsample_freq": 1}, "seed": 2026, "threads": 4}`.
- Artifacts: `artifacts/experiments/e013-lgbm-smooth-20261013T105912Z-b2e3a6`. OOF/source/submission verified.

### e020-lgbm-credit-leaves63-20261013T111434Z-b7d4ec

- Full OOF KS: 0.5065600085, mean KS: 0.5087601205, sample SD: 0.0122486093.
- Fold KS: 0.5040228130, 0.4948802834, 0.5229945630, 0.5016563552, 0.5202465877.
- Configuration: `{"device": "cpu", "early_stopping_rounds": 150, "exclude_features": [], "exclude_prefixes": [], "feature_view": "credit_v2", "gpu_devices": "0", "gpu_min_free_gib": 8, "gpu_ram_part": 0.08, "inner_fraction": 0.2, "max_iterations": 3000, "memory_limit": "1536MB", "model": "lightgbm", "n_splits": 5, "name": "e020-lgbm-credit-leaves63", "parameters": {"colsample_bytree": 0.8, "learning_rate": 0.025, "min_child_samples": 100, "num_leaves": 63, "reg_lambda": 15, "subsample": 0.85, "subsample_freq": 1}, "seed": 2026, "threads": 4}`.
- Artifacts: `artifacts/experiments/e020-lgbm-credit-leaves63-20261013T111434Z-b7d4ec`. OOF/source/submission verified.

### e021-lgbm-credit-positive-weight2-20261013T111435Z-25690a

- Full OOF KS: 0.5042757673, mean KS: 0.5068113089, sample SD: 0.0107264932.
- Fold KS: 0.5063138367, 0.4930224299, 0.5207913849, 0.5008668838, 0.5130620090.
- Configuration: `{"device": "cpu", "early_stopping_rounds": 150, "exclude_features": [], "exclude_prefixes": [], "feature_view": "credit_v2", "gpu_devices": "0", "gpu_min_free_gib": 8, "gpu_ram_part": 0.08, "inner_fraction": 0.2, "max_iterations": 3000, "memory_limit": "1536MB", "model": "lightgbm", "n_splits": 5, "name": "e021-lgbm-credit-positive-weight2", "parameters": {"colsample_bytree": 0.8, "learning_rate": 0.03, "min_child_samples": 70, "num_leaves": 31, "reg_lambda": 5.0, "scale_pos_weight": 2.0, "subsample": 0.85, "subsample_freq": 1}, "seed": 2026, "threads": 4}`.
- Artifacts: `artifacts/experiments/e021-lgbm-credit-positive-weight2-20261013T111435Z-25690a`. OOF/source/submission verified.

### e026-spline-newton-20261013T114336Z-1efbd7

- Full OOF KS: 0.5031035923, mean KS: 0.5056110256, sample SD: 0.0076381231.
- Fold KS: 0.5022500892, 0.4987823366, 0.5123033954, 0.4994687220, 0.5152505850.
- Configuration: `{"device": "cpu", "early_stopping_rounds": 150, "exclude_features": [], "exclude_prefixes": [], "feature_view": "credit_v2", "gpu_devices": "0", "gpu_min_free_gib": 8, "gpu_ram_part": 0.08, "inner_fraction": 0.2, "max_iterations": 3000, "memory_limit": "1536MB", "model": "spline_logistic", "n_splits": 5, "name": "e026-spline-newton", "parameters": {"C": 0.1, "feature_limit": 120, "knots": 5, "max_iter": 100, "solver": "newton-cholesky"}, "seed": 2026, "threads": 4}`.
- Artifacts: `artifacts/experiments/e026-spline-newton-20261013T114336Z-1efbd7`. OOF/source/submission verified.

### e024-spline-logistic-k5-20261013T113339Z-8e066a

- Full OOF KS: 0.5022856436, mean KS: 0.5041333837, sample SD: 0.0067903458.
- Fold KS: 0.5042304199, 0.4951362812, 0.5092991012, 0.5000920644, 0.5119090519.
- Configuration: `{"device": "cpu", "early_stopping_rounds": 150, "exclude_features": [], "exclude_prefixes": [], "feature_view": "credit_v2", "gpu_devices": "0", "gpu_min_free_gib": 8, "gpu_ram_part": 0.08, "inner_fraction": 0.2, "max_iterations": 3000, "memory_limit": "1536MB", "model": "spline_logistic", "n_splits": 5, "name": "e024-spline-logistic-k5", "parameters": {"C": 0.1, "feature_limit": 120, "knots": 5}, "seed": 2026, "threads": 4}`.
- Artifacts: `artifacts/experiments/e024-spline-logistic-k5-20261013T113339Z-8e066a`. OOF/source/submission verified.

### e025-spline-logistic-k0-20261013T113339Z-9b7e7c

- Full OOF KS: 0.4989568714, mean KS: 0.5021409982, sample SD: 0.0099846247.
- Fold KS: 0.5006034513, 0.4973360883, 0.5194037512, 0.4938799008, 0.4994817994.
- Configuration: `{"device": "cpu", "early_stopping_rounds": 150, "exclude_features": [], "exclude_prefixes": [], "feature_view": "credit_v2", "gpu_devices": "0", "gpu_min_free_gib": 8, "gpu_ram_part": 0.08, "inner_fraction": 0.2, "max_iterations": 3000, "memory_limit": "1536MB", "model": "spline_logistic", "n_splits": 5, "name": "e025-spline-logistic-k0", "parameters": {"C": 0.1, "feature_limit": 120, "knots": 0}, "seed": 2026, "threads": 4}`.
- Artifacts: `artifacts/experiments/e025-spline-logistic-k0-20261013T113339Z-9b7e7c`. OOF/source/submission verified.

### e012-lgbm-credit-no-repayment-20261013T105911Z-6765ef

- Full OOF KS: 0.4700478433, mean KS: 0.4714491488, sample SD: 0.0124067273.
- Fold KS: 0.4763943129, 0.4591751576, 0.4894949222, 0.4607275789, 0.4714537725.
- Configuration: `{"device": "cpu", "early_stopping_rounds": 150, "exclude_features": [], "exclude_prefixes": ["credit_repay_", "credit_loan_"], "feature_view": "credit_v2", "gpu_devices": "0", "gpu_min_free_gib": 8, "gpu_ram_part": 0.08, "inner_fraction": 0.2, "max_iterations": 3000, "memory_limit": "1536MB", "model": "lightgbm", "n_splits": 5, "name": "e012-lgbm-credit-no-repayment", "parameters": {"colsample_bytree": 0.8, "learning_rate": 0.03, "min_child_samples": 70, "num_leaves": 31, "reg_lambda": 5.0, "subsample": 0.85, "subsample_freq": 1}, "seed": 2026, "threads": 4}`.
- Artifacts: `artifacts/experiments/e012-lgbm-credit-no-repayment-20261013T105911Z-6765ef`. OOF/source/submission verified.

### e001-lgbm-all-20261013T082101Z-0b5b92

- Full OOF KS: 0.4208966794, mean KS: 0.4234041564, sample SD: 0.0106568397.
- Fold KS: 0.4286767693, 0.4075777344, 0.4304742083, 0.4174019103, 0.4328901597.
- Configuration: `{"device": "cpu", "early_stopping_rounds": 100, "exclude_features": [], "exclude_prefixes": [], "gpu_devices": "0", "gpu_min_free_gib": 8, "gpu_ram_part": 0.08, "inner_fraction": 0.2, "max_iterations": 2000, "memory_limit": "768MB", "model": "lightgbm", "n_splits": 5, "name": "e001-lgbm-all", "parameters": {"colsample_bytree": 0.85, "learning_rate": 0.03, "min_child_samples": 40, "num_leaves": 31, "reg_lambda": 2.0, "subsample": 0.8, "subsample_freq": 1}, "seed": 2026, "threads": 4}`.
- Artifacts: `artifacts/a100_runs/e001-lgbm-all-20261013T082101Z-0b5b92`. OOF/source/submission verified.

### e005-kumo-large-20261013T090703Z-5eda51

- Full OOF KS: 0.3888580213, mean KS: 0.3920221099, sample SD: 0.0189271789.
- Fold KS: 0.3899947561, 0.3788321571, 0.4242870049, 0.3893521095, 0.3776445221.
- Configuration: `{"context_rows": 16384, "estimators": 2, "exclude_features": [], "exclude_prefixes": [], "gpu_budget_gib": 8, "gpu_index": 0, "memory_limit": "768MB", "n_splits": 5, "name": "e005-kumo-large", "process_budget_gib": 4, "query_batch_size": 256, "seed": 2026, "size": "large", "threads": 4}`.
- Artifacts: `artifacts/a100_runs/e005-kumo-large-20261013T090703Z-5eda51`. OOF/source/submission verified.


<!-- END GENERATED LEADERBOARD -->

## E011 completed result

Run e011-catboost-credit-20261013T110907Z-1e2930 completed in 4911.7 seconds on the shared A100. Pooled KS 0.5107802079, mean KS 0.5135619641, SD 0.0127761944, AUC .8351251487. Improvement over E010 pooled KS is +0.0035739092. Full per-fold metrics/model artifacts are saved remotely. Target is not achieved. E015 run e015-kumo-smooth_v3-120-20261013T123915Z-5c135d and E017 run e017-tabiclv2-credit120-20261013T123939Z-ad6b9a also failed at the same preprocessing memory gate. No scores are assigned to them. The revised execution design is being piloted with the original E014 model/context parameters.

## 2026-10-14 research continuation

The previous goal turn was progress: new credit features and complete CV results improved OOF performance, with known resource failures retained. The interrupted observation did not establish model termination. Current A100 pilot status was checked through the Contents API after a transient WebSocket failure. Pilot e014-kumo-credit_v2-120-pilot-20261013T124414Z-c51949 completed CPU feature selection and produced all 512 validation/128 test predictions, but the supervisor reported RSS above 4 GiB during GPU execution/shutdown. It remains a failed resource check and is not assigned a CV score. Context/model/120-feature recipe must stay intact while execution memory accounting is investigated.

The log updater previously replaced the generated leaderboard through end-of-file, which could discard appended research sections. It now replaces only a bounded generated section, and a regression test checks that later hypotheses survive repeated updates. Lost research entries are reconstructed from configs, run artifacts and RESEARCH_STATE rather than fabricated results.

E020 capacity/regularization: pooled KS 0.5065600085. E021 positive weight 2: 0.5042757673. E024 cubic spline with LBFGS: 0.5022856436 with convergence warnings. E025 quantile logistic: 0.4989568714. E026 Newton spline: 0.5031035923. None surpassed the strongest current tree/ensemble result. E030 contractual maturity/amortization proxies: pooled KS 0.5089375072. E040 fixed .5/.5 E010+E030 blend: pooled KS 0.5097719433, mean 0.5116537340, SD 0.0109755511. E040 inference reproduced its submission exactly with SHA256 0521603bceea5ddde489cc7eb03f00f0992ecd2f5a1fba163fc1ba1ed4361a8c. E011 CatBoost remote result is higher at 0.5107802079 but local full-artifact import is pending.

## E031 repayment coverage specialists, registered 2026-10-15

Hypothesis: the absent-history segment has lower KS than the observed-history segment, so separate decision-time specialists may improve ranking. Route exclusively using credit_repay_loans > 0, with missing mapped to absent. Two LightGBM models per outer fold use the E030 683-feature recipe. Each segment selects its own iteration count only on its corresponding inner split, then refits only that segment of the outer training fold. Five original outer folds remain identical. No weight tuning or target-based routing. Models and iteration maps are saved for inference. CPU execution uses two threads and releases duplicate feature frames before training. No result is claimed until full OOF is complete.

## 2026-10-15 verified results and restored A100 access

The prior interrupted turn made progress through E031 implementation, training and foundation resource instrumentation. DNS initially failed for the A100 host, then recovered. Authenticated command execution succeeded. E011 full archive SHA256 aa5b22299b9142e60ef342cdd8c0da9e8473b82d0475e47f4ec91a7bfdffdb79 matched the server, all five fold model/output manifests passed, and source/70,000 OOF/30,000 test coverage passed the leaderboard verifier. Local CPU inference reproduced E011 exactly with SHA256 992993360de523e2a26e2e0c6f58a1c7f0cb5e9523e8eafc2df2dc404b045d00.

E031 separated repayment-history groups at an observable feature threshold. Pooled KS 0.5073556588, mean 0.5095920342, SD 0.0088497286. It did not improve E030 or E011. Its 30,000-row submission was reproduced byte-for-byte with SHA256 cf69f526b6d4ebc98c8d9c4cd25f7755ad4700128416873de8878625e909ff1c. Separate models reduce data available to each learner, and the observed routing difference did not justify promoting specialists.

E041 fixed 50/50 E011 plus E030: pooled KS 0.5113838817, mean 0.5142305736, SD 0.0116212025, AUC 0.8368497271. Gain versus E011 is +0.0006036738. No blend weights were selected using outer labels. This small exploratory improvement is not target attainment or an independent holdout result. Inference reproduced submission SHA256 bc1405912ea8b83dfccebb16d36b05fc4743b6df4f7f718da96be4d83f3e5c1d exactly.

Kumo pilot e014-kumo-credit_v2-120-pilot-20261014T173252Z-816070 completed selection and all 512 validation/128 test queries with unchanged Large/16,384-context/120-feature/two-estimator settings. Selector peak combined RSS 1,371,443,200 bytes. GPU-stage peak 3,517,243,392 bytes, below the unchanged 4 GiB gate. The earlier pilot remains failed. The controller now projects only IDs/labels and obtains candidate names from verified metadata, avoiding the full 633-column load. Per-second RSS/PSS/USS and failure summaries are preserved. Unit tests verify failure accounting, owned-child cleanup and context/validation isolation. Full E014, positive-quota E028 and TabICLv2 E017 are the next sequential queue, with GPU and reclaimable host-headroom guards. A successful pilot carries no full CV score.

The full foundation queue was launched as PID 2997059 with recorded creation time 1791999382.23. Full E014 run e014-kumo-credit_v2-120-20261014T173721Z-477a18 reached fold three under a verified live child PID 2997382. Fold-one KS 0.510120 and fold-two KS 0.511601 are partial results, not pooled CV claims. The leaderboard verifier now checks foundation completion/request hashes, all retained fold artifacts, context isolation, selected-feature fitting/inner IDs and fixed checkpoint assets across folds. Failed/pilot states remain unqualified and retain their configurations.

## E033 direct inner KS stopping, registered 2026-10-15

Hypothesis: AUC-based iteration selection can miss the best KS ranking point. Keep E030 features/parameters and all outer/inner folds, but stop and choose iterations using tie-aware KS on inner rows only. The custom metric is the only early-stopping metric. Outer labels never select iterations. No result is claimed before complete five-fold OOF.

## Account-capacity interruption and next iteration

Full E014 e014-kumo-credit_v2-120-20261014T173721Z-477a18 failed in fold three with worker SIGKILL (-9), after two completed folds at KS 0.510120 and 0.511601. Failed-stage peak own RSS 3,158,388,736 bytes was below its 4 GiB gate. The account reported oom_kill=78 but no event baseline existed for this occurrence, so account OOM is plausible rather than proven. No pooled CV result or submission is claimed. The waiting queue was stopped after verifying its own PID/create time, waiting state and no live children. No unrelated process was stopped. A replacement queue retains the same models with per-worker host-capacity checks, runtime reserve monitoring and before/after cgroup event counters. Failed resource outcomes stay registered, and corrected source starts a new run. E033 proceeds locally while host capacity is unavailable.

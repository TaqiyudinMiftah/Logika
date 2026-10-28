# Persistent research state

Updated: 2026-10-15 Asia/Jakarta.

## Objective and completion gates

Achieve robust pooled OOF KS strictly greater than 0.55. The user objective file resolves its later TARGET_KS placeholder to 0.55. The goal remains active. Historical outer-fold comparisons are exploratory model-selection evidence, not an untouched holdout.

Completion requires a frozen reproducible end-to-end training and inference pipeline, complete five-fold OOF, fold mean and sample SD, a fresh separately seeded CV repeat without subsequent tuning, chronological sensitivity, source/input/model provenance, leakage audits, sample-compatible predictions and retained successful/failed research records. Do not submit to Kaggle, use paid resources, install elevated system packages or use hidden labels.

## Current best verified candidate

E041 fixed 50/50 CatBoost E011 plus LightGBM E030 blend. Pooled OOF KS 0.5113838817086389, mean fold KS 0.5142305735927089, sample SD 0.011621202480878234, AUC 0.8368497271040635. Below the target. This is an exploratory small improvement of +0.0006036738 over E011.

Run: e041-fixed-catboost-contract-blend-20261014T173313Z-11ec39. Current recipe, fold metrics and artifact hashes are in configs/research/current_best.json. Test inference reproduced submission byte-for-byte, SHA256 bc1405912ea8b83dfccebb16d36b05fc4743b6df4f7f718da96be4d83f3e5c1d.

E011 is the strongest single model: pooled KS 0.5107802079060595, mean 0.513561964102532, SD 0.012776194395179017. Complete artifacts are imported locally in artifacts/a100_runs/e011-catboost-credit-20261013T110907Z-1e2930. Source snapshots, all five model/output manifests, OOF/sample coverage and metrics passed verification. Local inference reproduced SHA256 992993360de523e2a26e2e0c6f58a1c7f0cb5e9523e8eafc2df2dc404b045d00. CatBoost 1.2.10 was installed locally from the unchanged uv lock using the gpu extra, retaining existing packages. GPU training is nondeterministic and requires a fresh validation repeat before any robust win claim.

## Data, validation and competition constraints

70,000 labeled applications, 30,000 test applications, 9,507 positives. One selected application per borrower, train/test borrowers disjoint. IDs never enter model features. Event data must be available before selected decision_datetime. Repayment snapshots passed contractual/timestamp/calendar audits. Account features use completed prior months and e-commerce features use deduplicated completed orders. Provided competition data only.

Five target-by-decision-month stratified outer folds use sorted application IDs and seed 2026. Inner 80/20 splits with seed 2026 + fold select tree iterations and foundation features. All feature selection and fitted transformations use fitting/inner rows only. Kumo context uses only outer training rows. The tie-aware ROC empirical CDF KS is used, plus directional KS, AUC and log loss. Exact official evaluator code remains unavailable. Use the prior authenticated competition-rule snapshot and data/raw/DATA_DICTIONARY.md, recheck live rules when access allows.

## Completed hypotheses

- E001 original 81-feature LightGBM: pooled 0.4208966794.
- E005 frozen Kumo Large with 81 features: 0.3888580213. Label budget differs from full-data trees.
- E010 audited 633-feature credit view: 0.5072062987.
- E012 repayment ablation: 0.4700478433. Repayment is the largest observed improvement source.
- E013 801-feature provider smoothing: 0.5066121206, correlation with E010 0.9866174. No pooled gain.
- E020 stronger capacity/regularization: 0.5065600085. E021 positive weight 2: 0.5042757673.
- E024 spline logistic: 0.5022856436, convergence warnings retained. E025 quantile logistic: 0.4989568714. E026 Newton spline: 0.5031035923.
- E030 683-feature credit view with 50 matured-contract/amortization-proxy additions: 0.5089375072. Scheduled principal is a proxy rather than actual debt.
- E031 repayment-history specialists: 0.5073556588, mean 0.5095920342, SD 0.0088497286. Observable groups did not justify separate learners. All five models are saved and inference reproduced exact submission SHA256 cf69f526b6d4ebc98c8d9c4cd25f7755ad4700128416873de8878625e909ff1c.
- E040 equal E010/E030 blend: 0.5097719433. E041 equal E011/E030 blend is the current best.

See EXPERIMENT_LOG.md and configs/research/leaderboard.json for full configurations, per-fold outcomes and failed-run records.

## Foundation failures and repaired execution

Earlier E014 credit, E015 smoothing and E017 TabICLv2 failed the 4 GiB own-process RSS gate. The two-process selection pilot e014-kumo-credit_v2-120-pilot-20261013T124414Z-c51949 also failed near inference/shutdown. Keep every failed run and assign no full CV score.

A further fix projects only application_id and target into the controller and takes candidate names from cache metadata, retaining raw/cache hash validation. Its dataframe is 3,400,132 bytes. A separate CPU process selects 120 features from the 633 candidates using only development/inner rows and exits. The GPU process reads only selected features. The model/context/estimator count stays Large/16,384/two.

Pilot e014-kumo-credit_v2-120-pilot-20261014T173252Z-816070 completed all 512 validation and 128 test queries. CPU selection peak combined RSS 1,371,443,200 bytes. GPU-stage peak RSS 3,517,243,392 bytes, under the unchanged 4 GiB limit. PSS/USS are diagnostic only. Per-second process measurements and failure summaries are saved. Tests verify that a failed budget preserves diagnostics and reaps only its owned child. A pilot does not supply a CV score.

## Current execution and prioritized work

A100 access recovered after a DNS failure. Real checkout: /home/fariselhakim/gipta/projects/logika-ui-26. Kernel: Python (Gipta - LOGIKA UI 2026). Account cgroup RAM is 16 GiB. The last idle GPU observation showed 79,577 MiB free. The account has many resident processes, so use actual cgroup anon/kernel plus reclaimable-cache headroom rather than node RAM. The queue waits for GPU and host capacity and runs sequentially.

The first full repaired E014 run e014-kumo-credit_v2-120-20261014T173721Z-477a18 completed fold one at KS 0.510120 and fold two at 0.511601, then failed in fold three with worker return code -9. It has no pooled OOF score and remains unqualified. Measured peak own RSS during the failed stage was 3,158,388,736 bytes, below the 4 GiB limit. Account cgroup events included OOM kills, but there was no before/after counter baseline, so the specific cause is unproven. Host headroom later fell to about 3 GiB.

The original queue foundation-projection-20261015 was stopped only after verifying its PID creation identity, waiting_capacity state and absence of live child processes. Its history and failure artifacts remain intact. The supervisor now waits for account RAM before each worker, records event-count baselines and after values, and preserves a failure plus terminates only its owned worker if account reclaimable headroom falls below 0.25 GiB. Context/model/estimators and RSS budget are unchanged.

Replacement queue: artifacts/research_queue/foundation-capacity-20261015. Launcher: artifacts/foundation-capacity-20261015-launcher.json. Queue PID 3001883, creation time 1792000178.29. Its launcher was created successfully. Read its exact PID/create time and verify the live process before treating it as running. Its planned order is a new full E014 run, E028 positive-context sampling with prior correction, then E017 pinned TabICLv2. Do not resume the old failed run against changed source or restart capacity waits on observation timeouts.

Local E033 is running as e033-lgbm-inner-ks-stopping-20261014T174804Z-65a925, PID 1082635. Its sole model change from E030 is iteration selection using inner KS instead of AUC. Features, tree parameters and outer/inner splits remain the same. Do not claim a score until full OOF verifies. Inspect artifacts/e033_training.log and its status.json plus the exact live process. Keep its source unchanged until terminal status.

Next priorities:

1. Verify full E014 OOF and retained fold artifacts, then compare with E011 and fixed blends.
2. Evaluate E028 sampling and E017 alternative foundation model without silently reducing context/model or assigning a score to failed runs.
3. Let observed complementarity guide fixed blend or properly nested learned stacking experiments. Avoid fitting weights on outer validation labels.
4. Investigate foundation adaptation and an equal-label-budget tree control if frozen models plateau. Other model research is in docs/TABULAR_FOUNDATION_MODELS.md.
5. Promote a greater-than-0.55 candidate only after frozen separately seeded CV and chronological sensitivity checks.

## Resume instructions

Read this file, EXPERIMENT_LOG.md and the objective attachment first. Inspect configs/research/current_best.json and exact queue/run status plus live PID identity. Local helper is /home/giptarch/Downloads/Faris/.venv/bin/python with jupyter_remote.py or a100-term, using its existing credentials without printing/copying them. Remote training uses the project .venv directly, preserving installed Kumo extras. Keep source unchanged during live runs. Future changes require new run/source identities.

Commands: scripts/run_experiment.py, scripts/run_kumo.py, scripts/blend_runs.py, scripts/update_research.py and scripts/predict_run.py. Raw data, source snapshots and completed/failed artifact folders must be preserved. The log updater uses a bounded generated section and must retain appended manual research entries. Use Range downloads for large A100 artifacts, validate remote/local SHA256 and extract with tar safety checks.

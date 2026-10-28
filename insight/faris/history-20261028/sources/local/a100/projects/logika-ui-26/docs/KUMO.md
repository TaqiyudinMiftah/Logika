# Kumo experiments

## Primary recipe

E005 is frozen Kumo Tabular Large with 16,384 labeled context rows, two estimators, sequential estimator execution, BF16 inference and query batches of at most 256. The SDK is pinned to commit `329a707bbcf9ca147aa45a63fef8b2029f4ee37b`, pretrained assets to `nvidia/Kumo-Tabular` revision `v1.0.1`, and PyTorch to 2.7.1 with CUDA 12.6. Asset SHA256 values are saved in each fold. The pinned SDK's default classification preprocessing recipe is used.

The model uses in-context learning. `fit` prepares context representations and caches with zero gradient updates. The source of the labeled context is recorded application by application. It is sampled only from the outer training partition, stratified by target and decision month with seed `2026 + fold`. Outer validation and test rows never enter that context. The five outer folds are identical to the existing E001 design. Kumo has no boosting iteration selection or inner early stopping.

All 81 existing point-in-time features are used. The context consumes 16,384 of the 56,000 available outer training rows. This sampling is an explicit experimental choice. The runner fails rather than silently shrinking context or switching model. Only query batches may shrink on CUDA OOM, and each reduction is logged. Class-1 probabilities are selected by output name `1`, independently of column order.

## A100 commands

```bash
cd ~/gipta/projects/logika-ui-26
source scripts/a100_env.sh
uv sync --locked --extra gpu --extra kumo
.venv/bin/python scripts/run_kumo.py --pilot
.venv/bin/python scripts/run_kumo.py --config configs/experiments/e005_kumo_large.json
```

The runtime pilot uses the configured 16,384 context rows and both estimators, but only 512 validation queries and 128 test queries from fold one. Its folder includes `pilot` and `summary.json` contains `pilot: true`. It produces no competition submission and no CV score. Pilot predictions are execution evidence only.

For Jupyter, select **Python (Gipta - LOGIKA UI 2026)** and open `notebooks/03_kumo.ipynb`. The first execution cell defaults to the runtime pilot. Set `pilot=False` for the full five folds.

Each fold is a child process. Its exit releases pinned host caches and CUDA allocations. The controller monitors combined RSS of itself and child processes against the 4 GiB budget. PyTorch's allocator is capped at 8 GiB on GPU 0, with a free-memory check before loading. CUDA driver allocations can add overhead beyond the PyTorch allocator counter. These limits protect the shared account and are resource settings, not estimated model requirements. The account has a 16 GiB cgroup RAM allowance despite the node's much larger physical RAM.

The original remote path `~/Projects/logika-ui-26` is a compatibility symlink to the Gipta directory. The actual code, data and environment are in Gipta. The symlink preserves absolute interpreter references from the already installed environment.

## Full run outputs and restart

New runs are created under `artifacts/experiments/`. They save exact context/validation/test IDs, configuration, source snapshot, feature and raw-source hashes, checkpoint hashes, package versions, class-1 predictions, resource measurements and state. A completed full run includes 70,000 OOF rows and a 30,000-row submission in sample order. This creates a file without submitting it to Kaggle.

```bash
mkdir -p artifacts/logs
nohup .venv/bin/python scripts/run_kumo.py > artifacts/logs/e005.log 2>&1 < /dev/null &
tail -f artifacts/logs/e005.log
```

The log identifies the run folder. Inspect that folder's `status.json` and each fold's `worker.log` for progress. Resume only the same model/source/cache/configuration:

```bash
.venv/bin/python scripts/run_kumo.py --resume /absolute/path/to/artifacts/experiments/run-id
```

Completed folds are checksum-validated and reused. An unfinished fold restarts from its context. GPU reductions can vary numerically between executions, so seeds and hashes capture the procedure without promising bitwise identical predictions.

## Verified runtime pilot

The A100 pilot `e005-kumo-large-pilot-20261013T085811Z-4e37b6` completed on 2026-10-13. It used the exact Large / 16,384 context / 81 features / two-estimator recipe. All 512 validation and 128 test predictions were finite class-1 probabilities. Context fit took 280.8 seconds, validation queries 4.1 seconds and test queries 0.5 seconds. Total worker supervision took 345.8 seconds including loading and asset checks. Peak combined controller/worker RSS was 3.27 GiB. Peak PyTorch GPU allocation was 2.08 GiB and peak reservation was 2.59 GiB. Query batch size remained 256 with no OOM retries. Shared-node timings can vary.

The checkpoint SHA256 was `cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613`. Complete pilot artifacts are stored remotely under `artifacts/experiments/` and copied locally under `artifacts/a100_runs/`. This runtime check does not provide a full-CV performance estimate.

The full E005 run completed all five folds on the A100 as `e005-kumo-large-20261013T090703Z-5eda51`. Mean fold KS was 0.392022, sample standard deviation 0.018927, pooled OOF KS 0.388858 and pooled AUC 0.763496. The run contains 70,000 OOF predictions and a 30,000-row local submission. Its log is `artifacts/logs/e005-full-20261013.log`. Peak combined process RSS across folds was 3.47 GiB and cumulative fold execution time was 259.1 seconds with assets already cached.

The full run is copied to the laptop under `artifacts/a100_runs/`. Archive, source and fold-file checksums match. Local verification recomputed all metrics from CSV, checked labels against the raw training file, verified every application ID, and reproduced the submission as the five-fold average. Feature-source raw checksums also match. The verification record is `verified_local.json` in the local run folder.

This first Kumo recipe is below E001's mean fold KS 0.423404 and pooled OOF KS 0.420897. It is a measured starting reference. E005 uses 16,384 labeled context rows per fold, while E001 consumes all 56,000 outer-training rows, so the difference does not isolate model architecture. Next comparisons should control context size and feature view before adding adaptation or blending.

## Controlled comparisons

1. E005 Large versus E006 Small uses exactly the same contexts, features, folds and estimator count.
2. Change context size under a new experiment name only after capacity is measured. Compare identical sampled rows when context sizes are equal.
3. Run feature-family ablations using the same Kumo configuration, changing only `exclude_prefixes` or `exclude_features`.
4. Compare other frozen foundation models on the same contexts and folds before introducing fine-tuning. Inner validation is required for selecting adaptation steps, early stopping or learned preprocessing.
5. Evaluate ensemble gain from complete aligned OOF predictions. Report paired per-fold differences, directional KS, AUC, log loss, runtime and memory. Choosing blend weights from outer OOF results makes that result exploratory.

The E001 LightGBM reference used all outer-training rows. A comparison against E005 therefore changes both model family and number of consumed labels. A strict architecture comparison needs an additional tree arm using the exact Kumo context IDs.

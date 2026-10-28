# Kumo pipeline with nested correction training

This is one implemented pipeline change: train the original correction model on 56,000 honest out-of-fold predictions instead of a single 6,000-row calibration set. Its feature definitions, Kumo context size and ensemble size, correction penalty and shrinkage, and spline contribution remain fixed.

The preceding ranking-objective intervention failed: mean KS change −0.000584871 and mean AUC change −0.000014421 against the locked teammate pipeline. It is excluded here; all its artifacts remain in `../ranking-pipeline-20261020/`.

For each of the five frozen outer folds:

1. Recompute the original 50,000-row Kumo context predictions on the 6,000 calibration and 14,000 evaluation rows. Refit the original correction and retain the original spline model. This is the matched control for smaller GPU inference batches.
2. Split the 56,000 outer-training rows into ten stratified inner folds. For each inner fold, select exactly 50,000 of the available training rows and predict the 5,600 excluded rows. No context contains its query or the outer evaluation rows.
3. Train the unchanged correction on these 56,000 inner OOF scores. Evaluate once on the untouched outer fold, using exactly the matched control's Kumo and spline predictions.

There are 55 frozen-Kumo context fits and ten fitted correction models. Kumo's pretrained weights are unchanged; the checkpoint, complete context table and per-fit context indices retain each executable model. Folds and inner fits are not separate candidate recipes.

Two PC2 GPUs run one outer fold each at a time, with estimator batch size 1 and query batches of 512. An absolute 24-hour execution ceiling includes training, final comparison and packaging; it ends early on completion. This cap follows the user's permission for a full-day focused training run. Node380 was unavailable for admission because another workload occupied its VRAM; no unrelated process is altered.

The selected pipeline is packaged automatically with its preprocessing and weights only after complete five-fold assessment. A candidate must improve KS and AUC, satisfy fold stability and log-loss constraints, and pass adjusted paired uncertainty. If it fails, the incumbent is preserved and explicitly identified. No new submission is generated automatically.

All labels have been used historically. The inherited DPD/paid snapshots and global-reference assumptions remain uncertified for point-in-time availability. This is conditional development evidence; neither repeated CV nor AUC can guarantee a private-leaderboard increase.

## Operation

Host root: `/mnt/home/logika-faris-exploration/nested-calibration-20261020`

```sh
ssh faris@labkc-comm1 'tail -n 30 -F /mnt/home/logika-faris-exploration/nested-calibration-20261020/events.jsonl /mnt/home/logika-faris-exploration/nested-calibration-20261020/controller.log'
```

`"DONE": true, "status": "complete"` in the controller log indicates final reporting and packaging finished. A failed status is not a completed model. Read `REPORT.md`, `result.json`, and `model/manifest.json`.

Prediction accepts a parquet application feature table with the schema in `model/manifest.json`, unique `application_id`, and no target column:

```sh
CUDA_VISIBLE_DEVICES=0 /mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python \
  source/pipeline.py predict --root . --features applications.parquet --output predictions.parquet
```

It reuses the local hash-verified Kumo checkpoint, reconstructs the five retained contexts, applies each fitted correction and spline, and averages their probabilities in input order. New raw-table feature construction is outside this inference interface; the preserved inherited features are recorded in the baseline lock. GPU allocation must be available before manual inference.

To stop only this campaign, create `STOP` in its host root. The controller terminates its own process groups. Earlier campaigns and the root submission are not modified.

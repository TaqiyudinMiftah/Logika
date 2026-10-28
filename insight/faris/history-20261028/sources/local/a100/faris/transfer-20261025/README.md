# Logika research and submissions — 25 October 2026

**Continuation outcome on 25 October:** five additional fixed hypotheses (H012–H016) completed; none passed its frozen promotion rule. The highest observed local point is **0.531505932**, below **0.54**, with an unsupported gain of 0.000102127 over A. No submission was made after reset; the authenticated API still reports **3 available slots**. See [continuation report](CONTINUATION_REPORT.md), [status](CONTINUATION_STATUS.json), and [research state](research-state.json).

The preceding cycle completed11 hypotheses and2 submissions. It did not reach **0.54 local pooled OOF KS** or beat the best known public score. Its archive mirror is now verified:757 hashed files, including245 serialized artifacts, match PC2; see [LOCAL_DELIVERY_AUDIT.json](LOCAL_DELIVERY_AUDIT.json).

| Recipe | Local research OOF KS | Verified public KS | Kaggle reference |
|---|---:|---:|---:|
| Historical V25 | 0.530284660 | **0.56822** | 57444414 |
| Prior root periodicity+maturity refit | 0.530966531 | 0.56678 | 57482053 |
| B: maturity-only full refit | 0.530636104 | **0.56818** | 57604838 |
| A: periodicity+maturity, two Kumo seeds | 0.531403805 | 0.56656 | 57605161 |

B improves the prior root file by 0.00140 but trails V25 by 0.00004. A's higher local KS did not improve its public result. These are observed comparisons, not guarantees about private performance.

The highest point observed is now **0.531505932** from H016. Its paired adjusted interval versus A is **[-0.000896073, +0.001131765]**, its fresh-seed comparison loses, and the mean wins only two of five folds. It was not promoted. The selected local incumbent remains A's **0.531403805**; the public incumbent remains V25. H009's earlier 0.531409806 was also not promoted.

## Files to use

The exact downloaded V25 file is copied to the project root as [submission_best_public_V25_056822_verified.csv](/home/parasite/Project/Competition/logika-2026/submission_best_public_V25_056822_verified.csv). Its SHA-256 is `e7f0748c6c8ded776f0a6a03bdebcd0787cf6c0057d0b078c31196fb34f58efe`. Its schema, 30,000 unique test IDs, finite probability range and byte identity to the scored download are verified. The old root `submission.csv` remains unchanged.

[SUBMISSIONS.json](SUBMISSIONS.json), [KAGGLE_DOWNLOAD_IDENTITY_AUDIT.json](KAGGLE_DOWNLOAD_IDENTITY_AUDIT.json), and `kaggle_receipts/` retain the actual uploads, terminal scores, quota and downloaded-file hashes. The previously unidentified `submission.csv` at ref57482053 is resolved by an exact hash match to the old root file, SHA `922788d13cd8ec53b7cbd732496a9200150a499fd4363455414e2cc6230a6b06`.

## Completed fixed hypotheses

| ID | Test | Result and decision |
|---|---|---|
| H001 | Fresh Kumo seed; fixed correction resamples | Two-seed periodicity KS0.531403805 became A. Correction-resample averages lost KS. |
| H002 | Anchored smooth-KS correction | Two-seed KS0.529869496; small conditional gains versus its weaker matched baseline, not promoted. |
| H003 | Shallow tree correction | Two-seed KS0.527591255 despite improved AUC; reject. |
| H004 | Equal A/B average | KS0.529991678; reject. |
| H005 | Equal V25/B average | KS0.530313076, below B and only2/5 fold wins; reject. |
| H006 | Replace two Kumo inputs with maturity-aligned rates | Two-seed KS0.529927448, below matched control0.530068521; reject. |
| H007 | Replace only the 30-day Kumo input | KS0.530280480; fails original-seed gain and incumbent threshold; reject. |
| H008 | Smooth-KS correction with periodicity | Each seed improves, but their mean KS0.530936994 decreases; reject. |
| H009 | Average raw seeds before a shared correction | Logloss KS0.531409806 is an unsupported tiny point gain; direct-KS0.530903933; reject. |
| H010 | Five contexts inside every outer fold, with shared training-only priors | Ordinary KS0.528190778; direct-KS0.528059537; both fail. AUC improves but KS decreases. |
| H011 | Training-fitted percentile maps before averaging | KS0.528724411; below B, only1/5 fold win and intervals cross zero; reject. |
| H012 | Correct final mixture using context-only spline predictions for the correction rows | Two-seedKS0.529905572; reject. |
| H013 | Equal-class-loss variant of H012 with analytic prior adjustment | Two-seedKS0.528845312; reject. |
| H014 | Shared-application bureau posterior, replacing two existing downstream fields | OriginalKS0.530497857; static parity and measurement checks pass, default improvement rejected. |
| H015 | Balanced Kumo contexts vs same-size proportional contexts | Original KS0.527676778; fresh0.529070860; both below full-context controls; reject. |
| H016 | Direct smooth-KS objective in the 56k spline branch | Two-seed KS0.531505932; tiny point gain fails replication, fold consistency and uncertainty; reject. |

All hypotheses were frozen before their own complete evaluation. No partial-fold performance pruning, weight grid, hidden-label reconstruction or leaderboard perturbation was used. Public results that informed subsequent choices are recorded as adaptive evidence in [exposures.jsonl](exposures.jsonl).

## Validation scope and what H010 established

H001–H009 use the original seed31027 outer folds and 50kcontext/6kcorrection/14kvalidation roles. H010/H011 keep the same outer folds and one held-out prediction for every one of the70,000 applicants, but evaluate a distinct recipe: shared statistics fit on all56k outer-training covariates, a spline on56k training labels, and five separate50kcontext/6kcorrection models. Every validation applicant is excluded from every fit that predicts it. This scope differs from the earlier50kcontext-only priors.

The matched H010 ordinary one-context control scores **0.530290544**; its five-context average scores **0.528190778**. Their AUCs are0.847056771 and0.847350287. The result rejects the tested ensemble for KS; it does not establish an improvement from a training-graph match or prove a particular cause for earlier public transfer failures.

Every saved H010 head and spline was replayed. A separate fresh-context reconstruction from the retained feature table, label source, checkpoint and seed reproduced **all20,000 raw predictions and the14,000 validation pipeline predictions bit for bit**. See [ROLE_BAG_OOF_AUDIT.json](ROLE_BAG_OOF_AUDIT.json) and [ROLE_BAG_CONTEXT_RECONSTRUCTION_AUDIT.json](ROLE_BAG_CONTEXT_RECONSTRUCTION_AUDIT.json).

The last-slot rules were frozen separately. An ordinary five-context CV gain alone would not justify uploading, because B already averages five full-data models. H010's matched objective and H011's matched percentile comparisons both failed. No D, F or G full-data candidate was selected or launched. Their conditional source files are preparation artifacts, not runtime-verified delivered models.

## Retention and reproduction

Training ran on PC2: `ssh faris@100.68.29.126`, archive `/mnt/home/logika-faris-exploration/transfer-20261025`, pinned Python `/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python`.

There are **335 serialized artifact files**, including canonical/indexed Kumo contexts, correction heads, splines, tree models and CDF maps. This is a file count, not335 independent experiments. [CONTINUATION_RETENTION_MANIFEST.json](CONTINUATION_RETENTION_MANIFEST.json) inventories the current PC2 archive; [CONTINUATION_LOCAL_DELIVERY_AUDIT.json](CONTINUATION_LOCAL_DELIVERY_AUDIT.json) verifies its local mirror. The preceding [MODEL_RETENTION_MANIFEST.json](MODEL_RETENTION_MANIFEST.json) and its 245-file model count are preserved as an earlier snapshot. Completed and partial attempts are preserved. The local mirror includes the new fitted artifacts; original source data, feature bank, frozen checkpoint and some referenced historical contexts remain at their durable PC2 paths. This is not a standalone raw-data distribution.

A and B use64kcontext/6kcorrection per full-data member, with all70k labels used by their splines. Their five completed runs and replay audits are under `final/` and `candidate_B/final/`. Old OOF scores are research-recipe measurements, not measurements on the full-data refits.

Report entrypoints read saved predictions. Fit entrypoints reject existing output directories. Do not overwrite retained evidence to rerun a fit.

```sh
python source/robustness.py report
python source/role_bag_report.py
```

For exact fresh prediction replay of an indexed H010 member, preserve the original joint correction-plus-validation query order:

```sh
python source/predict_indexed_kumo.py --model role_bag/outer0/member0/kumo_model.joblib --rows role_bag/outer0/member0/raw_predictions.npz --row-key rows --head role_bag/outer0/member0/ordinary_head.joblib --spline role_bag/outer0/shared/spline.joblib --output replay_new_outer0_member0.npz
```

The first6000 output rows are correction rows; only the remaining14000 rows are validation. Optional feature-table, target-file and checkpoint overrides require exact matching hashes. The actual campaign replay is retained as `role_bag_replay_member0.npz` and its completion record.

The official Kaggle CLI wrapper reads a private credential outside the repository and redacts it from receipts. Existing submission intent records prevent accidental duplicate uploads. No credential is part of this package.

## Limits and why the preceding cycle stopped

All70k labels have extensive historical exposure. Fresh seeds, bootstraps and new role assignments do not create a fresh holdout. Reported intervals are conditional on fixed development predictions and exclude prior selection and refitting uncertainty. Raw repayment snapshot timing remains unresolved; the official v2 dictionary does not resolve it. These experiments add no new future fields.

The bounded tests are complete and no eligible third candidate remains. Broader search is not admitted by the current validation gate. The [research-analyzer skill](/home/parasite/.codex/skills/research-analyzer/SKILL.md) states: “Failed, stale, or inconclusive evidence blocks search.” This cycle therefore ends with the target explicitly unmet, rather than claiming a validated improvement. The original training deadline was06:10WIB and the submission deadline06:45WIB; the budget was **not exhausted** or restarted.

The preceding cycle's final readback showed2 submissions before reset,32total and1unused slot. The new day has3available slots. [FINAL_STATUS.json](FINAL_STATUS.json) preserves that preceding terminal state; [research-state.json](research-state.json) tracks the active continuation.

# Completed horizontal feature experiment — 26 October 2026

**Outcome: completed without improvement. The 0.54 local KS target was not reached. No Kaggle submission was made.**

The coordinator completed at **05:08 WIB on 26 October 2026**. A live audit at about **09:03 WIB** confirmed all expected predictions, row order, hashes, finite probabilities, repeated predictions and reported KS/AUC. All eight screening/confirmation worker services and the coordinator are inactive after completion.

| Matched fivefold procedure | Pooled KS | Difference from baseline |
|---|---:|---:|
| Existing maturity baseline | 0.5306361042 | — |
| Inner-selected Small feature-pack ensemble | 0.5261641847 | -0.0044719195 |
| Inner-selected Medium/Large ensemble | 0.5281372386 | -0.0024988656 |

The paired, class-stratified bootstrap 95% interval for the final difference is **[-0.00426331, -0.00043269]**. This is conditional uncertainty on adaptively reused development data, not independent confirmation or a forecast of leaderboard movement.

## Verified coverage

- 21,514 candidate feature definitions: 2,408 existing columns and 19,106 generated formulas.
- 290 feature packs, each using at most 100 columns.
- 1,450 completed Small/two-estimator screening fits: all 290 packs across all five folds.
- 30 Medium/eight-estimator and 30 Large/eight-estimator fits selected using correction data within each fold.
- 1,510 expected fits completed; zero failure records.
- 17,211 distinct feature IDs appeared in at least one fold after filtering. Per-fold totals ranged from 17,207 to 17,211. Feature-ID counts do not measure independent information; exact duplicate filtering operated within each pack.
- Recipes, seeds, hashes, predictions and metrics retained. No fitted weights or KV caches saved.

## What the result supports

The generated actuarial packs did not reach the shortlist under the frozen Small-model ranking rule. All larger-model shortlist entries came from the existing-feature families and the baseline control. Final selection kept the original raw baseline in folds 0 and 4, mixed a larger baseline-only model in fold 2, and used broader existing-feature ensembles in folds 1 and 3. The final changed folds all had negative outer KS differences.

This rejects the frozen horizontal pack-and-blend procedure as an improvement over its matched baseline. It does not isolate the marginal value of each generated feature. Standard semantic packs preserved 20 baseline anchors and replaced much of the existing 90-feature input with a new family. Their results therefore combine the effects of adding a new family and removing existing predictors. Screening used Small with two estimators; only shortlisted existing-feature packs received Medium/Large runs.

A useful next hypothesis would retain the full 90-feature baseline and add at most ten source-backed features at a time, then require stability across repeated inner selection. That is a proposed follow-up, not a completed test or a promised gain. No follow-up campaign was launched during this completion audit.

## Limits and references

The original outer labels have prior experiment exposure. Timestamp and delinquency-age consistency checks passed; exact upstream repayment snapshot capture semantics remain an assumption. The earlier broader requests to exhaust foundation models and actuarial model classes remain incomplete.

Authoritative evidence: [final audit](reports/FINAL_AUDIT.json), [final metrics and fold recipes](reports/confirmation_RESULT.json), [uncertainty](reports/UNCERTAINTY.json), [frozen inner selection](selection/SELECTION.json), [sources](SOURCE_LEDGER.json), and [original plan](PLAN.json). The authoritative remote archive is `/mnt/home/faris/logika-horizontal-20261026` on node2001. Node2002 and node2003 retain their worker outputs; Faris A100 remains stopped for this campaign.

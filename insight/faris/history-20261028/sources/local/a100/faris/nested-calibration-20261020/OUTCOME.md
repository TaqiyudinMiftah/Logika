# Verified outcome

The five-fold nested-calibration run completed in 91.46 minutes. It performed 55 frozen Kumo context fits and fitted ten correction models. Both GPU workers terminated normally, and all outputs remain on PC2.

| Metric | Matched incumbent | Nested correction | Change |
|---|---:|---:|---:|
| Pooled OOF KS | 0.5292498734 | 0.5288181780 | -0.0004316954 |
| ROC-AUC | 0.8472013286 | 0.8474156627 | +0.0002143340 |
| Log loss | 0.2925010399 | 0.2923669029 | -0.0001341370 |

KS improved in one of five folds; AUC improved in four. The worst fold KS change was -0.0036302722, exceeding the predeclared loss tolerance. The multiplicity-adjusted intervals were [-0.00302056, +0.00227578] for KS change and [-0.00012907, +0.00055530] for AUC change. Both include no improvement. The new model is rejected by the frozen rule.

The experiment supports a narrow observation: more honest calibration rows improved these pooled AUC and probability-loss estimates, but did not improve the competition's KS objective. It does not show that additional calibration data generally helps or hurts. Reused labels and unresolved inherited snapshot availability remain limitations.

## Selected deliverable

`results/model/` contains the retained incumbent correction and spline models, the Kumo feature definitions, fitted-context rows, and a hash manifest. The source entrypoint is `source/pipeline.py predict`. The verified Kumo Large checkpoint remains at `/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/assets/classifier.pt` on PC2; the manifest references that durable dependency.

The selected incumbent's historical aligned OOF is KS 0.5292994659 / AUC 0.8472026884. Its small difference from the matched control above comes from fresh Kumo execution with smaller GPU batching. The comparison used the new matched control so this numerical difference was not credited to the candidate.

`results/independent_verification.json` checks 152 artifact hashes, all 55 context boundaries, complete 70,000-row OOF coverage, and independently recomputed KS/AUC. Metric disagreement was at most 1.11e-16. `results/package_replay_check.json` separately exercises the packaged correction and spline models against original cached Kumo predictions.

All candidate models and inner OOF evidence remain under `/mnt/home/logika-faris-exploration/nested-calibration-20261020/folds/`. No new competition submission was generated. The root submission remains SHA-256 `1ba5b8677549f5d05fdda6d63fa5f978e731247275a74ff91f3bdd39ac068036`.

# Horizontal mechanism study

Status: complete. Eight fixed fusion recipes were evaluated using two training seeds and five matched outer folds. All 160 base fits and 50 head fits completed in 25.2 minutes; all fitted artifacts remain on PC2. The 24-hour allocation was a ceiling, not a requirement to add experiments.

No new recipe passed the preregistered KS and AUC criteria against both anchors. Horizontal probability fusion improved over our champion but underperformed the teammate baseline on both KS and AUC. No submission was produced or changed.

| Pipeline | Mean seed pooled KS | Mean seed pooled AUC |
|---|---:|---:|
| Our five-model champion, refitted on matched folds | 0.521232 | 0.842841 |
| Teammate Kumo + correction + spline, archived aligned OOF | 0.529299 | 0.847203 |
| Fixed 50:50 probability fusion | 0.526447 | 0.846572 |

The account additions did not show a reliable gain. Bureau additions improved standalone point estimates but their adjusted intervals include zero; their effect in the constrained ensemble was negligible. Specialist fusion improved AUC against the context-equal control, while its KS interval includes zero. Pricing residuals did not establish an advantage over calibration alone.

All labels have been used historically. The inherited repayment snapshots are not point-in-time certified. These are conditional research results, not a guarantee against overfitting or a prediction of the private leaderboard.

## Artifacts

- `evidence/RESEARCH.md`: archive synthesis, methods, counterevidence and sources.
- `results/REPORT.md`, `scoreboard.json`, `uncertainty.json`: complete findings.
- `results/independent_verification.json`: 870 artifact checksums, 30 independent metric checks, full aligned OOF coverage.
- Remote immutable bundle, fitted models and logs: `/mnt/home/logika-faris-exploration/horizontal-20261019/campaign` on `faris@labkc-comm1`.

## Commands

```sh
ssh faris@labkc-comm1 'tail -n 30 /mnt/home/logika-faris-exploration/horizontal-20261019/campaign/controller.log'
ssh faris@labkc-comm1 'cat /mnt/home/logika-faris-exploration/horizontal-20261019/campaign/result.json'
```

The terminal event is `"DONE": true`; `result.json` confirms `"status": "complete"`. No controller is expected to remain running.

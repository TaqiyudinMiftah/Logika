# Logika feature and ensemble research

This campaign is an unattended ordinary-Python experiment system, with no agents.
Read RULES.md and vendor/DATA_DICTIONARY.md. There are 70,000 labeled applications,
30,000 prediction-test applications, and 9,507 positives. The historical public-best
checkpoint is Logika v3 (0.55845); its historical OOF is 0.5190088513. These values
are not directly comparable with this campaign's newly frozen validation protocol.

Hold out 20% with target-by-decision-month stratification, seed 20261016. Run full
3-fold CV, seed 9173, on the 56,000 development rows. Model seed is 2026 plus fold.
Never use the confirmation partition for screening, stopping, fitted preprocessing,
feature selection, target encoding, or ensemble fitting. Earlier historical work
used these labels, so confirmation is fresh only to this campaign.

Register up to 1,024 feature/preprocessing representations times four lightweight
probes, with a separate maximum 8,192 fixed-ensemble catalogue and genuinely nested
learned ensemble tests. Keep full CV for every accepted recipe. Count logical recipes
once, separating pipelines from ensembles. EDA, folds, repeats, retries and calibration
checks do not add recipe counts. Report all failed, partial and untested outcomes.

The coordinator runs on PC2, retains one original 12-hour deadline, enforces host
budgets, verifies saved OOF independently and creates reports without human monitoring.
The search phase ends at hour 9.5; ensemble work ends at 10.5; confirmation ends at
11.5; the remainder is verification reserve. Start the clock only after deployment,
source/data verification, tests, backend smokes and resource-control checks pass.

Outputs are feature-effect evidence, coverage, reproducible recipes and an A100
handoff. Probe improvements are not guaranteed improvements for larger A100 models.

# Second research pass

This pass reuses completed experiment evidence to test a compact ensemble search and seed averaging. The requested 0.55 is treated as an internal KS target; it is not a public-score forecast.

## Frozen procedure

1. Verify source-bank hashes, OOF application IDs, labels, fold assignments, and completed prediction artifacts.
2. Collect the two highest-scoring distinct feature representations for each of LightGBM, CatBoost, XGBoost, and logistic regression, plus the five current submission components. This yields 10 distinct component recipes. The original scoreboard inventory contains repeated control references; its row count is not an independent experiment count.
3. Compare 32 predeclared fixed combinations using trimmed, arithmetic, logit, and geometric averaging. No learned weights or stackers are fitted on global OOF predictions.
4. Choose the highest pooled KS satisfying the declared fold-consistency gate. Evaluate that one structure with seeds 4917 and 9173 on all three original folds. Each fold retains its fitted model and preprocessing.
5. Average corresponding component probabilities across the original and two new seeds, then apply the selected ensemble rule. Adopt this bagging only if the predeclared development gate improves on the single-seed structure. Other new individual seeds are diagnostics and cannot be cherry-picked.
6. Freeze the final recipe before making reserved-partition predictions. Test only this nominee against the previously delivered candidate. Refit all 70,000 labels and generate a new CSV only if both development and confirmation promotion gates pass.

Development and confirmation labels have been reused historically. Bootstrap uncertainty is conditional on the selected predictions and does not correct adaptive-selection optimism. A positive exploratory promotion gate is not proof of a public leaderboard improvement.

## Execution and evidence

Code lives in `source/`; resolved settings in `plan.json`; data lineage and original experiment scores in `evidence/provenance.json`. `screening.json`, `seed_verification.json`, `frozen_recipe.json`, and `confirmation.json` record successive decisions when those stages run. `result.json` records the terminal outcome. A failed/interrupted stage instead leaves `failure.json` and available checkpoints; it cannot silently produce a successful submission.

PC2 runs eight numerical workers, one numerical-library thread each, inside a dedicated systemd user unit with CPU quota 10 cores, MemoryMax 40 GiB, zero swap allowance, and a one-hour safety cutoff. The latest user authorization permits expanded resources; these are conservative operational bounds, not changes to the expired campaigns. Only this new unit owns these jobs. The frozen clock survives restarts and cannot start another hour.

The Python 3.11 environment and package versions are recorded in `environment.json`. `resources.jsonl` samples aggregate process memory and CPU time; summing RSS can double-count shared pages. The systemd cgroup provides actual aggregate memory enforcement. Every fitted artifact retains parameters, selected features, row hashes, warnings, elapsed time, logs, model weights, and completed-file checksums.

Four unit tests cover combination order, identical-probability behavior, invalid predictions, and the fold-regression gate. Saved models are replayed on 256 test rows before any newly generated CSV is accepted, and the CSV is checked against exact official sample ID order.

```sh
# From ~/faris/submission-pass2-20261017 on PC2, using the existing environment:
../depth-20261017/.venv/bin/python source/test_run.py
../depth-20261017/.venv/bin/python source/prepare.py
../depth-20261017/.venv/bin/python source/run.py
```

The retained plan refers to the original read-only feature bank and the preceding submission's inputs. The source uses those paths directly. For a new run after the cutoff, use a new explicitly authorized output directory and freeze a new plan; do not edit the old clock.

No Kaggle submission or leaderboard probe is part of this pass.

## Completed outcome

No replacement was promoted. All 32 fixed combinations failed to exceed the existing trimmed ensemble. The best alternative was original-plus-CatBoost arithmetic averaging at KS 0.523413; it lost 0.00525 in one fold.

| Variant | Development OOF KS |
|---|---:|
| Existing seed 2026 | 0.525771439 |
| Additional seed 4917 | 0.521869446 |
| Additional seed 9173 | 0.523196201 |
| Three-seed component averaging | 0.524352901 |

The three-seed version lost 0.001418537 pooled KS and declined in two of three folds. The 0.55 target was not reached. These results expose seed sensitivity; they do not establish that seed 2026 will generalize better. Confirmation was not reopened after the failed development gate. The root submission.csv remains byte-for-byte unchanged.

All 30 new fold fits (five existing recipes, two new seeds, three folds) retained models and transformers. They are repeated-seed validation fits, not 30 new feature recipes. All artifact hashes, full OOF coverage, seed scores, and bagged KS were independently verified after transfer. See verification.json and seed_verification.json.

# Logika meta experiments

A plain-Python, unattended 12-hour feature/ensemble campaign. No agents, hosted
models, or competition submission API are used. Code and run artifacts remain
under the approved faris directories.

The immutable bank contains 2,329 numeric features and four categorical variables.
Historical v3 cohort priors are recomputed from each training fold's histories.
Preprocessing, feature selection, reduction and target encoding are fold-local.
Learned stacking uses inner OOF trained only on the corresponding outer training
partition. Fixed rank blends use CDFs from training predictions, not validation data.

## Commands

From this directory with the pinned environment:

```sh
python -m metaexp audit --workspace /path/to/logika-2026
python -m metaexp design
python -m metaexp deploy --workspace /path/to/logika-2026
```

After all deployment checks pass, execute on PC2 from its campaign directory:

```sh
.venv/bin/python -m metaexp launch
.venv/bin/python -m metaexp status
.venv/bin/python -m metaexp collect
.venv/bin/python -m metaexp report
.venv/bin/python -m metaexp resume
.venv/bin/python -m metaexp stop
```

`resume` preserves the original deadline. A changed source, dataset, split, or
configuration requires a new identity; expired deadlines are not reset implicitly.
Each attempt retains its command/configuration, source/environment and bank identity,
per-fold preprocessing audits, predictions, logs, metrics and resource samples.

PC1 uses the verified RX 6600 LE OpenCL backend for LightGBM when its GPU slot is
available, plus CPU work. The CPU and OpenCL implementations are tagged separately.
PC2/PC3 use one visible RTX 3060, at most seven experiment processes, CPUQuota=700%,
MemoryMax=20G and MemorySwapMax=0. The PC2 coordinator shares its service budget with
its local execution daemon. Resource admission leaves room for the controller and
file cache. PC1 has a 10 GiB project disk budget with a 3 GiB run-artifact allocation.

The catalogue is finite. Full 3-fold CV completion is required to rank a recipe;
there is no one-fold pruning. Timeouts, resource failures and unstarted work are
reported explicitly. A run count is not a fold count or an EDA check count.

The active campaign uses independent per-PC catalogue shards and worker pools,
with result collection and later ensemble/confirmation orchestration on PC2.
See `RUNNING.md` for the deployed scheduler upgrade, runtime proof, and recovery
command. Automatic restart preserves the upgraded entry point and original clock.

## Artifacts

- `history_inventory.json`: deduplicated history and verification states.
- `catalogue.json`: exact representations, pipeline recipes and planned coverage.
- `campaign.json`: original start/deadline and machine limits.
- `ledger.sqlite`: assignments and accepted/failure states.
- `runs/<id>/`: per-run evidence, fold predictions and logs.
- `reports/final_report.md`: measured outcomes and limitations.
- `reports/feature_effects.csv`: matched model/backend feature contrasts.
- `reports/confirmation.json`: frozen finalists, paired bootstrap and Holm results.
- `reports/untested_recipes.json`: every uncompleted logical pipeline.
- `reports/a100_handoff.json`: reusable feature recipes and preprocessing audits.

Preprocessing factories are in `metaexp.pipeline.Preprocessor`; attach a compatible
`Bank` and refit on the new training split before using a recipe with a larger model.
The confirmation-stage probe bundles are retained for diagnostic replay. The runner
does not train a final full-data competition model or generate new submissions.

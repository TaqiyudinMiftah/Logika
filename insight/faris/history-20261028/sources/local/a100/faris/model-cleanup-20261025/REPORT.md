# PC2 model cleanup result — 25 October 2026

Verified success: **True**. Finished: `2026-10-25T14:06:30Z`.

Scope: `/mnt/home/logika-faris-exploration` on `labkc-comm1`, reached with
`ssh faris@100.68.29.126` (port 22). Port 2001 refused the connection.

- Model, transformer, checkpoint and cached-weight files removed: **25,705**.
- Model aliases removed: **3**.
- Model copies stripped from mixed archives: **38**.
- Readable settings/metadata exports retained: **25,743**.
- Retained files checked: **113,112**.
- Retained-file verification failures: **0**.
- Remaining recognized model files: **0**.
- Filesystem free-space increase: **306.12 GiB**.
- Model allocation unlinked at its last hard link: **305.88 GiB**.
- Available filesystem space afterward: **636.12 GiB**.

The filesystem free-space change includes concurrent filesystem activity and new
cleanup metadata. Logical file sizes may count hard links repeatedly; they are
not an estimate of reclaimed space.

The historical Python 3.11 installation outside this root disappeared during
preparation, leaving existing virtual-environment interpreter links broken.
This task did not delete or repair that installation; system Python performed
the cleanup and recovered metadata exports.

Predictions, submissions, reports, code, configuration files, raw/prepared data,
and dependency environments were kept. Model settings are in `parameters/` on
PC2 and in `parameters.tar.gz` in the local receipt copy. Large learned weights
are not duplicated in these exports.

Verification methods: `{"device_inode_size_mtime_ns": 110615, "full_sha256": 2497}`.
Filesystem identity verification includes device/inode, size and nanosecond
modification time. It is not a full-content hash. Selected source/configuration
formats and submission CSVs also received full SHA-256 checks. Every retained
file member of the two rewritten archives was fully hashed before replacement.

Embedded metadata export errors: **1**;
archive-member metadata errors: **0**.
Existing configuration records and source references remain available for
formats whose configuration is not embedded as readable metadata.
The one model-export read error concerns an empty zero-byte file from an
interrupted teacher fit. Its boundary settings and intended champion recipe
were explicitly retained; no serialized model existed in that file.

Historical manifests and archive checksum sidecars remain historical evidence.
Use the cleanup archive receipts for replacement archive hashes. Old code paths
still describe training/inference recipes, but removed weights require refitting
or obtaining weights from a separate location. Other machines and local fitted
models were outside this cleanup scope. No reboot, poweroff or unrelated process
termination occurred.

## Removed model files by project

| Project | Model files removed | Logical GiB |
|---|---:|---:|
| account-conditional-fe-20261021 | 10 | 0.000 |
| account-dynamics-head-20261021 | 10 | 0.000 |
| account-trait-fe-20261021 | 10 | 0.000 |
| catboost-convergence-20261021 | 30 | 0.043 |
| context-model-retention-20261021 | 1 | 0.019 |
| coverage-information-fe-20261021 | 10 | 0.063 |
| coverage-transfer-20261021 | 71 | 0.000 |
| dicefest-checkpoints-20261023 | 6 | 0.274 |
| dicefest-dion-20261023 | 202 | 0.537 |
| dicefest-final-2315 | 41 | 8.842 |
| dicefest-formulas-20261023 | 108 | 0.000 |
| dicefest-reverse-20261023 | 258 | 86.543 |
| exploration-12h-20261017 | 23,585 | 24.067 |
| feature-only-20261021 | 20 | 0.000 |
| heldout-class-margin-20261021 | 5 | 0.000 |
| heldout-margin-fusion-20261021 | 10 | 0.000 |
| horizontal-20261019 | 210 | 0.049 |
| kumo-coverage-placement-20261021 | 39 | 93.353 |
| kumo-supervised-view-20261021 | 17 | 29.328 |
| loan-credibility-fe-20261021 | 10 | 0.000 |
| mature-denominator-fe-20261021 | 15 | 0.069 |
| mature-ma[REDACTED_CREDENTIAL] | 15 | 0.069 |
| maturity-placement-fe-20261021 | 5 | 0.000 |
| nested-calibration-20261020 | 20 | 0.000 |
| nested-correction-20261021 | 35 | 0.388 |
| ranking-pipeline-20261020 | 36 | 0.797 |
| repayment-severity-fe-20261021 | 15 | 0.000 |
| ri[REDACTED_CREDENTIAL] | 5 | 0.000 |
| ri[REDACTED_CREDENTIAL] | 177 | 57.856 |
| schematics-2026 | 17 | 0.460 |
| shared-weights | 1 | 0.796 |
| shopping-bands-fe-20261021 | 20 | 0.000 |
| shopping-periodicity-fe-20261021 | 25 | 0.000 |
| source-support-fe-20261021 | 10 | 0.000 |
| submission-final-20261018 | 31 | 0.017 |
| submission-full-periodicity-20261021 | 15 | 0.089 |
| submission-robust-20261018 | 115 | 0.029 |
| tabicl-compact-20261021 | 13 | 5.490 |
| teammate-approach-20261020 | 147 | 0.002 |
| transfer-20261025 | 335 | 0.146 |

# Node380 Kumo Large run

Remote project: `/home/taqiyudinmiftah/faris/kumo-large-node380-20261016`.
All code, the isolated Python environment, downloads, logs and new predictions
are stored inside that directory. The existing raw competition data at
`/home/taqiyudinmiftah/Logika/data/raw` is read only.

`scripts/run_node380.py` installs Python 3.11.13, PyTorch 2.7.1/CUDA 12.6,
the SDK at commit `329a707bbcf9ca147aa45a63fef8b2029f4ee37b`, and Kumo
checkpoint `nvidia/Kumo-Tabular` revision `v1.0.1`. It verifies CUDA and BF16,
builds the existing 388-feature view with the original aggregators, runs a
pilot, then runs all ten validation folds. Only Kumo is used as the learner.

The frozen model uses 16,384 labeled context rows per fold, two estimators,
BF16, and query batches of 256. It performs zero gradient updates. Fold seed
9173 and target/month context stratification are preserved. Each worker has
a 24 GiB CUDA allocation budget and a 32 GiB host RSS budget; the latter is
increased for node380. Training and validation IDs are kept separate and
checkpoint/input/source hashes are recorded.

From the local workspace, inspect the live run:

```sh
python a100/profile.py node380 exec --session term --cwd /home/taqiyudinmiftah/faris -- 'cat kumo-large-node380-20261016/artifacts/node380/status.json; tail -n 20 kumo-large-node380-20261016/launcher.log'
```

Detailed logs and fold artifacts are under `artifacts/node380/`.
`pilot/summary.json` confirms the pilot completed; `full/summary.json`
contains the actual pooled out-of-fold metrics after all ten folds finish.
The final CSV is copied to
`submissions_held/submission_kumo_large_logika_v3.csv`.
There are no Kaggle submission calls.

The deployment helper refuses a duplicate bootstrap record. Do not launch
a second process while the recorded launcher is live. A failed setup may be
resumed with this project's system-Python launcher after inspecting its
status and ensuring no previous launcher or child remains active.

Out-of-fold results have the existing caveat that historical v3 feature
selection reused training labels. Kumo uses fewer context labels per fold
than the historical tree models. No public leaderboard score is inferred.

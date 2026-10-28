# DSC LOGIKA UI 2026

Project workspace for the Kaggle competition [Data Science Competition (DSC) LOGIKA UI 2026](https://www.kaggle.com/competitions/data-science-competition-logika-ui-2026).

The task is to predict `default_probability` for each application in the test set. Kaggle evaluates predictions with the Kolmogorov–Smirnov statistic. The competition rules require Python, prohibit external data, and require all features to use information available at the application's `decision_datetime`.

## Environment

This project uses Python 3.11 and `uv`.

```bash
uv sync
uv run jupyter lab
```

The local environment lives in `.venv`. The dependency lockfile is `uv.lock`.

## Dataset

The Kaggle download contains 10 files and is about 949 MB. Keep the downloaded archive outside version control. From the Kaggle Data tab, download all files as a ZIP, then extract the expected files into `data/raw`:

```bash
uv run python scripts/prepare_data.py /path/to/competition.zip
```

The preparation script validates the archive and required filenames before extracting. Use `--overwrite` only when you want to replace existing raw files.

After extraction, inspect table row counts, schemas, null values, and the target distribution with:

```bash
uv run python scripts/inspect_data.py
```

## Baseline model

Build point-in-time features, evaluate a LightGBM model using the latest 20% of applications by `decision_datetime`, then refit on all labeled rows and write a local submission:

```bash
uv run python scripts/train_baseline.py
```

The feature builder uses DuckDB with a 768 MiB memory limit and one thread for CSV aggregation. It uses application data, customer profiles, previous applications, bureau observations, complete account months, and order-level e-commerce aggregates. Every event table is filtered to information available by the application cutoff. `repayment_history.csv` is excluded until its as-of semantics are established.

The run writes derived features to `data/processed/model_features.parquet`, the final model to `models/baseline_lgbm.txt`, validation metrics and diagnostics to `reports/`, and a sample-compatible CSV to `submissions/baseline_submission.csv`. The submission file is created locally and is not uploaded to Kaggle.

The reproducible entry point for Jupyter is `notebooks/01_baseline.ipynb`. Run it from the project environment with:

```bash
uv run jupyter lab
```

## Controlled experiments and A100

The experiment runner uses five folds jointly stratified by target and decision month, with application IDs sorted and seed 2026. Its outer and inner ID lists match Agung's current cross-validation implementation. Early stopping sees only each fold's inner validation set. The outer fold is scored after a refit on that fold's full training partition.

```bash
uv run --locked python scripts/run_experiment.py --config configs/experiments/e001_lgbm_all.json
```

The GPU extra adds CatBoost. On the A100, use the project environment and this command:

```bash
uv run --locked --extra gpu python scripts/run_experiment.py --config configs/experiments/e002_catboost_all.json
```

Each run writes an isolated directory under `artifacts/experiments/` with its configuration, raw-data and feature checksums, exact fold IDs, runtime, source snapshot, fold models, out-of-fold predictions, metrics and a five-model averaged submission. Completed folds can be reused with `--resume /absolute/path/to/run`. A changed input, source, configuration or fold assignment prevents resume.

Kumo is now the primary model. On the A100, the project is under `~/gipta/projects/logika-ui-26`. Use the locked CUDA environment:

```bash
cd ~/gipta/projects/logika-ui-26
source scripts/a100_env.sh
uv sync --locked --extra gpu --extra kumo
.venv/bin/python scripts/run_kumo.py --pilot
.venv/bin/python scripts/run_kumo.py
```

E005 uses frozen Kumo Large, 16,384 context rows sampled from each outer training fold, two estimators and BF16. The pilot keeps this complete model/context recipe but predicts only 512 validation and 128 test applications. It verifies execution and reports no CV score or submission. The full command creates five-fold OOF predictions and a test submission. E006 changes only model size to Small. Processes have explicit RAM and GPU budgets and release caches at fold exit. See [foundation model research](docs/TABULAR_FOUNDATION_MODELS.md) and [Kumo protocol](docs/KUMO.md).

The earlier temporal KS cannot be compared directly to shuffled five-fold CV scores.

See [A100 setup](docs/A100.md) for separate laptop and server commands and [experiment protocol](docs/EXPERIMENTS.md) for the team references and experiment questions. `notebooks/02_experiments.ipynb` runs the same pipeline from Jupyter.

Expected data files:

- `train_applications.csv` and `test_applications.csv`
- `customer_profiles.csv`
- `previous_applications.csv`
- `repayment_history.csv`
- `account_behavior.csv`
- `bureau_observations.csv`
- `ecommerce_transactions.csv`
- `sample_submission.csv`
- `DATA_DICTIONARY.md`

Raw and processed data, models, reports, artifacts, and submissions are ignored by Git. Do not add the private competition data or prediction files to a public repository.

## Layout

```text
data/raw/          Competition files
data/processed/    Derived features
notebooks/         Exploration and experiments
src/logika_ui_26/  Reusable project code
scripts/           Data preparation and project commands
models/            Saved model artifacts
submissions/       Kaggle CSV submissions
reports/           Analysis outputs
```

## Competition constraints to preserve

- Only use files provided by the committee. External datasets and other external data sources are prohibited.
- Use only information available at each application's `decision_datetime`.
- Exclude repayment or other records that occur after the prediction cutoff. The target period is future information.
- De-duplicate e-commerce order totals before aggregating because `order_total` repeats across order items.
- Submit exactly `application_id,default_probability`, one row per test application, with finite probabilities from 0 through 1.
- The notebook submitted to the committee must be reproducible and run in the designated environment.

# Research-derived submission pipeline

Selected design: five feature-view models, with the highest and lowest probability removed per application and the middle three averaged. The selected model settings exactly match completed experiment recipes.

| Fixed candidate | Development OOF KS | Fold 1 | Fold 2 | Fold 3 |
|---|---:|---:|---:|---:|
| existing_feature_ensemble | 0.525771 | 0.539039 | 0.508911 | 0.530661 |
| kumo | 0.523857 | 0.536013 | 0.511228 | 0.527259 |
| equal_kumo_features | 0.524770 | 0.538554 | 0.511606 | 0.529693 |
| equal_three_blocks | 0.524406 | 0.538582 | 0.510301 | 0.529710 |
| equal_kumo_depth | 0.524041 | 0.536734 | 0.511452 | 0.528036 |

Kumo Large context search, fine-tuning, stronger LightGBM and stronger logistic models were inspected. The five declared combinations were compared using complete aligned saved OOF predictions. Adding the strongest Kumo/depth components did not improve pooled OOF KS over the existing trimmed feature ensemble, so the final pipeline retains the established five-view design. No extra hyperparameter search or expired-campaign resume was performed.

Models:

- `w2p_e8dd515dcc04cd2e4959`: linear, cross_source; encoding=target, selection=gain, reducer=none, dimensions=64, history window=180.
- `w2p_c9402cdde8b7041a2de9`: lgb, reduction; encoding=onehot, selection=mi, reducer=svd, dimensions=16, history window=all.
- `w2p_dddfbf8272cdbfa280d1`: linear, shopping; encoding=target, selection=gain, reducer=none, dimensions=64, history window=all.
- `w2p_b7592b087e1f55a41563`: lgb, cross_source; encoding=onehot, selection=gain, reducer=none, dimensions=64, history window=60.
- `w2p_03f464270ee3d042b671`: lgb, reduction; encoding=onehot, selection=gain, reducer=pca, dimensions=128, history window=all.

Validation sequence: freeze recipe on development OOF; fit on 56,000 development labels; persist 14,000 reserved predictions; then read reserved labels and compare with the existing v3 reference. A predeclared fallback triggers only if confirmation KS is more than 0.01 below that reference. After assessment, refit the chosen five-view pipeline on all 70,000 supplied labels and predict the 30,000 competition applications.

The development OOF was reused during historical/adaptive selection. The reserved partition was also used by historical work, and its wave1 results were already viewed. Scores are research evidence, not an unbiased generalization estimate or a public leaderboard result.

## Reproduction

Run in the tested Python 3.11 environment with `source/requirements.lock`. The supplied read-only bank is the decision-time-safe feature artifact from the experiment campaigns, whose raw-source, feature-file and lineage hashes are verified before fitting.

```sh
python source/run_pipeline.py \
  --bank ../depth-20261017/stage/bank \
  --train ../../../dataset/train_applications.csv \
  --sample "../../../dataset/sample_submission(3).csv" \
  --recipe frozen_recipe.json --evidence evidence \
  --output reproduced-artifacts --workers 5
```

Run the command from this directory. A copy of the complete bank is shared with the earlier campaign rather than duplicating the 0.9 GiB matrix. Every phase stores fitted preprocessing, model weights, selected columns, resolved parameters, source/input hashes, warnings, elapsed time and raw probabilities. Retries reuse only artifacts with matching requests and file hashes.

The training run executes on PC2 under `~/faris/submission-research-20261017/`; `pipeline.log` records phases. Completed models and confirmation evidence are retained under `artifacts/`. The final CSV is read back, checked against exact sample ID order, and compared with independently reloaded model predictions for 256 test rows. No competition upload or submission API is called.

## Completed result

- Final output: `../../../submission.csv` and `artifacts/submission.csv`, 30,000 rows in official sample order.
- Development OOF KS: 0.525771439.
- Terminal reserved-partition KS: 0.530496149; matched v3 reference: 0.527854085.
- Paired difference: +0.002642064; 95% bootstrap interval [-0.003967909, 0.013051695]. This does not establish a reliable improvement.
- All 70,000 labels used in the final refit, with five retained fitted pipelines.
- The CSV was independently reaggregated locally from saved component predictions and all model checksums passed.
- SHA-256: `be059078ade95bc6758522e8d72849e3f793e57380e2c083e05c77c64335e192`.
- No Kaggle upload or leaderboard evaluation was performed.

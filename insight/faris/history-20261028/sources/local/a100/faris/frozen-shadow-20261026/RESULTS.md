# Frozen-feature modelling results

**Decision: retain the incumbent. The campaign completed, but KS 0.54 was not reached and no new submission was made.**

All **68 configurations × 5 folds = 340 successful planned fits** completed. The terminal evaluation finished **26 October 2026, 23:49:34 WIB**, about **6 hours 13 minutes** after the immutable modelling clock started. The 24 hours was a ceiling; the declared queue finished earlier. No scientific recipe was changed after terminal outer scoring.

| Outcome | Pooled validation KS |
|---|---:|
| Matched Large90 + fixed correction + maturity baseline | 0.530636104 |
| Preselected ensemble | 0.530332316 |
| Difference | -0.000303789 |
| Target | 0.540000000 |

The paired, class-stratified 2,000-bootstrap interval for the difference is **[-0.001267344, +0.000917130]**. Only **2/5 folds** improved. This supports no gain; the interval also does not establish significant harm. Validation uses previously reused development data, not an untouched audit.

## What the tested ideas showed

The original Large90 configuration exactly reproduced the matched baseline and was the best individual configuration. All 67 alternatives were below it. The family summaries below are terminal diagnostics, not new model selections. Every score includes the unchanged original correction head and maturity spline.

| Family | Best tested configuration | KS |
|---|---|---:|
| Global Kumo | `kumo_large_champion90` | 0.530636104 |
| BCE/pairwise/consistency correction | `rank_p0.05_c2.0_s7731` | 0.530275368 |
| Fixed-cluster local-context Kumo | `context_medium_16000` | 0.525850827 |
| CatBoost | `cat_master_d8_p20` | 0.524993554 |
| LightGBM | `lgb_master_l15_r0.04` | 0.524828129 |
| Logistic regression | `logistic_champion90_C0.01` | 0.524183004 |
| XGBoost | `xgb_master_d3` | 0.523787269 |
| ExtraTrees | `extra_champion90` | 0.519701209 |
| MLP | `mlp_champion90_w64` | 0.519374889 |

The learned additions were also compared against the original 90-column model with the same size and seed. Positive values below favor the new view.

| Kumo size | Added representation | Delta, seed 2026 | Delta, seed 7731 |
|---|---|---:|---:|
| medium | shadow91 | +0.000096359 | -0.000490740 |
| medium | latent99 | -0.001588475 | -0.001057741 |
| medium | full100 | -0.001244808 | -0.000725230 |
| large | shadow91 | -0.000716594 | -0.000080950 |
| large | latent99 | -0.000487638 | -0.000769478 |
| large | full100 | -0.000882252 | -0.000422330 |

Shadow-loan scores gave a small Medium gain in the first seed, which reversed in the second. Large shadow scores were negative in both seeds. Latent coordinates and the combined 100-column view decreased pooled KS in every matched comparison. The four local-context recipes also fell below the baseline. These are results for the tested implementations and settings, not a claim that the ideas can never work.

The ranking students trained on the same correction half used to rank candidate models, so their high ranking is training-informed. A separate half chose ensemble weights, yet the selected ensemble did not improve held-out pooled KS. The original selection rules were preserved and replayed exactly; they were not repaired after viewing outer scores.

## Completion and artifact audit

- All 340 successful fits have verified parameters, seeds, immutable feature hashes, prediction hashes and row order. Each configuration has all five folds.
- The frozen fold recipes were written before terminal scoring. Replaying the saved correction-only selection produced identical recipes and predictions.
- Independent NumPy KS recalculation agreed with recorded scores to better than 3e-13. Baseline predictions reproduced within 1.12e-16.
- Frozen feature tables, seeds, parameters, source versions, compact predictions and evaluation artifacts are retained. No fitted weights or KV caches were found in this campaign.
- Successful execution totals: 100 GPU fits and 240 CPU fits. The 26 verified results collected from 2002 before its outage were retained. Its five unverified attempts were replaced under the user's later instruction to use only 2001 and 2003.
- The earlier CPU Kumo feasibility attempt hit its 22 GiB RAM cap. A recovery bug repeated it, with 19 node-2001 memory kills recorded. Those unsuccessful attempts are disclosed separately, their evidence is retained, and CPU Kumo admission is disabled. They are not counted among the 340 successful fits.

The feature contract remains immutable: 116 master columns, fixed views of 90/91/99/100 columns for Kumo, and the original 18-column correction head. The predeclared queue of 68 configurations is complete. This is not a claim that every model in the older foundation/actuarial lists has been tested.

## Disposition

No full-data refit or Kaggle upload was warranted. The best recorded public submission remains **V25, KS 0.56822**. All original artifacts are preserved for review; no new feature search or post-hoc ensemble has been started.

Evidence: [terminal audit](TERMINAL_AUDIT.json), [evaluation result](evaluation/RESULT.json), [frozen recipe](evaluation/FROZEN.json), [fit audit](MODEL_RUN_AUDIT.json), [analysis](ANALYSIS.json), [source ledger](SOURCES.json), [resource incident](CPU_RESOURCE_LIMIT.json).

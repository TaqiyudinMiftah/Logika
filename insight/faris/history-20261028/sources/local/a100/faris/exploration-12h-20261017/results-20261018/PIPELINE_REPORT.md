# Best pipelines from the twelve-hour campaign

Reviewed on 18 October 2026, Asia/Jakarta. This is a post-campaign analysis of saved predictions, configurations and model artifacts. No further training, test prediction, submission generation or upload was performed.

The strongest candidate under the predeclared selection rule is **`ens_e62cfb5b6d636145`**, an equal-weight logit blend of five prediction streams. The same components combined by a simple probability average produced the highest observed original-CV and confirmation scores. These are promising small improvements; the campaign did not establish a statistically clear improvement or reach 0.55 development KS.

## Matched results

Higher KS is better. Original CV, the new training seed, and the second CV partition each cover all 56,000 development rows. Confirmation covers 14,000 reserved rows whose labels have historical reuse.

| Pipeline | Original OOF KS | New-seed OOF KS | Second-partition OOF KS | Confirmation KS |
|---|---:|---:|---:|---:|
| Existing five-model champion | 0.525771 | 0.521594 | 0.519099 | 0.530694 |
| **Logit blend, frozen primary** | **0.526020** | **0.523426** | **0.520846** | **0.531174** |
| Probability mean, secondary | 0.526033 | 0.523216 | 0.520741 | 0.532121 |
| Geometric mean, secondary | 0.525973 | 0.523420 | 0.520769 | 0.531265 |

The logit blend had the largest average matched KS improvement across the three development evaluations: **+0.001276**. It was frozen as primary before confirmation results were examined. Its original-fold differences were +0.001161, +0.000487 and -0.000316. All three aggregate development comparisons and confirmation favored it.

The probability mean's average development improvement was +0.001175. Its confirmation improvement was +0.001426, versus +0.000479 for the frozen primary. Selecting the mean because of that confirmation result would be a secondary, post-confirmation choice rather than the original selection rule.

The matched confirmation baseline above is the campaign's fresh refit, not the historical confirmation number 0.530496. The champion's original OOF predictions were reproduced exactly, with maximum difference 0.0. Its public score **0.56188** remains user-reported and belongs to the existing submission, not these new candidates.

## Exact five prediction streams

| Saved run | Role | Original OOF KS |
|---|---|---:|
| `pc2_job_000623` | Explainable Boosting Machine, `representation_17` | 0.520043 |
| `pc2_job_000853` | Nested logistic stack, `nested_9` | 0.521466 |
| `pc2_job_000855` | Nested ridge stack, `nested_11` | 0.521955 |
| `pc3_job_000424` | Elastic-net logistic regression, `account_00` | 0.518039 |
| `pc3_job_000846` | TabM, `bureau_13` | 0.519591 |

The primary combines these as `sigmoid(mean(logit(clip(p, 1e-8, 1-1e-8))))`. Each stream receives weight 0.2. The probability-mean alternative simply averages the same five streams.

Each nested stream contains the same three base recipes, with a different learned head:

1. Champion logistic component `w2p_e8dd515dcc04cd2e4959`.
2. Champion LightGBM/SVD component `w2p_c9402cdde8b7041a2de9`.
3. Repayment-view elastic-net recipe `eb3419a4bf925b93ca53`, with L2-only regularization, C=0.05 and positive-class weight 3.

The logistic head uses C=0.1; the ridge head uses alpha=100. Their heads were fitted on actual inner OOF predictions inside each outer training partition. The two nested streams are multi-model pipelines, so the overall ensemble is more complex than five individual learners.

## Features and preprocessing

All winning views retain the seven original sources: application, profile, previous applications, repayment, bureau, account behavior and shopping. Their view names indicate additional treatment, not exclusive use of one source.

- **EBM:** 180-day representation, median imputation, 1% tail clipping, rank transformation, frequency encoding, true gain selection of 128 features, then appended PCA components grouped by source. Its first outer fold had 339 output columns. The fitted first-fold selector retained no `new_` features; its success is evidence for representation/model diversity rather than proof that a newly added history block caused the gain. EBM used up to 1,000 rounds, four outer bags and ten interactions.
- **Account elastic net:** a 30-day account view with all-history summaries, median imputation, standard scaling, one-hot encoding, no dimensional reduction, C=0.005, L1 ratio 0.75 and class weight 3. First-fold output had 411 columns. New columns included persistent deficits, surplus/income/outflow/balance volatility and recent changes, and balance recovery.
- **TabM:** 365-day bureau view plus all-history summaries, median imputation, 0.5% clipping, signed-log transforms, missingness indicators, cross-fitted target encoding and 128 true-gain-selected features. The first fold retained matched-month provider disagreement and historical bureau-score changes. Architecture: width 256, four blocks, dropout 0.1, learning rate 1e-4 and 16 TabM members. Early stopping used an inner training split.
- **Nested repayment recipe:** all-history repayment features, 30-day decay, median imputation, one-hot encoding, true-gain selection of 128 features, standard scaling and no reducer, alongside the two preserved champion recipes.

Selected columns vary by training fold. Exact fitted selections, parameters, source paths, resource measurements and saved model counts are recorded in `results-audit-20261018.json`.

## Standalone and actuarial results

| Best completed family representative | OOF KS |
|---|---:|
| Nested ridge stack | 0.521955 |
| EBM, best individual learner | 0.520043 |
| TabM | 0.519591 |
| XGBoost | 0.518177 |
| Elastic-net logistic regression | 0.518039 |
| Native-categorical CatBoost | 0.517300 |
| Kumo Large | 0.516923 |
| Actuarial binomial logit GLM | 0.516360 |
| Actuarial complementary-log-log GLM | 0.516274 |
| Spline regression | 0.516159 |
| TabICLv2 | 0.515889 |
| WOE scorecard | 0.508706 |
| Credibility scorecard implementation | 0.352273 |

EBM is the best single learner and participates in the strongest ensemble. The explicit actuarial GLMs were competitive standalone alternatives but did not displace the best ensemble. The credibility-scorecard implementation was weak and sometimes unstable; this is evidence about the tested implementation, not a general verdict on credibility methods. The target supports default probability/frequency comparisons, not a loss-severity or survival claim.

Foundation-model coverage was limited: one completed Kumo configuration and 39 completed TabICLv2 configurations. Their best standalone scores did not beat the champion. The tested 95% champion / 5% Kumo probability blend scored 0.525743, a change of -0.000028. Most node380 results did not reach the central ensemble catalogue in time, so their ensemble value was not comprehensively tested. No confirmation-driven ensemble search was performed after the campaign.

## Confidence and limits

The frozen primary's paired confirmation bootstrap 95% interval for improvement is **[-0.005817, +0.006957]**. The probability mean's interval is **[-0.005495, +0.006992]**. Both include zero; all three Holm-adjusted comparison p-values are 1.0. Small positive point estimates do not establish a reliable gain.

Historical labels and adaptive development selection were reused. The full finalist ensemble lacks a chronological diagnostic because nested temporal retraining was not implemented; only its non-nested streams have temporal results. The campaign therefore supports a small, repeatable development improvement, not a proven shift-robust improvement or a predicted public score.

My first candidate for a later, explicitly requested refit is the frozen **logit blend**. Keep the probability mean as the strongest observed secondary alternative and retain the current champion as the benchmark.

## Verification, infrastructure and coverage

- Independently recomputed all finalist development and confirmation KS values using SciPy's two-sample KS statistic. Reconstructing the three finalist OOF blends from their archived component predictions matched saved arrays within 3.4e-16.
- Verified the five streams' saved models and transformers in the PC2 archive. CPU streams trained on PCs 2/3; TabM trained on an RTX 3060 on PC3. These winners do not require an A100. Their original three-fold job times were approximately 452, 807, 737, 490 and 152 seconds respectively; these are job durations from parallel execution, not a combined production benchmark.
- Recovered metadata and OOF predictions for 39 node380 runs omitted from the final central scoreboard. All 40 node outcomes were independently rescored; original node model artifacts were left in place. Their omitted scores do not change the standalone leader or the frozen finalist components.
- Reconciled total: **2,673 completed run records**, including controls and additional validation protocols; **2,620 distinct non-nested screening configurations**, plus **12 nested ensemble configurations** and **4,096 fixed ensemble recipes**. After reconciling node results, the 4,096-entry non-nested catalogue has 2,620 complete, 12 failed/unschedulable, 13 partial, 1,353 unstarted and 98 equivalent exclusions. Registration is not execution.
- Original machine counts: PC1 788, PC2 921, PC3 924 and node380 40 completed jobs. Node380 stopped because of its storage budget. Its models were not deleted.

The untouched copied `reports/` folder preserves the campaign's original outputs and its incomplete node aggregation. Supplemental node outcomes, prediction hashes and independent finalist audit sit alongside it. This review supersedes the original scoreboard's missing-node completion counts without rewriting the original evidence.

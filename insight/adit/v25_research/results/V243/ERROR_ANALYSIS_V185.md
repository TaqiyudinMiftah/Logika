# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 11 | 253 | 0.391304347826087 | 0.34782608695652173 |
| 0.200000 | 0 | 395 | 0.22531645569620254 | 0.22531645569620254 |
| 0.323686 | 6 | 473 | 0.12896405919661733 | 0.11627906976744186 |
| 0.400000 | 0 | 502 | 0.08764940239043825 | 0.08764940239043825 |
| 0.500000 | -3 | 483 | 0.06625258799171843 | 0.07246376811594203 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6079 | 3488 | +0.004398 | +0.000727 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003097 | -0.000182 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002962 | -0.000205 |
| aux_account_months | zero | 21000 | 2461 | +0.000572 | -0.000381 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000200 | -0.000116 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000403 | -0.000279 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000519 | -0.000153 |
| employment_type | variable_income | 15540 | 2180 | -0.000790 | -0.000387 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000817 | -0.000259 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.001094 | -0.001921 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001615 | -0.000110 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001725 | -0.000004 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001852 | -0.000136 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001865 | -0.000185 |
| employment_type | self_employed | 20987 | 2895 | -0.002323 | -0.000152 |
| baseline_risk_band | under005 | 29965 | 546 | -0.002378 | -0.000743 |
| employment_type | salaried | 33473 | 4432 | -0.003724 | -0.000121 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.004568 | -0.000665 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
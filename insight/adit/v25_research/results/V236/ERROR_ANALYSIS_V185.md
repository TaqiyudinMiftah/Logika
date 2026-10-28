# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 14 | 264 | 0.4053030303030303 | 0.3522727272727273 |
| 0.200000 | 9 | 388 | 0.24484536082474226 | 0.22164948453608246 |
| 0.323686 | 15 | 467 | 0.14775160599571735 | 0.11563169164882227 |
| 0.400000 | 3 | 510 | 0.08823529411764706 | 0.08235294117647059 |
| 0.500000 | -1 | 497 | 0.06237424547283702 | 0.06438631790744467 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.001392 | -0.000012 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.001311 | -0.000235 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001307 | -0.000163 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.001290 | -0.000037 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.000672 | +0.000596 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000322 | -0.000186 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000270 | -0.000234 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000167 | -0.000146 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000141 | -0.000193 |
| baseline_risk_band | under005 | 29965 | 546 | -0.000167 | -0.000339 |
| aux_account_months | zero | 21000 | 2461 | -0.000186 | -0.000165 |
| employment_type | variable_income | 15540 | 2180 | -0.000393 | -0.000324 |
| employment_type | self_employed | 20987 | 2895 | -0.000747 | -0.000084 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.001342 | -0.000029 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001402 | -0.000173 |
| employment_type | salaried | 33473 | 4432 | -0.001743 | -0.000182 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.002753 | -0.002777 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005479 | -0.001323 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
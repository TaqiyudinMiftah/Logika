# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 15 | 264 | 0.3977272727272727 | 0.3409090909090909 |
| 0.200000 | 8 | 435 | 0.23908045977011494 | 0.2206896551724138 |
| 0.323686 | 2 | 522 | 0.13218390804597702 | 0.12835249042145594 |
| 0.400000 | 7 | 554 | 0.10288808664259928 | 0.09025270758122744 |
| 0.500000 | -5 | 529 | 0.06805293005671077 | 0.07750472589792061 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| employment_type | variable_income | 15540 | 2180 | +0.001866 | -0.000284 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001105 | +0.000194 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.001039 | -0.000161 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000765 | +0.000064 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000711 | -0.000192 |
| baseline_risk_band | under005 | 29938 | 544 | +0.000689 | +0.000891 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.000385 | +0.000504 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.000200 | -0.000696 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000144 | +0.000185 |
| aux_account_months | zero | 21000 | 2461 | +0.000123 | -0.000354 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000128 | +0.000153 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001080 | -0.001312 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001131 | +0.000107 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001204 | +0.000234 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.001384 | +0.000821 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001515 | -0.000087 |
| employment_type | self_employed | 20987 | 2895 | -0.001550 | -0.000005 |
| employment_type | salaried | 33473 | 4432 | -0.002876 | +0.000150 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 10 | 254 | 0.3858267716535433 | 0.3464566929133858 |
| 0.200000 | -1 | 384 | 0.2265625 | 0.22916666666666666 |
| 0.323686 | 4 | 445 | 0.12808988764044943 | 0.11910112359550562 |
| 0.400000 | 9 | 490 | 0.09795918367346938 | 0.07959183673469387 |
| 0.500000 | 1 | 460 | 0.07173913043478261 | 0.06956521739130435 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003031 | -0.000153 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002490 | -0.000189 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.002354 | -0.001086 |
| aux_account_months | zero | 21000 | 2461 | +0.001705 | -0.000325 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000720 | -0.000039 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000448 | -0.000138 |
| employment_type | variable_income | 15540 | 2180 | -0.000107 | -0.000345 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.000367 | +0.000230 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000385 | -0.000181 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000422 | -0.000148 |
| baseline_risk_band | under005 | 29938 | 544 | -0.001154 | -0.000794 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001427 | -0.000109 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001460 | -0.000055 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001737 | -0.000067 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001763 | -0.000578 |
| employment_type | salaried | 33473 | 4432 | -0.002015 | -0.000097 |
| employment_type | self_employed | 20987 | 2895 | -0.002336 | -0.000041 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.003543 | -0.000036 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
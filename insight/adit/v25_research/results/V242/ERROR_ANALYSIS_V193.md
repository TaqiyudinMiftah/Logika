# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 12 | 268 | 0.3880597014925373 | 0.34328358208955223 |
| 0.200000 | -17 | 439 | 0.20273348519362186 | 0.24145785876993167 |
| 0.323686 | 5 | 513 | 0.12475633528265107 | 0.11500974658869395 |
| 0.400000 | 1 | 561 | 0.08377896613190731 | 0.08199643493761141 |
| 0.500000 | -7 | 557 | 0.06283662477558348 | 0.07540394973070018 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29938 | 544 | +0.004709 | +0.002069 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003994 | +0.000007 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003334 | -0.000099 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.002392 | -0.000067 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001929 | +0.000150 |
| aux_account_months | zero | 21000 | 2461 | +0.001741 | -0.000224 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.001418 | -0.000019 |
| employment_type | variable_income | 15540 | 2180 | -0.000923 | -0.000494 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.000961 | -0.000918 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001511 | +0.000565 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.001559 | -0.000034 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001750 | +0.000073 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001804 | +0.000072 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.001942 | +0.000602 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002541 | -0.000361 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002803 | -0.000105 |
| employment_type | self_employed | 20987 | 2895 | -0.002808 | +0.000124 |
| employment_type | salaried | 33473 | 4432 | -0.002887 | +0.000129 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -6 | 40 | 0.3 | 0.45 |
| 0.200000 | -8 | 60 | 0.2 | 0.3333333333333333 |
| 0.323686 | 0 | 84 | 0.16666666666666666 | 0.16666666666666666 |
| 0.400000 | 5 | 77 | 0.11688311688311688 | 0.05194805194805195 |
| 0.500000 | -4 | 86 | 0.046511627906976744 | 0.09302325581395349 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6079 | 3488 | +0.001918 | +0.000286 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.001897 | -0.000417 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000833 | +0.000010 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000767 | +0.000043 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000543 | -0.000016 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000501 | +0.000029 |
| employment_type | self_employed | 20987 | 2895 | +0.000442 | +0.000018 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000406 | -0.000015 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000379 | +0.000008 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000291 | +0.000011 |
| baseline_risk_band | under005 | 29965 | 546 | +0.000152 | +0.000097 |
| employment_type | salaried | 33473 | 4432 | +0.000122 | -0.000022 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000057 | -0.000016 |
| employment_type | variable_income | 15540 | 2180 | -0.000048 | -0.000009 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000145 | +0.000150 |
| aux_account_months | zero | 21000 | 2461 | -0.000226 | +0.000022 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000239 | -0.000019 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000891 | -0.000111 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
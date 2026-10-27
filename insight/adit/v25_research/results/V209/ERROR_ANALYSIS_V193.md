# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -5 | 41 | 0.34146341463414637 | 0.4634146341463415 |
| 0.200000 | -8 | 60 | 0.23333333333333334 | 0.36666666666666664 |
| 0.323686 | -5 | 74 | 0.12162162162162163 | 0.1891891891891892 |
| 0.400000 | 8 | 76 | 0.15789473684210525 | 0.05263157894736842 |
| 0.500000 | -4 | 81 | 0.037037037037037035 | 0.08641975308641975 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.003274 | -0.000382 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.001913 | +0.000280 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001270 | -0.000210 |
| baseline_risk_band | under005 | 29938 | 544 | +0.001062 | +0.000095 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000780 | -0.000008 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000603 | +0.000006 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000514 | -0.000004 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000270 | +0.000046 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000255 | +0.000065 |
| employment_type | self_employed | 20987 | 2895 | +0.000083 | +0.000017 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000075 | -0.000006 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000020 | -0.000024 |
| employment_type | salaried | 33473 | 4432 | -0.000005 | -0.000018 |
| employment_type | variable_income | 15540 | 2180 | -0.000357 | -0.000015 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000435 | -0.000018 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000531 | +0.000153 |
| aux_account_months | zero | 21000 | 2461 | -0.000669 | +0.000053 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000781 | -0.000019 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
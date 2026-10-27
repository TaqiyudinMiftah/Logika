# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 6 | 77 | 0.44155844155844154 | 0.36363636363636365 |
| 0.200000 | -8 | 131 | 0.21374045801526717 | 0.2748091603053435 |
| 0.323686 | 2 | 153 | 0.13725490196078433 | 0.12418300653594772 |
| 0.400000 | 8 | 146 | 0.1095890410958904 | 0.0547945205479452 |
| 0.500000 | -2 | 138 | 0.057971014492753624 | 0.07246376811594203 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002924 | +0.000044 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002161 | +0.000089 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.001789 | +0.000061 |
| aux_account_months | zero | 21000 | 2461 | +0.001658 | +0.000027 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001163 | -0.000099 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.000682 | +0.000005 |
| baseline_risk_band | under005 | 29965 | 546 | +0.000582 | +0.000560 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000555 | +0.000006 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.000145 | -0.000765 |
| employment_type | variable_income | 15540 | 2180 | +0.000111 | +0.000047 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000170 | +0.000122 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.000389 | +0.000170 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000471 | +0.000107 |
| employment_type | salaried | 33473 | 4432 | -0.000471 | +0.000026 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000843 | -0.000011 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000855 | +0.000016 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001102 | -0.000007 |
| employment_type | self_employed | 20987 | 2895 | -0.001630 | -0.000019 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
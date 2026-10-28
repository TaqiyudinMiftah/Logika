# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 9 | 251 | 0.38247011952191234 | 0.3466135458167331 |
| 0.200000 | -1 | 381 | 0.22572178477690288 | 0.2283464566929134 |
| 0.323686 | 9 | 446 | 0.13452914798206278 | 0.11434977578475336 |
| 0.400000 | 6 | 486 | 0.09259259259259259 | 0.08024691358024691 |
| 0.500000 | 1 | 464 | 0.07327586206896551 | 0.07112068965517242 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003262 | -0.000170 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003002 | -0.000211 |
| aux_account_months | zero | 21000 | 2461 | +0.002148 | -0.000356 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.000968 | -0.001154 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000497 | -0.000023 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000224 | -0.000136 |
| employment_type | variable_income | 15540 | 2180 | +0.000202 | -0.000339 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000004 | -0.000156 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000091 | -0.000157 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.000409 | +0.000225 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000885 | -0.000109 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000936 | -0.000047 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001244 | -0.000066 |
| baseline_risk_band | under005 | 29965 | 546 | -0.001882 | -0.000766 |
| employment_type | salaried | 33473 | 4432 | -0.001889 | -0.000101 |
| employment_type | self_employed | 20987 | 2895 | -0.001976 | -0.000040 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.003489 | -0.000018 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003924 | -0.000478 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
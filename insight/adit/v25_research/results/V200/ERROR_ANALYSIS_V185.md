# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 24 | 329 | 0.4012158054711246 | 0.3282674772036474 |
| 0.200000 | 16 | 515 | 0.2407766990291262 | 0.20970873786407768 |
| 0.323686 | 1 | 648 | 0.12808641975308643 | 0.12654320987654322 |
| 0.400000 | 1 | 673 | 0.09212481426448738 | 0.09063893016344725 |
| 0.500000 | -14 | 684 | 0.06432748538011696 | 0.0847953216374269 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.003964 | +0.000351 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003682 | +0.000320 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.003411 | +0.000095 |
| baseline_risk_band | under005 | 29965 | 546 | +0.003381 | +0.002558 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003035 | +0.000287 |
| aux_account_months | zero | 21000 | 2461 | +0.002100 | +0.000117 |
| employment_type | variable_income | 15540 | 2180 | +0.000207 | -0.000247 |
| employment_type | self_employed | 20987 | 2895 | +0.000029 | +0.000259 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.000024 | +0.001754 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000625 | +0.000120 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001152 | +0.000284 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001388 | +0.000279 |
| employment_type | salaried | 33473 | 4432 | -0.001755 | +0.000404 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001776 | +0.000065 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002204 | +0.000250 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002832 | +0.000770 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.002939 | +0.002945 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.008399 | -0.001769 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
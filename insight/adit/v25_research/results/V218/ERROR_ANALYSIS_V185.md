# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 7 | 247 | 0.3765182186234818 | 0.3481781376518219 |
| 0.200000 | -7 | 362 | 0.22099447513812154 | 0.24033149171270718 |
| 0.323686 | 16 | 434 | 0.14285714285714285 | 0.10599078341013825 |
| 0.400000 | 4 | 456 | 0.08552631578947369 | 0.07675438596491228 |
| 0.500000 | 2 | 443 | 0.0744920993227991 | 0.06997742663656885 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002800 | -0.000063 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002770 | -0.000098 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001682 | -0.000158 |
| aux_account_months | zero | 21000 | 2461 | +0.001061 | -0.000270 |
| baseline_risk_band | under005 | 29965 | 546 | +0.001027 | -0.000245 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000623 | -0.000002 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000022 | -0.000095 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000141 | -0.000118 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000225 | -0.000069 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000323 | -0.000042 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000454 | +0.000159 |
| employment_type | variable_income | 15540 | 2180 | -0.000510 | -0.000329 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.000685 | +0.000490 |
| employment_type | self_employed | 20987 | 2895 | -0.001176 | +0.000024 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002027 | -0.000005 |
| employment_type | salaried | 33473 | 4432 | -0.002431 | -0.000090 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.003672 | -0.001716 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.007223 | -0.000547 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
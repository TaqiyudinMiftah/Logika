# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -5 | 180 | 0.37222222222222223 | 0.4 |
| 0.200000 | -3 | 278 | 0.20863309352517986 | 0.21942446043165467 |
| 0.323686 | -2 | 321 | 0.1059190031152648 | 0.11214953271028037 |
| 0.400000 | 4 | 362 | 0.09944751381215469 | 0.08839779005524862 |
| 0.500000 | -1 | 329 | 0.06382978723404255 | 0.0668693009118541 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.004392 | -0.000015 |
| baseline_risk_band | under005 | 29853 | 535 | +0.002483 | -0.000544 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.002446 | +0.000314 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.001999 | +0.000585 |
| employment_type | salaried | 33473 | 4432 | +0.001684 | +0.000141 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000480 | -0.000108 |
| employment_type | variable_income | 15540 | 2180 | +0.000117 | -0.000149 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000349 | -0.000024 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000822 | -0.000006 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000879 | -0.000010 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.001034 | +0.000124 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001114 | +0.000062 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001166 | -0.000022 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001272 | +0.000082 |
| aux_account_months | zero | 21000 | 2461 | -0.001618 | +0.000083 |
| employment_type | self_employed | 20987 | 2895 | -0.001642 | -0.000059 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001790 | +0.000003 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002958 | +0.000008 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
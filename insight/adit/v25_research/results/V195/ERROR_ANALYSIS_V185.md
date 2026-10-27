# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 3 | 79 | 0.5063291139240507 | 0.46835443037974683 |
| 0.200000 | -5 | 138 | 0.2246376811594203 | 0.2608695652173913 |
| 0.323686 | 2 | 157 | 0.12101910828025478 | 0.10828025477707007 |
| 0.400000 | -2 | 186 | 0.08064516129032258 | 0.0913978494623656 |
| 0.500000 | 1 | 149 | 0.087248322147651 | 0.08053691275167785 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003144 | +0.000087 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002362 | +0.000074 |
| aux_account_months | zero | 21000 | 2461 | +0.002165 | -0.000085 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.001135 | +0.000577 |
| baseline_risk_band | under005 | 29965 | 546 | +0.000548 | +0.000872 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000496 | +0.000062 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000094 | +0.000056 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000000 | +0.000000 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000000 | +0.000000 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000000 | +0.000000 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000000 | +0.000000 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000000 | +0.000000 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000000 | +0.000000 |
| employment_type | variable_income | 15540 | 2180 | -0.000758 | -0.000227 |
| employment_type | salaried | 33473 | 4432 | -0.000773 | +0.000037 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.001035 | -0.000091 |
| employment_type | self_employed | 20987 | 2895 | -0.001367 | +0.000204 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.004080 | -0.000615 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
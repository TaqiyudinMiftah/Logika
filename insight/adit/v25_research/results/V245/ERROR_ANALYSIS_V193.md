# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 8 | 257 | 0.38132295719844356 | 0.35019455252918286 |
| 0.200000 | 2 | 403 | 0.228287841191067 | 0.22332506203473945 |
| 0.323686 | 4 | 476 | 0.1323529411764706 | 0.12394957983193278 |
| 0.400000 | 2 | 512 | 0.083984375 | 0.080078125 |
| 0.500000 | 1 | 505 | 0.06930693069306931 | 0.06732673267326733 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003785 | -0.000281 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003724 | -0.000268 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.002931 | +0.000500 |
| baseline_risk_band | under005 | 29938 | 544 | +0.001914 | -0.000741 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.001713 | -0.001547 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001129 | -0.000148 |
| aux_account_months | zero | 21000 | 2461 | +0.001098 | -0.000420 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000514 | -0.000227 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000898 | -0.000215 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.001218 | -0.000713 |
| employment_type | variable_income | 15540 | 2180 | -0.001276 | -0.000422 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001672 | -0.000078 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001836 | -0.000177 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001961 | -0.000136 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001975 | -0.000135 |
| employment_type | self_employed | 20987 | 2895 | -0.002751 | -0.000196 |
| employment_type | salaried | 33473 | 4432 | -0.002932 | -0.000138 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003735 | -0.000914 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
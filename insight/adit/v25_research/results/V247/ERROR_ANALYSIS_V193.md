# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 8 | 275 | 0.38181818181818183 | 0.3527272727272727 |
| 0.200000 | -2 | 416 | 0.2283653846153846 | 0.23317307692307693 |
| 0.323686 | 4 | 483 | 0.12836438923395446 | 0.12008281573498965 |
| 0.400000 | 6 | 526 | 0.09695817490494296 | 0.08555133079847908 |
| 0.500000 | 7 | 524 | 0.07633587786259542 | 0.06297709923664122 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003186 | -0.000295 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002950 | -0.000309 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.000488 | -0.001263 |
| aux_account_months | zero | 21000 | 2461 | +0.000475 | -0.000401 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000187 | -0.000146 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.000096 | -0.000003 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000315 | -0.000793 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000899 | -0.000273 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000968 | -0.000165 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001744 | -0.000878 |
| employment_type | salaried | 33473 | 4432 | -0.001852 | -0.000175 |
| baseline_risk_band | under005 | 29938 | 544 | -0.001864 | -0.001160 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002183 | -0.000117 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002569 | -0.000133 |
| employment_type | self_employed | 20987 | 2895 | -0.002626 | -0.000188 |
| employment_type | variable_income | 15540 | 2180 | -0.002642 | -0.000386 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002666 | -0.000145 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002918 | -0.000168 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
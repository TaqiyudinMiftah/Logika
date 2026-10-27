# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 20 | 277 | 0.4151624548736462 | 0.34296028880866425 |
| 0.200000 | 20 | 480 | 0.24791666666666667 | 0.20625 |
| 0.323686 | -9 | 589 | 0.12563667232597622 | 0.14091680814940577 |
| 0.400000 | 11 | 605 | 0.10413223140495868 | 0.0859504132231405 |
| 0.500000 | -10 | 613 | 0.06035889070146819 | 0.0766721044045677 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29938 | 544 | +0.008221 | +0.003342 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.005945 | +0.000826 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003829 | +0.000353 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003362 | +0.000407 |
| aux_account_months | zero | 21000 | 2461 | +0.002191 | +0.000384 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001346 | +0.000395 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000640 | +0.000080 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000228 | -0.001935 |
| employment_type | self_employed | 20987 | 2895 | -0.000538 | +0.000200 |
| employment_type | salaried | 33473 | 4432 | -0.000793 | +0.000560 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.000857 | +0.002512 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000877 | +0.000435 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001529 | +0.000353 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.001615 | +0.001768 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001774 | +0.000155 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002576 | +0.000266 |
| employment_type | variable_income | 15540 | 2180 | -0.002936 | -0.000147 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.003013 | +0.000887 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 19 | 280 | 0.4035714285714286 | 0.3357142857142857 |
| 0.200000 | 5 | 459 | 0.23093681917211328 | 0.22004357298474944 |
| 0.323686 | -15 | 549 | 0.11293260473588343 | 0.14025500910746813 |
| 0.400000 | -11 | 567 | 0.07936507936507936 | 0.09876543209876543 |
| 0.500000 | -9 | 575 | 0.06086956521739131 | 0.07652173913043478 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.005658 | -0.000797 |
| baseline_risk_band | under005 | 29938 | 544 | +0.005368 | +0.001272 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.004575 | +0.000256 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003362 | +0.000257 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002650 | +0.000211 |
| aux_account_months | zero | 21000 | 2461 | +0.001029 | +0.000036 |
| employment_type | self_employed | 20987 | 2895 | +0.000554 | +0.000230 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000018 | -0.000058 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000439 | +0.000135 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000860 | +0.001497 |
| employment_type | salaried | 33473 | 4432 | -0.001931 | +0.000080 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002056 | -0.000021 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002078 | +0.000107 |
| employment_type | variable_income | 15540 | 2180 | -0.002779 | -0.000110 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002788 | +0.000114 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002901 | -0.000792 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.005393 | +0.001474 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.006881 | +0.000295 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
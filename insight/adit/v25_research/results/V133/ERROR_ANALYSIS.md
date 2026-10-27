# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -5 | 110 | 0.36363636363636365 | 0.4090909090909091 |
| 0.200000 | 2 | 168 | 0.27380952380952384 | 0.2619047619047619 |
| 0.323686 | -18 | 202 | 0.0891089108910891 | 0.1782178217821782 |
| 0.400000 | 1 | 226 | 0.084070796460177 | 0.07964601769911504 |
| 0.500000 | -2 | 234 | 0.06837606837606838 | 0.07692307692307693 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.002831 | -0.001058 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000891 | +0.000113 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000541 | -0.000889 |
| aux_account_months | zero | 21000 | 2461 | +0.000355 | -0.000027 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000226 | +0.000030 |
| employment_type | salaried | 33473 | 4432 | +0.000004 | -0.000019 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.000028 | -0.001525 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000062 | +0.000088 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000672 | -0.000008 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000890 | +0.000464 |
| employment_type | variable_income | 15540 | 2180 | -0.000975 | -0.000274 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001011 | -0.000132 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001286 | -0.000171 |
| employment_type | self_employed | 20987 | 2895 | -0.001476 | -0.000094 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001555 | -0.000715 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001585 | -0.000049 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002276 | -0.000184 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.004606 | -0.000386 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
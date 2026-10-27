# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 4 | 158 | 0.4050632911392405 | 0.379746835443038 |
| 0.200000 | 6 | 258 | 0.2131782945736434 | 0.18992248062015504 |
| 0.323686 | -25 | 321 | 0.102803738317757 | 0.1806853582554517 |
| 0.400000 | 1 | 330 | 0.08484848484848485 | 0.08181818181818182 |
| 0.500000 | -11 | 337 | 0.05341246290801187 | 0.08605341246290801 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.004270 | -0.000123 |
| employment_type | variable_income | 15540 | 2180 | +0.003109 | -0.000400 |
| baseline_risk_band | under005 | 29853 | 535 | +0.002347 | +0.000827 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000860 | -0.001434 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000756 | -0.000214 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000012 | -0.000324 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000012 | -0.000185 |
| aux_account_months | zero | 21000 | 2461 | -0.000022 | -0.000195 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000081 | -0.000152 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000170 | -0.000658 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000414 | -0.000322 |
| employment_type | self_employed | 20987 | 2895 | -0.000455 | +0.000011 |
| employment_type | salaried | 33473 | 4432 | -0.000622 | -0.000123 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000647 | -0.000143 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001273 | -0.000179 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001305 | -0.000461 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.002419 | -0.000014 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.003072 | -0.001272 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 8 | 241 | 0.37344398340248963 | 0.34024896265560167 |
| 0.200000 | -8 | 362 | 0.21823204419889503 | 0.24033149171270718 |
| 0.323686 | 13 | 418 | 0.13875598086124402 | 0.1076555023923445 |
| 0.400000 | 8 | 449 | 0.0935412026726058 | 0.0757238307349666 |
| 0.500000 | 3 | 439 | 0.07517084282460136 | 0.0683371298405467 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003774 | -0.000007 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003749 | -0.000028 |
| aux_account_months | zero | 21000 | 2461 | +0.002309 | -0.000202 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.001796 | +0.000646 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001701 | -0.000170 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001116 | +0.000053 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000393 | -0.000011 |
| baseline_risk_band | under005 | 29965 | 546 | +0.000157 | +0.000099 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000193 | -0.000070 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000247 | -0.000033 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000367 | +0.000005 |
| employment_type | self_employed | 20987 | 2895 | -0.000458 | +0.000038 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000797 | +0.000393 |
| employment_type | variable_income | 15540 | 2180 | -0.001239 | -0.000289 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002032 | +0.000012 |
| employment_type | salaried | 33473 | 4432 | -0.002449 | -0.000011 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.003855 | -0.001337 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005043 | -0.000270 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
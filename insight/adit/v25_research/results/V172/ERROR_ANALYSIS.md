# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -2 | 86 | 0.3488372093023256 | 0.37209302325581395 |
| 0.200000 | 7 | 144 | 0.25 | 0.2013888888888889 |
| 0.323686 | -10 | 177 | 0.11864406779661017 | 0.1751412429378531 |
| 0.400000 | 2 | 211 | 0.11374407582938388 | 0.10426540284360189 |
| 0.500000 | 5 | 219 | 0.0776255707762557 | 0.0547945205479452 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.000955 | -0.001715 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000678 | +0.000237 |
| employment_type | variable_income | 15540 | 2180 | +0.000538 | +0.000263 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000382 | +0.000329 |
| employment_type | self_employed | 20987 | 2895 | +0.000290 | +0.000221 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000226 | -0.000056 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000191 | +0.000176 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000024 | +0.000175 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000019 | +0.000386 |
| aux_account_months | zero | 21000 | 2461 | +0.000000 | +0.000000 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000000 | +0.000000 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000003 | +0.000095 |
| employment_type | salaried | 33473 | 4432 | -0.000226 | +0.000018 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000253 | +0.000461 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000430 | +0.001143 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000506 | +0.000002 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.001298 | +0.001596 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003019 | -0.000239 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
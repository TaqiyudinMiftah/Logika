# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 0 | 61 | 0.3442622950819672 | 0.3442622950819672 |
| 0.200000 | 4 | 102 | 0.2647058823529412 | 0.22549019607843138 |
| 0.323686 | -7 | 128 | 0.1171875 | 0.171875 |
| 0.400000 | -7 | 137 | 0.051094890510948905 | 0.10218978102189781 |
| 0.500000 | -5 | 147 | 0.047619047619047616 | 0.08163265306122448 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.003486 | +0.001396 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.002482 | +0.001124 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.002304 | +0.000592 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001972 | +0.000340 |
| employment_type | variable_income | 15540 | 2180 | +0.001231 | +0.000148 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000828 | +0.000188 |
| employment_type | self_employed | 20987 | 2895 | +0.000538 | +0.000109 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000299 | +0.000208 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000295 | +0.000144 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000296 | +0.000181 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000481 | +0.000097 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000498 | +0.000121 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000501 | +0.000169 |
| employment_type | salaried | 33473 | 4432 | -0.000535 | +0.000180 |
| aux_account_months | zero | 21000 | 2461 | -0.000561 | +0.000115 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000607 | +0.000158 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001249 | +0.000199 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.001745 | +0.000403 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
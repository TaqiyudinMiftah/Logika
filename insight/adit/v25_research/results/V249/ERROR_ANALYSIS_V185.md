# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 12 | 268 | 0.3917910447761194 | 0.34701492537313433 |
| 0.200000 | -3 | 405 | 0.23209876543209876 | 0.23950617283950618 |
| 0.323686 | 5 | 481 | 0.12266112266112267 | 0.11226611226611227 |
| 0.400000 | 2 | 519 | 0.09633911368015415 | 0.09248554913294797 |
| 0.500000 | 5 | 507 | 0.07889546351084813 | 0.06903353057199212 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002438 | -0.000191 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002166 | -0.000221 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001404 | -0.000484 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.000507 | +0.000100 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.000337 | -0.000376 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000098 | -0.000115 |
| aux_account_months | zero | 21000 | 2461 | -0.000294 | -0.000346 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000874 | -0.000216 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001378 | -0.000189 |
| employment_type | variable_income | 15540 | 2180 | -0.001483 | -0.000358 |
| employment_type | self_employed | 20987 | 2895 | -0.002060 | -0.000135 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002249 | -0.000190 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.002332 | -0.001612 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002398 | -0.000049 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002430 | -0.000142 |
| employment_type | salaried | 33473 | 4432 | -0.002463 | -0.000160 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002478 | -0.000128 |
| baseline_risk_band | under005 | 29965 | 546 | -0.005094 | -0.001003 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
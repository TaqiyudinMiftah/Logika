# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 10 | 254 | 0.39763779527559057 | 0.35826771653543305 |
| 0.200000 | 4 | 381 | 0.24146981627296588 | 0.23097112860892388 |
| 0.323686 | 11 | 443 | 0.14446952595936793 | 0.11963882618510158 |
| 0.400000 | 1 | 476 | 0.08403361344537816 | 0.0819327731092437 |
| 0.500000 | -1 | 464 | 0.0625 | 0.06465517241379311 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.001931 | -0.000007 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.001692 | +0.000653 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001373 | -0.000147 |
| aux_allocation_count | zero | 23732 | 2895 | +0.001372 | -0.000002 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000925 | -0.000070 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000624 | -0.000153 |
| baseline_risk_band | under005 | 29938 | 544 | +0.000418 | -0.000142 |
| aux_account_months | zero | 21000 | 2461 | -0.000129 | -0.000146 |
| employment_type | variable_income | 15540 | 2180 | -0.000227 | -0.000319 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000351 | -0.000164 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000527 | -0.000125 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000665 | -0.000116 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000726 | -0.000117 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.001036 | -0.002070 |
| employment_type | salaried | 33473 | 4432 | -0.001337 | -0.000119 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.001607 | +0.000067 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001763 | -0.000985 |
| employment_type | self_employed | 20987 | 2895 | -0.002116 | -0.000020 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
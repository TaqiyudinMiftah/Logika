# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 7 | 256 | 0.3828125 | 0.35546875 |
| 0.200000 | 2 | 400 | 0.225 | 0.22 |
| 0.323686 | 9 | 478 | 0.13807531380753138 | 0.1192468619246862 |
| 0.400000 | -1 | 511 | 0.08023483365949119 | 0.0821917808219178 |
| 0.500000 | 1 | 507 | 0.06903353057199212 | 0.0670611439842209 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.004296 | -0.000303 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003955 | -0.000285 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.002922 | +0.000471 |
| aux_account_months | zero | 21000 | 2461 | +0.001540 | -0.000451 |
| baseline_risk_band | under005 | 29965 | 546 | +0.001153 | -0.000709 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000906 | -0.000133 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.000332 | -0.001631 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000183 | -0.000236 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000838 | -0.000683 |
| employment_type | variable_income | 15540 | 2180 | -0.000967 | -0.000416 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001122 | -0.000213 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001294 | -0.000178 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001452 | -0.000127 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001469 | -0.000135 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001618 | -0.000060 |
| employment_type | self_employed | 20987 | 2895 | -0.002392 | -0.000195 |
| employment_type | salaried | 33473 | 4432 | -0.002805 | -0.000142 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005896 | -0.000815 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
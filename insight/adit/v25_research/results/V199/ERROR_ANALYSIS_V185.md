# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 14 | 264 | 0.39015151515151514 | 0.3371212121212121 |
| 0.200000 | 8 | 427 | 0.234192037470726 | 0.2154566744730679 |
| 0.323686 | 7 | 520 | 0.13653846153846153 | 0.12307692307692308 |
| 0.400000 | 4 | 549 | 0.09836065573770492 | 0.09107468123861566 |
| 0.500000 | -5 | 531 | 0.0696798493408663 | 0.07909604519774012 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| employment_type | variable_income | 15540 | 2180 | +0.002175 | -0.000278 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.001270 | -0.000178 |
| aux_allocation_count | zero | 23732 | 2895 | +0.001222 | -0.000215 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000881 | +0.000196 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000637 | +0.000186 |
| aux_account_months | zero | 21000 | 2461 | +0.000565 | -0.000385 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000542 | +0.000079 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.000530 | +0.000538 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000395 | +0.000161 |
| baseline_risk_band | under005 | 29965 | 546 | -0.000381 | +0.000858 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000589 | +0.000107 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.000960 | -0.000733 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.001014 | +0.000863 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001150 | +0.000252 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001184 | -0.000096 |
| employment_type | self_employed | 20987 | 2895 | -0.001190 | -0.000004 |
| employment_type | salaried | 33473 | 4432 | -0.002749 | +0.000146 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003241 | -0.001212 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
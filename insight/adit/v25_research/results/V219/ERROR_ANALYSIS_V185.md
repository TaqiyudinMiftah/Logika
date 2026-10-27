# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 5 | 268 | 0.373134328358209 | 0.35447761194029853 |
| 0.200000 | 21 | 408 | 0.25980392156862747 | 0.20833333333333334 |
| 0.323686 | 1 | 522 | 0.12452107279693486 | 0.12260536398467432 |
| 0.400000 | 0 | 550 | 0.09636363636363636 | 0.09636363636363636 |
| 0.500000 | -1 | 563 | 0.06216696269982238 | 0.06394316163410302 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29965 | 546 | +0.007541 | +0.001248 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.004743 | -0.001323 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.003205 | +0.001363 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.002204 | +0.000207 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001821 | -0.002557 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.001600 | -0.001108 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000087 | -0.000278 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000255 | -0.000113 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000265 | -0.000245 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000325 | +0.000027 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000614 | +0.000120 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000678 | +0.000111 |
| employment_type | variable_income | 15540 | 2180 | -0.000898 | -0.000414 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001016 | +0.000013 |
| aux_account_months | zero | 21000 | 2461 | -0.001158 | +0.000010 |
| employment_type | salaried | 33473 | 4432 | -0.001751 | +0.000106 |
| employment_type | self_employed | 20987 | 2895 | -0.002321 | +0.000166 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.002390 | +0.000074 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
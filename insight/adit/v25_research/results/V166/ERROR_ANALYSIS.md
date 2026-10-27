# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 65 | 0.3384615384615385 | 0.3230769230769231 |
| 0.200000 | -9 | 111 | 0.15315315315315314 | 0.23423423423423423 |
| 0.323686 | -6 | 131 | 0.10687022900763359 | 0.15267175572519084 |
| 0.400000 | -2 | 160 | 0.1 | 0.1125 |
| 0.500000 | 3 | 137 | 0.072992700729927 | 0.051094890510948905 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.002180 | +0.000258 |
| light_basket_value_entropy_30d | Q4 | 12716 | 3119 | +0.001283 | -0.000075 |
| employment_type | variable_income | 15540 | 2180 | +0.000915 | +0.000115 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000761 | +0.000247 |
| light_basket_value_entropy_30d | Q2 | 12852 | 1854 | +0.000508 | +0.000006 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000376 | +0.000308 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000169 | +0.000002 |
| light_basket_value_entropy_30d | Q1 | 12739 | 1244 | -0.000244 | -0.000074 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000309 | -0.000049 |
| employment_type | self_employed | 20987 | 2895 | -0.000332 | -0.000038 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000378 | -0.000005 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000638 | -0.000016 |
| aux_account_months | zero | 21000 | 2461 | -0.000676 | -0.000050 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000681 | -0.000057 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000712 | -0.000047 |
| employment_type | salaried | 33473 | 4432 | -0.000783 | -0.000068 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000836 | -0.000538 |
| light_basket_value_entropy_30d | missing | 18868 | 963 | -0.000989 | +0.000116 |
| light_basket_value_entropy_30d | Q3 | 12825 | 2327 | -0.001092 | +0.000001 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001350 | +0.000013 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001383 | -0.000006 |
| baseline_risk_band | under005 | 29853 | 535 | -0.002190 | +0.000047 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002349 | -0.000013 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
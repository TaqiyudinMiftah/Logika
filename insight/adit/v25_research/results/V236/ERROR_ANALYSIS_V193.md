# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 15 | 265 | 0.41132075471698115 | 0.35471698113207545 |
| 0.200000 | 9 | 395 | 0.2481012658227848 | 0.22531645569620254 |
| 0.323686 | 10 | 465 | 0.14193548387096774 | 0.12043010752688173 |
| 0.400000 | 6 | 514 | 0.0933852140077821 | 0.08171206225680934 |
| 0.500000 | -1 | 493 | 0.060851926977687626 | 0.06288032454361055 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001532 | -0.000165 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.001258 | -0.000253 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.001059 | -0.000020 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000880 | +0.000010 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.000670 | +0.000545 |
| baseline_risk_band | under005 | 29938 | 544 | +0.000629 | -0.000327 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000390 | -0.000161 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000201 | -0.000194 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000272 | -0.000234 |
| aux_account_months | zero | 21000 | 2461 | -0.000629 | -0.000134 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000633 | -0.000194 |
| employment_type | variable_income | 15540 | 2180 | -0.000702 | -0.000330 |
| employment_type | self_employed | 20987 | 2895 | -0.001106 | -0.000085 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.001547 | -0.002721 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.001727 | -0.000035 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001733 | -0.000164 |
| employment_type | salaried | 33473 | 4432 | -0.001870 | -0.000178 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003318 | -0.001423 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -10 | 225 | 0.32 | 0.36444444444444446 |
| 0.200000 | -1 | 325 | 0.21846153846153846 | 0.22153846153846155 |
| 0.323686 | -19 | 377 | 0.09814323607427056 | 0.14854111405835543 |
| 0.400000 | -3 | 426 | 0.0892018779342723 | 0.09624413145539906 |
| 0.500000 | -1 | 395 | 0.0759493670886076 | 0.07848101265822785 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.004132 | +0.000220 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.003664 | +0.000854 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.003021 | -0.000043 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.002709 | -0.000028 |
| employment_type | salaried | 33473 | 4432 | +0.002416 | +0.000060 |
| employment_type | self_employed | 20987 | 2895 | +0.001259 | -0.000110 |
| employment_type | variable_income | 15540 | 2180 | +0.000597 | +0.000373 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000567 | +0.000073 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000179 | -0.000007 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000129 | +0.000022 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000039 | +0.000072 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000180 | +0.000284 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000311 | +0.000072 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000880 | +0.000094 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001188 | -0.000063 |
| aux_account_months | zero | 21000 | 2461 | -0.001802 | +0.000307 |
| baseline_risk_band | under005 | 29853 | 535 | -0.001849 | +0.000523 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002499 | +0.000151 |
| aux_allocation_count | zero | 23732 | 2895 | -0.003173 | +0.000143 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003373 | +0.000091 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.004251 | -0.000017 |
| orders90_exposure_band | zero | 3428 | 106 | -0.008423 | -0.000193 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
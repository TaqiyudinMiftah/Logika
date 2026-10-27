# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -5 | 143 | 0.35664335664335667 | 0.3916083916083916 |
| 0.200000 | 14 | 213 | 0.25821596244131456 | 0.19248826291079812 |
| 0.323686 | -5 | 274 | 0.10948905109489052 | 0.12773722627737227 |
| 0.400000 | -11 | 344 | 0.0755813953488372 | 0.10755813953488372 |
| 0.500000 | -6 | 338 | 0.0621301775147929 | 0.07988165680473373 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.004816 | -0.000728 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.003811 | -0.000486 |
| employment_type | salaried | 33473 | 4432 | +0.001723 | -0.000325 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.001709 | -0.000332 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.001312 | -0.000373 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.001207 | +0.000093 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000205 | -0.000363 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000170 | -0.000302 |
| employment_type | self_employed | 20987 | 2895 | +0.000126 | -0.000256 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000065 | -0.002492 |
| employment_type | variable_income | 15540 | 2180 | -0.000050 | -0.000094 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000322 | -0.000311 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000528 | +0.000010 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000714 | -0.000087 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000845 | -0.000326 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001346 | -0.000101 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001477 | -0.000302 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.001597 | -0.000295 |
| aux_account_months | zero | 21000 | 2461 | -0.001708 | +0.000045 |
| orders90_exposure_band | zero | 3428 | 106 | -0.002698 | +0.000133 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003431 | -0.001580 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.004007 | -0.001072 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
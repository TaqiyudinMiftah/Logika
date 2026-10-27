# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -10 | 242 | 0.3140495867768595 | 0.35537190082644626 |
| 0.200000 | 3 | 360 | 0.2361111111111111 | 0.22777777777777777 |
| 0.323686 | -15 | 415 | 0.1108433734939759 | 0.14698795180722893 |
| 0.400000 | 2 | 450 | 0.09777777777777778 | 0.09333333333333334 |
| 0.500000 | -5 | 417 | 0.07194244604316546 | 0.08393285371702638 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| beh_spectral_orders_periodicity_strength | Q2 | 11878 | 1863 | +0.007938 | +0.000215 |
| beh_spectral_spend_periodicity_strength | Q1 | 11919 | 1865 | +0.006561 | +0.000052 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.004839 | -0.000122 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.004588 | +0.000377 |
| beh_spectral_orders_power_7d | Q2 | 12110 | 1965 | +0.003959 | -0.000446 |
| beh_spectral_spend_power_14d | Q3 | 12033 | 1934 | +0.003444 | +0.000277 |
| beh_spectral_spend_power_7d | Q2 | 12262 | 1991 | +0.003420 | +0.000392 |
| beh_spectral_orders_power_30d | Q2 | 12179 | 1982 | +0.002749 | +0.000197 |
| beh_spectral_spend_power_30d | Q2 | 11994 | 2003 | +0.002684 | -0.000170 |
| beh_spectral_orders_power_30d | Q1 | 11913 | 1936 | +0.002648 | -0.000309 |
| beh_spectral_orders_dominant_cycle_days | Q3 | 12219 | 1954 | +0.002245 | +0.000419 |
| beh_spectral_spend_power_14d | Q4 | 11974 | 1917 | +0.001986 | -0.000377 |
| beh_spectral_spend_power_7d | Q1 | 12214 | 2109 | +0.001937 | -0.000176 |
| beh_spectral_spend_power_7d | Q4 | 11946 | 1858 | +0.001904 | -0.000415 |
| beh_spectral_spend_dominant_cycle_days | Q3 | 12161 | 1926 | +0.001795 | -0.000062 |
| beh_spectral_spend_periodicity_strength | Q3 | 12256 | 1976 | +0.001740 | -0.000054 |
| beh_spectral_orders_power_7d | Q4 | 12045 | 1878 | +0.001720 | +0.000199 |
| employment_type | variable_income | 15540 | 2180 | +0.001693 | +0.000319 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001608 | +0.000115 |
| employment_type | salaried | 33473 | 4432 | +0.001604 | -0.000119 |
| beh_spectral_orders_power_30d | Q4 | 11906 | 1917 | +0.001565 | -0.000570 |
| beh_spectral_orders_power_7d | Q3 | 12072 | 1975 | +0.001559 | -0.000287 |
| beh_spectral_spend_power_30d | Q3 | 11967 | 1899 | +0.001435 | +0.000048 |
| beh_spectral_spend_dominant_cycle_days | Q4 | 12455 | 2054 | +0.001415 | -0.000050 |
| beh_spectral_orders_power_14d | Q2 | 11970 | 2031 | +0.001002 | -0.000338 |
| beh_spectral_spend_power_30d | Q4 | 11976 | 1911 | +0.000969 | -0.000120 |
| beh_spectral_spend_power_7d | Q3 | 11842 | 1856 | +0.000912 | -0.000214 |
| beh_spectral_orders_dominant_cycle_days | Q2 | 11554 | 1885 | +0.000814 | -0.000432 |
| beh_spectral_orders_power_14d | Q4 | 11944 | 1863 | +0.000749 | +0.000082 |
| beh_spectral_orders_periodicity_strength | Q3 | 12170 | 1919 | +0.000717 | +0.000071 |
| beh_spectral_orders_power_7d | Q1 | 12037 | 1996 | +0.000701 | +0.000208 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000696 | -0.000679 |
| employment_type | self_employed | 20987 | 2895 | +0.000554 | -0.000217 |
| beh_spectral_window_supported | Q4 | 66586 | 8969 | +0.000545 | -0.000055 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000393 | -0.000009 |
| beh_spectral_spend_power_14d | Q1 | 12108 | 1995 | +0.000338 | -0.000405 |
| beh_spectral_spend_power_7d | missing | 21736 | 1693 | +0.000110 | +0.000067 |
| beh_spectral_spend_power_14d | missing | 21736 | 1693 | +0.000110 | +0.000067 |
| beh_spectral_spend_power_30d | missing | 21736 | 1693 | +0.000110 | +0.000067 |
| beh_spectral_spend_periodicity_strength | missing | 21736 | 1693 | +0.000110 | +0.000067 |
| beh_spectral_spend_dominant_cycle_days | missing | 21736 | 1693 | +0.000110 | +0.000067 |
| beh_spectral_orders_power_7d | missing | 21736 | 1693 | +0.000110 | +0.000067 |
| beh_spectral_orders_power_14d | missing | 21736 | 1693 | +0.000110 | +0.000067 |
| beh_spectral_orders_power_30d | missing | 21736 | 1693 | +0.000110 | +0.000067 |
| beh_spectral_orders_periodicity_strength | missing | 21736 | 1693 | +0.000110 | +0.000067 |
| beh_spectral_orders_dominant_cycle_days | missing | 21736 | 1693 | +0.000110 | +0.000067 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000009 | -0.000050 |
| beh_spectral_orders_dominant_cycle_days | Q4 | 12603 | 2051 | -0.000025 | +0.000001 |
| beh_spectral_orders_periodicity_strength | Q1 | 12138 | 2019 | -0.000038 | -0.000237 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000093 | -0.000089 |
| beh_spectral_orders_power_30d | Q3 | 12266 | 1979 | -0.000136 | +0.000345 |
| beh_spectral_spend_periodicity_strength | Q2 | 12042 | 1977 | -0.000136 | -0.000435 |
| beh_spectral_spend_dominant_cycle_days | Q1 | 11453 | 1776 | -0.001084 | +0.000074 |
| beh_spectral_spend_power_14d | Q2 | 12149 | 1968 | -0.001099 | +0.000152 |
| beh_spectral_spend_power_30d | Q1 | 12327 | 2001 | -0.001205 | -0.000141 |
| beh_spectral_window_supported | Q1 | 3414 | 538 | -0.001309 | -0.000310 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001496 | -0.000092 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.001505 | -0.000853 |
| beh_spectral_orders_dominant_cycle_days | Q1 | 11888 | 1924 | -0.001577 | -0.000317 |
| aux_account_months | zero | 21000 | 2461 | -0.001629 | +0.000057 |
| beh_spectral_spend_dominant_cycle_days | Q2 | 12195 | 2058 | -0.001634 | -0.000467 |
| beh_spectral_orders_power_14d | Q3 | 12154 | 1940 | -0.001812 | -0.000087 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001946 | -0.000144 |
| beh_spectral_orders_power_14d | Q1 | 12196 | 1980 | -0.002251 | -0.000075 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002324 | +0.000094 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.002683 | -0.000135 |
| baseline_risk_band | under005 | 29853 | 535 | -0.002920 | -0.000152 |
| aux_allocation_count | zero | 23732 | 2895 | -0.002942 | -0.000123 |
| beh_spectral_spend_periodicity_strength | Q4 | 12047 | 1996 | -0.002956 | +0.000156 |
| beh_spectral_orders_periodicity_strength | Q4 | 12078 | 2013 | -0.003477 | -0.000465 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.
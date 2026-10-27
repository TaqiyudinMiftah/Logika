# V167: V25 plus V80 spectral features only, without V19/V40 pricing

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities; extra_inputs=['beh_spectral_spend_power_7d', 'beh_spectral_spend_power_14d', 'beh_spectral_spend_power_30d', 'beh_spectral_spend_periodicity_strength', 'beh_spectral_spend_dominant_cycle_days', 'beh_spectral_orders_power_7d', 'beh_spectral_orders_power_14d', 'beh_spectral_orders_power_30d', 'beh_spectral_orders_periodicity_strength', 'beh_spectral_orders_dominant_cycle_days', 'beh_spectral_window_supported']; undefined exposure stays missing; event cache or training-only reference

OOF KS=0.53079546; delta vsV25=+0.00051080; AUC=0.84697838; deltaAUC=-0.00005090. KS wins 3/5.95% deltaKS CI=[-0.001652640823324003, 0.002958116915543895].

Correction train-validation KS gap=0.01610104 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V167_V25_modeling.csv` SHA256 `9f42a732b0001fa1d2bca9fab5be952a91c7c57952aebf9e48199bda90b6f0ba`.

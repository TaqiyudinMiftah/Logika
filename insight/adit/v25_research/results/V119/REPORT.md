# V119: Exact feature union V25 + V19 + V40 + V80, original frozen offset pipeline

Frozen Kumo Large90 + spline logit-offset correction12/new features + fixed spline77

**Rumus model:** `p=.75*sigmoid(z+strength*(beta0+beta_z*(z-mean_train)/sd_train+phi(x)@beta_features))+.25*p_spline77; z=logit(clip(p_Kumo,1e-6,1-1e-6)); mean test predictions of five folds`

**Objective:** `mean(logaddexp(0,z+Xbeta)-y*(z+Xbeta)) + penalty/2*sum(beta[1:]**2); fit full correction, apply strength only at inference`

**Preprocessing:** Signed log(sign(x)*log1p(abs(x))), correction-training median imputation, quadratic quantile spline3knots, standardization; missing bits and categorical one-hot; training-fold fit only

**Fold:** Original identical5fold StratifiedKFold seed31027; per fold50k Kumo context,6k correction fit,14k validation; no automatic competition submission

Penalti L2=0.003; strength=0.5; backbone100% frozen; spline77 tetap.

OOF KS=0.53039554; delta KS=+0.00109607; AUC=0.84679671; delta AUC=-0.00040598.

Bootstrap CI95% delta KS=[-0.001273662618276969, 0.003692242149582417]; delta AUC=[-0.0007425084138827975, -2.3475068379003053e-05].

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

**Bukti:** INCONCLUSIVE_OR_NEGATIVE. Risiko overfitting: HIGH_SEARCH_MULTIPLICITY; reduce only with independent time/seed evidence.

Validasi development berulang; CI bersyarat pada fitted scores, tanpa koreksi pemilihan kandidat. Skor public hanya untuk hash submission yang sama persis.

| Fold | KS | Delta KS | AUC | Delta AUC |
|---|---:|---:|---:|---:|
| 0 | 0.52358108 | -0.00460692 | 0.84366892 | -0.00020335 |
| 1 | 0.53460272 | -0.00285328 | 0.84961561 | -0.00040113 |
| 2 | 0.54156040 | +0.00149712 | 0.84953817 | -0.00081030 |
| 3 | 0.53238380 | +0.00216398 | 0.84341969 | -0.00038939 |
| 4 | 0.53163040 | +0.00302455 | 0.84787089 | -0.00024880 |

## Definisi fitur tambahan

### risk_conditional_negative_months

Metode/keluarga: conditional_allocation; status: FOLD_LOCAL.

Rumus: number of eligible observed negative-CF months

Sumber: Monthly eligible repayment events aligned to complete due-month account cashflow

Cutoff: Events observed at selected cutoff; priors fitted on correction-training borrowers only

Batasan: Allocation delay is not DPD. DPD is maximum by selected cutoff, not necessarily observed during due month. Monthly dependence can understate posterior uncertainty.

### risk_conditional_nonnegative_months

Metode/keluarga: conditional_allocation; status: FOLD_LOCAL.

Rumus: number of eligible observed nonnegative-CF months

Sumber: Monthly eligible repayment events aligned to complete due-month account cashflow

Cutoff: Events observed at selected cutoff; priors fitted on correction-training borrowers only

Batasan: Allocation delay is not DPD. DPD is maximum by selected cutoff, not necessarily observed during due month. Monthly dependence can understate posterior uncertainty.

### risk_underwriting_price_surprise

Metode/keluarga: pricing; status: FOLD_LOCAL.

Rumus: actual_annual_interest_rate - crossfit_E(rate|income,amount,burden,bureau,employment,purpose)

Sumber: Application/account/bureau/repayment and audited V7 inputs

Cutoff: All regressions cross-fitted on correction rows; no default labels

Batasan: Pricing is policy generated; residual is association, not causal risk. DPD branches quarantined.

### risk_underwriting_price_surprise_scaled

Metode/keluarga: pricing; status: FOLD_LOCAL.

Rumus: pricing residual / training residual SD

Sumber: Application/account/bureau/repayment and audited V7 inputs

Cutoff: All regressions cross-fitted on correction rows; no default labels

Batasan: Pricing is policy generated; residual is association, not causal risk. DPD branches quarantined.

### discoverq_pricing_dpd_surprise

Metode/keluarga: pricing_dpd; status: FOLD_LOCAL.

Rumus: -log(P(actual_APR_tier|credit characteristics plus current-snapshot historical DPD)); cross-fitted

Sumber: Raw basket masks and audited fold-local financial features

Cutoff: References fitted only on correction rows; nuisance3-fold crossfit; basket population capped before query decision month

Batasan: Tier is policy-generated; do not infer causal risk. No default labels in nuisance fitting.

### discoverq_pricing_dpd_expected_gap

Metode/keluarga: pricing_dpd; status: FOLD_LOCAL.

Rumus: actual_APR - expected_APR conditioned additionally on DPD; snapshot provenance unverified

Sumber: Raw basket masks and audited fold-local financial features

Cutoff: References fitted only on correction rows; nuisance3-fold crossfit; basket population capped before query decision month

Batasan: Tier is policy-generated; do not infer causal risk. No default labels in nuisance fitting.

### beh_spectral_spend_power_7d

Metode/keluarga: spectral; status: IMPLEMENTED.

Rumus: Hann-window detrended180-full-day spend periodogram; power_7d; bands6:8d,12:16d,25:35d; dominant period3:90d

Sumber: account_behavior, previous_applications, selected applications, repayment_history, ecommerce_transactions

Cutoff: Own events before selected decision; complete calendar account months; population references only correction customers with selected decision before query month

Batasan: At least8 completed orders and first recorded event before180d window; first event is only coverage proxy. Current incomplete calendar day excluded.

### beh_spectral_spend_power_14d

Metode/keluarga: spectral; status: IMPLEMENTED.

Rumus: Hann-window detrended180-full-day spend periodogram; power_14d; bands6:8d,12:16d,25:35d; dominant period3:90d

Sumber: account_behavior, previous_applications, selected applications, repayment_history, ecommerce_transactions

Cutoff: Own events before selected decision; complete calendar account months; population references only correction customers with selected decision before query month

Batasan: At least8 completed orders and first recorded event before180d window; first event is only coverage proxy. Current incomplete calendar day excluded.

### beh_spectral_spend_power_30d

Metode/keluarga: spectral; status: IMPLEMENTED.

Rumus: Hann-window detrended180-full-day spend periodogram; power_30d; bands6:8d,12:16d,25:35d; dominant period3:90d

Sumber: account_behavior, previous_applications, selected applications, repayment_history, ecommerce_transactions

Cutoff: Own events before selected decision; complete calendar account months; population references only correction customers with selected decision before query month

Batasan: At least8 completed orders and first recorded event before180d window; first event is only coverage proxy. Current incomplete calendar day excluded.

### beh_spectral_spend_periodicity_strength

Metode/keluarga: spectral; status: IMPLEMENTED.

Rumus: Hann-window detrended180-full-day spend periodogram; periodicity_strength; bands6:8d,12:16d,25:35d; dominant period3:90d

Sumber: account_behavior, previous_applications, selected applications, repayment_history, ecommerce_transactions

Cutoff: Own events before selected decision; complete calendar account months; population references only correction customers with selected decision before query month

Batasan: At least8 completed orders and first recorded event before180d window; first event is only coverage proxy. Current incomplete calendar day excluded.

### beh_spectral_spend_dominant_cycle_days

Metode/keluarga: spectral; status: IMPLEMENTED.

Rumus: Hann-window detrended180-full-day spend periodogram; dominant_cycle_days; bands6:8d,12:16d,25:35d; dominant period3:90d

Sumber: account_behavior, previous_applications, selected applications, repayment_history, ecommerce_transactions

Cutoff: Own events before selected decision; complete calendar account months; population references only correction customers with selected decision before query month

Batasan: At least8 completed orders and first recorded event before180d window; first event is only coverage proxy. Current incomplete calendar day excluded.

### beh_spectral_orders_power_7d

Metode/keluarga: spectral; status: IMPLEMENTED.

Rumus: Hann-window detrended180-full-day orders periodogram; power_7d; bands6:8d,12:16d,25:35d; dominant period3:90d

Sumber: account_behavior, previous_applications, selected applications, repayment_history, ecommerce_transactions

Cutoff: Own events before selected decision; complete calendar account months; population references only correction customers with selected decision before query month

Batasan: At least8 completed orders and first recorded event before180d window; first event is only coverage proxy. Current incomplete calendar day excluded.

### beh_spectral_orders_power_14d

Metode/keluarga: spectral; status: IMPLEMENTED.

Rumus: Hann-window detrended180-full-day orders periodogram; power_14d; bands6:8d,12:16d,25:35d; dominant period3:90d

Sumber: account_behavior, previous_applications, selected applications, repayment_history, ecommerce_transactions

Cutoff: Own events before selected decision; complete calendar account months; population references only correction customers with selected decision before query month

Batasan: At least8 completed orders and first recorded event before180d window; first event is only coverage proxy. Current incomplete calendar day excluded.

### beh_spectral_orders_power_30d

Metode/keluarga: spectral; status: IMPLEMENTED.

Rumus: Hann-window detrended180-full-day orders periodogram; power_30d; bands6:8d,12:16d,25:35d; dominant period3:90d

Sumber: account_behavior, previous_applications, selected applications, repayment_history, ecommerce_transactions

Cutoff: Own events before selected decision; complete calendar account months; population references only correction customers with selected decision before query month

Batasan: At least8 completed orders and first recorded event before180d window; first event is only coverage proxy. Current incomplete calendar day excluded.

### beh_spectral_orders_periodicity_strength

Metode/keluarga: spectral; status: IMPLEMENTED.

Rumus: Hann-window detrended180-full-day orders periodogram; periodicity_strength; bands6:8d,12:16d,25:35d; dominant period3:90d

Sumber: account_behavior, previous_applications, selected applications, repayment_history, ecommerce_transactions

Cutoff: Own events before selected decision; complete calendar account months; population references only correction customers with selected decision before query month

Batasan: At least8 completed orders and first recorded event before180d window; first event is only coverage proxy. Current incomplete calendar day excluded.

### beh_spectral_orders_dominant_cycle_days

Metode/keluarga: spectral; status: IMPLEMENTED.

Rumus: Hann-window detrended180-full-day orders periodogram; dominant_cycle_days; bands6:8d,12:16d,25:35d; dominant period3:90d

Sumber: account_behavior, previous_applications, selected applications, repayment_history, ecommerce_transactions

Cutoff: Own events before selected decision; complete calendar account months; population references only correction customers with selected decision before query month

Batasan: At least8 completed orders and first recorded event before180d window; first event is only coverage proxy. Current incomplete calendar day excluded.

### beh_spectral_window_supported

Metode/keluarga: spectral; status: IMPLEMENTED.

Rumus: 1 if first recorded transaction predates180-day full-calendar-day window

Sumber: account_behavior, previous_applications, selected applications, repayment_history, ecommerce_transactions

Cutoff: Own events before selected decision; complete calendar account months; population references only correction customers with selected decision before query month

Batasan: 

## Artefak

Submission: `v7/submissions/submission_V119_goal054_offset_v7.csv`

SHA256: `3211580568f6e7a9dc3a21b07505ba9e4e9b5b56ac068602aa542d2b7c6ce414`

OOF: `v7/runs/V119/OOF.csv`

Checkpoint: `v7/runs/V119/foldN_correction.joblib`

Provenance PIT tersertifikasi: False; fitur baru snapshot-dependent: True.

## Comparison against selected parents

Against **V25**: deltaKS=+0.00011088; deltaAUC=-0.00023257; KS wins 3/5; paired95% deltaKS CI=[-0.0019630984689662846, 0.0025804036925993947]; head training-validation gap delta=+0.00756565.

Against **V19**: deltaKS=-0.00008978; deltaAUC=-0.00016171; KS wins 2/5; paired95% deltaKS CI=[-0.002373673275251356, 0.002052179682314431]; head training-validation gap delta=+0.00637420.

Against **V40**: deltaKS=-0.00017456; deltaAUC=-0.00017407; KS wins 2/5; paired95% deltaKS CI=[-0.002452985001393293, 0.0021338717917877945]; head training-validation gap delta=+0.00740399.

Against **V80**: deltaKS=-0.00034138; deltaAUC=-0.00027215; KS wins 2/5; paired95% deltaKS CI=[-0.0020554808509000433, 0.0014437045988686592]; head training-validation gap delta=+0.00466410.

V25 remains public incumbent. Public leaderboard score for V119 is unknown. Selection on reused development folds and inherited DPD snapshot uncertainty remain.

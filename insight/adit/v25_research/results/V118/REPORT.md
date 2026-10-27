# V118: Exact V25 coverage plus V21 bureau-repayment disagreement feature union

Frozen Kumo Large90 + spline logit-offset correction12/new features + fixed spline77

**Rumus model:** `p=.75*sigmoid(z+strength*(beta0+beta_z*(z-mean_train)/sd_train+phi(x)@beta_features))+.25*p_spline77; z=logit(clip(p_Kumo,1e-6,1-1e-6)); mean test predictions of five folds`

**Objective:** `mean(logaddexp(0,z+Xbeta)-y*(z+Xbeta)) + penalty/2*sum(beta[1:]**2); fit full correction, apply strength only at inference`

**Preprocessing:** Signed log(sign(x)*log1p(abs(x))), correction-training median imputation, quadratic quantile spline3knots, standardization; missing bits and categorical one-hot; training-fold fit only

**Fold:** Original identical5fold StratifiedKFold seed31027; per fold50k Kumo context,6k correction fit,14k validation; no automatic competition submission

Penalti L2=0.003; strength=0.5; backbone100% frozen; spline77 tetap.

OOF KS=0.53017394; delta KS=+0.00087447; AUC=0.84695133; delta AUC=-0.00025136.

Bootstrap CI95% delta KS=[-0.0010806087005545805, 0.0029077189418285783]; delta AUC=[-0.00047824626275466133, -2.527056050134207e-05].

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

**Bukti:** INCONCLUSIVE_OR_NEGATIVE. Risiko overfitting: HIGH_SEARCH_MULTIPLICITY; reduce only with independent time/seed evidence.

Validasi development berulang; CI bersyarat pada fitted scores, tanpa koreksi pemilihan kandidat. Skor public hanya untuk hash submission yang sama persis.

| Fold | KS | Delta KS | AUC | Delta AUC |
|---|---:|---:|---:|---:|
| 0 | 0.52801413 | -0.00017387 | 0.84398213 | +0.00010987 |
| 1 | 0.53813434 | +0.00067834 | 0.84968834 | -0.00032839 |
| 2 | 0.54125993 | +0.00119664 | 0.84967104 | -0.00067743 |
| 3 | 0.53113853 | +0.00091872 | 0.84375206 | -0.00005702 |
| 4 | 0.53131741 | +0.00271156 | 0.84781922 | -0.00030047 |

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

### riskq_bureau_repayment_disagreement

Metode/keluarga: bureau_dpd; status: FOLD_LOCAL.

Rumus: actual_latest_provider_mean_score - crossfit_prediction(score|DPD,utilization,freshness,history_count)

Sumber: Application/account/bureau/repayment and audited V7 inputs

Cutoff: All regressions cross-fitted on correction rows; no default labels

Batasan: Pricing is policy generated; residual is association, not causal risk. DPD branches quarantined.

## Artefak

Submission: `v7/submissions/submission_V118_goal054_offset_v7.csv`

SHA256: `d29b19c2e339b40feaff26fe8c98d1a3b344a5da2637f236576952bf508d26d3`

OOF: `v7/runs/V118/OOF.csv`

Checkpoint: `v7/runs/V118/foldN_correction.joblib`

Provenance PIT tersertifikasi: False; fitur baru snapshot-dependent: True.

## Matched comparisons with selected parents

Against **V25**: deltaKS=-0.00011072, deltaAUC=-0.00007795, KS wins 4/5. 95% deltaKS CI=[-0.0009350792475468827, 0.0012505134023324056]. Head optimism gap delta=+0.00057992.

Against **V21**: deltaKS=+0.00000306, deltaAUC=-0.00009018, KS wins 3/5. 95% deltaKS CI=[-0.0010923276077740075, 0.0014756092297692144]. Head optimism gap delta=+0.00247263.

No automatic public-incumbent replacement; repeated-fold selection and inherited snapshot uncertainty remain.

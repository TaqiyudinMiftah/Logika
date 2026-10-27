# V25: Coverage-only control for conditional Empirical Bayes V14

Frozen Kumo Large90 + spline logit-offset correction12/new features + fixed spline77

**Rumus model:** `p=.75*sigmoid(z+strength*(beta0+beta_z*(z-mean_train)/sd_train+phi(x)@beta_features))+.25*p_spline77; z=logit(clip(p_Kumo,1e-6,1-1e-6)); mean test predictions of five folds`

**Objective:** `mean(logaddexp(0,z+Xbeta)-y*(z+Xbeta)) + penalty/2*sum(beta[1:]**2); fit full correction, apply strength only at inference`

**Preprocessing:** Signed log(sign(x)*log1p(abs(x))), correction-training median imputation, quadratic quantile spline3knots, standardization; missing bits and categorical one-hot; training-fold fit only

**Fold:** Original identical5fold StratifiedKFold seed31027; per fold50k Kumo context,6k correction fit,14k validation; no automatic competition submission

Penalti L2=0.003; strength=0.5; backbone100% frozen; spline77 tetap.

OOF KS=0.53028466; delta KS=+0.00098519; AUC=0.84702928; delta AUC=-0.00017341.

Bootstrap CI95% delta KS=[-0.001164102953087115, 0.0027360623485143385]; delta AUC=[-0.000382120881373188, 4.23609034069793e-05].

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

**Bukti:** INCONCLUSIVE_OR_NEGATIVE. Risiko overfitting: HIGH_SEARCH_MULTIPLICITY; reduce only with independent time/seed evidence.

Validasi development berulang; CI bersyarat pada fitted scores, tanpa koreksi pemilihan kandidat. Skor public hanya untuk hash submission yang sama persis.

| Fold | KS | Delta KS | AUC | Delta AUC |
|---|---:|---:|---:|---:|
| 0 | 0.52847991 | +0.00029191 | 0.84406978 | +0.00019752 |
| 1 | 0.53799917 | +0.00054317 | 0.84960495 | -0.00041178 |
| 2 | 0.54029824 | +0.00023495 | 0.84985691 | -0.00049156 |
| 3 | 0.53065206 | +0.00043224 | 0.84386336 | +0.00005428 |
| 4 | 0.52921975 | +0.00061390 | 0.84789671 | -0.00022299 |

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

## Artefak

Submission: `v7/submissions/submission_V25_goal054_offset_v7.csv`

SHA256: `e7f0748c6c8ded776f0a6a03bdebcd0787cf6c0057d0b078c31196fb34f58efe`

OOF: `v7/runs/V25/OOF.csv`

Checkpoint: `v7/runs/V25/foldN_correction.joblib`

Provenance PIT tersertifikasi: False; fitur baru snapshot-dependent: None.

# V00: Exact locked offset control

Frozen Kumo Large90 + spline logit-offset correction12/new features + fixed spline77

**Rumus model:** `p=.75*sigmoid(z+strength*(beta0+beta_z*(z-mean_train)/sd_train+phi(x)@beta_features))+.25*p_spline77; z=logit(clip(p_Kumo,1e-6,1-1e-6)); mean test predictions of five folds`

**Objective:** `mean(logaddexp(0,z+Xbeta)-y*(z+Xbeta)) + penalty/2*sum(beta[1:]**2); fit full correction, apply strength only at inference`

**Preprocessing:** Signed log(sign(x)*log1p(abs(x))), correction-training median imputation, quadratic quantile spline3knots, standardization; missing bits and categorical one-hot; training-fold fit only

**Fold:** Original identical5fold StratifiedKFold seed31027; per fold50k Kumo context,6k correction fit,14k validation; no automatic competition submission

Penalti L2=0.003; strength=0.5; backbone100% frozen; spline77 tetap.

OOF KS=0.52929947; delta KS=+0.00000000; AUC=0.84720269; delta AUC=+0.00000000.

Bootstrap CI95% delta KS=[0.0, 0.0]; delta AUC=[0.0, 0.0].

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

**Bukti:** EXACT_CONTROL. Risiko overfitting: HIGH_SEARCH_MULTIPLICITY; reduce only with independent time/seed evidence.

Validasi development berulang; CI bersyarat pada fitted scores, tanpa koreksi pemilihan kandidat. Skor public hanya untuk hash submission yang sama persis.

| Fold | KS | Delta KS | AUC | Delta AUC |
|---|---:|---:|---:|---:|
| 0 | 0.52818800 | +0.00000000 | 0.84387226 | +0.00000000 |
| 1 | 0.53745600 | +0.00000000 | 0.85001673 | +0.00000000 |
| 2 | 0.54006328 | +0.00000000 | 0.85034847 | +0.00000000 |
| 3 | 0.53021982 | +0.00000000 | 0.84380908 | +0.00000000 |
| 4 | 0.52860585 | +0.00000000 | 0.84811969 | +0.00000000 |

## Definisi fitur tambahan

Tidak ada fitur tambahan atau katalog tidak tersedia; lihat definisi baseline yang dikunci.

## Artefak

Submission: `v7/submissions/submission_V00_goal054_offset_v7.csv`

SHA256: `5d4d8b81e781354a3d1302aadfe93a1534e42e35279ba4867a7b4d0e1990dd33`

OOF: `v7/runs/V00/OOF.csv`

Checkpoint: `v7/runs/V00/foldN_correction.joblib`

Provenance PIT tersertifikasi: False; fitur baru snapshot-dependent: None.

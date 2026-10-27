# Five additional V25 experiments: V186-V190

Status: executable queue, after pending fixed-score V185. No numerical gains are assumed. Kumo100% frozen; original50kcontext,6k correction and14k outer-validation unchanged; original spline77 and75:25 final weights retained. Every run receives its own OOF, fivefold checkpoint, submission, paired300-resample CI, segment/error analysis, hash and GitHub report. Existing source data/snapshot uncertainties are inherited, not certified away.

The repeated development search already contains185 completed runs; conditional bootstrap does not correct model-selection/retraining uncertainty. V25 remains the public incumbent0.56822. No automatic Kaggle upload.

## V186: Inner-crossfitted restrained affine calibration before fixed spline fusion

V178 improves4/5folds but loses4 defaults at top20%; alignment across expert/fold scores may affect pooled ranking. V163 only class-prior shift was exactly baseline. This tests empirical crossfitted slope+intercept, not theoretical prior correction. Positive monotone calibration alone cannot improve standalone within-fold KS; any change must arise from fusion or fold alignment.

Formula: `3-fold inner OOF corrected Kumo on6k; fit a,b by meanBCE(a+b*logit(p_inner))+.02/2*(a^2+(b-1)^2), a in[-.5,.5], b in[.5,1.5]; pK=sigmoid(z25+.5*(a+(b-1)*z25)); final=.75*pK+.25*spline77`

## V187: Delete-one-third bagging of V25 correction heads

V167 train/heldout head KS gap .016101 vsV25 .011108. Test correction-estimation variance reduction rather than adding Kumo features, which decreased KS in V169:176. Distinct from V173 context bagging;50k Kumo context stays unchanged.

Formula: `Split6k correction into3 disjoint query blocks; train3 independent V25 heads on complementary4k rows; pK=mean_j sigmoid(logit(Kumo)+.5*X_j*beta_j); final=.75*pK+.25*spline77`

## V188: Baseline-orthogonalized spectral innovation residual

V167 spectral addition has strongest KS .530795 but higher head gap; V179 group ridge andV180 reliability residual did not improve KS. This tests only spectral basis innovation after ridge projection on existing V25 design, including missingness, rather than refitting redundant coefficients. All projections use6k correction only.

Formula: `B=(X25^T*X25/n+.01I)^(-1)*X25^T*S9/n; R=S9-X25*B; fit gamma by BCE(z25+R*gamma)+.03/2||gamma||^2+.2/2 mean((R*gamma)^2); eta=z25+.25*R*gamma; final=.75*sigmoid(eta)+.25*spline77`

## V189: Inner-OOF hierarchical employment and history calibration

V167 no-allocation segment deltaKS -.002942 vs +.004588 at1:3 observations; V178 zero-history remains -.000260. Partial pooling learns modest group bias from strictly inner-held-out correction predictions, not outer error-derived gates or separate small Kumo backbones. Sparse groups shrink to global correction.

Formula: `On inner OOF z fit additive b0+b_employment+b_history with meanBCE(z+delta)+.02/2*b0^2+.04/2*sum(b_group^2); history=zero/1_to3/over3/missing allocation. eta=z25+.5*clip(delta,-.25,.25); final=.75*sigmoid(eta)+.25*spline77; unseen levels use global fallback`

## V190: Regularized worst-group regret V25 correction

KS gains differ by history coverage while pooled improvements remain small. Test a group-robust regret objective with existing90-coefficient V25 head, preserving dimensionality; distinct from prior global L2 and ranking-surrogate objectives. Group regret baseline avoids prioritizing high-entropy groups solely because of prevalence.

Formula: `g=employment x allocation-present; merge correction groups n<200 to OTHER; Rg(beta)=mean_g[BCE(zK+Xbeta)-BCE(zK)]; minimize .05*log(mean_g exp(Rg/.05))+.003/2||beta[1:]||^2; pK=sigmoid(zK+.5*Xbeta); final=.75*pK+.25*spline77`

## Literature and limits

Calibration is motivated by Guo et al., ICML2017, https://proceedings.mlr.press/v70/guo17a.html . It does not promise a KS improvement and positive monotone calibration alone preserves within-fold rank. V186 tests calibration before mixing different scores.

V190 adapts the idea of regularized group robustness from Sagawa et al., ICLR2020, https://arxiv.org/abs/1911.08731 . It uses a different group-regret loss on this dataset, not a reproduction or a transferred performance claim.

## Execution

One real model worker at a time, two CPU threads. Frozen Kumo scores are reused, so these five jobs need no new GPU context allocation. Resource supervisor waits until at least0.9GiB effective available RAM (bounded conservative file-cache allowance) before each real fit; per-process RSS limit2GiB. No dummy memory buffers. A new shared GitHub HEAD stops unreviewed work; completed run is audited and pushed before next run.

Inner folds: stratified3fold seed52027+outer_fold, fit4000/query2000. No original spline predictions on those6000 rows are used for calibration because spline was trained on them. Baseline corrected Kumo and all encoder/scalers are refitted inside each inner split. Test and outer-validation features never fit new preprocessing. Original inherited latent-feature provenance remains a limitation.

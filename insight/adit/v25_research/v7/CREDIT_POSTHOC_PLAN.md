# V262: conservative credit-aware post-hoc on V25

User requested an actual V25 experiment and submission. The friend's exact method
has not been identified; this is our operational interpretation, not its reproduction.
All settings were fixed before opening V262 outer results. V25 is preserved byte-for-byte.

Use the existing seed31027 five folds (50k Kumo context / 6k correction / 14k validation).
Kumo backbone, context scores, V25 head and 77-input spline stay fixed. Three inner
folds (seed52027+outerfold, 4000fit/2000query) produce honest corrected-Kumo predictions
on the 6k correction rows. Do not train on spline predictions for these rows: the
spline already saw their labels. Inner heads are retained privately for replay.

Four interactions, with no additional raw data:

- `late30_x_allocation_reliability = z(late30_posterior)*(r-mean_train(r))`,
  `r=n_allocation/(n_allocation+10)`. Requires positive observed allocation count.
  This is a coverage proxy; allocation count is NOT the late30 posterior denominator.
- `installment_x_conditional_deficit = z(installment/income)*(q-mu)*m/(m+3)`.
  `m=n_negative+n_nonnegative`, `mu=(sum_train(n_negative)+.5)/(sum_train(m)+1)`,
  `q=(n_negative+2*mu)/(m+2)`. Counts refer to eligible repayment/account-matched
  borrower-months, not all account months. Require m>0 and observed burden.
- `installment_x_outflow_pressure = z(installment/income)*z(outflow/income)*a/(a+3)`,
  a=observed account months. Require both ratios observed and a>0. Existing account
  outflows are not added to loan payments, so this interaction introduces no
  double-counted obligation in an affordability denominator.
- `total_burden_x_sparse_allocation = z(total_burden/income)*(mean_train(r)-r)`.
  Observed n=0 is valid absence of history; missing n is unsupported.

Here z(v)=clip((signed_log1p(v)-training_median)/training_std,-3,3); references use
only the 6k correction rows. Undefined inputs get zero INTERACTION through explicit
support masks, not zero raw ratios. Clip interactions to[-4,4]. On supported training
rows project each against the original V25 basis using ridge .001. Subtract its
main-effect projection, clip residuals to[-4,4], and apply the same support masks.
This prevents a four-term experiment from simply re-fitting broad main effects.

Fit four coefficients gamma (no intercept):

`mean_BCE(inner_OOF_logit + R@gamma) + .03/2*||gamma||^2 + .2/2*mean((R@gamma)^2)`.

Inference: `eta=logit(p_V25_corrected)+.25*clip(R@gamma,-.3,.3)`;
`p=.75*sigmoid(eta)+.25*p_original_spline`; test is the mean of five fold predictions.
Maximum extra corrected-Kumo logit is .075. No weight or hyperparameter grid.

Prior residual audit found no statistically supported one-dimensional residual
after score matching; this warrants a small conditional experiment, not an assumed
gain. Peer credit-agreement tensor experiments were negative. Latest peer25k-context
deployment public .56793 remains below V25 .56822. Its training recipe is different
and is not silently imported into this experiment.

Evaluate pooled/fold KS and AUC, paired bootstrap, error segments and score direction.
All70k rows are reused development. Inner cross-fitting protects rowwise teacher
isolation, not selection from hundreds of previous experiments. Bootstrap intervals
exclude selection/refitting variability. Legacy DPD snapshot/global-reference
uncertainties are inherited; no new point-in-time certification is claimed.

Private artifacts: v7/runs/V262/{OOF.csv,fold*_correction.joblib,fold*_test.parquet},
v7/submissions/submission_V262_V25_conservative.csv. Public: source, formulas,
aggregate results, negative findings and artifact hashes. No automatic Kaggle upload.

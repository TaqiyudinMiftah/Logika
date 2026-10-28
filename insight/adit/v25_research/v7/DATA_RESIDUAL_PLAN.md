# Incremental information after existing V25 representation

Motivation: V243's eight support inputs lowered pooled KS from0.53028466 to0.52934705 and AUC from0.84702928 to0.84692008. Marginal association did not establish incremental information. V25 already encodes transaction exposure and source uncertainty.

V251 preserves the original head and adds8 support residuals. V252 preserves it and adds16 support/value residuals. V253 uses the same16 inputs without projection. These test a specific overlap/capacity hypothesis. They are not independent holdouts.

Let X be the existing90-column V25 correction basis and U standardized signedlog new inputs, both fit only on6k correction rows. For projected candidates A=argmin||U-XA||^2 via SVD with rcond1e-6, R=U-XA. Never rescale R to unit variance: that could amplify near-redundant noise. V253 takes R=U.

The baseline coefficients remain exactly fixed. Fit gamma by meanBCE(eta0+Rgamma)+.01/2||gamma||^2, eta0=KumoLogit+.5Xbeta_V25. At inference use eta0+.25Rgamma, then the unchanged75:25 original spline fusion. The original head has already seen these6k labels; only outer-held-out results can assess utility. No query target enters projection or scaling.

Undefined raw measurements remainNaN; model-basis centering supplies0 before projection and all explicit availability controls remain. The projected residual can be nonzero when raw evidence is unavailable, because it subtracts the training-expected value conditional onX. This is a model feature, not an imputed physical measurement.

Keep the same folds, all frozen Kumo/spline caches, paired300-bootstrap uncertainty and segment/error analyses. V252 versusV251 tests added values within the same method. V253 versusV252 tests projection; V253 versusV251 changes both coverage of values and projection, so do not assign that contrast to values alone. Pending experiments remain fully executable; V25 remains public incumbent0.56822.

Earlier related work: V26/V35 tested basket complexity and own-category price z-scores with different inputs/heads. The present raw-price batch is a refinement with matched coverage controls, not the first investigation of item prices. Reuse the current price cache; do not reread raw rows or refit Kumo for these tests.

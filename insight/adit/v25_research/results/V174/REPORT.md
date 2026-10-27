# V174: Empirical-Bayes allocation propensity and uncertainty inside frozen Kumo

Frozen Kumo Large;50k labeled context,8 estimators; disjoint6k offset-head training;14k outer validation; .75*sigmoid(logit(p_Kumo)+.5*Xbeta)+.25*p_original_spline77; mean5fold test probabilities

OOF KS=0.52929274; AUC=0.84698993; delta KS against cached V25=-0.00099192.95% paired delta CI=[-0.001801700967443226, 0.0014146063590179075].

All5folds evaluated. Frozen weights have no gradient updates. In-context fitting was rerun. Torch/CUDA runtime differs from archived control; V168 is a fresh runtime control. Development search bias remains, and a public score is unknown. V25 remains the public incumbent.

Error analysis is held-out only. Legacy snapshot provenance remains unverified.

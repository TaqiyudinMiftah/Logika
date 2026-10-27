# V173: Stratified context bagging: two40k frozen Kumo fits from identical50k eligible pool

Frozen Kumo Large;mean2 stratified contexts of40000 from eligible50k pool,8 estimators; disjoint6k offset-head training;14k outer validation; .75*sigmoid(logit(p_Kumo)+.5*Xbeta)+.25*p_original_spline77; mean5fold test probabilities

OOF KS=0.52860227; AUC=0.84705213; delta KS against cached V25=-0.00168239.95% paired delta CI=[-0.002703735448330538, 0.0009393480187653902].

All5folds evaluated. Frozen weights have no gradient updates. In-context fitting was rerun. Torch/CUDA runtime differs from archived control; V168 is a fresh runtime control. Development search bias remains, and a public score is unknown. V25 remains the public incumbent.

Error analysis is held-out only. Legacy snapshot provenance remains unverified.

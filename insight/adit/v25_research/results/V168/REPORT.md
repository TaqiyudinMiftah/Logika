# V168: Fresh frozen Kumo Large90 V25 control on node480 GPU

Frozen Kumo Large;50k labeled context,8 estimators; disjoint6k offset-head training;14k outer validation; .75*sigmoid(logit(p_Kumo)+.5*Xbeta)+.25*p_original_spline77; mean5fold test probabilities

OOF KS=0.53020201; AUC=0.84702960; delta KS against cached V25=-0.00008265.95% paired delta CI=[-0.00018183922106690674, 0.00014573602328103982].

All5folds evaluated. Frozen weights have no gradient updates. In-context fitting was rerun. Torch/CUDA runtime differs from archived control; V168 is a fresh runtime control. Development search bias remains, and a public score is unknown. V25 remains the public incumbent.

Error analysis is held-out only. Legacy snapshot provenance remains unverified.

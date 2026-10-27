# V175: Selective latent uncertainty and income-contradiction inputs

Frozen Kumo Large;50k labeled context,8 estimators; disjoint6k offset-head training;14k outer validation; .75*sigmoid(logit(p_Kumo)+.5*Xbeta)+.25*p_original_spline77; mean5fold test probabilities

OOF KS=0.52849337; AUC=0.84720164; delta KS against cached V25=-0.00179129.95% paired delta CI=[-0.0029290088079634846, 0.000652812706136102].

All5folds evaluated. Frozen weights have no gradient updates. In-context fitting was rerun. Torch/CUDA runtime differs from archived control; V168 is a fresh runtime control. Development search bias remains, and a public score is unknown. V25 remains the public incumbent.

Error analysis is held-out only. Legacy snapshot provenance remains unverified.

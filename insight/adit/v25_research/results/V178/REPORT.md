# V178: Conservative global-local frozen context experts by history coverage

Frozen Kumo Large;50k labeled context,8 estimators; disjoint6k offset-head training;14k outer validation; .75*sigmoid(logit(p_Kumo)+.5*Xbeta)+.25*p_original_spline77; mean5fold test probabilities; p_Kumo=.85*global_V168_90inputs+.15*local_history-coverage_expert before refitting18-input correction; local context excludes all correction/validation rows

OOF KS=0.53059739; AUC=0.84712938; delta KS against cached V25=+0.00031273.95% paired delta CI=[-0.0005809140359355452, 0.0010435568531322016].

All5folds evaluated. Frozen weights have no gradient updates. In-context fitting was rerun. Torch/CUDA runtime differs from archived control; V168 is a fresh runtime control. Development search bias remains, and a public score is unknown. V25 remains the public incumbent.

Error analysis is held-out only. Legacy snapshot provenance remains unverified.

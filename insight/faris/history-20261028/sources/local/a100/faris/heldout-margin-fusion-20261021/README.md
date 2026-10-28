# Reserved-row prediction-margin fusion (not yet admitted)

One prepared follow-up uses the new classifiers as inputs to the existing correction stage. It adds exactly two linear contrasts: each TabICL log-odds prediction minus Kumo's log-odds. Their normalization fits only the6,000 correction rows. The new coefficients receive10x the original ridge penalty; no weight grid is used.

Every backbone must exclude all6,000 correction labels and14,000 outer-evaluation labels from its50,000-row context. This is an honest heldout meta fit inside each outertraining partition, not a stack fitted to global OOF labels. The shared spline was trained on56,000 rows, so its predictions are excluded from meta fitting and retain the fixed25% outer blend.

Admission requires both new backbones' complete fivefold results, source hashes, rawprediction hashes, exact fitted-model replays, identical frozen row roles and allinput manifests. Missing sources do not trigger a silent smaller recipe. The current plan has training_admitted=false.

The convex objective's gradient passed an independent finite-difference check with error3.82e-8. That is a math check, not evidence of predictive improvement or leakage certification. No model has been fitted for this proposal.

The scripts keep training and reporting separate: only reporting opens outer labels after allfivefolds are complete. Historical label reuse remains disclosed. The original19:31WIB deadline and3h finalization reserve apply. No competition submission is generated.

# Matched tests of raw item-price information

Completed 1/8. Kumo100% frozen; identical five outer folds.

V243 is the common eight-input availability/count control. V244-V250 retain those same controls. Positive KS against V25 alone cannot establish information in price values.

| Experiment | KS | AUC | Delta KS vs V25 | Delta KS vs support | Decision |
|---|---:|---:|---:|---:|---|
| V243 | 0.52934705 | 0.84692008 | -0.00093761 | control | AVAILABILITY_CONTROL_ONLY |

The CSV records paired intervals, fold wins, model size and exact submission hashes. The JSON retains formulas, per-fold fits and held-out segment errors. Intervals exclude retraining, repeated selection and leaderboard uncertainty; these folds have already been used extensively.

The new raw item statistics use only each application's prior completed orders. No new population reference or target enters feature construction. The full model still inherits unresolved historical snapshot provenance from the baseline. V25 remains the public incumbent.

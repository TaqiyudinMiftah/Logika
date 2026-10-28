# Matched tests of raw item-price information

Completed 8/8. Kumo100% frozen; identical five outer folds.

V243 is the common eight-input availability/count control. V244-V250 retain those same controls. Positive KS against V25 alone cannot establish information in price values.

| Experiment | KS | AUC | Delta KS vs V25 | Delta KS vs support | Decision |
|---|---:|---:|---:|---:|---|
| V243 | 0.52934705 | 0.84692008 | -0.00093761 | control | AVAILABILITY_CONTROL_ONLY |
| V244 | 0.52950612 | 0.84690004 | -0.00077854 | +0.00015907 | UNIQUE_VALUE_NOT_ESTABLISHED |
| V245 | 0.52969473 | 0.84689528 | -0.00058993 | +0.00034769 | UNIQUE_VALUE_NOT_ESTABLISHED |
| V246 | 0.52902143 | 0.84693473 | -0.00126323 | -0.00032562 | UNIQUE_VALUE_NOT_ESTABLISHED |
| V247 | 0.52966272 | 0.84688871 | -0.00062194 | +0.00031567 | UNIQUE_VALUE_NOT_ESTABLISHED |
| V248 | 0.52940771 | 0.84690723 | -0.00087695 | +0.00006066 | UNIQUE_VALUE_NOT_ESTABLISHED |
| V249 | 0.52938034 | 0.84691434 | -0.00090432 | +0.00003329 | UNIQUE_VALUE_NOT_ESTABLISHED |
| V250 | 0.52954112 | 0.84687631 | -0.00074354 | +0.00019407 | UNIQUE_VALUE_NOT_ESTABLISHED |

The CSV records paired intervals, fold wins, model size and exact submission hashes. The JSON retains formulas, per-fold fits and held-out segment errors. Intervals exclude retraining, repeated selection and leaderboard uncertainty; these folds have already been used extensively.

The new raw item statistics use only each application's prior completed orders. No new population reference or target enters feature construction. The full model still inherits unresolved historical snapshot provenance from the baseline. V25 remains the public incumbent.

# Learned risk representation results

Point target .54 reached with nondecreasing AUC: False.
KS and AUC both exceed both matched controls: False.

| Pipeline | KS | AUC | Log loss |
|---|---:|---:|---:|
| baseline | 0.52924987 | 0.84720133 | 0.29250104 |
| historical | 0.52929947 | 0.84720269 | 0.29249967 |
| teacher | 0.52033244 | 0.84267157 | 0.29673751 |
| signal | 0.52758231 | 0.84690213 | 0.29277563 |
| null | 0.52863847 | 0.84716054 | 0.29252760 |

All five outer folds and all 150 teachers completed; query labels never trained context features.
All trained models, transformations, context inputs and checkpoint dependencies are retained.
The null augmentation checks width/permutation effects. The historical labels have repeated prior exposure.
No new competition submission was created. Inspect paired intervals and per-fold changes before any promotion.

# Measurement uncertainty study

Ten fitted actuarial GLMs; full five-fold 70k evaluation.

| Procedure | KS | AUC | Log loss |
|---|---:|---:|---:|
| baseline | 0.52929947 | 0.84720269 | 0.29249967 |
| plugin_PD | 0.50474734 | 0.83333682 | 0.30266408 |
| integrated_PD | 0.50526948 | 0.83314966 | 0.30294135 |
| plugin_replacement | 0.52699363 | 0.84639114 | 0.29318805 |
| integrated_replacement | 0.52689138 | 0.84638167 | 0.29321367 |
| plugin_small_addition | 0.52844815 | 0.84716768 | 0.29255134 |
| integrated_small_addition | 0.52855492 | 0.84716184 | 0.29255723 |

Primary comparison: integrated replacement versus retained incumbent. No blend-weight search.
.54 point target with AUC safeguard reached: False.
Models, raw sufficient statistics, posterior variances, priors and transformations are retained.
The familiar labels are reused; conditional bootstrap intervals do not establish fresh generalization.
No submission CSV was created.

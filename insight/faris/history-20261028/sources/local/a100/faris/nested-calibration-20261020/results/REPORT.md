# Nested correction pipeline

Selected: incumbent. New improvement accepted: False.

| Pipeline | KS | ROC-AUC | Log loss |
|---|---:|---:|---:|
| baseline | 0.529250 | 0.847201 | 0.292501 |
| candidate | 0.528818 | 0.847416 | 0.292367 |
| historical | 0.529299 | 0.847203 | 0.292500 |

Both KS and AUC must improve under the frozen criteria. Previously exposed labels and inherited snapshot uncertainty remain. No leaderboard improvement is guaranteed.

All frozen backbones, context indices, inner OOF predictions and fitted correctors are retained. No competition submission was generated.

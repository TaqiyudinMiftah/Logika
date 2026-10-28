# Matched teacher training size

.54 KS target with AUC safeguard: False.

| Pipeline | KS | AUC | Log loss |
|---|---:|---:|---:|
| matched_query | 0.52703397 | 0.84677620 | 0.29283406 |
| signal | 0.52758231 | 0.84690213 | 0.29277563 |
| null | 0.52863847 | 0.84716054 | 0.29252760 |
| historical | 0.52929947 | 0.84720269 | 0.29249967 |
| baseline | 0.52924987 | 0.84720133 | 0.29250104 |

All 70k development rows have one outer prediction. All models and cache references retained.
Only query teacher training size changed; no teacher or backbone was retrained.
All labels have historical exposure. No private leaderboard guarantee or competition CSV is claimed.

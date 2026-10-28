# Single-classification-margin anchor

Allfivefold fits completed with exact saved-model replays and full70,000-row OOF coverage. Every new coefficient was fitted only on itsreserved6,000 correction rows. Source model contexts exclude those labels and all14,000 outer-evaluation labels; allinput manifests and hashes matched. Baseline correction objectives were reproduced before admission.

| Pipeline | KS | AUC |
|---|---:|---:|
|Matched repaired Kumo90|0.530218537|0.847025850|
|One extra classification contrast|0.530234951|0.847011673|

The KS change is only+0.0000164; AUC and logloss are slightly worse. There is no demonstrated useful improvement and no promotion. The joint two-margin proposal remains separate and unadmitted until the regression-source recipe completes and its provenance passes. The0.54 target remains unmet. No submission was generated.

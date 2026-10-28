# Campaign meta-20261016-v1

Final report: True. Elapsed hours: 3.112.

Completed distinct fitted pipelines: 3977 / 4096. Fixed ensembles: 6972. Nested ensembles: 3.

Attempt states: {'completed': 10977, 'failed': 118, 'partial': 1}. Rejected evidence: 0.

Protocol: 56,000 development applications, full 3-fold OOF; 14,000 reserved for this campaign. Historical work used these labels.

Historical public-best Logika v3 score 0.55845 is not directly comparable with this campaign CV.

## Best fitted probes

| Recipe | Family | Model | OOF KS | Mean fold KS |

|---|---|---|---:|---:|

| p_8ba34bf89aa8f9bb1465 | reduction | lgb | 0.517647 | 0.518872 |

| p_4bba31a575638c4687c2 | reduction | cat | 0.516566 | 0.516978 |

| p_0fe6f7f1d8f4096a8aca | reduction | lgb | 0.516284 | 0.517467 |

| p_d9b466e59ef51a5aff17 | reduction | lgb | 0.516262 | 0.520139 |

| p_0cefb85f5e18176d13b8 | preprocessing | lgb | 0.516151 | 0.517790 |

| p_f3375b1c00e0c0ebb30b | account | lgb | 0.516124 | 0.517204 |

| p_0e9c2ac221ef4dca459c | repayment | lgb | 0.516042 | 0.516989 |

| p_6f35e94f7e370c3228ea | reduction | lgb | 0.516037 | 0.517093 |

| p_827b709f1f624d31b60c | reduction | cat | 0.515972 | 0.517468 |

| p_6a182a3daefc5fd734bd | shopping | lgb | 0.515945 | 0.516605 |

## Evidence and limits

See feature_effects.csv for matched per-model/backend changes, coverage.json for factor coverage and untested_recipes.json for unfinished work.

Positive/negative CV labels are exploratory, including multiple-change combinations. Confirmation and repeated seeds provide separate evidence.

No prediction-test labels, leaderboard queries, foundation-model runs or competition submissions are part of this campaign.

## Largest positive matched CV effects

- p_8ba34bf89aa8f9bb1465: lgb/cpu, delta KS +0.004941; {"reducer": "pca"}.

- p_040f3f57b221f2749891: linear/cpu, delta KS +0.004837; {"feature_count": 128, "select": "gain"}.

- p_827b709f1f624d31b60c: cat/cpu, delta KS +0.003771; {"dimensions": 8, "reducer": "group_pca"}.

- p_896e3982655748c9b37d: cat/cpu, delta KS +0.003645; {"dimensions": 16, "reducer": "group_pca"}.

- p_0fe6f7f1d8f4096a8aca: lgb/opencl, delta KS +0.003578; {"dimensions": 32, "reducer": "svd"}.

## Largest negative matched CV effects

- p_106ab1a52c329dcee47b: lgb/cpu, delta KS -0.512612; {"source_mask": 2}.

- p_26925cf7b336b1091ca3: lgb/cpu, delta KS -0.512612; {"append": false, "clip": 0, "representation": "persistence", "window": 365}.

- p_3bdf408d85ec7bad0b77: lgb/cpu, delta KS -0.512612; {"append": false, "half_life": 365, "representation": "disagreement", "window": 60}.

- p_4908779df02a853c4bbb: lgb/cpu, delta KS -0.512612; {"append": false, "half_life": 90, "representation": "disagreement", "window": 90}.

- p_5f1f09b06802cf08f06c: lgb/cpu, delta KS -0.512612; {"append": false, "representation": "persistence"}.

## Frozen confirmation

Primary nominee selected before confirmation: e_4e29e9816b48a5f01e9c

- p_8ba34bf89aa8f9bb1465: delta KS -0.000593, 95% paired bootstrap CI [-0.010612789698037011, 0.008504924240003365], Holm significant positive: False.

- p_0fe6f7f1d8f4096a8aca: delta KS -0.004368, 95% paired bootstrap CI [-0.014357046432511294, 0.006529928284533485], Holm significant positive: False.

- p_0cefb85f5e18176d13b8: delta KS -0.002988, 95% paired bootstrap CI [-0.01157053249843629, 0.0058474722327402015], Holm significant positive: False.

- p_9f0f1a26e14547f45310: delta KS -0.004927, 95% paired bootstrap CI [-0.012051840290599173, 0.005739653600388413], Holm significant positive: False.

- p_f51b6c6ef5d373a6825b: delta KS -0.007358, 95% paired bootstrap CI [-0.015004995174172189, 0.002205852653709605], Holm significant positive: False.

- p_4bba31a575638c4687c2: delta KS -0.002752, 95% paired bootstrap CI [-0.010431203225676427, 0.006981247205730761], Holm significant positive: False.

- p_1368591cbf912532e12f: delta KS -0.006773, 95% paired bootstrap CI [-0.014106317080126196, 0.0027195895131168183], Holm significant positive: False.

- p_e509e9f8befbf29df2b5: delta KS -0.004345, 95% paired bootstrap CI [-0.012483082211593074, 0.005456977785279098], Holm significant positive: False.

- e_4e29e9816b48a5f01e9c: delta KS +0.000161, 95% paired bootstrap CI [-0.008144723008700875, 0.008350316882040857], Holm significant positive: False.

- e_7471157a035ebd5e46b4: delta KS +0.001430, 95% paired bootstrap CI [-0.007107674372730422, 0.010518375297535396], Holm significant positive: False.

- e_fc08ef9be7d762af20c3: delta KS +0.003247, 95% paired bootstrap CI [-0.004594634159469655, 0.010721228977192621], Holm significant positive: False.

- e_fab93730203e21cb2dbf: delta KS +0.000226, 95% paired bootstrap CI [-0.009650348025249784, 0.00962133849363658], Holm significant positive: False.

# Campaign meta-20261016-v1

Final report: False. Elapsed hours: 1.536.

Completed distinct fitted pipelines: 2977 / 4096. Fixed ensembles: 0. Nested ensembles: 0.

Attempt states: {'completed': 2981, 'failed': 90}. Rejected evidence: 0.

Protocol: 56,000 development applications, full 3-fold OOF; 14,000 reserved for this campaign. Historical work used these labels.

Historical public-best Logika v3 score 0.55845 is not directly comparable with this campaign CV.

## Best fitted probes

| Recipe | Family | Model | OOF KS | Mean fold KS |

|---|---|---|---:|---:|

| p_8ba34bf89aa8f9bb1465 | reduction | lgb | 0.517647 | 0.518872 |

| p_0fe6f7f1d8f4096a8aca | reduction | lgb | 0.516284 | 0.517467 |

| p_d9b466e59ef51a5aff17 | reduction | lgb | 0.516262 | 0.520139 |

| p_0cefb85f5e18176d13b8 | preprocessing | lgb | 0.516151 | 0.517790 |

| p_f3375b1c00e0c0ebb30b | account | lgb | 0.516124 | 0.517204 |

| p_0e9c2ac221ef4dca459c | repayment | lgb | 0.516042 | 0.516989 |

| p_6f35e94f7e370c3228ea | reduction | lgb | 0.516037 | 0.517093 |

| p_6a182a3daefc5fd734bd | shopping | lgb | 0.515945 | 0.516605 |

| p_1368591cbf912532e12f | repayment | lgb | 0.515899 | 0.517606 |

| p_f3eedb62f03f28b365d5 | reduction | lgb | 0.515889 | 0.517023 |

## Evidence and limits

See feature_effects.csv for matched per-model/backend changes, coverage.json for factor coverage and untested_recipes.json for unfinished work.

Positive/negative CV labels are exploratory, including multiple-change combinations. Confirmation and repeated seeds provide separate evidence.

No prediction-test labels, leaderboard queries, foundation-model runs or competition submissions are part of this campaign.

## Largest positive matched CV effects

- p_8ba34bf89aa8f9bb1465: lgb/cpu, delta KS +0.004941; {"reducer": "pca"}.

- p_040f3f57b221f2749891: linear/cpu, delta KS +0.004837; {"feature_count": 128, "select": "gain"}.

- p_896e3982655748c9b37d: cat/cpu, delta KS +0.003645; {"dimensions": 16, "reducer": "group_pca"}.

- p_0fe6f7f1d8f4096a8aca: lgb/opencl, delta KS +0.003578; {"dimensions": 32, "reducer": "svd"}.

- p_d9b466e59ef51a5aff17: lgb/cpu, delta KS +0.003556; {"reducer": "svd", "select": "gain"}.

## Largest negative matched CV effects

- p_106ab1a52c329dcee47b: lgb/cpu, delta KS -0.512612; {"source_mask": 2}.

- p_26925cf7b336b1091ca3: lgb/cpu, delta KS -0.512612; {"append": false, "clip": 0, "representation": "persistence", "window": 365}.

- p_3bdf408d85ec7bad0b77: lgb/cpu, delta KS -0.512612; {"append": false, "half_life": 365, "representation": "disagreement", "window": 60}.

- p_4908779df02a853c4bbb: lgb/cpu, delta KS -0.512612; {"append": false, "half_life": 90, "representation": "disagreement", "window": 90}.

- p_5f1f09b06802cf08f06c: lgb/cpu, delta KS -0.512612; {"append": false, "representation": "persistence"}.

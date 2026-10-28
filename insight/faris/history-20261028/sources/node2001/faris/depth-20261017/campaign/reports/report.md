# Experiment report: depth-20261017-pc2

State: **stopped**. Stop reason: user_stop.

## Outcomes

{"complete": 18, "pruned": 0, "failed": 0, "interrupted": 4, "timeout": 0, "invalid": 0, "attempts": 22, "retries": 0, "jobs": 23, "unique_configurations": 22, "screen_jobs": 22, "confirmation_jobs": 0, "unschedulable": 0, "budget_skipped": 0, "unstarted": 1, "pending": 0, "running": 0}

## Coverage

| Family | Complete configs | Candidate pool | Pair coverage | Denominator |
|---|---:|---:|---:|---|
| lgb_depth | 5 | 4100 | 68/629 | sampled_reference_pool |
| cat_depth | 7 | 4097 | 86/443 | sampled_reference_pool |
| xgb_depth | 6 | 4100 | 104/629 | sampled_reference_pool |

## Verified scoreboards

### screen:d6642c4a07cfc09a4684f4213b8de40901a5fc1bfa1f244a2df14d44e2c86d4a:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_000016 | lgb_depth | 0.517502034 | 54.075 | `runs/job_000016/attempt_001` |
| job_000013 | lgb_depth | 0.517199049 | 56.371 | `runs/job_000013/attempt_001` |
| job_000010 | cat_depth | 0.515110954 | 11.509 | `runs/job_000010/attempt_001` |
| job_000009 | cat_depth | 0.515099419 | 12.748 | `runs/job_000009/attempt_001` |
| job_000006 | lgb_depth | 0.513844664 | 47.038 | `runs/job_000006/attempt_001` |
| job_000015 | xgb_depth | 0.513542256 | 10.895 | `runs/job_000015/attempt_001` |
| job_000005 | xgb_depth | 0.513529443 | 19.896 | `runs/job_000005/attempt_001` |
| job_000000 | lgb_depth | 0.513433672 | 42.674 | `runs/job_000000/attempt_001` |
| job_000003 | lgb_depth | 0.513300094 | 33.135 | `runs/job_000003/attempt_001` |
| job_000011 | xgb_depth | 0.513200927 | 11.906 | `runs/job_000011/attempt_001` |
| job_000008 | xgb_depth | 0.512894367 | 10.435 | `runs/job_000008/attempt_001` |
| job_000002 | xgb_depth | 0.512429757 | 9.424 | `runs/job_000002/attempt_001` |
| job_000001 | cat_depth | 0.512099262 | 9.406 | `runs/job_000001/attempt_001` |
| job_000018 | cat_depth | 0.511764133 | 62.759 | `runs/job_000018/attempt_001` |
| job_000007 | cat_depth | 0.511402607 | 10.435 | `runs/job_000007/attempt_001` |
| job_000012 | cat_depth | 0.510716086 | 141.530 | `runs/job_000012/attempt_001` |
| job_000004 | cat_depth | 0.510655317 | 21.238 | `runs/job_000004/attempt_001` |
| job_000019 | xgb_depth | 0.507704503 | 27.049 | `runs/job_000019/attempt_001` |

## Matched ablations

See summary.json for matched control/treatment differences. Unmatched experiments are not presented as ablations.

## Limitations and unresolved questions

- Search scores are selection-biased; confirmation results are reported separately.
- CPU seconds are sampled process-tree lower bounds; GPU allocation-seconds are not measured busy GPU time.
- Pair coverage concerns declared categorical factors within families; it does not prove higher-order or global optimality.
- No protected confirmation protocol supplied; no independent generalization claim.

## Reproduction

Each attempt contains spec.json, process.json, environment.json, and immutable source references. Replay one with:

`/home/faris/faris/depth-20261017/.venv/bin/python /home/faris/faris/depth-20261017/campaign/runtime/campaign.py replay /home/faris/faris/depth-20261017/campaign JOB_ID --output NEW_DIRECTORY`

All retained model and validation artifacts are indexed in records/*.json. No submission files are produced.

# Experiment report: depth-20261017-pc3

State: **stopped**. Stop reason: user_stop.

## Outcomes

{"complete": 22, "pruned": 0, "failed": 0, "interrupted": 2, "timeout": 0, "invalid": 0, "attempts": 24, "retries": 0, "jobs": 25, "unique_configurations": 24, "screen_jobs": 24, "confirmation_jobs": 0, "unschedulable": 0, "budget_skipped": 0, "unstarted": 1, "pending": 0, "running": 0}

## Coverage

| Family | Complete configs | Candidate pool | Pair coverage | Denominator |
|---|---:|---:|---:|---|
| linear_depth | 11 | 702 | 54/246 | exact_finite_space |
| cat_depth | 5 | 4098 | 45/443 | sampled_reference_pool |
| xgb_depth | 6 | 4101 | 104/629 | sampled_reference_pool |

## Verified scoreboards

### screen:d6642c4a07cfc09a4684f4213b8de40901a5fc1bfa1f244a2df14d44e2c86d4a:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_000003 | linear_depth | 0.515068801 | 20.074 | `runs/job_000003/attempt_001` |
| job_000012 | cat_depth | 0.514795375 | 46.098 | `runs/job_000012/attempt_001` |
| job_000015 | cat_depth | 0.514061022 | 43.152 | `runs/job_000015/attempt_001` |
| job_000013 | xgb_depth | 0.513653839 | 14.004 | `runs/job_000013/attempt_001` |
| job_000005 | xgb_depth | 0.513385678 | 20.723 | `runs/job_000005/attempt_001` |
| job_000016 | xgb_depth | 0.512955548 | 13.716 | `runs/job_000016/attempt_001` |
| job_000004 | cat_depth | 0.512794912 | 43.726 | `runs/job_000004/attempt_001` |
| job_000023 | linear_depth | 0.512672402 | 23.957 | `runs/job_000023/attempt_001` |
| job_000002 | xgb_depth | 0.512561933 | 11.369 | `runs/job_000002/attempt_001` |
| job_000009 | xgb_depth | 0.512555141 | 12.107 | `runs/job_000009/attempt_001` |
| job_000011 | linear_depth | 0.512440412 | 22.692 | `runs/job_000011/attempt_001` |
| job_000007 | cat_depth | 0.51205337 | 40.037 | `runs/job_000007/attempt_001` |
| job_000017 | linear_depth | 0.511947753 | 29.826 | `runs/job_000017/attempt_001` |
| job_000006 | linear_depth | 0.511091646 | 9.694 | `runs/job_000006/attempt_001` |
| job_000001 | cat_depth | 0.511029959 | 39.755 | `runs/job_000001/attempt_001` |
| job_000014 | linear_depth | 0.510939632 | 23.026 | `runs/job_000014/attempt_001` |
| job_000018 | linear_depth | 0.510060112 | 11.491 | `runs/job_000018/attempt_001` |
| job_000008 | linear_depth | 0.509991617 | 13.999 | `runs/job_000008/attempt_001` |
| job_000000 | linear_depth | 0.509843341 | 7.948 | `runs/job_000000/attempt_001` |
| job_000010 | linear_depth | 0.509743652 | 14.810 | `runs/job_000010/attempt_001` |

## Matched ablations

See summary.json for matched control/treatment differences. Unmatched experiments are not presented as ablations.

## Limitations and unresolved questions

- Search scores are selection-biased; confirmation results are reported separately.
- CPU seconds are sampled process-tree lower bounds; GPU allocation-seconds are not measured busy GPU time.
- Pair coverage concerns declared categorical factors within families; it does not prove higher-order or global optimality.
- No protected confirmation protocol supplied; no independent generalization claim.

## Reproduction

Each attempt contains spec.json, process.json, environment.json, and immutable source references. Replay one with:

`/mnt/home/faris/faris/depth-20261017/.venv/bin/python /mnt/home/faris/faris/depth-20261017/campaign/runtime/campaign.py replay /mnt/home/faris/faris/depth-20261017/campaign JOB_ID --output NEW_DIRECTORY`

All retained model and validation artifacts are indexed in records/*.json. No submission files are produced.

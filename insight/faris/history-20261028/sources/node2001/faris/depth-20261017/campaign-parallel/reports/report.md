# Experiment report: depth-20261017-pc2

State: **stopped**. Stop reason: controller_error: FileNotFoundError: [Errno 2] No such file or directory: '/home/faris/faris/depth-20261017/campaign-parallel/runs/job_000272/attempt_001/result.json.687237.tmp'.

## Outcomes

{"complete": 297, "pruned": 0, "failed": 0, "interrupted": 10, "timeout": 0, "invalid": 0, "attempts": 307, "retries": 0, "jobs": 320, "unique_configurations": 307, "screen_jobs": 307, "confirmation_jobs": 0, "unschedulable": 0, "budget_skipped": 0, "unstarted": 13, "pending": 0, "running": 0}

## Coverage

| Family | Complete configs | Candidate pool | Pair coverage | Denominator |
|---|---:|---:|---:|---|
| lgb_depth | 221 | 4100 | 629/629 | sampled_reference_pool |
| cat_depth | 38 | 4097 | 313/443 | sampled_reference_pool |
| xgb_depth | 38 | 4100 | 482/629 | sampled_reference_pool |

## Verified scoreboards

### screen:d6642c4a07cfc09a4684f4213b8de40901a5fc1bfa1f244a2df14d44e2c86d4a:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_000240 | lgb_depth | 0.519539603 | 105.981 | `runs/job_000240/attempt_001` |
| job_000164 | lgb_depth | 0.518506129 | 91.696 | `runs/job_000164/attempt_001` |
| job_000178 | lgb_depth | 0.518247618 | 133.212 | `runs/job_000178/attempt_001` |
| job_000198 | lgb_depth | 0.517924024 | 139.975 | `runs/job_000198/attempt_001` |
| job_000275 | lgb_depth | 0.517886918 | 136.706 | `runs/job_000275/attempt_001` |
| job_000125 | lgb_depth | 0.517849041 | 84.967 | `runs/job_000125/attempt_001` |
| job_000230 | cat_depth | 0.517811109 | 208.499 | `runs/job_000230/attempt_001` |
| job_000161 | lgb_depth | 0.517733953 | 97.021 | `runs/job_000161/attempt_001` |
| job_000156 | lgb_depth | 0.517640768 | 146.585 | `runs/job_000156/attempt_001` |
| job_000126 | lgb_depth | 0.517636258 | 87.524 | `runs/job_000126/attempt_001` |
| job_000131 | cat_depth | 0.517565139 | 161.569 | `runs/job_000131/attempt_001` |
| job_000269 | lgb_depth | 0.517542894 | 100.071 | `runs/job_000269/attempt_001` |
| job_000295 | lgb_depth | 0.517534755 | 89.172 | `runs/job_000295/attempt_001` |
| job_000183 | lgb_depth | 0.517508826 | 90.257 | `runs/job_000183/attempt_001` |
| job_000012 | lgb_depth | 0.517502034 | 86.991 | `runs/job_000012/attempt_001` |
| job_000181 | lgb_depth | 0.517455262 | 124.458 | `runs/job_000181/attempt_001` |
| job_000188 | lgb_depth | 0.51744502 | 151.642 | `runs/job_000188/attempt_001` |
| job_000218 | lgb_depth | 0.517227494 | 103.114 | `runs/job_000218/attempt_001` |
| job_000009 | lgb_depth | 0.517199049 | 91.645 | `runs/job_000009/attempt_001` |
| job_000204 | lgb_depth | 0.517169366 | 135.625 | `runs/job_000204/attempt_001` |

## Matched ablations

See summary.json for matched control/treatment differences. Unmatched experiments are not presented as ablations.

## Limitations and unresolved questions

- Search scores are selection-biased; confirmation results are reported separately.
- CPU seconds are sampled process-tree lower bounds; GPU allocation-seconds are not measured busy GPU time.
- Pair coverage concerns declared categorical factors within families; it does not prove higher-order or global optimality.
- No protected confirmation protocol supplied; no independent generalization claim.

## Reproduction

Each attempt contains spec.json, process.json, environment.json, and immutable source references. Replay one with:

`/home/faris/faris/depth-20261017/.venv/bin/python /home/faris/faris/depth-20261017/campaign-parallel/runtime/campaign.py replay /home/faris/faris/depth-20261017/campaign-parallel JOB_ID --output NEW_DIRECTORY`

All retained model and validation artifacts are indexed in records/*.json. No submission files are produced.

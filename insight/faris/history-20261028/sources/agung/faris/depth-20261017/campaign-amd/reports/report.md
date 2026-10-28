# Experiment report: depth-20261017-amd

State: **finished**. Stop reason: search_exhausted.

## Outcomes

{"complete": 98, "pruned": 0, "failed": 0, "interrupted": 0, "timeout": 0, "invalid": 0, "attempts": 98, "retries": 0, "jobs": 330, "unique_configurations": 98, "screen_jobs": 98, "confirmation_jobs": 0, "unschedulable": 0, "budget_skipped": 232, "unstarted": 0, "pending": 0, "running": 0}

## Coverage

| Family | Complete configs | Candidate pool | Pair coverage | Denominator |
|---|---:|---:|---:|---|
| lgb_depth | 98 | 4101 | 561/668 | sampled_reference_pool |

## Verified scoreboards

### screen:d6642c4a07cfc09a4684f4213b8de40901a5fc1bfa1f244a2df14d44e2c86d4a:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_000038 | lgb_depth | 0.518592 | 25.243 | `runs/job_000038/attempt_001` |
| job_000074 | lgb_depth | 0.518206415 | 64.089 | `runs/job_000074/attempt_001` |
| job_000086 | lgb_depth | 0.517192078 | 32.309 | `runs/job_000086/attempt_001` |
| job_000045 | lgb_depth | 0.516865843 | 41.378 | `runs/job_000045/attempt_001` |
| job_000063 | lgb_depth | 0.516559751 | 44.950 | `runs/job_000063/attempt_001` |
| job_000054 | lgb_depth | 0.516476808 | 31.659 | `runs/job_000054/attempt_001` |
| job_000095 | lgb_depth | 0.516366877 | 27.747 | `runs/job_000095/attempt_001` |
| job_000089 | lgb_depth | 0.516066642 | 18.672 | `runs/job_000089/attempt_001` |
| job_000070 | lgb_depth | 0.515941888 | 33.348 | `runs/job_000070/attempt_001` |
| job_000061 | lgb_depth | 0.515934683 | 30.098 | `runs/job_000061/attempt_001` |
| job_000062 | lgb_depth | 0.515707741 | 34.671 | `runs/job_000062/attempt_001` |
| job_000072 | lgb_depth | 0.515684795 | 41.249 | `runs/job_000072/attempt_001` |
| job_000026 | lgb_depth | 0.515585932 | 52.026 | `runs/job_000026/attempt_001` |
| job_000036 | lgb_depth | 0.515561225 | 20.536 | `runs/job_000036/attempt_001` |
| job_000077 | lgb_depth | 0.515489747 | 23.854 | `runs/job_000077/attempt_001` |
| job_000091 | lgb_depth | 0.515427055 | 27.787 | `runs/job_000091/attempt_001` |
| job_000084 | lgb_depth | 0.515234632 | 18.519 | `runs/job_000084/attempt_001` |
| job_000037 | lgb_depth | 0.515108493 | 37.916 | `runs/job_000037/attempt_001` |
| job_000048 | lgb_depth | 0.514983397 | 52.240 | `runs/job_000048/attempt_001` |
| job_000073 | lgb_depth | 0.514924622 | 85.844 | `runs/job_000073/attempt_001` |

## Matched ablations

See summary.json for matched control/treatment differences. Unmatched experiments are not presented as ablations.

## Limitations and unresolved questions

- Search scores are selection-biased; confirmation results are reported separately.
- CPU seconds are sampled process-tree lower bounds; GPU allocation-seconds are not measured busy GPU time.
- Pair coverage concerns declared categorical factors within families; it does not prove higher-order or global optimality.
- No protected confirmation protocol supplied; no independent generalization claim.

## Reproduction

Each attempt contains spec.json, process.json, environment.json, and immutable source references. Replay one with:

`/home/agung/faris/depth-20261017/.venv/bin/python /home/agung/faris/depth-20261017/campaign-amd/runtime/campaign.py replay /home/agung/faris/depth-20261017/campaign-amd JOB_ID --output NEW_DIRECTORY`

All retained model and validation artifacts are indexed in records/*.json. No submission files are produced.

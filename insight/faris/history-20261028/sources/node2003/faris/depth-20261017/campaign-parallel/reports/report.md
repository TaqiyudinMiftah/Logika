# Experiment report: depth-20261017-pc3

State: **finished**. Stop reason: search_exhausted.

## Outcomes

{"complete": 477, "pruned": 0, "failed": 2, "interrupted": 0, "timeout": 1, "invalid": 0, "attempts": 480, "retries": 0, "jobs": 531, "unique_configurations": 480, "screen_jobs": 480, "confirmation_jobs": 0, "unschedulable": 0, "budget_skipped": 51, "unstarted": 0, "pending": 0, "running": 0}

## Coverage

| Family | Complete configs | Candidate pool | Pair coverage | Denominator |
|---|---:|---:|---:|---|
| linear_depth | 421 | 702 | 246/246 | exact_finite_space |
| cat_depth | 25 | 4098 | 293/443 | sampled_reference_pool |
| xgb_depth | 31 | 4101 | 459/629 | sampled_reference_pool |

## Verified scoreboards

### screen:d6642c4a07cfc09a4684f4213b8de40901a5fc1bfa1f244a2df14d44e2c86d4a:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_000427 | linear_depth | 0.519222855 | 147.645 | `runs/job_000427/attempt_001` |
| job_000303 | linear_depth | 0.518323193 | 48.294 | `runs/job_000303/attempt_001` |
| job_000042 | linear_depth | 0.518175151 | 55.247 | `runs/job_000042/attempt_001` |
| job_000225 | cat_depth | 0.51809713 | 137.051 | `runs/job_000225/attempt_001` |
| job_000326 | linear_depth | 0.517914895 | 98.988 | `runs/job_000326/attempt_001` |
| job_000364 | linear_depth | 0.51789628 | 119.323 | `runs/job_000364/attempt_001` |
| job_000081 | linear_depth | 0.517470123 | 46.260 | `runs/job_000081/attempt_001` |
| job_000293 | linear_depth | 0.517448291 | 116.789 | `runs/job_000293/attempt_001` |
| job_000150 | cat_depth | 0.517427285 | 129.575 | `runs/job_000150/attempt_001` |
| job_000319 | linear_depth | 0.517341756 | 69.466 | `runs/job_000319/attempt_001` |
| job_000289 | linear_depth | 0.517338126 | 147.926 | `runs/job_000289/attempt_001` |
| job_000084 | linear_depth | 0.517330867 | 36.543 | `runs/job_000084/attempt_001` |
| job_000118 | linear_depth | 0.517248391 | 36.568 | `runs/job_000118/attempt_001` |
| job_000405 | linear_depth | 0.517192132 | 73.919 | `runs/job_000405/attempt_001` |
| job_000335 | linear_depth | 0.517151631 | 52.293 | `runs/job_000335/attempt_001` |
| job_000320 | linear_depth | 0.517097241 | 45.024 | `runs/job_000320/attempt_001` |
| job_000312 | linear_depth | 0.516994569 | 115.213 | `runs/job_000312/attempt_001` |
| job_000310 | linear_depth | 0.516962549 | 50.877 | `runs/job_000310/attempt_001` |
| job_000119 | linear_depth | 0.516833878 | 36.577 | `runs/job_000119/attempt_001` |
| job_000447 | linear_depth | 0.516820598 | 55.615 | `runs/job_000447/attempt_001` |

## Matched ablations

See summary.json for matched control/treatment differences. Unmatched experiments are not presented as ablations.

## Limitations and unresolved questions

- Search scores are selection-biased; confirmation results are reported separately.
- CPU seconds are sampled process-tree lower bounds; GPU allocation-seconds are not measured busy GPU time.
- Pair coverage concerns declared categorical factors within families; it does not prove higher-order or global optimality.
- No protected confirmation protocol supplied; no independent generalization claim.

## Reproduction

Each attempt contains spec.json, process.json, environment.json, and immutable source references. Replay one with:

`/mnt/home/faris/faris/depth-20261017/.venv/bin/python /mnt/home/faris/faris/depth-20261017/campaign-parallel/runtime/campaign.py replay /mnt/home/faris/faris/depth-20261017/campaign-parallel JOB_ID --output NEW_DIRECTORY`

All retained model and validation artifacts are indexed in records/*.json. No submission files are produced.

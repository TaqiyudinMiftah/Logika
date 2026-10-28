# Experiment report: depth-20261017-pc1

State: **finished**. Stop reason: search_exhausted.

## Outcomes

{"complete": 780, "pruned": 0, "failed": 0, "interrupted": 0, "timeout": 0, "invalid": 0, "attempts": 780, "retries": 0, "jobs": 830, "unique_configurations": 780, "screen_jobs": 780, "confirmation_jobs": 0, "unschedulable": 0, "budget_skipped": 50, "unstarted": 0, "pending": 0, "running": 0}

## Coverage

| Family | Complete configs | Candidate pool | Pair coverage | Denominator |
|---|---:|---:|---:|---|
| linear_depth | 645 | 702 | 246/246 | exact_finite_space |
| lgb_depth | 135 | 4101 | 665/668 | sampled_reference_pool |

## Verified scoreboards

### screen:d6642c4a07cfc09a4684f4213b8de40901a5fc1bfa1f244a2df14d44e2c86d4a:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_000393 | linear_depth | 0.519222855 | 100.231 | `runs/job_000393/attempt_001` |
| job_000338 | linear_depth | 0.518323193 | 57.970 | `runs/job_000338/attempt_001` |
| job_000023 | linear_depth | 0.518175151 | 57.935 | `runs/job_000023/attempt_001` |
| job_000602 | lgb_depth | 0.518080096 | 127.369 | `runs/job_000602/attempt_001` |
| job_000408 | linear_depth | 0.517914895 | 83.507 | `runs/job_000408/attempt_001` |
| job_000375 | linear_depth | 0.51789628 | 91.579 | `runs/job_000375/attempt_001` |
| job_000718 | lgb_depth | 0.517710774 | 297.653 | `runs/job_000718/attempt_001` |
| job_000429 | lgb_depth | 0.51751339 | 154.731 | `runs/job_000429/attempt_001` |
| job_000064 | linear_depth | 0.517470123 | 60.188 | `runs/job_000064/attempt_001` |
| job_000315 | linear_depth | 0.517448291 | 50.183 | `runs/job_000315/attempt_001` |
| job_000326 | linear_depth | 0.517341756 | 47.234 | `runs/job_000326/attempt_001` |
| job_000318 | linear_depth | 0.517338126 | 49.940 | `runs/job_000318/attempt_001` |
| job_000068 | linear_depth | 0.517330867 | 47.299 | `runs/job_000068/attempt_001` |
| job_000109 | linear_depth | 0.517227727 | 54.065 | `runs/job_000109/attempt_001` |
| job_000311 | linear_depth | 0.5172255 | 71.369 | `runs/job_000311/attempt_001` |
| job_000361 | linear_depth | 0.517192132 | 81.230 | `runs/job_000361/attempt_001` |
| job_000412 | linear_depth | 0.517188682 | 94.160 | `runs/job_000412/attempt_001` |
| job_000341 | linear_depth | 0.517151631 | 56.102 | `runs/job_000341/attempt_001` |
| job_000334 | linear_depth | 0.517097241 | 58.930 | `runs/job_000334/attempt_001` |
| job_000304 | lgb_depth | 0.517096073 | 120.280 | `runs/job_000304/attempt_001` |

## Matched ablations

See summary.json for matched control/treatment differences. Unmatched experiments are not presented as ablations.

## Limitations and unresolved questions

- Search scores are selection-biased; confirmation results are reported separately.
- CPU seconds are sampled process-tree lower bounds; GPU allocation-seconds are not measured busy GPU time.
- Pair coverage concerns declared categorical factors within families; it does not prove higher-order or global optimality.
- No protected confirmation protocol supplied; no independent generalization claim.

## Reproduction

Each attempt contains spec.json, process.json, environment.json, and immutable source references. Replay one with:

`/home/agung/faris/depth-20261017/.venv/bin/python /home/agung/faris/depth-20261017/campaign/runtime/campaign.py replay /home/agung/faris/depth-20261017/campaign JOB_ID --output NEW_DIRECTORY`

All retained model and validation artifacts are indexed in records/*.json. No submission files are produced.

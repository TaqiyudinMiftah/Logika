# Experiment report: explore12h-pc3

State: **stopped**. Stop reason: user_stop.

## Outcomes

{"complete": 53, "pruned": 0, "failed": 1, "interrupted": 6, "timeout": 0, "invalid": 0, "attempts": 60, "retries": 0, "jobs": 88, "unique_configurations": 60, "screen_jobs": 60, "confirmation_jobs": 0, "unschedulable": 0, "budget_skipped": 0, "unstarted": 28, "pending": 0, "running": 0}

## Coverage

| Family | Complete configs | Candidate pool | Pair coverage | Denominator |
|---|---:|---:|---:|---|
| act_cloglog | 14 | 64 | 0/0 | exact_finite_space |
| act_logit | 10 | 75 | 0/0 | exact_finite_space |
| cat_native | 1 | 107 | 0/0 | exact_finite_space |
| cred_scorecard | 7 | 80 | 0/0 | exact_finite_space |
| deepsets | 1 | 100 | 0/0 | exact_finite_space |
| ebm | 2 | 48 | 0/0 | exact_finite_space |
| elastic | 1 | 71 | 0/0 | exact_finite_space |
| extra | 4 | 72 | 0/0 | exact_finite_space |
| ft | 0 | 118 | 0/0 | exact_finite_space |
| gru | 1 | 124 | 0/0 | exact_finite_space |
| resmlp | 1 | 118 | 0/0 | exact_finite_space |
| rff | 2 | 64 | 0/0 | exact_finite_space |
| spline | 2 | 60 | 0/0 | exact_finite_space |
| tabm | 1 | 87 | 0/0 | exact_finite_space |
| tabr | 0 | 94 | 0/0 | exact_finite_space |
| woe_scorecard | 5 | 68 | 0/0 | exact_finite_space |
| xgb | 1 | 90 | 0/0 | exact_finite_space |

## Verified scoreboards

### screen:86ecc034dd9d3bd05ba26549e15f947413a817a50efc5cc29666d191b8cec081:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_000013 | tabm | 0.516242482 | 107.240 | `runs/job_000013/attempt_001` |
| job_000057 | ebm | 0.514181788 | 246.590 | `runs/job_000057/attempt_001` |
| job_000016 | xgb | 0.513871832 | 21.401 | `runs/job_000016/attempt_001` |
| job_000059 | act_cloglog | 0.513523999 | 24.456 | `runs/job_000059/attempt_001` |
| job_000005 | ebm | 0.513519794 | 79.961 | `runs/job_000005/attempt_001` |
| job_000085 | act_logit | 0.513087491 | 28.526 | `runs/job_000085/attempt_001` |
| job_000002 | cat_native | 0.512566747 | 49.534 | `runs/job_000002/attempt_001` |
| job_000063 | act_logit | 0.511258249 | 10.802 | `runs/job_000063/attempt_001` |
| job_000076 | act_cloglog | 0.510909482 | 16.436 | `runs/job_000076/attempt_001` |
| job_000072 | act_cloglog | 0.51030743 | 25.167 | `runs/job_000072/attempt_001` |
| job_000061 | act_logit | 0.510300872 | 28.426 | `runs/job_000061/attempt_001` |
| job_000006 | elastic | 0.508996183 | 463.387 | `runs/job_000006/attempt_001` |
| job_000004 | deepsets | 0.508068777 | 66.986 | `runs/job_000004/attempt_001` |
| job_000010 | resmlp | 0.506043445 | 18.517 | `runs/job_000010/attempt_001` |
| job_000009 | gru | 0.506037821 | 62.071 | `runs/job_000009/attempt_001` |
| job_000084 | act_cloglog | 0.505917756 | 35.829 | `runs/job_000084/attempt_001` |
| job_000029 | woe_scorecard | 0.505523097 | 25.404 | `runs/job_000029/attempt_001` |
| job_000069 | act_logit | 0.505155606 | 33.958 | `runs/job_000069/attempt_001` |
| job_000049 | act_logit | 0.505119131 | 30.677 | `runs/job_000049/attempt_001` |
| job_000073 | act_logit | 0.504475284 | 39.352 | `runs/job_000073/attempt_001` |

## Matched ablations

See summary.json for matched control/treatment differences. Unmatched experiments are not presented as ablations.

## Limitations and unresolved questions

- Search scores are selection-biased; confirmation results are reported separately.
- CPU seconds are sampled process-tree lower bounds; GPU allocation-seconds are not measured busy GPU time.
- Pair coverage concerns declared categorical factors within families; it does not prove higher-order or global optimality.
- ft: H_ft; matched champion and feature-block exploration
- tabr: H_tabr; matched champion and feature-block exploration

## Reproduction

Each attempt contains spec.json, process.json, environment.json, and immutable source references. Replay one with:

`/mnt/home/faris/faris/exploration-12h-20261017/.venv/bin/python /mnt/home/faris/faris/exploration-12h-20261017/campaign/runtime/campaign.py replay /mnt/home/faris/faris/exploration-12h-20261017/campaign JOB_ID --output NEW_DIRECTORY`

All retained model and validation artifacts are indexed in records/*.json. No submission files are produced.

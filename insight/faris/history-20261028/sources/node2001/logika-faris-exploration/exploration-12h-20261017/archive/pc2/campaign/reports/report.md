# Experiment report: explore12h-pc2

State: **stopped**. Stop reason: user_stop.

## Outcomes

{"complete": 74, "pruned": 0, "failed": 1, "interrupted": 6, "timeout": 0, "invalid": 0, "attempts": 81, "retries": 0, "jobs": 105, "unique_configurations": 81, "screen_jobs": 81, "confirmation_jobs": 0, "unschedulable": 0, "budget_skipped": 0, "unstarted": 24, "pending": 0, "running": 0}

## Coverage

| Family | Complete configs | Candidate pool | Pair coverage | Denominator |
|---|---:|---:|---:|---|
| act_cloglog | 18 | 65 | 0/0 | exact_finite_space |
| act_logit | 15 | 74 | 0/0 | exact_finite_space |
| cat_native | 1 | 115 | 0/0 | exact_finite_space |
| champion | 5 | 5 | 0/0 | exact_finite_space |
| cred_scorecard | 6 | 71 | 0/0 | exact_finite_space |
| deepsets | 1 | 97 | 0/0 | exact_finite_space |
| ebm | 1 | 56 | 0/0 | exact_finite_space |
| elastic | 0 | 55 | 0/0 | exact_finite_space |
| extra | 4 | 66 | 0/0 | exact_finite_space |
| ft | 0 | 96 | 0/0 | exact_finite_space |
| gru | 1 | 102 | 0/0 | exact_finite_space |
| resmlp | 1 | 103 | 0/0 | exact_finite_space |
| rff | 5 | 62 | 0/0 | exact_finite_space |
| spline | 7 | 66 | 0/0 | exact_finite_space |
| tabm | 1 | 87 | 0/0 | exact_finite_space |
| tabr | 0 | 96 | 0/0 | exact_finite_space |
| woe_scorecard | 7 | 62 | 0/0 | exact_finite_space |
| xgb | 1 | 93 | 0/0 | exact_finite_space |
| nested | 0 | 12 | 0/0 | exact_finite_space |

## Verified scoreboards

### screen:86ecc034dd9d3bd05ba26549e15f947413a817a50efc5cc29666d191b8cec081:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_000003 | champion | 0.51809361 | 48.981 | `runs/job_000003/attempt_001` |
| job_000070 | champion | 0.517260269 | 32.726 | `runs/job_000070/attempt_001` |
| job_000079 | champion | 0.516722491 | 117.973 | `runs/job_000079/attempt_001` |
| job_000057 | champion | 0.516431438 | 77.184 | `runs/job_000057/attempt_001` |
| job_000064 | act_logit | 0.516359672 | 25.012 | `runs/job_000064/attempt_001` |
| job_000052 | act_cloglog | 0.516273567 | 13.673 | `runs/job_000052/attempt_001` |
| job_000014 | tabm | 0.516240612 | 121.302 | `runs/job_000014/attempt_001` |
| job_000097 | champion | 0.515791205 | 70.612 | `runs/job_000097/attempt_001` |
| job_000084 | act_logit | 0.514953838 | 30.153 | `runs/job_000084/attempt_001` |
| job_000006 | ebm | 0.51396888 | 413.031 | `runs/job_000006/attempt_001` |
| job_000063 | act_cloglog | 0.513892371 | 33.447 | `runs/job_000063/attempt_001` |
| job_000017 | xgb | 0.513006417 | 22.025 | `runs/job_000017/attempt_001` |
| job_000002 | cat_native | 0.51258051 | 59.962 | `runs/job_000002/attempt_001` |
| job_000011 | resmlp | 0.511816115 | 21.758 | `runs/job_000011/attempt_001` |
| job_000076 | spline | 0.511771913 | 37.918 | `runs/job_000076/attempt_001` |
| job_000055 | act_cloglog | 0.51078726 | 31.843 | `runs/job_000055/attempt_001` |
| job_000068 | act_cloglog | 0.50894093 | 32.901 | `runs/job_000068/attempt_001` |
| job_000010 | gru | 0.508326696 | 77.491 | `runs/job_000010/attempt_001` |
| job_000086 | act_logit | 0.508197324 | 43.032 | `runs/job_000086/attempt_001` |
| job_000005 | deepsets | 0.507765918 | 80.453 | `runs/job_000005/attempt_001` |

## Matched ablations

See summary.json for matched control/treatment differences. Unmatched experiments are not presented as ablations.

## Limitations and unresolved questions

- Search scores are selection-biased; confirmation results are reported separately.
- CPU seconds are sampled process-tree lower bounds; GPU allocation-seconds are not measured busy GPU time.
- Pair coverage concerns declared categorical factors within families; it does not prove higher-order or global optimality.
- elastic: H_elastic; matched champion and feature-block exploration
- ft: H_ft; matched champion and feature-block exploration
- tabr: H_tabr; matched champion and feature-block exploration
- nested: H7 genuine nested ensembles

## Reproduction

Each attempt contains spec.json, process.json, environment.json, and immutable source references. Replay one with:

`/mnt/home/logika-faris-exploration/exploration-12h-20261017/.venv/bin/python /mnt/home/logika-faris-exploration/exploration-12h-20261017/campaign/runtime/campaign.py replay /mnt/home/logika-faris-exploration/exploration-12h-20261017/campaign JOB_ID --output NEW_DIRECTORY`

All retained model and validation artifacts are indexed in records/*.json. No submission files are produced.

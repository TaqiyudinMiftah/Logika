# Experiment report: explore12h-pc1

State: **stopped**. Stop reason: user_stop.

## Outcomes

{"complete": 217, "pruned": 0, "failed": 0, "interrupted": 13, "timeout": 0, "invalid": 0, "attempts": 230, "retries": 0, "jobs": 245, "unique_configurations": 230, "screen_jobs": 230, "confirmation_jobs": 0, "unschedulable": 0, "budget_skipped": 0, "unstarted": 15, "pending": 0, "running": 0}

## Coverage

| Family | Complete configs | Candidate pool | Pair coverage | Denominator |
|---|---:|---:|---:|---|
| act_cloglog | 33 | 65 | 0/0 | exact_finite_space |
| act_logit | 29 | 63 | 0/0 | exact_finite_space |
| cred_scorecard | 27 | 65 | 0/0 | exact_finite_space |
| ebm | 1 | 51 | 0/0 | exact_finite_space |
| elastic | 0 | 68 | 0/0 | exact_finite_space |
| extra | 5 | 65 | 0/0 | exact_finite_space |
| rff | 53 | 60 | 0/0 | exact_finite_space |
| spline | 26 | 57 | 0/0 | exact_finite_space |
| woe_scorecard | 25 | 59 | 0/0 | exact_finite_space |
| lgb_cpu | 8 | 110 | 0/0 | exact_finite_space |
| lgb_opencl | 10 | 110 | 0/0 | exact_finite_space |

## Verified scoreboards

### screen:86ecc034dd9d3bd05ba26549e15f947413a817a50efc5cc29666d191b8cec081:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_000118 | act_cloglog | 0.516214091 | 34.968 | `runs/job_000118/attempt_001` |
| job_000122 | act_logit | 0.514971394 | 39.565 | `runs/job_000122/attempt_001` |
| job_000003 | ebm | 0.514689845 | 614.188 | `runs/job_000003/attempt_001` |
| job_000097 | act_logit | 0.514589384 | 47.840 | `runs/job_000097/attempt_001` |
| job_000107 | spline | 0.51418132 | 42.024 | `runs/job_000107/attempt_001` |
| job_000086 | act_logit | 0.514138178 | 35.380 | `runs/job_000086/attempt_001` |
| job_000217 | act_cloglog | 0.514066124 | 22.567 | `runs/job_000217/attempt_001` |
| job_000212 | act_logit | 0.513623688 | 25.213 | `runs/job_000212/attempt_001` |
| job_000083 | act_logit | 0.513597526 | 23.548 | `runs/job_000083/attempt_001` |
| job_000137 | act_logit | 0.513481971 | 31.958 | `runs/job_000137/attempt_001` |
| job_000147 | act_cloglog | 0.513459313 | 34.751 | `runs/job_000147/attempt_001` |
| job_000010 | lgb_opencl | 0.513192211 | 67.166 | `runs/job_000010/attempt_001` |
| job_000114 | act_cloglog | 0.513059856 | 47.150 | `runs/job_000114/attempt_001` |
| job_000045 | act_cloglog | 0.512938101 | 27.692 | `runs/job_000045/attempt_001` |
| job_000096 | act_cloglog | 0.512822025 | 47.319 | `runs/job_000096/attempt_001` |
| job_000143 | spline | 0.512760555 | 33.479 | `runs/job_000143/attempt_001` |
| job_000077 | act_cloglog | 0.51261316 | 45.191 | `runs/job_000077/attempt_001` |
| job_000138 | act_logit | 0.51242999 | 40.973 | `runs/job_000138/attempt_001` |
| job_000196 | act_cloglog | 0.511781868 | 14.580 | `runs/job_000196/attempt_001` |
| job_000157 | lgb_cpu | 0.511406758 | 106.694 | `runs/job_000157/attempt_001` |

## Matched ablations

See summary.json for matched control/treatment differences. Unmatched experiments are not presented as ablations.

## Limitations and unresolved questions

- Search scores are selection-biased; confirmation results are reported separately.
- CPU seconds are sampled process-tree lower bounds; GPU allocation-seconds are not measured busy GPU time.
- Pair coverage concerns declared categorical factors within families; it does not prove higher-order or global optimality.
- elastic: H_elastic; matched champion and feature-block exploration

## Reproduction

Each attempt contains spec.json, process.json, environment.json, and immutable source references. Replay one with:

`/home/agung/faris/exploration-12h-20261017/.venv/bin/python /home/agung/faris/exploration-12h-20261017/campaign/runtime/campaign.py replay /home/agung/faris/exploration-12h-20261017/campaign JOB_ID --output NEW_DIRECTORY`

All retained model and validation artifacts are indexed in records/*.json. No submission files are produced.

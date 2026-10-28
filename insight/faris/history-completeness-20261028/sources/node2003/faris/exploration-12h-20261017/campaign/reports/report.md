# Experiment report: explore12h-pc3

State: **finished**. Stop reason: deadline.

## Outcomes

{"complete": 924, "pruned": 0, "failed": 5, "interrupted": 0, "timeout": 2, "invalid": 0, "attempts": 938, "retries": 7, "jobs": 1087, "unique_configurations": 923, "screen_jobs": 923, "confirmation_jobs": 2, "unschedulable": 0, "budget_skipped": 156, "unstarted": 0, "pending": 0, "running": 0}

## Coverage

| Family | Complete configs | Candidate pool | Pair coverage | Denominator |
|---|---:|---:|---:|---|
| act_cloglog | 64 | 64 | 0/0 | exact_finite_space |
| act_logit | 75 | 75 | 0/0 | exact_finite_space |
| cat_native | 48 | 107 | 0/0 | exact_finite_space |
| cred_scorecard | 80 | 80 | 0/0 | exact_finite_space |
| deepsets | 36 | 100 | 0/0 | exact_finite_space |
| ebm | 48 | 48 | 0/0 | exact_finite_space |
| elastic | 71 | 71 | 0/0 | exact_finite_space |
| extra | 72 | 72 | 0/0 | exact_finite_space |
| ft | 27 | 118 | 0/0 | exact_finite_space |
| gru | 48 | 124 | 0/0 | exact_finite_space |
| resmlp | 36 | 118 | 0/0 | exact_finite_space |
| rff | 64 | 64 | 0/0 | exact_finite_space |
| spline | 60 | 60 | 0/0 | exact_finite_space |
| tabm | 58 | 87 | 0/0 | exact_finite_space |
| tabr | 0 | 94 | 0/0 | exact_finite_space |
| woe_scorecard | 68 | 68 | 0/0 | exact_finite_space |
| xgb | 61 | 90 | 0/0 | exact_finite_space |

## Verified scoreboards

### screen:86ecc034dd9d3bd05ba26549e15f947413a817a50efc5cc29666d191b8cec081:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_000846 | tabm | 0.519590939 | 152.001 | `runs/job_000846/attempt_001` |
| job_000030 | xgb | 0.518176623 | 23.275 | `runs/job_000030/attempt_001` |
| job_000424 | elastic | 0.518039399 | 490.029 | `runs/job_000424/attempt_001` |
| job_000668 | tabm | 0.517974371 | 62.301 | `runs/job_000668/attempt_001` |
| job_000803 | tabm | 0.517695913 | 279.347 | `runs/job_000803/attempt_001` |
| job_000531 | elastic | 0.517651891 | 70.761 | `runs/job_000531/attempt_001` |
| job_000918 | tabm | 0.517451632 | 59.986 | `runs/job_000918/attempt_001` |
| job_000815 | cat_native | 0.517299961 | 111.520 | `runs/job_000815/attempt_001` |
| job_000833 | tabm | 0.517163797 | 182.762 | `runs/job_000833/attempt_001` |
| job_000665 | elastic | 0.517114743 | 60.211 | `runs/job_000665/attempt_001` |
| job_000667 | elastic | 0.51706364 | 426.841 | `runs/job_000667/attempt_001` |
| job_000824 | tabm | 0.517032555 | 150.272 | `runs/job_000824/attempt_001` |
| job_000605 | tabm | 0.517027111 | 97.264 | `runs/job_000605/attempt_001` |
| job_000326 | elastic | 0.517017461 | 87.091 | `runs/job_000326/attempt_001` |
| job_000627 | elastic | 0.516970689 | 153.252 | `runs/job_000627/attempt_001` |
| job_000130 | ebm | 0.516914609 | 153.262 | `runs/job_000130/attempt_001` |
| job_000873 | tabm | 0.516755391 | 247.074 | `runs/job_000873/attempt_001` |
| job_000027 | tabm | 0.516718736 | 133.413 | `runs/job_000027/attempt_001` |
| job_000819 | tabm | 0.516676941 | 97.121 | `runs/job_000819/attempt_001` |
| job_000838 | tabm | 0.516661434 | 152.725 | `runs/job_000838/attempt_001` |
### reseed:86ecc034dd9d3bd05ba26549e15f947413a817a50efc5cc29666d191b8cec081:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_001079 | elastic | 0.518080727 | 288.412 | `runs/job_001079/attempt_001` |
| job_001082 | tabm | 0.517945622 | 135.512 | `runs/job_001082/attempt_001` |
### repeat:a4ed8aa03f8bd00899777cff0a900905d3595e88c7b679036ca2a00ce138e181:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_001083 | tabm | 0.516328011 | 138.691 | `runs/job_001083/attempt_001` |
| job_001080 | elastic | 0.516296801 | 380.825 | `runs/job_001080/attempt_001` |
### temporal:86ecc034dd9d3bd05ba26549e15f947413a817a50efc5cc29666d191b8cec081:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_001084 | tabm | 0.512808081 | 53.738 | `runs/job_001084/attempt_001` |
| job_001081 | elastic | 0.507844156 | 68.720 | `runs/job_001081/attempt_001` |
### confirm:6fccc5c672a802ab759367b7c319aefaca88f4e369dfdd1e4d7d46b717badeb5:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_001086 | tabm | 0.530268108 | 71.216 | `runs/job_001086/attempt_001` |
| job_001085 | elastic | 0.524590331 | 160.941 | `runs/job_001085/attempt_001` |

## Matched ablations

See summary.json for matched control/treatment differences. Unmatched experiments are not presented as ablations.

## Limitations and unresolved questions

- Search scores are selection-biased; confirmation results are reported separately.
- CPU seconds are sampled process-tree lower bounds; GPU allocation-seconds are not measured busy GPU time.
- Pair coverage concerns declared categorical factors within families; it does not prove higher-order or global optimality.
- tabr: H_tabr; matched champion and feature-block exploration

## Reproduction

Each attempt contains spec.json, process.json, environment.json, and immutable source references. Replay one with:

`/mnt/home/faris/faris/exploration-12h-20261017/.venv/bin/python /mnt/home/faris/faris/exploration-12h-20261017/campaign/runtime/campaign.py replay /mnt/home/faris/faris/exploration-12h-20261017/campaign JOB_ID --output NEW_DIRECTORY`

All retained model and validation artifacts are indexed in records/*.json. No submission files are produced.

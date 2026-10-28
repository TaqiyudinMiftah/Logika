# Experiment report: explore12h-pc1

State: **finished**. Stop reason: deadline.

## Outcomes

{"complete": 788, "pruned": 0, "failed": 0, "interrupted": 0, "timeout": 0, "invalid": 0, "attempts": 801, "retries": 13, "jobs": 788, "unique_configurations": 776, "screen_jobs": 776, "confirmation_jobs": 3, "unschedulable": 0, "budget_skipped": 0, "unstarted": 0, "pending": 0, "running": 0}

## Coverage

| Family | Complete configs | Candidate pool | Pair coverage | Denominator |
|---|---:|---:|---:|---|
| act_cloglog | 65 | 65 | 0/0 | exact_finite_space |
| act_logit | 63 | 63 | 0/0 | exact_finite_space |
| champion | 3 | 3 | 0/0 | exact_finite_space |
| cred_scorecard | 65 | 65 | 0/0 | exact_finite_space |
| ebm | 51 | 51 | 0/0 | exact_finite_space |
| elastic | 68 | 68 | 0/0 | exact_finite_space |
| extra | 65 | 65 | 0/0 | exact_finite_space |
| rff | 60 | 60 | 0/0 | exact_finite_space |
| spline | 57 | 57 | 0/0 | exact_finite_space |
| woe_scorecard | 59 | 59 | 0/0 | exact_finite_space |
| lgb_cpu | 110 | 110 | 0/0 | exact_finite_space |
| lgb_opencl | 110 | 110 | 0/0 | exact_finite_space |

## Verified scoreboards

### screen:86ecc034dd9d3bd05ba26549e15f947413a817a50efc5cc29666d191b8cec081:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_000245 | champion | 0.517564134 | 103.733 | `runs/job_000245/attempt_001` |
| job_000677 | elastic | 0.517128973 | 145.782 | `runs/job_000677/attempt_001` |
| job_000264 | ebm | 0.516954464 | 117.577 | `runs/job_000264/attempt_001` |
| job_000538 | ebm | 0.516840094 | 147.998 | `runs/job_000538/attempt_001` |
| job_000328 | ebm | 0.516769442 | 152.611 | `runs/job_000328/attempt_001` |
| job_000246 | champion | 0.516722491 | 124.319 | `runs/job_000246/attempt_001` |
| job_000433 | elastic | 0.516614896 | 118.976 | `runs/job_000433/attempt_001` |
| job_000441 | elastic | 0.516520059 | 136.019 | `runs/job_000441/attempt_001` |
| job_000449 | elastic | 0.516393094 | 37.132 | `runs/job_000449/attempt_001` |
| job_000557 | elastic | 0.516371441 | 192.630 | `runs/job_000557/attempt_001` |
| job_000551 | ebm | 0.516357048 | 91.361 | `runs/job_000551/attempt_001` |
| job_000247 | champion | 0.516347615 | 75.872 | `runs/job_000247/attempt_001` |
| job_000585 | lgb_cpu | 0.516288373 | 75.419 | `runs/job_000585/attempt_001` |
| job_000118 | act_cloglog | 0.516214091 | 34.968 | `runs/job_000118/attempt_001` |
| job_000244 | spline | 0.516159359 | 38.771 | `runs/job_000244/attempt_001` |
| job_000617 | ebm | 0.516098949 | 143.885 | `runs/job_000617/attempt_001` |
| job_000226 | ebm | 0.516080459 | 551.930 | `runs/job_000226/attempt_002` |
| job_000647 | ebm | 0.515959693 | 506.964 | `runs/job_000647/attempt_001` |
| job_000667 | lgb_cpu | 0.515753399 | 75.872 | `runs/job_000667/attempt_001` |
| job_000620 | elastic | 0.515652128 | 178.485 | `runs/job_000620/attempt_001` |
### reseed:86ecc034dd9d3bd05ba26549e15f947413a817a50efc5cc29666d191b8cec081:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_000779 | champion | 0.516722491 | 78.668 | `runs/job_000779/attempt_001` |
| job_000782 | champion | 0.515011265 | 49.182 | `runs/job_000782/attempt_001` |
| job_000776 | champion | 0.513816274 | 57.212 | `runs/job_000776/attempt_001` |
### repeat:a4ed8aa03f8bd00899777cff0a900905d3595e88c7b679036ca2a00ce138e181:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_000777 | champion | 0.51817276 | 58.293 | `runs/job_000777/attempt_001` |
| job_000780 | champion | 0.514433741 | 76.783 | `runs/job_000780/attempt_001` |
| job_000783 | champion | 0.513157388 | 50.317 | `runs/job_000783/attempt_001` |
### temporal:86ecc034dd9d3bd05ba26549e15f947413a817a50efc5cc29666d191b8cec081:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_000778 | champion | 0.508467532 | 22.660 | `runs/job_000778/attempt_001` |
| job_000784 | champion | 0.50382684 | 23.539 | `runs/job_000784/attempt_001` |
| job_000781 | champion | 0.499070707 | 33.555 | `runs/job_000781/attempt_001` |
### confirm:6fccc5c672a802ab759367b7c319aefaca88f4e369dfdd1e4d7d46b717badeb5:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_000787 | champion | 0.528074388 | 21.568 | `runs/job_000787/attempt_001` |
| job_000785 | champion | 0.523922206 | 23.798 | `runs/job_000785/attempt_001` |
| job_000786 | champion | 0.522799651 | 33.366 | `runs/job_000786/attempt_001` |

## Matched ablations

See summary.json for matched control/treatment differences. Unmatched experiments are not presented as ablations.

## Limitations and unresolved questions

- Search scores are selection-biased; confirmation results are reported separately.
- CPU seconds are sampled process-tree lower bounds; GPU allocation-seconds are not measured busy GPU time.
- Pair coverage concerns declared categorical factors within families; it does not prove higher-order or global optimality.

## Reproduction

Each attempt contains spec.json, process.json, environment.json, and immutable source references. Replay one with:

`/home/agung/faris/exploration-12h-20261017/.venv/bin/python /home/agung/faris/exploration-12h-20261017/campaign/runtime/campaign.py replay /home/agung/faris/exploration-12h-20261017/campaign JOB_ID --output NEW_DIRECTORY`

All retained model and validation artifacts are indexed in records/*.json. No submission files are produced.

# Experiment report: explore12h-synthetic-smoke

State: **finished**. Stop reason: search_exhausted.

## Outcomes

{"complete": 4, "pruned": 0, "failed": 0, "interrupted": 0, "timeout": 0, "invalid": 0, "attempts": 4, "retries": 0, "jobs": 4, "unique_configurations": 4, "screen_jobs": 4, "confirmation_jobs": 0, "unschedulable": 0, "budget_skipped": 0, "unstarted": 0, "pending": 0, "running": 0}

## Coverage

| Family | Complete configs | Candidate pool | Pair coverage | Denominator |
|---|---:|---:|---:|---|
| cpu | 4 | 4 | 0/0 | exact_finite_space |

## Verified scoreboards

### screen:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a:full

| Job | Family | Score | Seconds | Evidence |
|---|---|---:|---:|---|
| job_000001 | cpu | 1 | 0.797 | `runs/job_000001/attempt_001` |
| job_000002 | cpu | 1 | 1.432 | `runs/job_000002/attempt_001` |
| job_000003 | cpu | 1 | 0.786 | `runs/job_000003/attempt_001` |
| job_000000 | cpu | 0.935483871 | 1.085 | `runs/job_000000/attempt_001` |

## Matched ablations

See summary.json for matched control/treatment differences. Unmatched experiments are not presented as ablations.

## Limitations and unresolved questions

- Search scores are selection-biased; confirmation results are reported separately.
- CPU seconds are sampled process-tree lower bounds; GPU allocation-seconds are not measured busy GPU time.
- Pair coverage concerns declared categorical factors within families; it does not prove higher-order or global optimality.
- No protected confirmation protocol supplied; no independent generalization claim.

## Reproduction

Each attempt contains spec.json, process.json, environment.json, and immutable source references. Replay one with:

`/mnt/home/logika-faris-exploration/exploration-12h-20261017/.venv/bin/python /mnt/home/logika-faris-exploration/exploration-12h-20261017/deployment-smoke/campaign/runtime/campaign.py replay /mnt/home/logika-faris-exploration/exploration-12h-20261017/deployment-smoke/campaign JOB_ID --output NEW_DIRECTORY`

All retained model and validation artifacts are indexed in records/*.json. No submission files are produced.

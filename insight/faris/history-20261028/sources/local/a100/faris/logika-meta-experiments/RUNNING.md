# Campaign launched

Started: 2026-10-16T17:29:39.888708+07:00 (Asia/Jakarta).

Deadline: 2026-10-17T05:29:39.888708+07:00 (Asia/Jakarta).

All three hosts passed deployment and acceptance tests. Startup readback confirms active workers and the configured CPU/RAM/swap groups. The first complete three-fold recipes were collected and independently verified.

Registered: 1,024 representations, 4,096 pipeline recipes, and up to 8,192 fixed ensemble comparisons, with separate nested learned-ensemble checks. Registration is not completion; final reports retain actual counts and unfinished work.

The coordinator runs independently on PC2 in `/home/faris/faris/logika-meta-experiments/meta-20261016-v1`. Its automatic report is `reports/final_report.md`; the A100 handoff is `reports/a100_handoff.json`.

Status on PC2, from that directory:

```sh
.venv/bin/python -m metaexp status
```

See `launch_verification.json` for the timestamped startup snapshot and `README.md` for the CLI and artifact layout. The laptop can disconnect. No shutdown, reboot, or termination of other workloads is part of the system.

## Independent parallel execution upgrade

Each PC now stores its entire assigned remainder of the catalogue and starts
experiments from its own persistent queue. PC1 runs up to 16 workers; PC2 and PC3
run up to 7 each, including GPU work. The resource limits and original deadline
above remain unchanged. Each machine admits at most one GPU experiment at a time.
CPU CatBoost/XGBoost work can use otherwise spare CPU slots; the backend is retained
in the evidence, matched comparisons, and finalist confirmation.

PC2 collects and independently verifies results, computes ensemble comparisons,
and coordinates confirmation/reporting. Local feature-search execution does not
depend on its queue refills or a working collection connection. Disconnected nodes
retain evidence locally; PC1 pauses admission at its disk budget until verified
archives can be transferred and compacted. No recipe is reassigned merely because
its owner is disconnected.

The initial autonomous delivery installed 299 tasks on PC1, 373 on PC2, and 371 on
PC3. These are the remaining owned tasks at migration time, including work already
in flight; they do not add to the catalogue's logical experiment count.

Live verification paused only the campaign collector for 35 seconds. During that
pause PC1 started 6 new jobs, PC2 started 7, and PC3 started 11. All three executors
continued, with 16/7/7 active workers and effective systemd CPU/RAM/swap limits.
The verification snapshot recorded 3,302 distinct verified full-CV pipelines.
This is a timestamped progress count, not the final campaign total.

`autonomous_execution_verification.json` contains the pause/resume and per-node
evidence; `autonomous_campaign_snapshot.json` contains the ledger and delivery
snapshot. `scheduler_tests.py` passed 13 orchestration checks. The frozen training
source hash, catalogue, data identity, splits, and campaign clock were preserved.
User-service lingering is enabled on all three PCs so their services survive the
last SSH session closing. No other workload was stopped.

The live service restarts into `scheduler_parallel.py` automatically. If manual
recovery is needed before the original deadline, use this on PC2 from its campaign
directory to preserve the upgraded scheduler:

```sh
.venv/bin/python upgrade_scheduler.py
```

Use the normal `status` and `stop` commands above. Do not redeploy the original
launcher over this active campaign. The new implementation and source revision
records are `scheduler_parallel.py`, `autonomous_shards.py`, `node_executor.py`,
`rpc_parallel.py`, and `scheduler_revision.json`.

## Second wave uses the remaining original window

A separate development-only extension is deployed at
`~/faris/logika-meta-experiments/meta-20261016-v1-wave2/` on each PC. It adds 4,096
representations and 16,384 new pipeline recipes, plus repeated controls. Its
deadline remains 17 October, 05:29 WIB, with the same host resource limits.

The first-wave nomination and confirmation evidence remain frozen. New results
use development CV and are reported separately. Read current status from the
second-wave directory; the parent status is historical. See
`extensions/wave2/README.md` and `research_extension.json` on PC2.

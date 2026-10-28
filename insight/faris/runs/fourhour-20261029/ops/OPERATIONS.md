# Four-hour campaign operations

Operational record only. Scientific `source/campaign.py`, PLAN, SOURCE_SEAL and CLOCK remain immutable. Parent six-hour study is complete and unchanged. This campaign has54registered designs and540planned meta fits, not a completion claim.

| Boundary | UTC | User time |
|---|---|---|
| Start | 2026-10-28 23:55:12 UTC | 2026-10-29 06:55:12 WIB |
| Admissions end | 2026-10-29 03:10:12 UTC | 2026-10-29 10:10:12 WIB |
| Models end | 2026-10-29 03:35:12 UTC | 2026-10-29 10:35:12 WIB |
| All computation ends | 2026-10-29 03:55:12 UTC | 2026-10-29 10:55:12 WIB |

The user service started25seconds before the first-fit clock; its14400second RuntimeMaxSec is conservatively earlier than the scientific ceiling. Never reset/extend either. Complete and audit early when every admitted task is terminal; never pad runtime. No newbasefits, paidallocations, weights/KVcaches orKaggle calls occur in this campaign. The separate fixed-winner submission belongs to another authorized run.

Canonical root: `faris@100.68.29.126:/mnt/home/faris/logika-fourhour-20261029`. Node2001service `logika-fourhour-20261029.service`; node2003worker `logika-fourhour-worker-20261029.service`. Scientific and runtime gates passed23:55UTC, with1.31s/1.78s ordinary runtimepilot fits. Both labs run at2fit slots,2threads per fit and12GiB/400percent service ceilings. Only actual`/mnt/home/faris` homes are allowed.

Initial source prepublication commit: `efb40dce6573ce849b6324ae252831302413f04c`. Original clock SHA256: `ab43e25c8e5e84ab5cb106405b3f03b87470023c66e1df47ffb2710b07326e49`. Source-seal SHA256: `8200c0bd8079c48a4395d414f6fa1de67730640f2667f7b63424162416de4369`. Root owns Github synchronization and the30minute app follow-up.

`ops/backup_progress.py` is the workstation-only incremental private collector. Root launched PID17728 and records its identity in`private/BACKUP_PID`; inspect the actual PID before acting, as processes can exit. Do not launch a duplicate. It copies compact prediction/metadata evidence under`INCREMENTAL_EVIDENCE/`, validates completed RESULT/FROZEN/prediction hashes, and writes`private/PRIVATE_BACKUP_MANIFEST.json` plus backup receipts. It forecasts changed transfer bytes and preserves512MiB free. If space is insufficient, it defers while canonical evidence remains intact. It never copies worker packets, raw data, labels or fitted weights.

`ops/publish.py` publishes only sanitized aggregate JSON/Markdown plus operational source, withholding case-level predictions and labels. It fetches peer main, preserves peer commits, scans secrets, normally pushes without force and verifies the remote commit. Namespace is`insight/faris/runs/fourhour-20261029`. Initial root publication must be complete before launch, and root must explicitly authorize publisher launch to avoid concurrent Git writes. At this record's creation publisher launch is pending root's separate submission-outcome publication.

Workstation commands, run from competition root only after checking PIDfiles:

```sh
python3 a100/faris/fourhour-20261029/ops/backup_progress.py --once
python3 a100/faris/fourhour-20261029/ops/publish.py --local /home/parasite/Project/Competition/logika-2026/a100/faris/fourhour-20261029 --repo /home/parasite/Project/Competition/logika-2026/.history-publish-20261028 --once
```

Use existing daemons for ordinary synchronization. Do not run either`--once` concurrently with its daemon. Each helper holds its own lock. Publisher PIDfile is`private/publication/PUBLISHER_PID`; latest verified commit is in`private/publication/LATEST_RECEIPT.json`.

Terminal scientific files are`FINAL_SELECTION.json`, `FINAL_EVALUATION.json`, `ARTIFACT_CONSISTENCY_AUDIT.json` and`COMPLETION.json`. Publisher recognizes this exact completion filename and direct`runs/task/RESULT.json` layout. It collects final private compact evidence under`FINAL_EVIDENCE/`, hardlinking identical existing incremental evidence, checks hashes, and writes`private/publication/FINAL_PRIVATE_MANIFEST.json`, `COLLECTION_RECEIPT.json` and`FINAL_RECEIPT.json`. Backup produces`private/FINAL_BACKUP_RECEIPT.json` after matching the terminal audit's completed-fit count. The final report requires these receipts and the finalremotehash.

Interpretation remains reused adaptive development: all70krows previously exposed; no untouched audit. Hash/arithmetic replay validates artifacts, not statistical independence. Unsupported or negative scientific results and failures/partial designs must all be retained. No automatic refit, submission or leaderboard probing follows this study.

# Current MoE campaign handoff — 28 October 2026

The user explicitly chose a new **12-hour ceiling**. This campaign is active;
older completed frozen-feature and architecture campaigns remain sealed and expired.
The latest user continuation reopened feature engineering and modelling and allowed
Colab A100 account2, as recorded in the project and publication AGENTS.md.

- Local root: `a100/faris/feature-subspace-moe-20261028/`.
- Canonical lab root: `faris@100.68.29.126:/mnt/home/faris/logika-feature-subspace-moe-20261028/`.
- Main service: `logika-feature-subspace-moe-20261028.service`; after admitted spillover,
  completion resumes with `logika-moe28-postspillover-resume.service`.
- Immutable clock SHA256: `693c430a2cdfe6cfc6c540e7e1e9480a0ae8944763b4cb1598da90b673456bac`.
- Started 28Oct11:28:37 UTC; admission ends18:58:37 UTC; **all computation ends
  28Oct23:28:37 UTC / 29Oct06:28:37 WIB**. Never reset or extend the clock.

The plan tests four source-backed semantic feature domains, existing versus expanded
views (up to99 predictors per Kumo expert), fixed versus conditional soft gating,
Kumo/LightGBM/CatBoost, two seeds and five original folds. There are2090 candidate
definitions and260 predeclared expert fits. All180 CPU fits and3 lab Kumo fits
completed without failures. Lab GPUs became occupied by unrelated users; leave them alone.

The user-authorized A100 account2 spillover is preregistered. Isolated session:
`logika-moe-20261028-12h`. Its private state/config is in `colab_private/`; never
publish that directory. The actual owned runtime has an80GiB A100 and pinned
Python3.11/Kumo0.2.1/PyTorch2.7.1 environment. It caches only the original public
pretrained checkpoint, whose SHA256 is
`cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613`.
No fitted weights or KV caches are serialized. Every new file in the VM is under its
actual `/root` home. Node2002 is still forbidden; lab work stays under `/mnt/home/faris`.

Read `colab_private/CONTROLLER_STATE.json`, `STREAM_PROGRESS.json`, original
`CLOCK.json`, canonical `LIVE_STATE.json`, and current owned processes. Source/config/
contract/clock are immutable. Only an external execution adapter maps the logical
checkpoint pathname to identical local checkpoint bytes. Before broad admission,
a label-free replay must meet preregistered MAE<=.001,max error<=.02,correlation>=.999
with exact query rows/checkpoint. The first replay passed: MAE0.0001474258,max error0.0019853413,correlation0.9999987807
on exact20000 query rows. See `COLAB_REPLAY_GATE.json`. The A100 cohort was admitted
28Oct12:59:01 UTC; the original waiting coordinator is now inactive, with old lab
Kumo attempts retired and the180 CPU fits retained.

On admission the controller reconciles only owned lab units, retains old Kumo
attempts in `retired_lab_runs/` plus hashes, retains180 CPU fits, and runs all80
Kumo fits on one consistent A100 backend. No old lab outputs are mixed into the
replacement cohort. Private inputs stream during training; each fold's full sealed
manifest/feature/role hashes must verify before a worker can use it. Exact full
archive hashes are checked on collection. Healthy workers must not be interrupted.

The workstation controller is `ops/colab_controller.py` (exclusive lock); do not
launch duplicates. It collects predictions incrementally to private local storage
and to the canonical lab archive, verifies all260 fits, stops its own paid runtime,
and resumes original sealed composition/evaluation using only the original remaining
clock. The notebook kernel also has a release watchdog at the original deadline.
If transport/setup fails, preserve evidence and reconcile actual processes before
repairing; never interpret resource failure as negative model evidence.

Scientific selection is fixed: `kumo__expanded__gated`, two-seed mean. Matched local
baseline0.5306361042400267, publicV25 incumbent0.56822. Gain rule: deltaKS>=.0005,
paired conditional95% lower>0,4of5folds positive,both seeds positive. All70k cases
have prior development exposure; there is no untouched audit. Local0.54 is a
separate target and does not establish public improvement. No new score is claimed.

`logika-moe28-diagnostic-watch.service` runs the predeclared learned-global-weight
control. `logika-moe28-artifact-audit-watch.service` independently reads back the
260 fits, original seeds/row boundaries, source/prediction hashes, baseline/cohort
KS, and conditional interval. This is artifact consistency, not new independent
data. A supported primary requires audited frozen refit within the remaining
original clock; no automatic submission follows a partial run or point estimate.

Every plan and completed/failed/partial result belongs in
`git@github.com:TaqiyudinMiftah/Logika.git` under
`insight/faris/runs/feature-subspace-moe-20261028/`. Workstation
`ops/publish_progress.py` preserves peer commits, scans for secrets, normally pushes
aggregate evidence and verifies the remote commit. Credentials, raw data, case
labels/predictions and fitted weights remain private. It waits for terminal diagnostic
and artifact audit, then collects compact final evidence including retired lab
attempts. Do not stage concurrently with this publisher; inspect its PID/lock and
pause only that owned local process around a manual publication.

# Completed MoE campaign handoff — 28 October 2026

**COMPLETE. No supported gain. No active model workers or paid Colab runtime should
be resumed.** See RESULTS.md, ANALYSIS.json and FINAL_LOCAL_COLLECTION_AUDIT.json.

-260 planned expert fits completed; zero expert-fit failures or unresolved work.
- Frozen primary KS0.5295070116 versus matched anchor0.5306361042.
- Best secondary point0.5306791136 is inconsistent across folds/seeds and not promoted.
- Local0.54 target unmet; retained public V25 score0.56822 is a different artifact.
- No refit or submission; all70k cases are reused development, without untouched audit.
- All80 Kumo fits used account2 A100. Three superseded lab attempts remain separately
  preserved with hashes. The180 CPU fits were retained.
- Colab session logika-moe-20261028-12h was stopped after verified collection at
  28Oct14:46:27 UTC /21:46:27 WIB. Final live account meter:0CU/hour,zero assignments.
  Observed rounded balance decrease:15.18CU. No further allocation is needed for this completed comparison. Its unused
  ceiling does not justify restarting sealed work.
- Artifact/score recomputation audit PASS; local260-fit and3-retired-output collection
  audit PASS.1262 private final artifacts,171349676bytes. No fitted weights/KV caches.

Local root: a100/faris/feature-subspace-moe-20261028/.
Canonical archive: faris@100.68.29.126:/mnt/home/faris/logika-feature-subspace-moe-20261028/.
Private predictions: FINAL_EVIDENCE/ locally; runs/,composed/,evaluation/ on lab.
Full private hash manifest: publication_private/FINAL_PRIVATE_MANIFEST.json.
Public namespace: insight/faris/runs/feature-subspace-moe-20261028/ in
 git@github.com:TaqiyudinMiftah/Logika.git.

Original immutable clock SHA256:
693c430a2cdfe6cfc6c540e7e1e9480a0ae8944763b4cb1598da90b673456bac.
Start28Oct11:28:37 UTC; admission18:58:37; total deadline23:28:37 UTC.
Never reset, extend, relaunch or overwrite source/config/contract/clock/FROZEN.
The former main coordinator, A100 controller, postspillover coordinator, diagnostic
watcher, artifact watcher and publisher completed their work. Reconcile actual
service/process state before any repair; do not create a new VM to inspect this
stopped session. Read artifacts first.

Newer user instructions reopened feature/modelling work and permitted account2;
older completed campaigns' freezes and expired clocks remain unchanged. Node2002
is still retired. Preserve source/parameters/seeds/preprocessing/compact predictions;
never serialize new fitted weights/KV caches or publish raw cases/credentials.
Fetch peers before a new study and before any normal non-forced push. This
completed study's selection and negative result cannot be rewritten after seeing
its outer scores. Any new work must be separately declared and honestly labeled
as adaptive reused-development research.

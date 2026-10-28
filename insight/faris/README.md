# Faris experiment history and ongoing work

This workspace records Faris/Codex/Antigravity experiments alongside the existing
[Adit research](../adit/). The shared repository is
`git@github.com:TaqiyudinMiftah/Logika.git`.

## Historical import

The [28 October archive](history-20261028/README.md) preserves experiment sources,
configurations, seeds, aggregate reports, failure logs and provenance. Its source
inventory maps mirrored copies and private evidence to published records. It does
not publish credentials, raw data, case-level labelled OOF or fitted models.

The discovered accessible historical trees have been inventoried and imported. Read
[IMPORT_STATUS.json](history-20261028/IMPORT_STATUS.json) for exactly which sources
have been included and which verification/push stages have passed. Coverage is
bounded by the documented discovered roots, including the retired-node limitation.
File/source-reference counts are not independent experiment counts.

The [coverage correction](history-completeness-20261028/README.md) adds 1,067
missing files found during a later root/vendor audit. The earlier universal
completeness claim was too broad; coverage is limited to documented accessible
roots. Node2002 remains unverified beyond surviving mirrors.

## Completed MoE campaign — 28 October

The [feature-subspace MoE study](runs/feature-subspace-moe-20261028/RESULTS.md)
completed all260 planned expert fits across5folds and2seeds. Frozen primary KS
was0.5295070116 versus matched anchor0.5306361042. The best secondary point was
0.5306791136, inconsistent across folds/seeds and not promoted. Target0.54 was
not reached; retained publicV25 remains0.56822. No refit or submission was made.

All260 outputs and3 superseded lab attempts were saved and hash-verified. Source,
seeds,row boundaries,baseline/cohort KS and conditional uncertainty audits passed;
all70k cases still constitute reused development, not an independent audit. The
A100 account2 runtime was released at28October14:46:27 UTC after collection,
with about15.18CU observed consumption. No paid runtime or campaign worker remains
active. The original12hour clock and scientific seals are unchanged. See the
[current handoff](runs/feature-subspace-moe-20261028/CURRENT_HANDOFF.md).

## Completed Antigravity continuation

The user asked to continue Antigravity conversation
`73181457-58b8-405a-bb23-a8afbd02f051`, then required the historical import and peer
sync. The latest instruction explicitly authorizes refit work in parallel while
the import runs. [Continuation provenance](history-20261028/ANTIGRAVITY_CONTINUATION.json)
records the latest explicit scope: features and modelling may both change; the
requested highest exploratory mix should receive test predictions; Colab A100
account 2 is allowed. These instructions supersede the old feature/A100 freeze
for this new continuation, without altering sealed completed campaigns.

Antigravity's full-OOF-selected routing/blend/shift scores are optimistic. Its later
cross-fitted selection check did not establish an improvement; it also is not a
fresh untouched audit of the whole historical research process. The user explicitly
requested the exploratory mix CSV after this was explained. Creating that artifact
does not establish KS 0.54 or a public gain. The ALL66 refit completed; its public KS **0.55983** is below V25 **0.56822**.
See [the run record](runs/agy-all66-refit-20261028/README.md). It is rejected
for promotion.

## Every new experiment

1. Fetch and read peer changes; compare configuration/data/feature/fold/backbone
   identity against the shared registry. Give intentional replications a reason.
2. Record the hypothesis and exact protocol before execution with
   `python tools/log_experiment.py start --id ID --hypothesis TEXT --config CONFIG.json --protocol PROTOCOL.json`.
3. Run with private case-level artifacts and preserve failures as well as successes.
4. Record the outcome using
   `python tools/log_experiment.py finish --id ID --status completed --results RESULTS.json --private-artifact /private/path/OOF.npz`.
   Use `failed`, `incomplete` or `cancelled` when appropriate.
5. Review sanitized public changes, fetch again, reconcile concurrent peer commits,
   commit, and use `python tools/sync_experiments.py push --receipt /private/path/sync.json`.
   The helper never force pushes and verifies the remote commit.

The existing [shared sync protocol](../adit/v25_research/SYNC_PROTOCOL.md) remains
authoritative for peer coordination and public/private evidence separation.

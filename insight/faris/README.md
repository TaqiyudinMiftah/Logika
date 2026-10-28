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

## Current continuation

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
does not establish KS 0.54 or a public gain. No new refit is claimed until logged
runtime evidence exists.

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

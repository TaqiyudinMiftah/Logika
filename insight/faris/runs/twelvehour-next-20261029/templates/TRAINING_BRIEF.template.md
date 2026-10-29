# Continuous training brief — <slug>

This sidecar records the campaign; it supplements the project's `AGENTS.md`.
Include its full content in every worker prompt and read applicable project
instructions before executing the assignment.

- Competition root and worker working directory: <competition_root>
- Metric: <metric_name>
- Direction: <direction>
- Train data, read only: <train_path>
- Test for prediction, read only: <test_path>
- Compute: <compute_description>
- Campaign start UTC: <start_utc>
- Campaign end UTC: <end_utc>
- Original hours: <hours>
- Live deadline check: `DEADLINE.md`
- Supplied rules file: <rules_file>
- Rules SHA-256: <rules_sha256>

## Supplied competition rules

<pasted_rules>

## Campaign boundaries

- Train only on supplied train labels. Test targets remain unseen. Do not decode
  hidden labels from caches, artifacts, embeddings, or prediction files.
- No submissions, uploads, leaderboard calls, or leaderboard probing during this
  campaign. Prediction files are held for a subsequent explicit user command.
- Default team bound: at most three running workers plus one coordinator; use a
  lower number if the runtime or measured compute allows fewer.
- Start workers in the competition root. Forks inherit the parent's currently
  persisted transcript; fresh sessions have only their supplied prompt.
- Each worker writes only its assigned `models/<run_id>/` folder. Read shared data,
  project instructions, frozen splits, code, and prior evidence without modifying
  them. The coordinator owns shared ledgers, worker state, and held predictions.
- Record one hypothesis before training and retain executable code, logs, exact
  split assignments, OOF predictions, package versions, and finite fold scores.
  A reply or score JSON alone does not establish a completed experiment.
- The coordinator independently recomputes local metrics before accepting gains.
  Only the coordinator promotes a prediction file to `submissions_held/` after
  row count, ID order, headers, finiteness, and task constraint checks.
- Check the preserved deadline before each run. Reserve time for verification and
  reporting, and return partial status if interrupted or out of time.

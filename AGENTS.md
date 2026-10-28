# Shared Logika experiment record

- Before every new modelling experiment, fetch the latest `main`, read peer changes
  since the last sync receipt, and check existing experiment signatures for duplication.
  Preserve the scoped instructions and sync protocol under `insight/adit/v25_research`.
- Log the hypothesis, exact configuration, seeds, data/feature/fold identities, source
  commit and intended validation before execution. Log completed, failed and partial
  outcomes, aggregate metrics, uncertainty and limitations afterward. New Faris runs
  belong under `insight/faris/runs/<stable-experiment-id>/`.
- Use `tools/log_experiment.py start` and `finish` or equivalent records with the same
  fields. Maintain negative results. A file count or repeated assessment is not an
  independent experiment count. Do not select on outer labels and call that independent.
- Keep credentials, raw data, case-level labels/OOF, fitted weights and KV caches out
  of this public repository. Retain them privately with size/hash/location references.
  The user requested parameter/source/prediction evidence retention rather than new
  fitted-weight dumps. Sanitize logs before staging them.
- Before pushing, fetch again, preserve concurrent peer commits, and push only as a
  normal fast-forward. Never force push, reset another person's checkout, overwrite
  peer results or silently drop experiments. Verify the remote commit after pushing.
- The 28 October Antigravity continuation explicitly reopened feature AND modelling
  changes and later permitted Colab A100 account 2. Those instructions supersede the
  older Faris handoff's feature/A100 restrictions for that continuation. Do not rewrite
  the immutable completed campaigns or assume their expired clocks are reusable.
- Prefer RTK for supported commands when reduced output is enough. Preserve errors,
  diagnostics and exit status; keep raw output available. Use ordinary structured
  commands when exact evidence is needed.

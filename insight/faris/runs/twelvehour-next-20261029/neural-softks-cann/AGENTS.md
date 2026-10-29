# Command output

- Prefer RTK for supported commands when reduced output is sufficient.
- For recurring, verbose custom commands, create and reuse a narrowly scoped RTK
  filter when the benefit justifies the setup.
- Match the specific script and operation. Preserve errors, useful diagnostics and
  exit status; keep raw output accessible. Validate filters against captured output
  without repeating remote actions. Use standard commands when unsuitable.

# Shared Logika experiments

- The user requires every experiment to be recorded in
  `git@github.com:TaqiyudinMiftah/Logika.git`, including negative and partial results.
- Pull/fetch peer changes before modelling; inspect their configs, protocol, results
  and registry to avoid duplicate work. Fetch again before pushing; preserve peer
  commits, never force push, and verify the remote commit afterward.
- The current publication checkout is `.history-publish-20261028/`. The older
  `Logika/` checkout contains pre-existing local modifications; preserve them.
- Read the publication checkout's `AGENTS.md`, `insight/faris/README.md` and the
  scoped peer sync protocol. Record config, source, seed, data/features/folds,
  validation, aggregate metrics and private-artifact hashes before/after each run.
- Keep credentials, raw data, case-level labels/OOF and fitted weights private.
- The latest user instructions in Antigravity conversation
  `73181457-58b8-405a-bb23-a8afbd02f051` reopened features and modelling and allowed
  Colab A100 account2. Preserve old completed campaign seals and expired clocks;
  do not mistake their earlier freeze for this continuation's current scope.

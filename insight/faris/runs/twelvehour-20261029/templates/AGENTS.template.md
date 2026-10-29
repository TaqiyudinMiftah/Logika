# AGENTS.md — <slug>

For continuous training, read `TRAINING_BRIEF.md` and `DEADLINE.md` in this
competition root. Every worker must receive the full brief with its assignment.
The coordinator selects an available native, Muse MCP, or CLI agent route and
checks each returned status and artifact. Agents share the workspace; workers
have disjoint output folders under `models/`.

Train labels are the only label source. Shared data is read only. No submissions,
uploads, or leaderboard probing during the experiment loop.

# <run_id> — <area>

<full_training_brief>

## Assignment

- Root and required working directory: <competition_root>
- Read project instructions: <competition_root>/AGENTS.md and applicable nested files.
- Goal: <goal>
- One hypothesis: <hypothesis>
- Frozen split and evaluator: <split_and_evaluator>
- Current verified best: <current_best>
- Relevant measured learnings: <relevant_learnings>
- Assigned area: <area>
- Allowed writes: <competition_root>/models/<run_id>/ only.
- Shared files/data and earlier runs are read only.
- Deadline: reread <competition_root>/DEADLINE.md before starting training.

Read the existing local learnings and outcomes. Research this narrow hypothesis
only as needed using available official sources; if a connector or optional skill
is missing, write grounded notes directly. Record the hypothesis before execution.

Save research/hypothesis notes, runnable code, `train.log`, exact split assignments,
OOF predictions, package versions, and `scores.json` using the continuous-training
results schema. Recompute mean and population standard deviation from finite fold
scores. Retain any trained model and prediction artifacts needed for reproduction.

Check your own record without writing shared reports:
`python3 "<skill_root>/scripts/scoreboard.py" "<competition_root>" --run "<run_id>" --check-only`
Read back the actual files and recompute your metric from saved OOF predictions.
Do not manufacture evidence files or treat score JSON as proof of training.

Return status (`completed`, `partial`, or `failed`), run ID, measured metric/split,
code/log/OOF paths, and remaining limitations. A completed reply is a claim the
coordinator will verify; it does not promote a result or permit submission.

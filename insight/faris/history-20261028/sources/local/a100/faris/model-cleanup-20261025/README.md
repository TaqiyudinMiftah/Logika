# PC2 model cleanup — 25 October 2026

User-authorized scope: delete model artifacts under
`/mnt/home/logika-faris-exploration` on `labkc-comm1`, while keeping outputs,
model settings and code. This includes all projects inside that directory.

Connection verified on 25 October: `ssh faris@100.68.29.126` (port 22).
The requested port 2001 refused the connection. No host change was made.

During preparation, the historical Python 3.11 interpreter outside this root
disappeared; its running process showed a deleted executable. Existing virtual
environment links to that installation are therefore broken. This cleanup did
not delete or repair that installation. System Python handled the cleanup and
metadata recovery, using retained pure-Python joblib code when needed.

## Removal rules

- Delete fitted model binaries, foundation-model weight/context caches,
  checkpoints, fitted preprocessing/selection/reduction objects, and serialized
  tree models explicitly recognized by their file headers.
- Export embedded architecture/settings/hyperparameters and compact linear
  coefficients where readable, retaining references to existing recipe files.
  Large learned weight arrays are described by shape/type, not copied.
- Remove the same artifacts from mixed delivery archives while preserving every
  other member byte for byte. Keep a receipt with the old/new archive hashes.
- Unlink model aliases only; never follow directory symlinks during traversal
  or delete files outside the named root.

## Kept

Submissions; OOF, validation and test predictions; reports and metrics; source
code and notebooks; JSON/YAML/TOML configuration records; datasets and feature
banks; prepared train/test pickles; scheduler/RNG state; dependency environments;
and smoke-test transport files. A directory named `models` is not itself a
removal rule: predictions and settings inside it are kept.

## Evidence

`removal-plan.json` freezes exact removal paths and file identities.
`parameters/` contains one readable metadata/settings export per model file,
with separate records for stripped archive members. `deletions.jsonl` records
the files actually removed. `REPORT.json` is the final verification receipt;
its `success` field must be true before claiming completion.

For every retained file, verification compares inode/device, byte length and
nanosecond modification time. Source files, Markdown, YAML/TOML/INI/CFG records
and submission CSVs up to 8 MiB additionally receive full SHA-256 checks.
Other datasets, prediction arrays and JSON records receive filesystem identity
checks, **not** full-content hashes. Retained members of rewritten archives are fully hashed and
compared before the original archive is replaced. No retained research file is
opened for writing except the two explicitly rewritten mixed archives.

Historical model manifests and old archive `.sha256` files remain historical
evidence. Their model-existence claims no longer describe the cleaned archive;
use cleanup receipts for its post-cleanup state. Inference requires retraining
or obtaining weights from a separate location.

No reboot, poweroff, unrelated-process termination, training, Kaggle upload or
cleanup of another machine/local model archive is authorized by this task.

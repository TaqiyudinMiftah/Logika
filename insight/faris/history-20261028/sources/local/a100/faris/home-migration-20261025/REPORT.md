# Faris home migration — 25 October 2026

Verified success: **True**.

The account's default home, `$HOME`, and working directory on a fresh SSH login
are `/mnt/home/faris`. Login verification used `ssh faris@100.68.29.126`.
UID/GID and shell were preserved; no reboot or logout was required.
The normal account tool refused to edit a logged-in account, so the home field
alone was updated through the locked `vipw` editor. Password/shadow fields were
not changed. A fresh authenticated SSH login verified the result.

## Paths

| Purpose | Current path |
|---|---|
| Default home | `/mnt/home/faris` |
| Logika workspace hub | `/mnt/home/faris/Logika` |
| Logika experiment archive | `/mnt/home/faris/logika-faris-exploration` |
| Current Logika arena | `/mnt/home/faris/logika-model-arena-20261025` |
| Live remote storage mount | `/mnt/home/faris/logika-model-arena-20261025/storage` |
| Original home data backup | `/home/faris.pre-migration-20261025` |

`/home/faris` is now a compatibility symlink to `/mnt/home/faris`.
**Keep this link. Do not recursively delete it or its target.** The original
directory was atomically exchanged with this symlink and remains in the backup
path above; this task did not delete that backup. Old-data backup removal has
not been authorized or performed. The initial copy was made by the user, then
incrementally synchronized; the two services were stopped for the final
license-plate data synchronization. This is not a claim that every byte of
the entire home was independently checksummed.

The former `/mnt/home/logika-faris-exploration` path is also a compatibility
symlink to the archive inside the new home. Existing experiment commands keep
working. `Logika/` is a workspace hub linking the archive, current arena, feature
work and older campaigns; no unrelated repository or model weights were fetched.

## Dependency repair

The Python 3.11.13 installation had already been copied under
`/mnt/home/faris/faris/logika-meta-experiments/meta-20261016-v1/.python/`.
Its missing old path was first restored through a compatibility link.
Copied virtual-environment interpreter links, activation scripts, console
launchers, editable-install pointers and configuration paths were updated for
the new home, with private rollback copies.

**22 environment invocations checked; 0 failed.**
Checks covered old and new invocation paths and actual imports of installed
NumPy, pandas, SciPy, scikit-learn, joblib, LightGBM, CatBoost, XGBoost and PyTorch,
plus SSL, SQLite, secrets and the previously missing encoding module. No model
training was started. This is an import/runtime check, not an exhaustive test
of every optional package or training workflow.

## Running services and storage

The user explicitly authorized stopping/restarting the two license-plate user
services and migrating the 11 affected CVAT containers. The services now run
from `/mnt/home/faris/license-plate`. Their original units have path overrides.
The 11 containers retain their exact prior images and use bind sources under
the new home. The database, Redis, ClickHouse, UI and OPA containers retained
their original container identities and remained running.
The coordinator's existing paused work state was preserved across restart;
no paused processing campaign was resumed.

The SSHFS filesystem was bind-mounted at the new location, verified as Faris,
and unmounted from the old physical directory. The original SSHFS process
remained running; the remote data was not copied or deleted. Root access to
this user-owned FUSE filesystem remains restricted by its existing policy;
access it as `faris`, without `sudo`.

Authenticated license-plate TLS API: **HTTP 200**;
certificate verification: **True**.
CVAT application API: **HTTP 200**.

The mount is currently live. No reboot test or new automatic boot mount was
performed. No reboot, poweroff, other-user session termination, database
recreation, model retraining or competition submission occurred.

## Evidence and rollback records

On PC2: `/mnt/home/faris/.home-migration-20261025/` (private permissions).
`REPORT.json` records live account, path, environment, service and container
checks. `environment-path-changes.json`, `environment-checks.json`,
`home-switch.json`, Docker mount/config records, sync logs, launcher backups,
and preserved-image overrides record the changes. Private keys, API tokens and
other secret contents are not included in this report.

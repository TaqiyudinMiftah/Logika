# Historical coverage correction — 28 October 2026

The earlier statement that every experiment had been pushed was too broad. This
append-only audit found omitted discovery roots and custom vendor sources. It
adds **1,067 distinct public source/evidence files**: 1,000 older metadata/log
files from node2003, one custom feature-builder script, and 66 node2001 refit
logs/receipts. Negative results, failures and partial attempts are retained.

The [initial import](../history-20261028/README.md) contains 438,732 distinct public
blobs. The two history imports now contain **439,799** public blobs. These are
**file counts, not independent experiment counts**. Repeated assessments, aliases,
completed fits and scientific experiments must not be added together.

This supplement inventories 26,330 source records: 1,067 new public files, 20,790
aliases to published content, 4,455 private references, and 18 excluded tree entries.
Most of the seven omitted node2003 directories were mirrored already. Direct
inspection also covered node2001's Antigravity joint/refit roots and the legacy
submission/linear evidence archives. Published mirrors are referenced rather than
copied again. No new model fitting or competition submission occurred in this audit.

Read [COVERAGE_CORRECTION.json](COVERAGE_CORRECTION.json) for roots and limitations,
[VERIFICATION.json](VERIFICATION.json) for the exact new file/hash list, and
[CONTENT_AUDIT.json](CONTENT_AUDIT.json) for byte verification of every referenced
public mapping. `SOURCE_INVENTORY.jsonl.gz` maps each discovered source file to
its published path/archive member or private reference. Original and transformed
hashes are separate; missing private hashes are never invented.

Coverage remains bounded by the documented accessible roots. Node2002 was not
accessed; deleted or undiscovered experiments cannot be certified. Credentials,
raw data, case-level labels/OOF and fitted weights remain private. Completed
campaign seals and the original archive verification are preserved.

The latest ALL66 result is recorded as a failed public comparison: **0.55983**
versus V25 **0.56822**, delta **-0.00839**. See the
[exact-artifact feedback](../runs/agy-all66-refit-20261028/PUBLIC_FEEDBACK_20261028.json).
Fetched peer evidence independently associates Kaggle submission 57730355 with
the published CSV SHA-256. V25 remains the public incumbent.

Peer changes were pulled and reviewed before publication. The
[peer review receipt](PEER_SYNC_REVIEW.json) distinguishes queued V254–V256 designs
from completed results. `SECRET_SCAN_RESULT.json` records the scoped privacy scan.

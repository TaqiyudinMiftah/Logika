"""Summarize a completed cleanup receipt without changing research artifacts."""
from pathlib import Path
import json,collections
out=Path(__file__).resolve().parent
r=json.loads((out/'REPORT.json').read_text())
p=json.loads((out/'removal-plan.json').read_text())
by=collections.defaultdict(lambda:[0,0])
for m in p['models']:
    v=by[m['path'].split('/')[0]];v[0]+=1;v[1]+=m['bytes']
rows=['| Project | Model files removed | Logical GiB |','|---|---:|---:|']
for name,(n,size)in sorted(by.items()):rows.append(f'| {name} | {n:,} | {size/2**30:.3f} |')
text=f'''# PC2 model cleanup result — 25 October 2026

Verified success: **{r['success']}**. Finished: `{r['finished_utc']}`.

Scope: `/mnt/home/logika-faris-exploration` on `labkc-comm1`, reached with
`ssh faris@100.68.29.126` (port 22). Port 2001 refused the connection.

- Model, transformer, checkpoint and cached-weight files removed: **{r['deleted_model_files']:,}**.
- Model aliases removed: **{r['deleted_model_links']:,}**.
- Model copies stripped from mixed archives: **{r['removed_archive_model_members']:,}**.
- Readable settings/metadata exports retained: **{r['parameter_exports']:,}**.
- Retained files checked: **{r['retained_files_verified']:,}**.
- Retained-file verification failures: **{len(r['retained_hash_failures'])}**.
- Remaining recognized model files: **{len(r['remaining_model_files'])}**.
- Filesystem free-space increase: **{r['net_freed_gib']:.2f} GiB**.
- Model allocation unlinked at its last hard link: **{r['model_allocated_gib_unlinked_at_last_hardlink']:.2f} GiB**.
- Available filesystem space afterward: **{r['free_after_bytes']/2**30:.2f} GiB**.

The filesystem free-space change includes concurrent filesystem activity and new
cleanup metadata. Logical file sizes may count hard links repeatedly; they are
not an estimate of reclaimed space.

The historical Python 3.11 installation outside this root disappeared during
preparation, leaving existing virtual-environment interpreter links broken.
This task did not delete or repair that installation; system Python performed
the cleanup and recovered metadata exports.

Predictions, submissions, reports, code, configuration files, raw/prepared data,
and dependency environments were kept. Model settings are in `parameters/` on
PC2 and in `parameters.tar.gz` in the local receipt copy. Large learned weights
are not duplicated in these exports.

Verification methods: `{json.dumps(r['retained_verification_methods'],sort_keys=True)}`.
Filesystem identity verification includes device/inode, size and nanosecond
modification time. It is not a full-content hash. Selected source/configuration
formats and submission CSVs also received full SHA-256 checks. Every retained
file member of the two rewritten archives was fully hashed before replacement.

Embedded metadata export errors: **{r['embedded_export_errors']}**;
archive-member metadata errors: **{len(r.get('archive_parameter_export_errors',[]))}**.
Existing configuration records and source references remain available for
formats whose configuration is not embedded as readable metadata.
The one model-export read error concerns an empty zero-byte file from an
interrupted teacher fit. Its boundary settings and intended champion recipe
were explicitly retained; no serialized model existed in that file.

Historical manifests and archive checksum sidecars remain historical evidence.
Use the cleanup archive receipts for replacement archive hashes. Old code paths
still describe training/inference recipes, but removed weights require refitting
or obtaining weights from a separate location. Other machines and local fitted
models were outside this cleanup scope. No reboot, poweroff or unrelated process
termination occurred.

## Removed model files by project

'''+ '\n'.join(rows)+'\n'
(out/'REPORT.md').write_text(text)
print(text.split('## Removed model files by project')[0])

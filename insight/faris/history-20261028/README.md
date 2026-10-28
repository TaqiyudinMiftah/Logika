# Historical Logika import — 28 October 2026

This is an append-only import of the available Faris experiment archives, assembled
on top of the teammate's updated `main` rather than replacing their work. The
initial fetched base was `56b056b98692d05425d8ce917ca6410148b35a4b` (V250).

Read [IMPORT_STATUS.json](IMPORT_STATUS.json) for the current source coverage and
counts. The inventory index is `SOURCE_INVENTORY_INDEX.json`; it lists bounded gzip JSONL shards.
Bulk remote trial metadata is retained in `bundles/trial-records-*.tar.gz`.
Each published record points either to `public_path` or to `public_archive` plus
`archive_member`, with its own content SHA-256. `BUNDLE_INDEX.json` records archive hashes.
Every discovered source file is classified; a public file points to its preserved
repository path/hash, an identical copy points to the same public blob, and private
evidence retains its original location, byte count and available digest. Private
hashes are not invented when byte verification was intentionally omitted.

Public files keep their original contents unless credential redaction, case-array
omission, stripping notebook outputs/attachments, exclusion of serialized model structures or compression is necessary.
Transformations and both original/public hashes are recorded. The private originals
are unchanged. Large environments, dependencies, caches and unrelated competitions
are excluded at their tree boundaries and recorded as such.

This inventory is an artifact/source inventory, **not an independent experiment
counter**. Completed fits, failed attempts, repeated assessments, configs, aliases
and mirrored files must not be added together as distinct experiments. Missing or
inaccessible sources are reported rather than silently called complete.

The [Antigravity continuation record](ANTIGRAVITY_CONTINUATION.json) and
[aggregate console excerpts](ANTIGRAVITY_AGGREGATE_LOG.jsonl) cover the recent joint
feature/modelling work. Raw user conversations and private reasoning are not
published. The excerpts are logs, not independent validation or promotion evidence.

Older handoffs and sealed campaign reports remain historical snapshots. In
particular, the 28 October continuation reopened feature/model changes and later
allowed Colab A100 account 2; an old handoff's earlier freeze is not current scope.

Public/private separation follows the existing shared protocol: source, parameters,
aggregate metrics and hashes are public; raw data, case-level labels/OOF, credentials
and fitted weights stay on the original private machines.

To inspect a bundled record without extracting the archive:

```sh
tar -xOzf bundles/trial-records-0001.tar.gz 'node2001/path/from/archive_member.json'
```

The member name above is an example; use the exact `archive_member` recorded in
the inventory. Sources remain private at their original absolute paths.

The imported snapshot includes the local workspace, Antigravity scratch sources,
node2001, node2003 and the surviving Agung archives. See `SOURCE_COVERAGE.json`
for the actual roots. Retired node2002 was not accessed. Third-party uv caches and
embedded Python runtimes are retained only as environment references;
`ENVIRONMENT_CACHE_EXCLUSION.json` documents the correction. Every published
content mapping is checked by `tools/verify_experiment_history.py`.

"""Verify every public mapping in an appended historical inventory."""
import argparse
import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def verify(repo, supplement):
    repo = repo.resolve()
    root = repo / supplement
    receipt = json.loads((root / 'VERIFICATION.json').read_text())
    inventory = root / 'SOURCE_INVENTORY.jsonl.gz'
    assert digest(inventory) == receipt['inventory_sha256']
    direct, bundled, archived, seen = {}, {}, set(), set()
    counts, kinds = collections.Counter(), collections.Counter()
    records = 0
    with gzip.open(inventory, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            records += 1
            kinds[row['kind']] += 1
            assert row['kind'] not in {'scan_error', 'missing_root'}, row
            if row['kind'] != 'file':
                continue
            key = (row['source'], row['path'])
            assert key not in seen, key
            seen.add(key)
            publication = row['publication']
            counts[publication] += 1
            if publication == 'private_reference':
                continue
            assert publication in {'archived', 'deduplicated_alias'}
            expected = row['public_sha256']
            if 'public_path' in row:
                relative = row['public_path']
                path = (repo / relative).resolve()
                assert path.is_relative_to(repo), relative
                assert direct.setdefault(relative, expected) == expected
                if publication == 'archived':
                    archived.add(relative)
            else:
                archive = row['public_archive']
                assert (repo / archive).resolve().is_relative_to(repo)
                members = bundled.setdefault(archive, {})
                assert members.setdefault(row['archive_member'], expected) == expected
    assert records == receipt['records']
    assert dict(counts) == receipt['publication_counts']
    assert archived == {x['public_path'] for x in receipt['new_public_files']}
    source_files = {str(x.relative_to(repo)) for x in (root / 'sources').rglob('*') if x.is_file()}
    assert source_files == archived, (source_files - archived, archived - source_files)
    for relative, expected in direct.items():
        assert digest(repo / relative) == expected, relative
    bundle_index = json.loads((repo / 'insight/faris/history-20261028/BUNDLE_INDEX.json').read_text())
    bundle_hashes = {r['public_archive']: r['sha256'] for r in bundle_index['bundles']}
    checked_members = 0
    for relative, expected_members in bundled.items():
        assert digest(repo / relative) == bundle_hashes[relative], relative
        pending = dict(expected_members)
        with tarfile.open(repo / relative, 'r|gz') as archive:
            for member in archive:
                if member.name not in pending:
                    continue
                content = archive.extractfile(member).read()
                assert hashlib.sha256(content).hexdigest() == pending.pop(member.name), member.name
                checked_members += 1
                if not pending:
                    break
        assert not pending, (relative, list(pending)[:5])
    result = {
        'status': 'PASS',
        'verified_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'inventory_sha256': receipt['inventory_sha256'],
        'records': records,
        'record_kinds': dict(kinds),
        'publication_counts': dict(counts),
        'new_public_files': len(archived),
        'direct_public_mappings_verified': len(direct),
        'existing_archives_hash_verified': len(bundled),
        'existing_archive_members_verified': checked_members,
        'unresolved_missing_or_changed_files': 0,
        'scope': 'Byte verification of the documented supplement; not proof of undiscovered or inaccessible experiments.',
        'verifier_source_sha256': digest(Path(__file__)),
    }
    (root / 'CONTENT_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--supplement', default='insight/faris/history-completeness-20261028')
    args = parser.parse_args()
    verify(args.repo, args.supplement)

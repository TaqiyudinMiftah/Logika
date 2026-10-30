"""Prepare unchanged private packet bytes before any paid allocation."""
from pathlib import Path
import hashlib
import json
import tarfile

R = Path(__file__).resolve().parents[1]
P = R / 'private'
P.mkdir(exist_ok=True)
packets = Path('/mnt/home/faris/logika-twelvehour-next-20261029/packets')
records = {}
for kind, names in [
    ('BASE', ['features.parquet', 'roles.npz', 'inner_roles.npz', 'anchor.npz', 'MANIFEST.json']),
    ('WIDE', ['all_features.parquet']),
]:
    target = P / ('PACKETS_' + kind + '.tar')
    if not target.exists():
        temporary = target.with_suffix('.tmp')
        with tarfile.open(temporary, 'w', dereference=True) as archive:
            for fold in range(5):
                for name in names:
                    path = packets / f'fold{fold}' / name
                    archive.add(path, arcname=f'packets/fold{fold}/{name}')
        temporary.replace(target)
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    records[target.name] = dict(bytes=target.stat().st_size, sha256=digest.hexdigest(),
                                private_original_packet_bytes=True)
(P / 'PAYLOAD_MANIFEST.json').write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps(records))


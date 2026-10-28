"""Stream an append-only private download; verify each sealed packet before use."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
from colab_transport import LOCAL, PRIVATE, REMOTE, OPTIONS

manifest = json.loads((PRIVATE / 'input_bundle.json').read_text())
expected = manifest['bytes']
source = PRIVATE / 'inputs.tar.gz'
assert time.time() < 1793213917.3252351
command = f"bash -o pipefail -c 'tee {REMOTE}/COLAB_INPUTS_PRIVATE.tar.gz | tar -xz -C {REMOTE} --wildcards \"packets/*\"'"
log = (PRIVATE / 'stream_transfer.log').open('ab')
process = subprocess.Popen(['ssh', '-T', *OPTIONS, 'root@colab', command], stdin=subprocess.PIPE,
                           stdout=log, stderr=log)
offset, digest = 0, hashlib.sha256()
try:
    with source.open('rb') as stream:
        while offset < expected:
            assert time.time() < 1793213917.3252351 - 1800, 'Original admission reserve exhausted'
            available = source.stat().st_size - offset
            if available <= 0:
                assert process.poll() is None, 'Remote stream failed; private log retained'
                time.sleep(2)
                continue
            block = stream.read(min(4 * 1024**2, available, expected - offset))
            assert block
            process.stdin.write(block)
            process.stdin.flush()
            digest.update(block)
            offset += len(block)
            (PRIVATE / 'STREAM_PROGRESS.json').write_text(json.dumps({
                'bytes': offset, 'expected': expected, 'epoch': time.time()}) + '\n')
    process.stdin.close()
    assert process.wait(timeout=120) == 0
    assert digest.hexdigest() == manifest['sha256']
    (PRIVATE / 'STREAM_COMPLETE.json').write_text(json.dumps({
        'bytes': offset, 'sha256': digest.hexdigest(), 'epoch': time.time()}) + '\n')
    print('Private input stream complete; original archive hash matches.', flush=True)
finally:
    if process.poll() is None:
        process.terminate()
    log.close()

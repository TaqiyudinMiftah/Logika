"""Upload hash-verified Kumo source and detach its node380 launcher."""
from pathlib import Path
import hashlib
import json
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from connection_profiles import activated, load_profile

SOURCE = HERE / 'kumo-large-node380-20261016'
RELATIVE = 'faris/kumo-large-node380-20261016'
REMOTE = '/home/taqiyudinmiftah/' + RELATIVE


def main():
    files = sorted([*SOURCE.glob('scripts/*.py'), SOURCE / 'artifacts/optimization/tenfold/locked_recipe.json'])
    hashes = {str(file.relative_to(SOURCE)): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
    archive = HERE / 'node380_code.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for file in files:
            tar.add(file, arcname=str(file.relative_to(SOURCE)), recursive=False)
    manifest = {'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'files': hashes}
    manifest_path = HERE / 'node380_code_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    with activated(load_profile('node380')) as remote:
        client = remote.JupyterClient(remote.DEFAULT_BASE_URL, remote.load_token(Path(remote.DEFAULT_SYSINFO)))
        client.upload_small(archive, RELATIVE + '/code.tar.gz')
        client.upload_small(manifest_path, RELATIVE + '/code_manifest.json')
        kernel = client.session_kernel('term')
        bootstrap = '''
from pathlib import Path
import hashlib,json,os,subprocess,tarfile,time
root=Path(REMOTE_ROOT)
record=root/'bootstrap.json'
if record.exists():
    raise RuntimeError('Existing bootstrap record; inspect before restarting')
manifest=json.loads((root/'code_manifest.json').read_text())
assert hashlib.sha256((root/'code.tar.gz').read_bytes()).hexdigest()==manifest['sha256']
with tarfile.open(root/'code.tar.gz','r:gz') as tar:
    assert {m.name for m in tar.getmembers()}==set(manifest['files'])
    assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in tar.getmembers())
    tar.extractall(root)
for name,want in manifest['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==want,name
with (root/'launcher.log').open('ab') as log:
    p=subprocess.Popen(['/opt/conda/bin/python','-u',str(root/'scripts/run_node380.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(pid=p.pid,proc_start_ticks=Path('/proc/%s/stat'%p.pid).read_text().split()[21],root=str(root),archive_sha256=manifest['sha256'],launched_epoch=time.time())
record.write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
'''.replace('REMOTE_ROOT', repr(REMOTE))
        code, output = client.run_python(kernel, bootstrap, timeout=30)
        print(output)
        if code:
            raise RuntimeError('Remote bootstrap failed')
        (HERE / 'node380_deployment.json').write_text(output)


if __name__ == '__main__':
    main()

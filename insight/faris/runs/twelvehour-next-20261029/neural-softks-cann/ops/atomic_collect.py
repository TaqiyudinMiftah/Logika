"""Collect complete attempt directories without exposing a half-uploaded RESULT.

Run on canonical node2001. Source may be a local directory or host:/absolute/path.
No model code, retry, prediction edits or deletion is performed.
"""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, tempfile, time

R = Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    q=path.with_name(path.name+f'.{os.getpid()}.tmp')
    q.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');q.replace(path)

def check(folder):
    result=folder/'RESULT.json'
    if result.exists():
        v=json.loads(result.read_text())
        # An upload can be observed before all files arrive. Wait for next pass.
        if not (folder/'START.json').exists() or not (folder/'prediction.npz').exists():return False
        assert v['status']=='complete'
        assert sha(folder/'START.json')==v['start_sha256']
        assert sha(folder/'prediction.npz')==v['prediction_sha256']
        assert v['weights_saved'] is False and v['KV_cache_saved'] is False
        return True
    return (folder/'FAILURE.json').exists() or (folder/'INTERRUPTION.json').exists()

def publish(source, owner):
    base=R/'private/collection'/owner;base.mkdir(parents=True,exist_ok=True)
    incoming=base/'incoming';incoming.mkdir(exist_ok=True)
    # The caller supplies a narrowly scoped SSH transport, not shell commands.
    args=['rsync','-az','--exclude=__pycache__/***']
    transport=os.environ.get('LOGIKA_COLLECT_RSH')
    if transport:args+=['-e',transport]
    subprocess.run([*args,source.rstrip('/')+'/',str(incoming)+'/'],check=True,capture_output=True,timeout=300)
    complete=failed=waiting=0
    for folder in sorted(incoming.glob('*/attempt*')):
        if not folder.is_dir() or folder.is_symlink():continue
        if not check(folder):waiting+=1;continue
        target=R/'runs'/folder.parent.name/folder.name
        if target.exists():
            # Immutable terminal directories must agree. Ignore unrelated live logs.
            for name in ['START.json','RESULT.json','prediction.npz','FAILURE.json','INTERRUPTION.json']:
                p=folder/name;q=target/name
                if p.exists():assert q.exists() and sha(p)==sha(q),f'Conflicting terminal evidence: {q}'
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            staging=target.parent/(target.name+f'.collect-{os.getpid()}')
            assert not staging.exists()
            shutil.copytree(folder,staging)
            assert check(staging)
            staging.rename(target)
        if (target/'RESULT.json').exists():complete+=1
        else:failed+=1
    result=dict(epoch=time.time(),owner=owner,complete=complete,failed=failed,awaiting_atomic_completion=waiting,
                predictions_private=True,source=source,terminal_attempts_verified_before_publication=True)
    dump(base/'RECEIPT.json',result);return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source',required=True);a.add_argument('--owner',required=True)
    v=a.parse_args();print(json.dumps(publish(v.source,v.owner)))

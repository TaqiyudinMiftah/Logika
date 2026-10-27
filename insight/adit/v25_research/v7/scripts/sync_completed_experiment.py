"""Local Windows publisher: requires a peer-reviewed expected main HEAD."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import quote
import requests

ROOT=Path(__file__).resolve().parents[2]
V=ROOT/'v7'


def main(ident,expected,resume=False):
    checkout=ROOT.parent/'shared_repo_sync_20261027'
    def git(*args):
        return subprocess.run(['git','-C',str(checkout),*args],check=True,capture_output=True,text=True).stdout.strip()
    git('fetch','origin','main')
    remote=git('rev-parse','origin/main')
    assert remote==expected,'Peer pushed newer results; review before publishing'
    changed=git('status','--porcelain')
    if changed:
        changed_paths=git('diff','--name-only','HEAD').splitlines()+git('ls-files','--others','--exclude-standard').splitlines()
        assert resume and all(name.startswith('insight/adit/v25_research/') for name in changed_paths),'Review local checkout changes before publishing'
    if git('rev-parse','HEAD')!=remote:git('merge','--ff-only','origin/main')
    sys.path.insert(0,str(ROOT.parent))
    from node380_remote import protect
    vault=json.loads(protect((Path.home()/'.codex/private/node480_session.dpapi').read_bytes(),decrypt=True))
    session=requests.Session();session.cookies.update(vault['cookies'])
    base=vault['base_url'].rstrip('/')+'/api/contents/'
    def download(name,optional=False):
        r=session.get(base+quote('adit/Logika/insight/goal054_offset_baseline_20261019/'+name)+'?content=1',timeout=20)
        if optional and r.status_code==404:return
        r.raise_for_status()
        target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(r.json()['content'].replace('\r\n','\n').encode('utf-8'))
    download(f'v7/runs/{ident}/RESULT.json')
    result=json.loads((V/'runs'/ident/'RESULT.json').read_text())
    files=['v7/reports/EXPERIMENT_MASTER_LOG.json','v7/modeling/reports/MODELING_EXPERIMENTS.csv','v7/reports/MODELING_QUEUE.json']
    files += [f'v7/runs/{ident}/{name}' for name in ['RESULT.json','ERROR_ANALYSIS.json','ERROR_ANALYSIS.md','ERROR_SEGMENTS.csv','REPORT.md','MATCHED_CONTROL.json','MODELING_METHOD.json']]
    files += [result['submission']]
    for name in files:
        download(name)
    for name in ['GPU_CACHE_PARITY.json','PROCESS_ATTEMPTS.json','PREVIOUS_ATTEMPT_FAILURE.json','CONTEXT_INPUT_AUDIT.json']:
        download(f'v7/runs/{ident}/{name}',optional=True)
    download('v7/kumo_gpu/WEIGHTS_MANIFEST.json',optional=True)
    assert hashlib.sha256((ROOT/result['submission']).read_bytes()).hexdigest()==result['submission_sha256']
    subprocess.run([sys.executable,str(V/'scripts/prepare_public_progress.py')],check=True,capture_output=True,text=True)
    source=ROOT.parent/'github_sync_20261027/insight/adit/v25_research'
    destination=checkout/'insight/adit/v25_research'
    assert destination.resolve().is_relative_to(checkout.resolve())
    shutil.copytree(source,destination,dirs_exist_ok=True)
    (destination/'.gitattributes').write_bytes(b'* text eol=lf\n')
    review=dict(checked_UTC=datetime.now(timezone.utc).isoformat(),reviewed_head=expected,repository='TaqiyudinMiftah/Logika',branch='main',latest_experiment=ident,peer_new_commit=False,protocol='Expected HEAD independently reviewed before this run; push rejects concurrent advancement')
    (destination/'PEER_REVIEW.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    node=json.loads(protect((Path.home()/'.codex/private/node480_fariselhakim.dpapi').read_bytes(),decrypt=True))
    kaggle=json.loads((Path.home()/'Downloads/kaggle (1).json').read_text(encoding='utf-8-sig'))
    manifest=[]
    for f in sorted(destination.rglob('*')):
        if not f.is_file() or '__pycache__' in f.parts or f.suffix=='.pyc':continue
        text=f.read_text(encoding='utf-8').replace('\r\n','\n')
        assert node['password'] not in text and kaggle['key'] not in text,'Credential detected'
        f.write_bytes(text.encode('utf-8'))
        if f.name!='PUBLIC_MANIFEST.json':manifest.append(dict(path=f.relative_to(destination).as_posix(),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    (destination/'PUBLIC_MANIFEST.json').write_bytes(json.dumps(dict(prepared_UTC=datetime.now(timezone.utc).isoformat(),files=manifest,credential_files_included=False,raw_training_rows_included=False),indent=2).encode())
    git('add','--','insight/adit/v25_research')
    assert all(name.startswith('insight/adit/v25_research/') for name in git('diff','--cached','--name-only').splitlines())
    git('fetch','origin','main');assert git('rev-parse','origin/main')==expected,'Peer pushed while preparing; inspect and merge before commit'
    body=checkout.parent/f'commit_{ident}.txt'
    body.write_text(f"Record {ident} and held-out error analysis against V25\n\nPooled KS{result['OOF']['KS']:.10f}, AUC{result['OOF']['AUC']:.10f}; deltaKS{result['delta_KS']:+.10f} against fixed V25. Preserve negative results, five-fold checks, paired uncertainty, formula, source and separately hashed submission. Keep V25 as public incumbent; no hidden-test score is claimed.\n\nCheck shared main before and after preparation; publish only the adit research namespace.\n",encoding='utf-8')
    git('commit','-F',str(body));head=git('rev-parse','HEAD');git('push','origin','HEAD:main')
    assert git('ls-remote','origin','refs/heads/main').split()[0]==head
    receipt=dict(status='PUSHED',experiment=ident,commit=head,previous_head=expected,KS=result['OOF']['KS'],AUC=result['OOF']['AUC'],url='https://github.com/TaqiyudinMiftah/Logika/commit/'+head,checked_UTC=datetime.now(timezone.utc).isoformat())
    (V/'modeling/reports'/f'GITHUB_SYNC_{ident}.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('experiment');parser.add_argument('--expected-head',required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args();main(args.experiment,args.expected_head,args.resume)

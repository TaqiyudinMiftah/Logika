"""Publish a reviewed research note/plan inside the authorized adit namespace."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'


def main(expected,body_file):
    checkout=ROOT.parent/'shared_repo_sync_20261027'
    def git(*args):return subprocess.check_output(['git','-C',str(checkout),*args],text=True).strip()
    git('fetch','origin','main');assert git('rev-parse','origin/main')==expected and not git('status','--porcelain')
    subprocess.run([sys.executable,str(V/'scripts/prepare_public_progress.py')],check=True,stdout=subprocess.DEVNULL)
    source=ROOT.parent/'github_sync_20261027/insight/adit/v25_research';destination=checkout/'insight/adit/v25_research';shutil.copytree(source,destination,dirs_exist_ok=True)
    (destination/'PEER_REVIEW.json').write_text(json.dumps(dict(checked_UTC=datetime.now(timezone.utc).isoformat(),reviewed_head=expected,peer_new_commit=False,update_kind='research_plan_and_generalization_review'),indent=2),encoding='utf-8')
    sys.path.insert(0,str(ROOT.parent));from node380_remote import protect
    node=json.loads(protect((Path.home()/'.codex/private/node480_fariselhakim.dpapi').read_bytes(),decrypt=True));kaggle=json.loads((Path.home()/'Downloads/kaggle (1).json').read_text(encoding='utf-8-sig'));manifest=[]
    for file in sorted(destination.rglob('*')):
        if not file.is_file() or '__pycache__' in file.parts:continue
        text=file.read_text(encoding='utf-8').replace('\r\n','\n');assert node['password'] not in text and kaggle['key'] not in text,'Credential detected';file.write_bytes(text.encode())
        if file.name!='PUBLIC_MANIFEST.json':manifest.append(dict(path=file.relative_to(destination).as_posix(),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    (destination/'PUBLIC_MANIFEST.json').write_bytes(json.dumps(dict(prepared_UTC=datetime.now(timezone.utc).isoformat(),files=manifest,raw_training_rows_included=False,credential_files_included=False),indent=2).encode())
    git('add','--','insight/adit/v25_research');assert all(p.startswith('insight/adit/v25_research/') for p in git('diff','--cached','--name-only').splitlines())
    git('fetch','origin','main');assert git('rev-parse','origin/main')==expected
    git('commit','-F',str(Path(body_file).resolve()));git('push','origin','HEAD:main');head=git('rev-parse','HEAD');assert git('ls-remote','origin','refs/heads/main').split()[0]==head
    receipt=dict(status='PUSHED',commit=head,previous_head=expected);(V/'reports/RESEARCH_NOTE_SYNC.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--expected-head',required=True);p.add_argument('--body-file',required=True);a=p.parse_args();main(a.expected_head,a.body_file)

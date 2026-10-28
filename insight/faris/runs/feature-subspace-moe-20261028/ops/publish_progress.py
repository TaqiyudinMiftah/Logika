"""Synchronize public aggregate evidence; never copy labels or predictions to Git.

Runs on the user's workstation. Authentication stays in its existing Git setup.
Peer changes are fetched and preserved; any namespace conflict stops publication.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run(args, cwd=None, timeout=180):
    return subprocess.run(args,cwd=cwd,text=True,capture_output=True,check=True,timeout=timeout)


def dump(value,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def publish(local,repo,once=False):
    private=local/'publication_private';private.mkdir(exist_ok=True)
    lock=(private/'publisher.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    mirror=private/'remote_aggregate_mirror';mirror.mkdir(exist_ok=True)
    destination=repo/'insight/faris/runs/feature-subspace-moe-20261028'
    sys.path.insert(0,str(repo/'tools'))
    from publish_experiment_history import public_content
    errors=0
    while True:
        try:
            run(['rsync','-az','--include=*/','--include=*.json','--include=*.md','--include=*.jsonl',
                 '--exclude=*','faris@100.68.29.126:/mnt/home/faris/logika-feature-subspace-moe-20261028/',str(mirror)+'/'],timeout=180)
            # Keep raw operational aggregate originals private as well as public
            # transformations. No arrays/dataframe files match this rsync filter.
            for p in mirror.rglob('*'):
                if not p.is_file():continue
                relative=p.relative_to(mirror)
                if any(part in {'source','publication_private','evaluation_inputs','failures'}for part in relative.parts):continue
                content,policy=public_content(p)
                if content is None:continue
                target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
            state=json.loads((mirror/'LIVE_STATE.json').read_text())if(mirror/'LIVE_STATE.json').exists()else{'stage':'launch_pending'}
            clock=json.loads((mirror/'CLOCK.json').read_text())if(mirror/'CLOCK.json').exists()else{}
            record_path=destination/'RUN.json'
            if record_path.exists():
                record=json.loads(record_path.read_text())
                record.pop('actual_data_training_started',None)
                record.update(status=state['stage'],computation_clock_started=bool(clock),
                              completed_expert_fits=state.get('completed',0),clock=clock,
                              current_state_evidence='LIVE_STATE.json')
                if(mirror/'evaluation/RESULT.json').exists():
                    record['new_results']='evaluation/RESULT.json'
                dump(record,record_path)
            run(['git','fetch','origin','main'],repo)
            head=run(['git','rev-parse','HEAD'],repo).stdout.strip();remote=run(['git','rev-parse','origin/main'],repo).stdout.strip()
            if head!=remote:
                changed=run(['git','diff','--name-only',head+'...'+remote],repo).stdout.splitlines()
                namespace='insight/faris/runs/feature-subspace-moe-20261028/'
                assert not any(p.startswith(namespace)for p in changed),'Peer changed the campaign namespace; inspect rather than overwrite.'
                ancestor=subprocess.run(['git','merge-base','--is-ancestor',head,remote],cwd=repo).returncode==0
                run(['git','merge','--quiet','--ff-only','origin/main']if ancestor else['git','merge','--quiet','--no-edit','origin/main'],repo)
            scan=private/'secret_scan.json'
            scanner=local.parents[0]/'github-history-20261028/private/scanner/gitleaks'
            run([str(scanner),'dir',str(destination),'--config',str(repo/'.gitleaks.toml'),
                 '--ignore-gitleaks-allow','--redact=100','--no-banner','--no-color',
                 '--report-format','json','--report-path',str(scan)],timeout=180)
            assert json.loads(scan.read_text())==[],'Public evidence scan needs review'
            run(['git','add','-f',str(destination.relative_to(repo))],repo)
            staged=run(['git','diff','--cached','--name-only'],repo).stdout.splitlines()
            assert all(p.startswith('insight/faris/runs/feature-subspace-moe-20261028/')for p in staged),'Unrelated staged work detected'
            if staged:
                message=private/'commit_message.txt'
                message.write_text('Record MoE campaign '+state['stage']+' evidence\n\n'
                                   'Preserve all planned, completed, failed and partial fit metadata.\n'
                                   'Case-level arrays and fitted weights remain private.\n')
                run(['git','commit','--quiet','--file',str(message)],repo)
                run(['git','push','origin','HEAD:refs/heads/main'],repo,180)
            head=run(['git','rev-parse','HEAD'],repo).stdout.strip()
            verified=run(['git','ls-remote','origin','refs/heads/main'],repo).stdout.split()[0]
            assert verified==head,'Remote moved concurrently; retain receipt and inspect on next sync'
            receipt=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote_commit=verified,
                         stage=state['stage'],completed=state.get('completed'),expected=state.get('expected'),
                         findings=0,metadata_files=len(list(mirror.rglob('*.json'))))
            dump(receipt,private/'LATEST_RECEIPT.json');print(json.dumps(receipt),flush=True);errors=0
            terminal=state['stage']in{'complete','failed'}
            if terminal:
                # Terminal compact prediction evidence is collected privately.
                command="from pathlib import Path;import json;p=Path('/mnt/home/faris/logika-feature-subspace-moe-20261028');print(json.dumps([n for n in ['evaluation','composed','runs','logs','failures']if(p/n).is_dir()]))"
                import shlex
                present=json.loads(run(['ssh','-o','BatchMode=yes','faris@100.68.29.126','python3 -c '+shlex.quote(command)],timeout=30).stdout)
                for name in present:
                    target=local/'FINAL_EVIDENCE'/name;target.mkdir(parents=True,exist_ok=True)
                    run(['rsync','-az','faris@100.68.29.126:/mnt/home/faris/logika-feature-subspace-moe-20261028/'+name+'/',str(target)+'/'],timeout=300)
                dump(dict(receipt,compact_evidence_collected=True),private/'FINAL_RECEIPT.json')
            if once or terminal or clock and time.time()>clock['deadline_epoch']+300:return
        except Exception as error:
            errors+=1
            dump(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=str(error),consecutive_errors=errors),private/'PUBLICATION_ERROR.json')
            print(json.dumps(dict(publication_error=str(error),count=errors)),flush=True)
            if once or errors>=5:raise
        time.sleep(300)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--local',type=Path,required=True)
    parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--once',action='store_true')
    a=parser.parse_args();publish(a.local.resolve(),a.repo.resolve(),a.once)

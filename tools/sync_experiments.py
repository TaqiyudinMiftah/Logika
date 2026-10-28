"""Fetch shared research state or push a reviewed fast-forward and verify it."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['before','push'])
    parser.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--receipt',required=True,help='Private durable receipt path; not an endlessly self-changing Git artifact')
    args=parser.parse_args();repo=Path(args.repo).resolve()
    previous=Path(args.receipt)
    old=json.loads(previous.read_text())if previous.exists()else {}
    subprocess.run(['git','-C',str(repo),'fetch','origin','main'],check=True)
    remote=git(repo,'rev-parse','origin/main');head=git(repo,'rev-parse','HEAD')
    if args.action=='before':
        print(git(repo,'log','--oneline',(old.get('remote_commit')or remote)+'..'+remote))
        if head!=remote:
            print('Read and reconcile fetched peer commits before executing. No checkout was reset.')
    else:
        assert not git(repo,'status','--porcelain'),'Commit the complete reviewed experiment record before pushing'
        subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',remote,'HEAD'],check=True)
        subprocess.run(['git','-C',str(repo),'push','origin','HEAD:refs/heads/main'],check=True)
        verified=git(repo,'ls-remote','origin','refs/heads/main').split()[0]
        assert verified==head,'Remote moved concurrently; inspect instead of claiming this push is the current HEAD'
        remote=verified
    previous.parent.mkdir(parents=True,exist_ok=True)
    previous.write_text(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                   'action':args.action,'local_commit':head,'remote_commit':remote},indent=2)+'\n')
    print(json.dumps({'action':args.action,'remote_commit':remote,'receipt':str(previous)}))

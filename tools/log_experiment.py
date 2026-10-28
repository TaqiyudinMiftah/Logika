"""Append public, sanitized experiment records; keep case-level artifacts private."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from publish_experiment_history import clean_json


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(clean_json(value),indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def record(args):
    repo=Path(args.repo).resolve()
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}',args.id),'Use a stable safe experiment ID'
    root=repo/'insight/faris/runs'/args.id
    state=root/'RUN.json'
    if args.action=='start':
        assert not state.exists(),'Existing experiment ID; preserve it rather than overwrite'
        config_path=Path(args.config);protocol_path=Path(args.protocol)
        config=json.loads(config_path.read_text());protocol=json.loads(protocol_path.read_text())
        signature=hashlib.sha256(json.dumps({'config':config,'protocol':protocol},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        duplicates=[]
        for path in (repo/'insight/faris/runs').glob('*/RUN.json'):
            old=json.loads(path.read_text())
            if old.get('signature_sha256')==signature:duplicates.append(old['id'])
        assert not duplicates or args.replication_reason,('Duplicate signature requires an explicit replication reason',duplicates)
        head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        value={'id':args.id,'owner':args.owner,'status':'planned','created_UTC':utc(),
               'hypothesis':args.hypothesis,'source_commit':head,'signature_sha256':signature,
               'configuration_source_sha256':sha(config_path),'protocol_source_sha256':sha(protocol_path),
               'replication_reason':args.replication_reason,'duplicate_signatures':duplicates,
               'configuration':config,'protocol':protocol,'private_artifacts':[],
               'independent_confirmation_claimed':False}
        dump(state,value)
    else:
        assert state.exists(),'Start the experiment before recording an outcome'
        value=json.loads(state.read_text())
        assert value['status']=='planned','Preserve terminal records; use another ID for another attempt'
        result=json.loads(Path(args.results).read_text())
        artifacts=[]
        for name in args.private_artifact or []:
            path=Path(name).resolve();assert path.is_file(),path
            artifacts.append({'private_path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'uploaded':False})
        value.update(status=args.status,finished_UTC=utc(),results=result,
                     notes=args.notes,private_artifacts=artifacts,
                     results_source_sha256=sha(args.results))
        dump(state,value)
    print(json.dumps({'id':value['id'],'status':value['status'],'path':str(state),'signature_sha256':value['signature_sha256']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]))
    sub=parser.add_subparsers(dest='action',required=True)
    start=sub.add_parser('start');start.add_argument('--id',required=True);start.add_argument('--owner',default='faris')
    start.add_argument('--hypothesis',required=True);start.add_argument('--config',required=True);start.add_argument('--protocol',required=True)
    start.add_argument('--replication-reason')
    finish=sub.add_parser('finish');finish.add_argument('--id',required=True)
    finish.add_argument('--status',choices=['completed','failed','incomplete','cancelled'],required=True)
    finish.add_argument('--results',required=True);finish.add_argument('--notes',default='')
    finish.add_argument('--private-artifact',action='append')
    record(parser.parse_args())

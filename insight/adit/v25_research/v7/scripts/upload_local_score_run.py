"""Upload locally verified derived scores; preserve exact bytes on node480."""
from pathlib import Path
import argparse,base64,hashlib,json,sys,time
from urllib.parse import quote
import requests
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT.parent))
from node380_remote import protect
from node480_task import BASE

def main(ident):
    out=ROOT/'v7/runs'/ident;r=json.loads((out/'RESULT.json').read_text());a=json.loads((out/'REQUIREMENTS_AUDIT.json').read_text())
    assert r['execution_host']=='local_windows' and a['status']=='PASS' and a['submission_sha256']==r['submission_sha256']
    assert a['measured_OOF']['KS']==r['OOF']['KS']
    vault=json.loads(protect((Path.home()/'.codex/private/node480_session.dpapi').read_bytes(),decrypt=True));s=requests.Session();s.cookies.update(vault['cookies'])
    api=vault['base_url'].rstrip('/')+'/api/contents/';static=vault['base_url'].rstrip('/')+'/files/'
    from node480_task import client
    c=client()
    if c.request('GET','api/contents/'+BASE+f'v7/runs?content=1').get('type')!='directory':raise ValueError('Unexpected target')
    directory=api+quote(BASE+f'v7/runs/{ident}')
    response=s.get(directory+'?content=0',timeout=20)
    if response.status_code==404:c.request('PUT','api/contents/'+BASE+f'v7/runs/{ident}',dict(type='directory'))
    else:response.raise_for_status()
    files=[ROOT/'v7/scripts/upload_local_score_run.py',ROOT/'v7/scripts/sync_completed_experiment.py',ROOT/'v7/scripts/prepare_public_progress.py']
    if r.get('controller_source')=='v7/scripts/data_fe_worker.py':
        from summarize_data_research import summarize
        summarize()
        files += [ROOT/'v7/scripts/summarize_data_research.py']
        files += [ROOT/'v7/data_fe/reports'/name for name in ['BATCH_RESULTS.json','BATCH_RESULTS.csv','BATCH_RESULTS.md']]
    if r.get('controller_source')=='v7/scripts/score_fusion_local.py':files += [ROOT/'v7/score_fusion_plan.json',ROOT/'v7/scripts/score_fusion_local.py']
    files += [ROOT/'v7/reports'/name for name in ['BEST_DEVELOPMENT_MODEL.json','DEVELOPMENT_PARETO_FRONTIER.json'] if (ROOT/'v7/reports'/name).exists()]
    files += [p for p in sorted(out.iterdir()) if p.is_file() and p.name not in ['RESULT.json','STATUS.json','adapter.lock'] and '.tmp' not in p.name]
    files += [ROOT/r['submission'],out/'RESULT.json',out/'STATUS.json'];manifest=[]
    for path in files:
        rel=path.relative_to(ROOT).as_posix();payload=path.read_bytes();digest=hashlib.sha256(payload).hexdigest()
        c.request('PUT','api/contents/'+BASE+rel,dict(type='file',format='base64',content=base64.b64encode(payload).decode()))
        actual=hashlib.sha256()
        with s.get(static+quote(BASE+rel),stream=True,timeout=(10,40)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(128*1024):actual.update(chunk)
        assert actual.hexdigest()==digest,rel
        manifest.append(dict(path=rel,sha256=digest,bytes=len(payload)));print('UPLOADED_VERIFIED',rel,flush=True)
    result=dict(status='PASS',experiment=ident,UTC=time.time(),all_uploaded_bytes_verified=True,files=manifest)
    (ROOT/'v7/reports'/f'LOCAL_UPLOAD_{ident}.json').write_bytes(json.dumps(result,indent=2).encode())
    # Metadata-only merge retains the node's complete archive; it loads no ML
    # stack and does not start another model in the saturated RAM cgroup.
    command="python - <<'PY'\nimport subprocess,sys\np=subprocess.run([sys.executable,'/home/fariselhakim/adit/Logika/insight/goal054_offset_baseline_20261019/v7/scripts/master_log.py'],timeout=120);raise SystemExit(p.returncode)\nPY"
    assert c.execute(command,'python3')==0,'Uploaded result must enter the authoritative node ledger'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('experiment');args=p.parse_args();main(args.experiment)

"""Stream authorized frozen-score artifacts to Windows with low server memory."""
from pathlib import Path
import hashlib,json,sys,time
from urllib.parse import quote
import requests
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
from node380_remote import protect
from node480_task import BASE

def main():
    vault=json.loads(protect((Path.home()/'.codex/private/node480_session.dpapi').read_bytes(),decrypt=True))
    session=requests.Session();session.cookies.update(vault['cookies']);base=vault['base_url'].rstrip('/')+'/files/'
    files=['data/train_features.parquet','data/test_features.parquet','data/sample_submission.csv']
    for fold in range(5):files += [f'artifacts/fold{fold}/indices.npz',f'v7/extension/data/fold{fold}/validation.parquet']
    for source in ['V25','V167','V178']:
        files += [f'v7/runs/{source}/OOF.csv',f'v7/runs/{source}/RESULT.json']
        for fold in range(5):files += [f'v7/runs/{source}/fold{fold}_{part}.parquet' for part in ['validation','test']]
    records=[]
    for rel in files:
        path=ROOT/rel;assert path.resolve().is_relative_to(ROOT.resolve())
        if path.exists() and path.stat().st_size>0:
            records.append(dict(path=rel,bytes=path.stat().st_size,already_mirrored=True));continue
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.download')
        url=base+quote(BASE+rel)
        with session.get(url,stream=True,timeout=(10,40)) as response:
            response.raise_for_status();assert 'text/html' not in response.headers.get('Content-Type',''),'Unexpected login or error page'
            with tmp.open('wb') as f:
                for chunk in response.iter_content(128*1024):
                    if chunk:f.write(chunk)
        assert tmp.stat().st_size>0
        if rel.endswith('.parquet'):
            with tmp.open('rb') as f:
                assert f.read(4)==b'PAR1';f.seek(-4,2);assert f.read(4)==b'PAR1'
        tmp.replace(path);records.append(dict(path=rel,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print('MIRRORED',rel,path.stat().st_size,flush=True)
    report=dict(status='COMPLETE',UTC=time.time(),transfer='Authenticated Jupyter static-file streaming;128KiB chunks, no remote pandas/model process',private_files_not_for_GitHub=True,files=records)
    (ROOT/'v7/reports/LOCAL_SCORE_MIRROR.json').write_text(json.dumps(report,indent=2));print('MIRROR_COMPLETE',len(records),flush=True)

if __name__=='__main__':main()

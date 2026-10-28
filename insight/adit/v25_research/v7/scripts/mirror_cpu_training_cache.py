"""Stream existing private frozen caches; no model process on the full node."""
from pathlib import Path
import hashlib,json,sys,time
from urllib.parse import quote
ROOT=Path(__file__).resolve().parents[2]
V=ROOT/'v7'
sys.path.insert(0,str(ROOT.parent))
from node480_task import client,BASE


def main():
    c=client();c.RETRIES=2
    files=[]
    for fold in range(5):
        files += [f'v7/runs/V25/fold{fold}_correction.joblib']
        files += [f'artifacts/fold{fold}/kumo_{part}.parquet' for part in ['holdout','oof','test']]
        files += [f'artifacts/fold{fold}/spline_{part}.parquet' for part in ['oof','test']]
        files += [f'v7/extension/data/fold{fold}/{part}.parquet' for part in ['correction','validation','test']]
    rows=[]
    for rel in files:
        p=ROOT/rel;assert p.resolve().is_relative_to(ROOT.resolve())
        if not p.exists():
            p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.cpu-mirror.tmp')
            with c.opener.open(c.base_url.rstrip('/')+'/files/'+quote(BASE+rel),timeout=45) as response,tmp.open('wb') as f:
                while True:
                    b=response.read(128*1024)
                    if not b:break
                    f.write(b)
            assert tmp.stat().st_size>0
            tmp.replace(p);print('CPU_CACHE_MIRRORED',rel,p.stat().st_size,flush=True)
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        rows.append(dict(file=rel,bytes=p.stat().st_size,sha256=digest))
    (V/'reports/CPU_CACHE_MIRROR.json').write_text(json.dumps(dict(status='COMPLETE',UTC=time.time(),files=rows,source='Authenticated node480 private cache; stream128KiB',raw_data_publication=False),indent=2),encoding='utf-8')
    print('CPU_CACHE_MIRROR_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()

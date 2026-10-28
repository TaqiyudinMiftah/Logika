from pathlib import Path
import datetime,hashlib,json,os
import numpy as np
R=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def dump(value,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp')
    q.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n');q.replace(p)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def root_path(name):return Path(json.loads((R/'PATHS.json').read_text())[name])
def ks(y,p):
    from sklearn.metrics import roc_curve
    f,t,_=roc_curve(y,p,drop_intermediate=False);return float(np.max(abs(t-f)))

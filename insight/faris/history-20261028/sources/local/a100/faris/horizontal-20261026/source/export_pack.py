"""Export one context-preprocessed pack without evaluation labels or wide bank."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from features import Features, R

p=argparse.ArgumentParser()
p.add_argument('--pack',required=True);p.add_argument('--fold',type=int,required=True)
p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
f=Features(a.fold);pack=next(p for p in f.catalog['packs'] if p['id']==a.pack)
frame,dropped=f.pack(pack);rows=np.r_[f.ix['context'],f.ix['correction'],f.ix['validation']]
frame=frame.iloc[rows].reset_index(drop=True)
for c in frame:
    if not pd.api.types.is_numeric_dtype(frame[c]):
        vocabulary=set(frame[c].iloc[:50000].dropna().astype(str))
        frame[c]=frame[c].astype('string').where(frame[c].astype('string').isin(vocabulary))
frame.to_parquet(out/'frame.parquet',index=False)
np.savez_compressed(out/'roles.npz',rows=rows,y_context=f.ix['y_context'])
manifest={'pack':a.pack,'fold':a.fold,'columns':list(frame),'dropped':dropped,
          'outer_labels_included':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'files':{n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ['frame.parquet','roles.npz']}}
(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'pack':a.pack,'fold':a.fold,'bytes':sum(p.stat().st_size for p in out.iterdir())}))

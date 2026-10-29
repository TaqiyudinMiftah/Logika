"""Regenerate only frozen features, first proving original-membership parity."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import gc,hashlib,importlib.util,json,sys,time
import numpy as np,pandas as pd
R=Path('/mnt/home/faris/logika-overnight-refit-20261030');OLD=Path('/mnt/home/faris/logika-submit-highest-ks-20261029');DEV=Path('/mnt/home/faris/logika-twelvehour-next-20261029/neural-cann');BROAD=Path('/mnt/home/faris/logika-sixhour-broad-20261028')
sys.path.insert(0,str(OLD/'source'))
from prepare import ProductionBank
from common import read,sha
spec=importlib.util.spec_from_file_location('actuarial',BROAD/'source/features_actuarial.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def dump(p,v):p.write_text(json.dumps(v,indent=2,default=str)+'\n')
def build(fold,ctx,parity,columns):
 bank=ProductionBank(fold,ctx,parity=parity);schema=read(BROAD/'FEATURE_SCHEMA.json');definitions={s['id']:s for s in schema['specs']}
 needed=set(columns)-{x for x in columns if x.startswith('a6_')}
 mapping={s['inputs'][0]:s['id']for s in schema['specs']if s['op']=='identity'and len(s['inputs'])==1};mapping.update(schema.get('extra_input_ids',{}))
 for name in mod.required_inputs():
  if name in mapping:needed.add(mapping[name])
 values={}
 for name in needed:
  value=np.asarray(bank.compute(definitions[name]))
  if pd.api.types.is_numeric_dtype(value.dtype):value=value.astype('float32');value[~np.isfinite(value)]=np.nan
  values[name]=value
 frame=pd.DataFrame(values);extra,audit=mod.expand(frame,schema,ctx)
 for c in extra:frame[c]=extra[c]
 assert set(columns).issubset(frame),set(columns)-set(frame)
 return frame[columns],bank.ids,audit

def main():
 R.mkdir(exist_ok=True);(R/'packets').mkdir(exist_ok=True)
 y=pd.read_parquet('/mnt/home/faris/logika-faris-exploration/ranking-pipeline-20261020/input/data/train_features.parquet',columns=['application_id','target'])
 reports=[]
 for f in range(5):
  out=R/f'packets/fold{f}';out.mkdir(exist_ok=True)
  if(out/'MANIFEST.json').exists():reports.append(read(out/'MANIFEST.json'));continue
  m=read(DEV/f'packets/fold{f}/MANIFEST.json');k=read(OLD/f'packets/fold{f}/MANIFEST.json')['columns'];cols=list(dict.fromkeys(m['views']['actuarial99']+k))
  roles=np.load(DEV/f'packets/fold{f}/roles.npz');inner=np.load(DEV/f'packets/fold{f}/inner_roles.npz')['inner_train']
  candidate,ids,audit=build(f,inner,True,cols);reference=pd.read_parquet(DEV/f'packets/fold{f}/features.parquet',columns=m['views']['actuarial99'])
  for c in reference:
   a=candidate[c].iloc[:70000];b=reference[c]
   if pd.api.types.is_numeric_dtype(a.dtype):assert np.array_equal(a.to_numpy(),b.to_numpy(),equal_nan=True),(f,c,'feature parity')
   else:assert a.fillna('__NA__').equals(b.fillna('__NA__')),(f,c,'category parity')
  del candidate,reference;gc.collect()
  frame,ids,audit=build(f,np.arange(70000),False,cols);assert ids[:70000].tolist()==y.application_id.astype(str).tolist()
  frame.to_parquet(out/'features.parquet',index=False);np.save(out/'y.npy',y.target.to_numpy(np.int8));np.save(out/'test_ids.npy',ids[70000:])
  report=dict(status='PASS',fold=f,rows=len(frame),train_rows=70000,test_rows=30000,views={'actuarial99':m['views']['actuarial99'],'global99':k},feature_sha256=sha(out/'features.parquet'),labels_sha256=sha(out/'y.npy'),test_ids_sha256=sha(out/'test_ids.npy'),original_70000_rows_bit_exact_under_original_inner_membership=True,production_statistics_training_only=True,actuarial_audit=audit,source_sha256=sha(Path(__file__)))
  dump(out/'MANIFEST.json',report);reports.append(report);print(json.dumps({'fold':f,'status':'production_parity_pass','columns':len(cols)}),flush=True);del frame;gc.collect()
 dump(R/'PREPARATION_AUDIT.json',dict(status='PASS',folds=reports,query_labels_available=False))
if __name__=='__main__':main()

"""Freeze a small final-recipe comparison and its retained model references."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,sys,hashlib,copy,shutil
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import ks_2samp
ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT.parent/'exploration-12h-20261017'
sys.path.insert(0,str(CAMPAIGN));sys.path.insert(0,str(CAMPAIGN/'source'))
from supervisor import records
from utils import read,write,sha,combine

def ks(y,p):return float(ks_2samp(p[y==0],p[y==1]).statistic)

def main():
 ROOT.joinpath('evidence').mkdir(exist_ok=True)
 if (ROOT/'plan.json').exists():
  print('Already frozen:',read(ROOT/'plan.json')['selected']['id']);return
 candidates=read(ROOT/'candidate_plan.json'); rows=records(); table=read(CAMPAIGN/'reports/frozen_finalists.json');parents=table['nominees'][0]['spec']['parents'];original={r['id']:r for r in rows if r['job']['protocol']=='screen'}
 deployment=read(CAMPAIGN/'deployment.json');bank=Path(deployment['hosts']['pc2']['bank']);meta=pd.read_parquet(bank/'metadata.parquet');dev=np.load(bank/'development_rows.npy');y=np.load(bank/'development_y.npy');fold=np.load(bank/'folds.npy');cfg=read(CAMPAIGN/'source/configurations.json');arrays={};references={}
 for protocol in ['screen','reseed','repeat','confirm']:
  a={}
  for key in parents:
   r=original[key];matches=[v for v in rows if v['host']==r['host'] and v['job']['protocol']==protocol and v['job']['family']==r['job']['family'] and v['job']['params']==r['job']['params']];assert len(matches)==1,(key,protocol,len(matches));rr=matches[0];z=np.load(Path(rr['folder'])/'validation.npz');expected=np.load(bank/'confirmation_rows.npy') if protocol=='confirm' else dev;assert np.array_equal(z['rows'],expected);a[key]=z['prediction'];assert np.isfinite(a[key]).all()
   if protocol=='confirm':references[key]={'folder':rr['folder'],'job':rr['job'],'configuration':cfg.get(rr['job']['params']['config']),'validation_sha256':sha(Path(rr['folder'])/'validation.npz')}
  a['champion']=np.load(CAMPAIGN/'evidence/champion_oof.npz')['existing_feature_ensemble'] if protocol=='screen' else np.load(CAMPAIGN/'reports'/('champion_'+protocol+'.npy'));arrays[protocol]=a
 evaluated=[]
 for spec in candidates['candidates']:
  scores={};consistent=True
  for protocol in ['screen','reseed','repeat']:
   p=combine(spec,arrays[protocol]);b=arrays[protocol]['champion'];f=fold if protocol!='repeat' else np.asarray(read(CAMPAIGN/'evidence/repeat.json')['folds']);delta=ks(y,p)-ks(y,b);ds=[ks(y[f==i],p[f==i])-ks(y[f==i],b[f==i]) for i in range(3)];scores[protocol]={'ks':ks(y,p),'delta':delta,'fold_delta':ds};consistent &= delta>0
   if protocol=='screen':consistent &= sum(d>0 for d in ds)>=2 and min(ds)>=-.005
  evaluated.append({'spec':spec,'development':scores,'eligible':bool(consistent),'mean_development_delta':float(np.mean([v['delta'] for v in scores.values()]))})
 # This is explicit reuse for selection under the new user request, not an independent assessment.
 yc=np.load(bank/'confirmation_y.npy');bc=ks(yc,arrays['confirm']['champion'])
 for result in evaluated:
  if result['eligible']:
   p=combine(result['spec'],arrays['confirm']);result['reused_validation']={'ks':ks(yc,p),'baseline_ks':bc,'delta':ks(yc,p)-bc}
 eligible=[v for v in evaluated if v['eligible']];assert eligible,'No proposed recipe passes the development gates'
 eligible.sort(key=lambda v:(v['reused_validation']['ks'],v['mean_development_delta'],-len(v['spec']['parents'])),reverse=True);selected=eligible[0]
 write(ROOT/'evidence/recipe_comparison.json',{'predeclared_catalogue_sha256':sha(ROOT/'candidate_plan.json'),'evaluated':evaluated,'selected':selected,'confirmation_is_selection_data':True,'independent_private_score_available':False})
 selected_parents=selected['spec']['parents'];selected_arrays={k:arrays['confirm'][k] for k in selected_parents};np.savez_compressed(ROOT/'evidence/selected_validation.npz',rows=np.load(bank/'confirmation_rows.npy'),prediction=combine(selected['spec'],selected_arrays))
 for protocol in ['screen','reseed','repeat']:
  np.savez_compressed(ROOT/'evidence'/('selected_'+protocol+'.npz'),rows=dev,prediction=combine(selected['spec'],arrays[protocol]))
 snapshot=ROOT/'research_source';snapshot.mkdir(exist_ok=True)
 for name,checksum in read(CAMPAIGN/'source_manifest.json').items():
  assert sha(CAMPAIGN/'source'/name)==checksum
  target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(CAMPAIGN/'source'/name,target)
 shutil.copy2(CAMPAIGN/'source/requirements.lock',snapshot/'requirements.lock')
 train=CAMPAIGN/'raw/train_applications.csv';sample=CAMPAIGN/'raw/sample_submission(3).csv';test=CAMPAIGN/'raw/test_applications.csv';manifest=read(bank/'manifest.json');extra=read(CAMPAIGN/'extras/manifest.json')
 for name,checksum in manifest['files'].items():assert sha(bank/name)==checksum,('bank',name)
 for name,checksum in extra['files'].items():assert sha(CAMPAIGN/'extras'/name)==checksum,('extras',name)
 assert sha(train)==manifest['raw_sha256']['train_applications'];assert sha(test)==manifest['raw_sha256']['test_applications'];assert extra['base_bank_id']==manifest['bank_id']
 all_rows=np.flatnonzero(meta.split.to_numpy()=='train');test_rows=np.flatnonzero(meta.split.to_numpy()=='test');labels=pd.read_csv(train).set_index('application_id').target.reindex(meta.application_id.iloc[all_rows]).to_numpy(dtype=int);assert len(all_rows)==70000 and len(test_rows)==30000;assert np.array_equal(labels[pd.Index(all_rows).get_indexer(dev)],y)
 plan={'id':'submission-final-20261018','campaign':str(CAMPAIGN),'bank':str(bank),'extras':str(CAMPAIGN/'extras'),'train':str(train),'test':str(test),'sample':str(sample),'bank_id':manifest['bank_id'],'selected':selected['spec'],'metrics':selected,'references':{k:references[k] for k in selected_parents},'shared_stack_components':['fd1fd49493df80697827','d2f4719a69d44e93a849','eb3419a4bf925b93ca53'],'seed':2026,'inner_folds':3,'inner_seed':2726,'cpu_workers':4,'gpu_uuid':'GPU-24a9fffc-77a7-e8dd-d49b-d6f06cb1966a','gpu_cpus':2,'gpu_memory_fraction':.25,'safety_seconds':7200,'no_upload':True,'confirmation_is_selection_data':True,'input_sha256':{'train':sha(train),'test':sha(test),'sample':sha(sample),'bank_manifest':sha(bank/'manifest.json'),'extras_manifest':sha(CAMPAIGN/'extras/manifest.json')}}
 plan['stream_configs']={k:references[k]['job']['params']['config'] for k in selected_parents if references[k]['configuration'] is not None}
 plan['tabm_config']=next(cid for cid in plan['stream_configs'].values() if cfg[cid]['model']=='tabm')
 source_files=[p for p in snapshot.rglob('*') if p.is_file() and '__pycache__' not in p.parts];write(ROOT/'research_source_manifest.json',{str(p.relative_to(snapshot)):sha(p) for p in source_files});write(ROOT/'plan.json',plan)
 write(ROOT/'input_verification.json',{'passed':True,'bank_id':manifest['bank_id'],'development_rows':len(dev),'full_training_rows':len(all_rows),'test_rows':len(test_rows),'source_manifest_sha256':sha(ROOT/'research_source_manifest.json'),'input_sha256':plan['input_sha256']})
 print(json.dumps({'selected':selected,'candidates':len(evaluated),'eligible':len(eligible),'input_verification':'passed'},indent=2))

if __name__=='__main__':main()

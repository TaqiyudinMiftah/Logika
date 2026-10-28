"""Copy one exact existing baseline into adit; never modify its source project."""
from pathlib import Path
import hashlib,json,os,shutil,sys,importlib.metadata
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,train_test_split
from baseline import ROOT,sha,dump,fingerprint,ids_hash

SOURCE=Path('/home/taqiyudinmiftah/Logika')
OLD=Path('/home/taqiyudinmiftah/adit/Logika/insight/large_latent_reproduction_20261018')
GOAL=SOURCE/'artifacts/optimization/goal054'
NAME='kumo_offset_l20p003_unweighted_d0p5'
CSV='submission_goal054_'+NAME+'_w1.csv'
BACKBONE='cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613'
EXPECTED='5d4d8b81e781354a3d1302aadfe93a1534e42e35279ba4867a7b4d0e1990dd33'


def main():
 (ROOT/'reports').mkdir(exist_ok=True);items=[]
 def progress(stage,**extra):
  v=dict(stage=stage,pid=os.getpid(),UTC=datetime.now(timezone.utc).isoformat(),**extra)
  dump(v,ROOT/'reports/IMPORT_PROGRESS.json');print(json.dumps(v),flush=True)
 def copy(source,destination):
  source=Path(source);dest=ROOT/destination;dest.parent.mkdir(parents=True,exist_ok=True)
  digest=sha(source)
  if dest.exists():assert sha(dest)==digest,f'Existing baseline asset differs:{dest}'
  else:shutil.copy2(source,dest);assert sha(dest)==digest
  items.append(dict(source=str(source),destination=destination,sha256=digest,bytes=dest.stat().st_size))
 def generated(path,source):
  items.append(dict(source=source,destination=path,sha256=sha(ROOT/path),bytes=(ROOT/path).stat().st_size))
 progress('COPYING_EXACT_SUBMISSION')
 assert sha(SOURCE/CSV)==EXPECTED;copy(SOURCE/CSV,'submissions/'+CSV)
 definition=json.loads((GOAL/'cv5_seed31027'/NAME/'definition.json').read_text())
 spline_defs=json.loads((SOURCE/'artifacts/reverse_engineering/validation_definitions.json').read_text());spline=next(v for v in spline_defs if v['name']=='reverse_spline')
 columns=list(dict.fromkeys(definition['context_columns']+spline['columns']+definition['correction_columns']))
 progress('COPYING_FEATURE_VALUES',features=len(columns))
 frame=pd.read_parquet(SOURCE/'data/processed/application_features_reverse.parquet',columns=['application_id','customer_id','target','split']+columns)
 train=frame[frame.split.eq('train')].reset_index(drop=True);test=frame[frame.split.eq('test')].reset_index(drop=True)
 assert len(train)==70000 and len(test)==30000 and train.customer_id.is_unique and test.customer_id.is_unique
 assert set(train.customer_id).isdisjoint(test.customer_id)
 assert fingerprint(train,['application_id','target']+definition['columns'])==definition['training_sha256']
 assert fingerprint(test,['application_id']+definition['columns'])==definition['test_sha256']
 (ROOT/'data').mkdir(exist_ok=True)
 for label,part in [('train',train),('test',test)]:
  p='data/'+label+'_features.parquet';part.to_parquet(ROOT/p,index=False);generated(p,'Exact selected source parquet values')
 copy(SOURCE/'data/raw/sample_submission.csv','data/sample_submission.csv')
 feature_rows=[dict(feature=c,dtype=str(frame[c].dtype),Kumo=c in definition['context_columns'],correction=c in definition['correction_columns'],spline=c in spline['columns']) for c in columns]
 pd.DataFrame(feature_rows).to_csv(ROOT/'data/FEATURES.csv',index=False);generated('data/FEATURES.csv','Source recipe feature roles')
 for name in ['context_correction_definitions.json','submission_metadata.json','independent_verification.json','paired_uncertainty.json','numerical_source_control.json']:
  copy(GOAL/name,'provenance/'+name)
 copy(GOAL/'cv5_seed31027'/NAME/'definition.json','provenance/definition.json')
 for name in ['research_context_correction054.py','check_context_source054.py','uncertainty_goal054.py','research_goal054.py','hybrid_kumo.py','kumo_tabular.py','improve_kumo.py','reverse_synthetic_patterns.py','build_features_v3.py','build_features_v4.py','build_features_v5.py']:
  p=SOURCE/'scripts'/name
  if p.exists():copy(p,'provenance/original_scripts/'+name)
 dictionary=SOURCE/'data/raw/DATA_DICTIONARY.md'
 if dictionary.exists():copy(dictionary,'provenance/DATA_DICTIONARY.md')
 fold_rows=[];y=train.target.to_numpy(int);folds=list(StratifiedKFold(5,shuffle=True,random_state=31027).split(train,y))
 original_spline=SOURCE/'artifacts/optimization/kumo_hybrid/cv5_seed31027/reverse_spline'
 for fold,(ti,vi) in enumerate(folds):
  progress('COPYING_FOLD_ASSETS',fold=fold)
  ci,_=train_test_split(ti,train_size=50000,stratify=y[ti],random_state=2026+fold);hi=np.setdiff1d(ti,ci)
  meta=json.loads((GOAL/'cv5_seed31027'/NAME/f'fold{fold}_metadata.json').read_text());dest=f'artifacts/fold{fold}/'
  for key,indices in [('training',ti),('validation',vi),('context',ci),('correction_training',hi)]:assert ids_hash(train,indices)==meta[key+'_ids_sha256']
  (ROOT/dest).mkdir(parents=True,exist_ok=True);np.savez(ROOT/dest/'indices.npz',train=ti,validation=vi,context=ci,correction=hi);generated(dest+'indices.npz','Reconstructed original fixed split and context sampling; verified ID hashes')
  source=GOAL/'cv5_seed31027'/NAME
  for ending,new in [('model.joblib','correction.joblib'),('metadata.json','metadata.json'),('oof.parquet','final_oof.parquet'),('test.parquet','final_test.parquet')]:copy(source/f'fold{fold}_{ending}',dest+new)
  context=GOAL/'context_holdout_base/cv5_seed31027'
  for kind in ['holdout','oof','test']:copy(context/f'fold{fold}_{kind}.parquet',dest+f'kumo_{kind}.parquet')
  copy(context/f'fold{fold}_metadata.json',dest+'kumo_metadata.json')
  for kind in ['oof','test']:copy(original_spline/f'fold{fold}_{kind}.parquet',dest+f'spline_{kind}.parquet')
  candidates=list(original_spline.glob(f'fold{fold}_*.joblib'));assert len(candidates)==1,(fold,candidates)
  copy(candidates[0],dest+'spline.joblib');copy(original_spline/f'fold{fold}_metadata.json',dest+'spline_metadata.json')
  assert sha(ROOT/dest/'correction.joblib')==meta['model_sha256']
  fold_rows.append(dict(fold=fold,context_rows=len(ci),correction_rows=len(hi),validation_rows=len(vi),metrics=meta['metrics']))
 progress('COPYING_FROZEN_BACKBONE',bytes=854850034)
 copy(OLD/'models/pretrained/large/classifier.pt','checkpoints/kumo_large/classifier.pt')
 assert sha(ROOT/'checkpoints/kumo_large/classifier.pt')==BACKBONE
 packages={p:importlib.metadata.version(p) for p in ['torch','structured-data-models','numpy','pandas','scipy','scikit-learn','joblib','pyarrow','threadpoolctl']}
 (ROOT/'requirements-lock.txt').write_text('\n'.join(k+'=='+v for k,v in packages.items())+'\n')
 dump(dict(python=sys.version,packages=packages),ROOT/'reports/ENVIRONMENT.json')
 receipt=json.loads((GOAL/'submission_metadata.json').read_text());uncertainty=json.loads((GOAL/'paired_uncertainty.json').read_text())
 lock=dict(baseline_id='goal054_offset_unweighted_half',created_UTC=datetime.now(timezone.utc).isoformat(),source_project=str(SOURCE),
  submission_file=CSV,submission_sha256=EXPECTED,public_score=.56470,public_score_source='User report for this exact filename; bytes matched source submission metadata',
  Kumo=dict(size='large',features=definition['context_columns'],frozen_fraction=1.,context_rows=50000,estimators=8,estimator_batch_size=4,query_batch_size=8192,seed='2026+fold',dtype='float16',checkpoint_sha256=BACKBONE),
  correction=dict(features=definition['correction_columns'],rows=6000,penalty=.003,strength=.5,balance='unweighted',spline_knots=3,spline_degree=2,intercept_penalized=False),
  original_spline=dict(features=spline['columns'],rows=56000,weight=.25,specification=spline),
  validation=dict(splits=5,shuffle=True,seed=31027,rows=70000,selected_metrics=receipt['validation'],conditional_uncertainty=uncertainty),
  formula='p_final=.75*sigmoid(logit(clip(p_Kumo,1e-6,1-1e-6))+.5*X_correction@beta)+.25*p_spline; test averages5 folds',
  trusted_point_in_time=False,source_global_reference_assumptions_retained=True,development_folds_reused=True,
  no_new_hyperparameter_or_feature_selection=True,active_experiment_queue=[])
 dump(lock,ROOT/'BASELINE_LOCK.json');dump(dict(artifacts=items,source_project_read_only=True,source_csv_sha256=EXPECTED),ROOT/'reports/IMPORT_MANIFEST.json')
 dump(dict(status='IMPORTED_PENDING_REPLAY',public_score=.56470,OOF=receipt['validation'],folds=fold_rows),ROOT/'reports/BASELINE_METRICS.json')
 progress('IMPORT_COMPLETE',artifacts=len(items),features=len(columns),submission_sha256=EXPECTED)


if __name__=='__main__':main()

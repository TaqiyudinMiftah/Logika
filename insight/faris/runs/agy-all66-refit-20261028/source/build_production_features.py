"""Faithful100k production views with explicit legacy context/query asymmetries.

All64k-cohort priors reuse the verified earlier full refit; this builder never
fits default labels. It preserves separately the mature/frozen and legacy views.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse,ast,gc,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
R=Path(__file__).resolve().parents[1];A=Path('/mnt/home/faris/logika-faris-exploration');F=A/'submission-full-periodicity-20261021';FRZ=Path('/mnt/home/faris/logika-frozen-shadow-20261026');J=Path('/mnt/home/faris/logika-joint-20261028')
import sys
sys.path.insert(0,str(F/'source'))
from maturity_features import MaturityFeatures,ALIASES

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def write(frame,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp.parquet');frame.to_parquet(q,index=False);q.replace(p);return sha(p)
def load_static_assignments(path,names):
 tree=ast.parse(Path(path).read_text());ns={}
 for node in tree.body:
  if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id in names for t in node.targets):exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
 return ns

def main(fold):
 dest=R/f'features/fold{fold}';dest.mkdir(parents=True,exist_ok=True)
 apps=pd.read_parquet(F/'applications.parquet').drop(columns=['row'],errors='ignore');assert len(apps)==100000 and apps.application_id.is_unique
 original=np.load(FRZ/f'frozen/fold{fold}/roles.npz');cor=original['correction'];ctx=np.setdiff1d(np.arange(70000),cor);qrows=np.r_[cor,np.arange(70000,100000)]
 oldfull=np.load(F/f'input/artifacts/fold{fold}/indices.npz');assert np.array_equal(ctx,oldfull['context'])and np.array_equal(cor,oldfull['correction'])
 np.savez_compressed(dest/'roles.npz',context=ctx,correction=cor,test=np.arange(70000,100000),query=qrows)
 small=pd.read_parquet(F/f'feature_build/fold{fold}/features.parquet');assert small.application_id.tolist()==apps.application_id.tolist()and'target'not in small
 complete=json.loads((F/f'feature_build/fold{fold}/complete.json').read_text());assert sha(F/f'feature_build/fold{fold}/features.parquet')==complete['feature_table_sha256']
 cov=pd.read_parquet(F/'coverage.parquet').set_index('application_id').loc[apps.application_id].reset_index(drop=True)
 spectral=pd.read_parquet(A/'shopping-periodicity-fe-20261021/true_features.parquet').set_index('application_id').loc[apps.application_id].drop(columns=['customer_id'],errors='ignore').reset_index(drop=True)
 for c in cov:small[c]=cov[c].to_numpy()
 for c in spectral:small[c]=spectral[c].to_numpy()
 history=pd.read_csv(A/'exploration-12h-20261017/raw/repayment_history.csv',usecols=['customer_id','installment_id','due_date','max_days_past_due'])
 factory=MaturityFeatures('eligible_denominators').fit(apps,history,apps.application_id.iloc[ctx]);mature=factory.transform(apps,history);factory.save(dest/'MATURITY_PARAMETERS.json')
 gate=json.loads((FRZ/'FEATURE_FREEZE.json').read_text());champ=gate['views']['champion90'];head=json.loads((FRZ/'frozen/fold0/MANIFEST.json').read_text())['head_columns'];master=list(dict.fromkeys(champ+head))
 assert len(champ)==90 and len(master)==106
 frozen=small.copy()
 for c in ['re_late_1_posterior','re_late_30_posterior']:frozen[c]=mature[c].to_numpy()
 # Original64k base remains separate. Legacyquery's mature means are applied
 # only to correction/test rows when packets are built below.
 files={}
 files['legacy_original.parquet']=write(small,dest/'legacy_original.parquet')
 files['champion_current.parquet']=write(frozen[master],dest/'champion_current.parquet')
 legacy=small.copy()
 for c in ['re_late_1_posterior','re_late_30_posterior']:legacy.loc[qrows,c]=mature[c].iloc[qrows].to_numpy()
 files['legacy_head_query.parquet']=write(legacy,dest/'legacy_head_query.parquet')
 mat=legacy.copy()
 for c,_,_ in ALIASES:mat[c]=mature[c].mask(small[c].isna()).to_numpy()
 files['legacy_mature_alias.parquet']=write(mat,dest/'legacy_mature_alias.parquet')
 core=legacy.copy()
 for t in [1,30]:core[f'v3_repay_all_late_{t}_posterior_24']=mature[f're_late_{t}_posterior'].to_numpy()
 files['legacy_core_maturity.parquet']=write(core,dest/'legacy_core_maturity.parquet')
 # Reconstruct the exact selectedCB block and oldfloat32/'MISSING' convention.
 static=load_static_assignments(R/'source/vendor/unified_views_original.py',{'cb_Extra_str','cb_Extra_fspec'})
 screen=pd.read_csv(J/'inner_oof_residual_screen.csv');scores=dict(zip(screen.name,screen.max_abs_z));frzcols=pq.read_schema(FRZ/'frozen/fold0/features.parquet').names
 avail=[c for c in pq.read_schema(A/'catboost-convergence-20261021/data/fold0/train.parquet').names if c in pq.read_schema(A/'catboost-convergence-20261021/data/fold0/validation.parquet').names and c not in ['target','application_id']and c not in frzcols]
 ranked=sorted([c for c in avail if c in scores and c not in static['cb_Extra_fspec']],key=lambda c:scores[c],reverse=True)[:160]
 selected=[c for c in dict.fromkeys(static['cb_Extra_str']+static['cb_Extra_fspec']+ranked)if c in avail]
 broadpath=F/f'feature_build/fold{fold}/workspace/data/processed/application_features_v5.parquet';schema=pq.read_schema(broadpath).names
 broad=pd.read_parquet(broadpath,columns=['application_id']+[c for c in selected if c in schema]);broad=broad.set_index('application_id').loc[apps.application_id].reset_index(drop=True)
 cb=pd.DataFrame(index=np.arange(100000))
 for c in selected:
  value=broad[c]if c in broad else small[c]
  if pd.api.types.is_numeric_dtype(value):cb[c]=value.astype(np.float32)
  else:cb[c]=value.astype('string').fillna('MISSING').astype(str)
 raw=np.load(R/'features/raw_domain_features.npz');assert np.array_equal(raw['application_id'],apps.application_id.to_numpy(str));rawframe=pd.DataFrame(raw['data'],columns=[str(c)for c in raw['names']])
 extras=[]
 for path in json.loads(Path('/mnt/home/faris/logika-horizontal-20261026/EXTRA_SOURCES.json').read_text()):
  table=pd.read_parquet(path).set_index('application_id').loc[apps.application_id].drop(columns=['customer_id'],errors='ignore').reset_index(drop=True);table.columns=['hz_'+c for c in table];extras.append(table)
 expanded=pd.concat([frozen[master],rawframe,*extras,cb],axis=1);expanded=expanded.loc[:,~expanded.columns.duplicated()]
 for c in spectral:expanded[c]=spectral[c].to_numpy()
 expanded.insert(0,'customer_id',apps.customer_id.to_numpy());expanded.insert(0,'application_id',apps.application_id.to_numpy())
 files['expanded_base.parquet']=write(expanded,dest/'expanded_base.parquet')
 # Build immediately runnable simplebackbonepackets; no learnedteacher columns
 # are required by these threeviews.
 config=json.loads((R/'ENGINE_CONFIGS.json').read_text())['models'];y=pd.read_parquet(A/'ranking-pipeline-20261020/input/data/train_features.parquet',columns=['target']).target.to_numpy(int)
 variants={'fe_cov':legacy,'fe_mat':mat,'tr_mat_orig':core,'champ_s7731':frozen}
 for model,table in variants.items():
  spec=config[model];columns=spec['columns'];packet=R/f'packets/{model}/fold{fold}';packet.mkdir(parents=True,exist_ok=True)
  context=table.iloc[ctx][columns].reset_index(drop=True).copy();context['target']=y[ctx]
  query=table.iloc[qrows][columns].reset_index(drop=True).copy();assert 'target'not in query
  write(context,packet/'context.parquet');write(query,packet/'query.parquet');np.save(packet/'query_rows.npy',qrows);np.save(packet/'context_rows.npy',ctx)
  seed=spec['seed_base']+spec['fold_seed_increment']*fold
  meta={'model':model,'fold':fold,'engine':spec['engine'],'seed':seed,'num_estimators':8,'columns':columns,'n_context':len(ctx),'n_query':len(qrows),'roles_sha256':sha(dest/'roles.npz'),'context_sha256':sha(packet/'context.parquet'),'query_sha256':sha(packet/'query.parquet'),'query_rows_sha256':sha(packet/'query_rows.npy'),'fitted_weights_saved':False,'policy':'64kcontext excluding original6kcorrection; correction+30ktestquery; exact historicalperview maturity overlays preserved'}
  dump(packet/'meta.json',meta)
 # Specialists use the same fixedview definitions and C2augmentation rule.
 spec=config['pop_spec'];is_a=expanded['pop_A'].to_numpy()>0.5;is_c2=expanded['pop_C2'].to_numpy()>0.5;is_bc=(~is_a)&(~is_c2)
 for part,mask in [('A',is_a),('BC1',is_bc),('C2',is_c2)]:
  selected_ctx=ctx[mask[ctx]];selected_query=qrows[mask[qrows]]
  if part=='BC1':
   pool=ctx[is_c2[ctx]];rank=np.argsort(expanded['repayment_installment_count'].to_numpy()[pool],kind='mergesort')[:5044];selected_ctx=np.r_[selected_ctx,pool[rank]]
  part_spec=spec['specialists'][part];columns=part_spec['columns'];packet=R/f'packets/pop_spec/fold{fold}/{part}';packet.mkdir(parents=True,exist_ok=True)
  context=expanded.iloc[selected_ctx][columns].reset_index(drop=True).copy();context['target']=y[selected_ctx];query=expanded.iloc[selected_query][columns].reset_index(drop=True).copy()
  write(context,packet/'context.parquet');write(query,packet/'query.parquet');np.save(packet/'context_rows.npy',selected_ctx);np.save(packet/'query_rows.npy',selected_query)
  dump(packet/'meta.json',{'model':'pop_spec','part':part,'fold':fold,'engine':spec['engine'],'seed':part_spec['seed_base']+fold*spec['fold_seed_increment'],'num_estimators':8,'columns':columns,'n_context':len(selected_ctx),'n_query':len(selected_query),'context_sha256':sha(packet/'context.parquet'),'query_sha256':sha(packet/'query.parquet'),'context_rows_sha256':sha(packet/'context_rows.npy'),'query_rows_sha256':sha(packet/'query_rows.npy'),'fitted_weights_saved':False,'C2_near_boundary_augmentation':5044 if part=='BC1' else 0})
 manifest={'status':'base_features_and_simple_packets_ready','fold':fold,'rows':100000,'context_rows':64000,'correction_rows':6000,'test_rows':30000,'learned_features_pending':True,'source_sha256':sha(__file__),'raw_parity_sha256':sha(R/'features/RAW_PARITY.json'),'roles_sha256':sha(dest/'roles.npz'),'files':files,'selected_cb_columns':selected,'master_columns':master,'feature_fit_membership':complete['fit_ids_sha256'],'target_labels_used_for_features':False,'no_old_campaigns_modified':True}
 dump(dest/'BASE_MANIFEST.json',manifest);print(json.dumps({'stage':'base_features_ready','fold':fold,'columns':len(expanded.columns),'packets':list(variants)}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);main(p.parse_args().fold)

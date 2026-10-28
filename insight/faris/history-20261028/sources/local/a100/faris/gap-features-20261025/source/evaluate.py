"""Frozen matched feature experiments; original five folds, two cached Kumo seeds."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
import sys,json,time,datetime,hashlib,shutil,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
R=Path(__file__).resolve().parents[1]
OLD=Path('/mnt/home/logika-faris-exploration/transfer-20261025')
sys.path.insert(0,str(OLD/'source'))
import robustness as rb
import baseline as b
import feature_head,spline_rolebag

def dump(x,p): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def guard(start):
    assert not (R/'STOP').exists()
    plan=json.loads((R/'PLAN.json').read_text())
    prior=json.loads((R/'features/GATE.json').read_text())['seconds']
    assert time.time()-start+prior < plan['research_budget']['max_elapsed_compute_seconds']-plan['research_budget']['verification_refit_reserve_seconds']
    assert shutil.disk_usage(R).free>plan['resource_caps']['minimum_free_bytes']

def run():
    start=time.time(); gate=json.loads((R/'features/GATE.json').read_text()); assert gate['status']=='passed'
    for name,digest in gate['files'].items(): assert b.sha(R/'features'/name)==digest
    true=pd.read_parquet(R/'features/true.parquet').set_index('application_id')
    null=pd.read_parquet(R/'features/alignment_null.parquet').set_index('application_id')
    ref=np.load(OLD/'ROBUSTNESS_OOF.npz'); y=ref['target']; folds=ref['fold']
    cases={**gate['families'],'alignment_null':gate['families']['alignment']}
    out=R/'matched';out.mkdir(exist_ok=False)
    predictions={f'{c}__{s}':np.full(len(y),np.nan) for c in cases for s in ['original','fresh','mean']}
    replay=[];diagnostics={}
    dump({'started_UTC':now(),'source_sha256':b.sha(__file__),'cases':cases,'seeds':['original','fresh'],
          'outer_labels_for_fit':False,'models':'B maturity head18 plus 8 fields, spline77 plus8; raw Kumo unchanged',
          'primary':'mean of two Kumo seeds','bootstrap':'1000 paired stratified draws, fixed seed251025; per-family 98.333% interval (3 real families)',
          'status':'running'},out/'PLAN.json')
    for f in range(5):
        guard(start)
        original,ix,q,context,yc,core,raw,frame,mature,models=rb.inputs(f)
        assert np.array_equal(original.target,y)
        query,table=rb.stages(q,frame,mature,'maturity')
        oldhead,oldsp=models['maturity']; vp=ix['validation']
        fr=np.load(OLD/f'robustness/fold{f}/raw_seed_repeat.npz')
        assert np.array_equal(fr['rows'],np.r_[ix['correction'],vp])
        bases={'original':raw,'fresh':fr['prediction']}
        spbase=b.predict_spline(oldsp,table.iloc[vp])
        for seed,base in bases.items():
            h=oldhead if seed=='original' else joblib.load(OLD/f'robustness/fold{f}/maturity__fresh_kumo_seed.joblib')
            p=.75*b.corrected_probability(h,query.iloc[6000:],base[6000:])[0]+.25*spbase
            error=float(abs(p-ref['maturity__original' if seed=='original' else 'maturity__fresh_kumo_seed'][vp]).max())
            assert error<2e-14,(f,seed,error)
            replay.append({'fold':f,'seed':seed,'max_error':error})
        if f==0:
            # Fixed descriptive bins; no feature selection from these diagnostics.
            x=true.loc[original.application_id]
            for c in sum(gate['families'].values(),[]):
                z=x[c].to_numpy();mask=np.isfinite(z);bins=pd.qcut(pd.Series(z),5,duplicates='drop')
                d=pd.DataFrame({'bin':bins.astype(str),'y':y,'residual':y-ref['maturity__two_seed_mean']})
                diagnostics[c]={'available':int(mask.sum()),'unique':int(pd.Series(z).nunique()),
                    'bins':d.groupby('bin',observed=True).agg(n=('y','size'),positives=('y','sum'),residual=('residual','mean')).reset_index().to_dict('records')}
            dump({'kind':'reused-development descriptive residual EDA; all families frozen first','fields':diagnostics},R/'EDA.json')
        for case,cols in cases.items():
            guard(start); dest=out/case/f'fold{f}';dest.mkdir(parents=True)
            feature=null if case=='alignment_null' else true
            qt=query.copy();ft=table.copy()
            for c in cols:
                qt[c]=feature.loc[qt.application_id,c].to_numpy()
                ft[c]=feature.loc[ft.application_id,c].to_numpy()
            training=ft.iloc[ix['train']].copy();training['target']=y[ix['train']]
            sp=spline_rolebag.fit(training,oldsp['columns']+cols)
            joblib.dump(sp,dest/'spline.joblib',compress=3)
            spred=b.predict_spline(sp,ft.iloc[vp])
            assert np.array_equal(spred,b.predict_spline(joblib.load(dest/'spline.joblib'),ft.iloc[vp]))
            for seed,base in bases.items():
                training=qt.iloc[:6000].copy();training['target']=yc
                head=feature_head.fit(training,base[:6000],oldhead['columns']+cols,.003,.5)
                joblib.dump(head,dest/f'head_{seed}.joblib',compress=3)
                p=.75*b.corrected_probability(head,qt.iloc[6000:],base[6000:])[0]+.25*spred
                loaded=joblib.load(dest/f'head_{seed}.joblib')
                assert np.array_equal(p,.75*b.corrected_probability(loaded,qt.iloc[6000:],base[6000:])[0]+.25*spred)
                predictions[f'{case}__{seed}'][vp]=p
            predictions[f'{case}__mean'][vp]=.5*(predictions[f'{case}__original'][vp]+predictions[f'{case}__fresh'][vp])
            np.savez_compressed(dest/'predictions.npz',rows=vp,**{s:predictions[f'{case}__{s}'][vp] for s in ['original','fresh','mean']})
            dump({'status':'complete','fold':f,'case':case,'validation_labels_in_fit':False,
                'files':{p.name:b.sha(p)for p in dest.iterdir()},'completed_UTC':now()},dest/'complete.json')
            print(json.dumps({'stage':'fits_complete','case':case,'fold':f,'elapsed_seconds':time.time()-start}),flush=True)
    metrics={}
    basemap={'original':'maturity__original','fresh':'maturity__fresh_kumo_seed','mean':'maturity__two_seed_mean'}
    for name,p in predictions.items():
        assert np.isfinite(p).all()
        case,seed=name.split('__');base=ref[basemap[seed]]
        metrics[name]={**b.scores(y,p),'delta_KS':b.scores(y,p)['KS']-b.scores(y,base)['KS'],
            'fold_deltas':[b.scores(y[folds==f],p[folds==f])['KS']-b.scores(y[folds==f],base[folds==f])['KS']for f in range(5)]}
    np.savez_compressed(R/'OOF.npz',target=y,fold=folds,**predictions)
    dump({'status':'fits_complete','metrics':metrics,'control_replay':replay,'independent_confirmation':False,'seconds':time.time()-start},R/'RESULT.json')
    print(json.dumps(metrics,indent=2),flush=True)
    dump({'feature_build_seconds':gate['seconds'],'matched_fit_seconds':time.time()-start,
          'union_elapsed_compute_seconds':gate['seconds']+time.time()-start,'completed_UTC':now()},R/'COMPUTE_LEDGER.json')

if __name__=='__main__':
    try:run()
    except BaseException:
        dump({'error':traceback.format_exc(),'UTC':now()},R/f'FAILURE_{int(time.time())}.json');raise

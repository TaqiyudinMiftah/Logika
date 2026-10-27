"""Re-use frozen Small context scores for matched capacity ablations."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,io,json,sys,traceback
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path[:0]=[str(ROOT/'scripts'),str(V/'scripts')]
from baseline import dump,sha,corrected_probability
from experiments import fit_head
from conservative_adapters import main as evaluate,SPECTRAL
from stability_worker import postprocess,alternate_head


def small_cache(f):
    frames=[];checks=[]
    for part,key in [('holdout','h'),('oof','val'),('test','test')]:
        p=V/'runs/V207'/f"fold{f['fold']}_kumo_{part}.parquet";d=pd.read_parquet(p);assert d.application_id.tolist()==f[key].application_id.tolist();frames.append(d);checks.append(dict(file=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
    d=pd.concat(frames);assert d.application_id.is_unique
    return d.set_index('application_id').prediction,checks


def fit_adapter(f,spec,columns):
    model=dict(kind=spec['method'],spec=spec,fold=f['fold'])
    if spec.get('fixed_score_fusion'):return model,dict(gradient_max_abs=0.,iterations=0),0
    if f.get('synthetic_preflight'):
        assert f['h'].application_id.str.startswith('SYNTHETIC_').all();p=f['synthetic_small'];checks=[]
    else:p,checks=small_cache(f)
    k=p.reindex(f['h'].application_id).to_numpy(float);model['raw_small']=p
    if spec['method']=='small_reduced_knots':head=alternate_head(f['h'],k,columns,spec)
    else:head=fit_head(f['h'],k,columns,.003,.5)
    model['head']=head
    return model,dict(gradient_max_abs=head['gradient_max_abs'],iterations=head['iterations'],raw_small_cache=checks),len(head['coefficients'])


def probability(model,frame,k,head):
    if model['spec'].get('fixed_score_fusion'):return corrected_probability(head,frame,k)
    p=model['raw_small'].reindex(frame.application_id).to_numpy(float);assert np.isfinite(p).all()
    return corrected_probability(model['head'],frame,p)


def final_prediction(model,f,p,tp):
    if not model['spec'].get('fixed_score_fusion'):return p,tp
    output=[]
    for part,key in [('validation','val'),('test','test')]:
        spec=model['spec'];sources=spec['sources'];values={};hashes={}
        for source in sources:
            path=V/'runs'/source/f"fold{f['fold']}_{part}.parquet";d=pd.read_parquet(path);assert d.application_id.tolist()==f[key].application_id.tolist();values[source]=d.prediction.to_numpy();hashes[source]=sha(path)
        if spec['method']=='capacity_coverage_blend':
            n=f[key].aux_allocation_count.to_numpy(float);n=np.where(np.isfinite(n)&(n>0),n,0.);w=.15*10/(n+10)
            pred=(1-w)*values['V193']+w*values['V207']
        else:pred=sum(weight*values[source] for source,weight in zip(sources,spec['weights']))
        output.append(pred);model['sources_'+part+'_sha256']=hashes
    return tuple(output)


def preflight():
    rng=np.random.default_rng(55027);n=1000;k=np.clip(rng.beta(2,8,n),.001,.999)
    h=pd.DataFrame(dict(application_id=[f'SYNTHETIC_{i}' for i in range(n)],a=rng.lognormal(size=n),employment_type=np.resize(['salaried','self_employed','variable_income'],n),target=rng.binomial(1,k)))
    for c in SPECTRAL:h[c]=rng.random(n)
    f=dict(h=h,kh=k,fold=0,synthetic_preflight=True,synthetic_small=pd.Series(k,index=h.application_id));base=fit_head(h,k,['a','employment_type'],.003,.5);checks=[]
    for spec in json.loads((V/'capacity_plan.json').read_text()):
        if spec['worker']!='capacity_followups.py' or spec.get('fixed_score_fusion'):continue
        columns=['a','employment_type']+(SPECTRAL if spec.get('spectral_inputs')=='stable9' else []);m,audit,count=fit_adapter(f,spec,columns);p,_=probability(m,h.iloc[:50],k[:50],base);assert np.isfinite(p).all()
        stream=io.BytesIO();joblib.dump(m,stream);stream.seek(0);again,_=probability(joblib.load(stream),h.iloc[:50],k[:50],base);assert np.array_equal(p,again);checks.append(dict(method=spec['method'],parameters=count,replay_exact=True))
    n=np.array([0.,1.,10.,100.]);w=.15*10/(n+10);assert np.all(w>0)&np.all(w<=.15)&np.all(np.diff(w)<0)
    dump(dict(status='PASS',synthetic_only=True,checks=checks,source_sha256=sha(__file__)),V/'capacity/reports/PREFLIGHT.json');print('CAPACITY_PREFLIGHT_PASS',flush=True)


def main(ident):
    spec=next(s for s in json.loads((V/'capacity_plan.json').read_text()) if s['id']==ident)
    for source in spec.get('depends_on',[]):assert json.loads((V/'runs'/source/'RESULT.json').read_text())['status']=='COMPLETE'
    pre=json.loads((V/'capacity/reports/PREFLIGHT.json').read_text());assert pre['status']=='PASS' and pre['source_sha256']==sha(__file__)
    methods=[spec['method']] if spec.get('spectral_inputs')=='stable9' else []
    return evaluate(ident,plan_file='capacity_plan.json',adapter_fit=fit_adapter,adapter_probability=probability,method_formulas={spec['method']:spec['formula']},spectral_methods=methods,spectral_columns=SPECTRAL,worker_source='capacity_followups.py',final_prediction=final_prediction,postprocess=postprocess)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--preflight',action='store_true');args=p.parse_args()
    try:
        if args.preflight:preflight()
        else:assert args.run;main(args.run)
    except Exception:
        if args.run:dump(dict(status='FAILED',traceback=traceback.format_exc()),V/'runs'/args.run/'STATUS.json')
        raise

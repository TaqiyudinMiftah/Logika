"""Compare complementary frozen population contexts before correction."""
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
from conservative_adapters import main as evaluate
from stability_worker import postprocess


def fit_adapter(f,spec,columns):
    if f.get('synthetic_preflight'):raw=f['synthetic_contexts'];checks=[]
    else:
        parts=[];checks=[]
        for part,key in [('holdout','h'),('oof','val'),('test','test')]:
            scores=[]
            for source in spec['raw_sources']:
                result=json.loads((V/'runs'/source/'RESULT.json').read_text());assert result['frozen_Kumo_fraction']==1. and result['all_outer_indices_identical']
                p=V/'runs'/source/f"fold{f['fold']}_kumo_{part}.parquet";d=pd.read_parquet(p);assert d.application_id.tolist()==f[key].application_id.tolist();scores.append(d.prediction.to_numpy());checks.append(dict(file=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
            parts.append(pd.Series(np.mean(scores,axis=0),index=f[key].application_id))
        raw=pd.concat(parts);assert raw.index.is_unique
    h=raw.reindex(f['h'].application_id).to_numpy(float);assert np.isfinite(h).all();head=fit_head(f['h'],h,columns,.003,.5)
    return dict(raw_context_prediction=raw,head=head,raw_sources=spec['raw_sources']),dict(gradient_max_abs=head['gradient_max_abs'],iterations=head['iterations'],raw_source_checks=checks),len(head['coefficients'])


def probability(model,frame,k,base_head):
    raw=model['raw_context_prediction'].reindex(frame.application_id).to_numpy(float);assert np.isfinite(raw).all();return corrected_probability(model['head'],frame,raw)


def preflight():
    rng=np.random.default_rng(58027);n=1000;k=rng.beta(2,8,n);h=pd.DataFrame(dict(application_id=[f'SYNTHETIC_{i}' for i in range(n)],a=rng.lognormal(size=n),employment_type=np.resize(['salaried','self_employed'],n),target=rng.binomial(1,k)));f=dict(h=h,synthetic_preflight=True,synthetic_contexts=pd.Series(k,index=h.application_id))
    specifications=json.loads((V/'population_fusion_plan.json').read_text());assert all(len(a['raw_sources'])>=2 and len(set(a['raw_sources']))==len(a['raw_sources']) for a in specifications)
    assert np.allclose(np.mean([[.1,.8],[.3,.4],[.2,.6]],axis=0),[.2,.6])
    spec=specifications[0];model,audit,count=fit_adapter(f,spec,['a','employment_type']);p,_=probability(model,h,k,None);assert np.isfinite(p).all();buf=io.BytesIO();joblib.dump(model,buf);buf.seek(0);q,_=probability(joblib.load(buf),h,k,None);assert np.array_equal(p,q)
    dump(dict(status='PASS',synthetic_only=True,checkpoint_replay_exact=True,source_sha256=sha(__file__)),V/'population_fusion/reports/PREFLIGHT.json');print('POPULATION_FUSION_PREFLIGHT_PASS',flush=True)


def main(ident):
    spec=next(s for s in json.loads((V/'population_fusion_plan.json').read_text()) if s['id']==ident);pre=json.loads((V/'population_fusion/reports/PREFLIGHT.json').read_text());assert pre['status']=='PASS' and pre['source_sha256']==sha(__file__)
    return evaluate(ident,plan_file='population_fusion_plan.json',adapter_fit=fit_adapter,adapter_probability=probability,method_formulas={spec['method']:spec['formula']},spectral_methods=[],worker_source='population_fusion.py',postprocess=postprocess)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--preflight',action='store_true');args=p.parse_args()
    try:
        if args.preflight:preflight()
        else:assert args.run;main(args.run)
    except Exception:
        if args.run:dump(dict(status='FAILED',traceback=traceback.format_exc()),V/'runs'/args.run/'STATUS.json')
        raise

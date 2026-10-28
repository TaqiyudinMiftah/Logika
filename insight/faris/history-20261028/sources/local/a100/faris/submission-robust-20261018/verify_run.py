"""Independently reconstruct fit boundaries and transferred artifact integrity."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import hashlib,json,re
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parent


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ident(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def main():
    plan=read(ROOT/'benchmark_plan.json');local_data=ROOT.parents[2]/'dataset'
    train_path=local_data/'train_applications.csv' if local_data.exists() else Path(plan['train'])
    train=pd.read_csv(train_path);y=train.target.to_numpy(dtype=np.int64);rows=np.arange(len(train))
    assert len(rows)==70000 and sha(train_path)==plan['input_sha256']['train']
    for name,h in plan['source_manifest'].items():assert sha(ROOT/'source'/name)==h
    for name,h in plan['code_sha256'].items():assert sha(ROOT/name)==h
    outer=list(StratifiedKFold(3,shuffle=True,random_state=20261019).split(rows,y))
    count=0;files=0
    for folder in sorted((ROOT/'fits').iterdir()):
        if not folder.is_dir():continue
        request=read(folder/'request.json');complete=read(folder/'complete.json')
        match=re.fullmatch(r'outer(\d)_(inner(\d)|refit)_(champion|addition)(\d)',folder.name)
        if match:
            f=int(match[1]);ti,vi=outer[f]
            if match[3] is not None:
                g=int(match[3]);inner=list(StratifiedKFold(3,shuffle=True,random_state=20261019+117*(f+1)).split(ti,y[ti]));a,b=inner[g];fitrows=rows[ti[a]];query=rows[ti[b]]
                assert not np.intersect1d(fitrows,rows[vi]).size
                assert not np.intersect1d(query,rows[vi]).size
            else:fitrows=rows[ti];query=rows[vi]
        elif folder.name.startswith('final_addition'):
            fitrows=rows;query=np.arange(70000,100000)
        else:raise AssertionError(('Unknown fit',folder.name))
        assert not np.intersect1d(fitrows,query).size
        assert request['fit_rows']==ident(fitrows.tolist())
        assert request['query_rows']==ident(query.tolist())
        assert request['labels']==ident(y[fitrows].tolist())
        assert request['plan_sha']==sha(ROOT/'benchmark_plan.json')
        prediction=np.load(folder/'prediction.npz');assert np.array_equal(prediction['rows'],query)
        assert len(prediction['prediction'])==len(query) and np.isfinite(prediction['prediction']).all()
        for name,h in complete['files'].items():assert sha(folder/name)==h;files+=1
        count+=1
    expected=(len(plan['champion_components'])+len(plan['candidate_components']))*12+1
    assert count==expected,('Missing or unexpected fitted jobs',count,expected)
    z=np.load(ROOT/'evidence/nested_oof.npz');assert np.array_equal(z['rows'],rows) and np.array_equal(z['target'],y)
    assert np.isfinite(z['prediction']).all()
    for f,(_,vi) in enumerate(outer):assert (z['fold'][vi]==f).all()
    complete=read(ROOT/'artifacts/complete.json')
    secondary=complete.get('decision_type')=='fixed_secondary_candidate'
    if secondary:
        chosen=read(ROOT/'fixed_comparison.json')['selected']
        assert chosen==complete['addition']
        fixed=np.load(ROOT/'evidence/fixed_selected_oof.npz')
        for key in ['rows','target','fold']:assert np.array_equal(fixed[key],z[key])
        expected_fixed=(1-chosen['weight'])*z['champion']+chosen['weight']*z[chosen['component']]
        assert np.max(np.abs(expected_fixed-fixed['prediction']))<1e-12
        assert read(ROOT/'artifacts/pipeline_index.json')['fixed_comparison_sha256']==sha(ROOT/'fixed_comparison.json')
    for name,h in complete['models'].items():assert sha(ROOT/name)==h
    sample_path=local_data/'sample_submission(3).csv' if local_data.exists() else Path(plan['sample'])
    sample=pd.read_csv(sample_path);csv=pd.read_csv(ROOT/'artifacts/submission.csv')
    assert len(csv)==30000 and csv.columns.tolist()==sample.columns.tolist()
    assert csv.application_id.tolist()==sample.application_id.tolist() and csv.application_id.is_unique
    assert np.isfinite(csv.default_probability).all() and csv.default_probability.between(0,1).all()
    assert sha(ROOT/'artifacts/submission.csv')==complete['submission_sha256']
    streams=np.load(ROOT/'artifacts/streams.npz');test_path=local_data/'test_applications.csv' if local_data.exists() else Path(plan['test'])
    test=pd.read_csv(test_path);ix=pd.Index(test.application_id).get_indexer(csv.application_id);assert (ix>=0).all()
    w=complete['addition']['weight'];p=(1-w)*streams['champion']+w*streams['addition']
    error=float(np.max(np.abs(p[ix]-csv.default_probability.to_numpy())));assert error<1e-12
    result={'passed':True,'verified_fits':count,'verified_fit_artifact_files':files,'retained_model_files':len(complete['models']),
        'outer_validation_excluded_from_all_inner_fits':True,'fit_and_query_rows_disjoint':True,
        'full_nested_prediction_coverage':70000,'prediction_rows':30000,'sample_order_verified':True,
        'reaggregation_max_error':error,'all_models_remotely_replayed':complete['all_rows_model_replay'],
        'submission_sha256':complete['submission_sha256'],'historical_selection_reuse':True,
        'fixed_secondary_oof_matches_selected_recipe':secondary,'primary_nested_gate_passed':complete['primary_nested_gate_passed'],
        'quality_status':complete['quality_status'],'no_upload':True}
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

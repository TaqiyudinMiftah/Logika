"""Independent post-run evidence audit; never tunes or trains candidates."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp,rankdata
from scipy.special import logit

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'source'))
import baseline as baseline

def read(path):return json.loads(Path(path).read_text())
def sha(path):return baseline.sha(path)
def ids(frame,rows):return hashlib.sha256('\n'.join(frame.application_id.iloc[rows]).encode()).hexdigest()

def verify():
    state=read(root/'state.json');assert state['status']=='complete','Not complete: retain partial evidence'
    result=read(root/'result.json');plan=read(root/'plan.json');inp=Path(plan['input'])
    seal=read(root/'seal.json');hashes=0
    for name,digest in seal['files'].items():assert sha(root/name)==digest;hashes+=1
    for name,digest in seal['inputs'].items():assert sha(Path(name))==digest;hashes+=1
    data=pd.read_parquet(inp/'data/train_features.parquet');y=data.target.to_numpy(int)
    pooled=np.load(root/'oof.npz');seen=np.zeros(len(data),int)
    assert len(data)==70000 and data.application_id.is_unique
    assert np.array_equal(y,pooled['target'])
    teachers=0;contexts=0;max_head_replay_error=0.
    for fold in range(5):
        directory=root/f'folds/fold{fold}';ix=dict(np.load(inp/f'artifacts/fold{fold}/indices.npz'))
        context,correction,validation=[ix[key] for key in ['context','correction','validation']]
        query=np.r_[correction,validation]
        assert not np.intersect1d(context,query).size
        assert not np.intersect1d(correction,validation).size
        for name,digest in read(directory/'complete.json')['files'].items():
            assert sha(directory/name)==digest;hashes+=1
        feature=np.load(directory/'teacher_features.npz')
        assert np.array_equal(feature['context'],context) and np.array_equal(feature['query'],query)
        mapping=pd.Series(np.arange(len(context)),index=context)
        inner_seen=np.zeros(len(context),int)
        for part in [*[f'inner{j}' for j in range(5)],'full']:
            if part=='full':fitrows=context;queryrows=query
            else:
                ix_inner=np.load(directory/(part+'_indices.npz'))
                fitrows=ix_inner['fit'];queryrows=ix_inner['query']
                assert len(fitrows)==40000 and len(queryrows)==10000
                assert np.array_equal(np.sort(np.r_[fitrows,queryrows]),np.sort(context))
                positions=mapping.loc[queryrows].to_numpy();inner_seen[positions]+=1
                assert np.all(feature['context_inner_fold'][positions]==int(part[5:]))
            assert not np.intersect1d(fitrows,queryrows).size
            assert not np.intersect1d(fitrows,query).size
            for component in range(5):
                folder=directory/f'teachers/{part}/teacher{component}'
                meta=read(folder/'complete.json')
                for name,digest in meta['files'].items():assert sha(folder/name)==digest;hashes+=1
                bundle=joblib.load(folder/'model.joblib')
                assert np.array_equal(bundle['preprocessing'].fit_rows,fitrows)
                boundary=read(folder/'boundary.json')
                assert boundary['fit_ids_sha']==ids(data,fitrows)
                assert boundary['query_ids_sha']==ids(data,queryrows)
                prediction=np.load(folder/'prediction.npz')
                assert np.array_equal(prediction['rows'],queryrows)
                expected=(feature['query_predictions'][:,component] if part=='full'
                          else feature['context_predictions'][positions,component])
                assert np.array_equal(expected,prediction['prediction'])
                teachers+=1
        assert np.all(inner_seen==1)
        for prefix in ['context','query']:
            probabilities=feature[prefix+'_predictions']
            trimmed=np.sort(probabilities,axis=1)[:,1:4].mean(axis=1)
            expected=np.column_stack([logit(np.clip(trimmed,1e-6,1-1e-6)),
                logit(np.clip(probabilities,1e-6,1-1e-6)).std(axis=1)]).astype(np.float32)
            assert np.array_equal(expected,feature[prefix+'_features'])
        null=np.load(directory/'null_indices.npz');permutation=null['context_permutation']
        assert np.array_equal(np.sort(permutation),np.arange(len(context)))
        labels=feature['context_inner_fold']
        assert np.array_equal(labels,labels[permutation]),'Null donor/recipient teacher boundary'
        spline=pd.read_parquet(inp/f'artifacts/fold{fold}/spline_oof.parquet').prediction.to_numpy()
        for variant in ['signal','null']:
            folder=directory/variant
            for name,digest in read(folder/'complete.json')['files'].items():assert sha(folder/name)==digest;hashes+=1
            assert (folder/'fitted_context.pt').stat().st_size>0
            boundary=read(folder/'boundary.json')
            assert boundary['context_ids_sha']==ids(data,context)
            assert boundary['correction_ids_sha']==ids(data,correction)
            assert boundary['validation_ids_sha']==ids(data,validation)
            pred=np.load(folder/'prediction.npz');assert np.array_equal(pred['rows'],query)
            head=joblib.load(folder/'correction.joblib')
            repeated=.75*baseline.corrected_probability(head,data.iloc[validation].drop(columns='target'),
                pred['raw'][6000:])[0]+.25*spline
            error=float(np.max(abs(repeated-pred['final'])));assert error<1e-12
            max_head_replay_error=max(max_head_replay_error,error)
            assert np.array_equal(pred['final'],pooled[variant][validation]);contexts+=1
        assert np.all(pooled['fold'][validation]==fold);seen[validation]+=1
    assert np.all(seen==1) and teachers==150 and contexts==10
    independently_recomputed={}
    for name in ['baseline','historical','teacher','signal','null']:
        pred=pooled[name];assert np.isfinite(pred).all()
        ks=float(ks_2samp(pred[y==1],pred[y==0]).statistic)
        n1=y.sum();auc=float((rankdata(pred)[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1)))
        assert abs(ks-result['metrics'][name]['KS'])<1e-12
        assert abs(auc-result['metrics'][name]['AUC'])<1e-12
        independently_recomputed[name]=dict(KS=ks,AUC=auc)
    output=dict(status='passed',hashes_verified=hashes,fitted_teachers=teachers,fitted_Kumo_contexts=contexts,
        distinct_outer_OOF_rows=int((seen==1).sum()),null_context_recipient_boundaries_checked=250000,
        max_full_head_replay_error=max_head_replay_error,independent_metrics=independently_recomputed,
        KS_threshold_reached=independently_recomputed['signal']['KS']>=.54,
        point_target_with_AUC_safeguard=(independently_recomputed['signal']['KS']>=.54 and
            independently_recomputed['signal']['AUC']>=independently_recomputed['historical']['AUC']),
        historical_label_reuse=True,does_not_certify_fresh_generalization=True,
        Kumo_cache_replay_scope='Preflight exercised actual cache reload; main full caches and their inputs verified by hash and provenance, not rerun by this CPU audit.')
    (root/'INDEPENDENT_AUDIT.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':verify()
